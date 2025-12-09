/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package datastore

import (
	"context"
	"errors"
	"fmt"
	"net"
	"reflect"
	"strconv"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	podutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/pod"
)

var (
	errPoolNotSynced = errors.New("InferencePool is not initialized in data store")
	AllPodsPredicate = func(_ datalayer.Endpoint) bool { return true }
)

// The datastore is a local cache of relevant data for the given InferencePool (currently all pulled from k8s-api)
type Datastore interface {
	// InferencePool operations
	// PoolSet sets the given pool in datastore. If the given pool has different label selector than the previous pool
	// that was stored, the function triggers a resync of the pods to keep the datastore updated. If the given pool
	// is nil, this call triggers the datastore.Clear() function.
	PoolSet(ctx context.Context, reader client.Reader, endpointPool *datalayer.EndpointPool) error
	PoolGet() (*datalayer.EndpointPool, error)
	PoolHasSynced() bool
	PoolLabelsMatch(podLabels map[string]string) bool

	// InferenceObjective operations
	ObjectiveSet(infObjective *v1alpha2.InferenceObjective)
	ObjectiveGet(objectiveName string) *v1alpha2.InferenceObjective
	ObjectiveDelete(namespacedName types.NamespacedName)
	ObjectiveGetAll() []*v1alpha2.InferenceObjective

	// InferenceModelRewrite operations
	ModelRewriteSet(infModelRewrite *v1alpha2.InferenceModelRewrite)
	ModelRewriteDelete(namespacedName types.NamespacedName)
	ModelRewriteGet(modelName string) (*v1alpha2.InferenceModelRewriteRule, string)
	ModelRewriteGetAll() []*v1alpha2.InferenceModelRewrite

	// PodList lists pods matching the given predicate.
	PodList(predicate func(backendmetrics.PodMetrics) bool) []backendmetrics.PodMetrics
	PodUpdateOrAddIfNotExist(pod *corev1.Pod) bool
	PodDelete(podName string)

	// Clears the store state, happens when the pool gets deleted.
	Clear()
}

// NewDatastore creates a new data store.
// TODO: modelServerMetricsPort is being deprecated
func NewDatastore(parentCtx context.Context, epFactory datalayer.EndpointFactory, modelServerMetricsPort int32, opts ...DatastoreOption) Datastore {
	// Initialize with defaults
	store := &datastore{
		parentCtx:              parentCtx,
		pool:                   nil,
		mu:                     sync.RWMutex{},
		objectives:             make(map[string]*v1alpha2.InferenceObjective),
		modelRewrites:          newModelRewriteStore(),
		pods:                   &sync.Map{},
		modelServerMetricsPort: modelServerMetricsPort,
		epf:                    epFactory,
	}

	// Apply options
	for _, opt := range opts {
		opt(store)
	}

	return store
}

type datastore struct {
	// parentCtx controls the lifecycle of the background metrics goroutines that spawn up by the datastore.
	parentCtx context.Context
	// mu is used to synchronize access to pool, objectives, and rewrites.
	mu   sync.RWMutex
	pool *datalayer.EndpointPool
	// key: InferenceObjective name, value: *InferenceObjective
	objectives map[string]*v1alpha2.InferenceObjective
	// modelRewrites store for InferenceModelRewrite objects.
	modelRewrites *modelRewriteStore
	// key: types.NamespacedName, value: backendmetrics.PodMetrics
	pods *sync.Map
	// modelServerMetricsPort metrics port from EPP command line argument
	// used only if there is only one inference engine per pod
	modelServerMetricsPort int32 // TODO: deprecating
	epf                    datalayer.EndpointFactory
}

func (ds *datastore) Clear() {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.pool = nil
	ds.objectives = make(map[string]*v1alpha2.InferenceObjective)
	ds.modelRewrites = newModelRewriteStore()
	// stop all pods go routines before clearing the pods map.
	ds.pods.Range(func(_, v any) bool {
		ds.epf.ReleaseEndpoint(v.(backendmetrics.PodMetrics))
		return true
	})
	ds.pods.Clear()
}

// /// Pool APIs ///
func (ds *datastore) PoolSet(ctx context.Context, reader client.Reader, endpointPool *datalayer.EndpointPool) error {
	if endpointPool == nil {
		ds.Clear()
		return nil
	}
	logger := log.FromContext(ctx)
	ds.mu.Lock()
	defer ds.mu.Unlock()

	oldEndpointPool := ds.pool
	ds.pool = endpointPool
	if oldEndpointPool == nil || !reflect.DeepEqual(oldEndpointPool.Selector, endpointPool.Selector) {
		logger.V(logutil.DEFAULT).Info("Updating endpoints", "selector", endpointPool.Selector)
		// A full resync is required to address two cases:
		// 1) At startup, the pod events may get processed before the pool is synced with the datastore,
		//    and hence they will not be added to the store since pool selector is not known yet
		// 2) If the selector on the pool was updated, then we will not get any pod events, and so we need
		//    to resync the whole pool: remove pods in the store that don't match the new selector and add
		//    the ones that may have existed already to the store.
		if err := ds.podResyncAll(ctx, reader); err != nil {
			return fmt.Errorf("failed to update pods according to the pool selector - %w", err)
		}
	}

	return nil
}

func (ds *datastore) PoolGet() (*datalayer.EndpointPool, error) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	if !ds.PoolHasSynced() {
		return nil, errPoolNotSynced
	}
	return ds.pool, nil
}

func (ds *datastore) PoolHasSynced() bool {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return ds.pool != nil
}

func (ds *datastore) PoolLabelsMatch(podLabels map[string]string) bool {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	if ds.pool == nil {
		return false
	}
	poolSelector := labels.SelectorFromSet(ds.pool.Selector)
	podSet := labels.Set(podLabels)
	return poolSelector.Matches(podSet)
}

// /// InferenceObjective APIs ///
func (ds *datastore) ObjectiveSet(infObjective *v1alpha2.InferenceObjective) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.objectives[infObjective.Name] = infObjective
}

func (ds *datastore) ObjectiveGet(objectiveName string) *v1alpha2.InferenceObjective {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return ds.objectives[objectiveName]
}

func (ds *datastore) ObjectiveDelete(namespacedName types.NamespacedName) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	delete(ds.objectives, namespacedName.Name)
}

func (ds *datastore) ObjectiveGetAll() []*v1alpha2.InferenceObjective {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	res := make([]*v1alpha2.InferenceObjective, 0, len(ds.objectives))
	for _, v := range ds.objectives {
		res = append(res, v)
	}
	return res
}

func (ds *datastore) ModelRewriteSet(infModelRewrite *v1alpha2.InferenceModelRewrite) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.modelRewrites.set(infModelRewrite)
}

func (ds *datastore) ModelRewriteDelete(namespacedName types.NamespacedName) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.modelRewrites.delete(namespacedName)
}

func (ds *datastore) ModelRewriteGet(modelName string) (*v1alpha2.InferenceModelRewriteRule, string) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return ds.modelRewrites.getRule(modelName)
}

func (ds *datastore) ModelRewriteGetAll() []*v1alpha2.InferenceModelRewrite {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return ds.modelRewrites.getAll()
}

// /// Pods/endpoints APIs ///
// TODO: add a flag for callers to specify the staleness threshold for metrics.
// ref: https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/1046#discussion_r2246351694
func (ds *datastore) PodList(predicate func(datalayer.Endpoint) bool) []datalayer.Endpoint {
	res := []datalayer.Endpoint{}

	ds.pods.Range(func(k, v any) bool {
		ep := v.(datalayer.Endpoint)
		if predicate(ep) {
			res = append(res, ep)
		}
		return true
	})

	return res
}

func (ds *datastore) PodUpdateOrAddIfNotExist(pod *corev1.Pod) bool {
	if ds.pool == nil {
		return true
	}

	labels := make(map[string]string, len(pod.GetLabels()))
	for key, value := range pod.GetLabels() {
		labels[key] = value
	}

	modelServerMetricsPort := 0
	if len(ds.pool.TargetPorts) == 1 {
		modelServerMetricsPort = int(ds.modelServerMetricsPort)
	}
	pods := []*datalayer.EndpointMetadata{}
	for idx, port := range ds.pool.TargetPorts {
		metricsPort := modelServerMetricsPort
		if metricsPort == 0 {
			metricsPort = port
		}
		pods = append(pods,
			&datalayer.EndpointMetadata{
				NamespacedName: types.NamespacedName{
					Name:      pod.Name + "-rank-" + strconv.Itoa(idx),
					Namespace: pod.Namespace,
				},
				PodName:     pod.Name,
				Address:     pod.Status.PodIP,
				Port:        strconv.Itoa(port),
				MetricsHost: net.JoinHostPort(pod.Status.PodIP, strconv.Itoa(metricsPort)),
				Labels:      labels,
			})
	}

	result := true
	for _, endpointMetadata := range pods {
		var ep datalayer.Endpoint
		existing, ok := ds.pods.Load(endpointMetadata.NamespacedName)
		if !ok {
			ep = ds.epf.NewEndpoint(ds.parentCtx, endpointMetadata, ds)
			ds.pods.Store(endpointMetadata.NamespacedName, ep)
			result = false
		} else {
			ep = existing.(backendmetrics.PodMetrics)
		}
		// Update endpoint properties if anything changed.
		ep.UpdateMetadata(endpointMetadata)
	}
	return result
}

func (ds *datastore) PodDelete(podName string) {
	ds.pods.Range(func(k, v any) bool {
		ep := v.(datalayer.Endpoint)
		if ep.GetMetadata().PodName == podName {
			ds.pods.Delete(k)
			ds.epf.ReleaseEndpoint(ep)
		}
		return true
	})
}

func (ds *datastore) podResyncAll(ctx context.Context, reader client.Reader) error {
	logger := log.FromContext(ctx)
	podList := &corev1.PodList{}
	if err := reader.List(ctx, podList, &client.ListOptions{
		LabelSelector: labels.SelectorFromSet(ds.pool.Selector),
		Namespace:     ds.pool.Namespace,
	}); err != nil {
		return fmt.Errorf("failed to list pods - %w", err)
	}

	activePods := make(map[string]bool)
	for _, pod := range podList.Items {
		if !podutil.IsPodReady(&pod) {
			continue
		}
		namespacedName := types.NamespacedName{Name: pod.Name, Namespace: pod.Namespace}
		activePods[pod.Name] = true
		if !ds.PodUpdateOrAddIfNotExist(&pod) {
			logger.V(logutil.DEFAULT).Info("Pod added", "name", namespacedName)
		} else {
			logger.V(logutil.DEFAULT).Info("Pod already exists", "name", namespacedName)
		}
	}

	// Remove pods that don't belong to the pool or not ready any more.
	ds.pods.Range(func(k, v any) bool {
		ep := v.(datalayer.Endpoint)
		if exist := activePods[ep.GetMetadata().PodName]; !exist {
			logger.V(logutil.VERBOSE).Info("Removing pod", "pod", ep.GetMetadata().PodName)
			ds.PodDelete(ep.GetMetadata().PodName)
		}
		return true
	})

	return nil
}

type DatastoreOption func(*datastore)

func WithEndpointPool(pool *datalayer.EndpointPool) DatastoreOption {
	return func(d *datastore) {
		d.pool = pool
	}
}
