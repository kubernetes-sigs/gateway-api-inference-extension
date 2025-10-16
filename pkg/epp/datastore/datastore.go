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

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	podutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/pod"
)

var (
	errPoolNotSynced = errors.New("InferencePool is not initialized in data store")
)

// The datastore is a local cache of relevant data for the given InferencePool (currently all pulled from k8s-api)
type Datastore interface {
	// InferencePool operations
	// PoolSet sets the given pool in datastore. If the given pool has different label selector than the previous pool
	// that was stored, the function triggers a resync of the pods to keep the datastore updated. If the given pool
	// is nil, this call triggers the datastore.Clear() function.
	PoolSet(ctx context.Context, reader client.Reader, pool *v1.InferencePool) error
	PoolGet() (*v1.InferencePool, error)
	PoolHasSynced() bool
	PoolLabelsMatch(podLabels map[string]string) bool

	// InferenceObjective operations
	ObjectiveSet(infObjective *v1alpha2.InferenceObjective)
	ObjectiveGet(objectiveName string) *v1alpha2.InferenceObjective
	ObjectiveDelete(namespacedName types.NamespacedName)
	ObjectiveGetAll() []*v1alpha2.InferenceObjective

	// PodList lists pods matching the given predicate.
	PodList(predicate func(backendmetrics.PodMetrics) bool) []backendmetrics.PodMetrics
	PodUpdateOrAddIfNotExist(pod *corev1.Pod) bool
	PodDelete(namespacedName types.NamespacedName)
	PodRemove(podName string)

	// Clears the store state, happens when the pool gets deleted.
	Clear()
}

func NewDatastore(parentCtx context.Context, epFactory datalayer.EndpointFactory, modelServerMetricsPort int32) Datastore {
	store := &datastore{
		parentCtx:              parentCtx,
		poolAndObjectivesMu:    sync.RWMutex{},
		objectives:             make(map[string]*v1alpha2.InferenceObjective),
		pods:                   &sync.Map{},
		modelServerMetricsPort: modelServerMetricsPort,
		epf:                    epFactory,
	}
	return store
}

type datastore struct {
	// parentCtx controls the lifecycle of the background metrics goroutines that spawn up by the datastore.
	parentCtx context.Context
	// poolAndObjectivesMu is used to synchronize access to pool and the objectives map.
	poolAndObjectivesMu sync.RWMutex
	pool                *v1.InferencePool
	// key: InferenceObjective.Spec.ModelName, value: *InferenceObjective
	objectives map[string]*v1alpha2.InferenceObjective
	// key: types.NamespacedName, value: backendmetrics.PodMetrics
	pods *sync.Map
	// modelServerMetricsPort metrics port from EPP command line argument
	// used only if there is only one inference engine per pod
	modelServerMetricsPort int32
	epf                    datalayer.EndpointFactory
}

func (ds *datastore) Clear() {
	ds.poolAndObjectivesMu.Lock()
	defer ds.poolAndObjectivesMu.Unlock()
	ds.pool = nil
	ds.objectives = make(map[string]*v1alpha2.InferenceObjective)
	// stop all pods go routines before clearing the pods map.
	ds.pods.Range(func(_, v any) bool {
		ds.epf.ReleaseEndpoint(v.(backendmetrics.PodMetrics))
		return true
	})
	ds.pods.Clear()
}

// /// InferencePool APIs ///
func (ds *datastore) PoolSet(ctx context.Context, reader client.Reader, pool *v1.InferencePool) error {
	if pool == nil {
		ds.Clear()
		return nil
	}
	logger := log.FromContext(ctx)
	ds.poolAndObjectivesMu.Lock()
	defer ds.poolAndObjectivesMu.Unlock()

	oldPool := ds.pool
	ds.pool = pool
	if oldPool == nil || !reflect.DeepEqual(pool.Spec.Selector, oldPool.Spec.Selector) {
		logger.V(logutil.DEFAULT).Info("Updating inference pool endpoints", "selector", pool.Spec.Selector)
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

func (ds *datastore) PoolGet() (*v1.InferencePool, error) {
	ds.poolAndObjectivesMu.RLock()
	defer ds.poolAndObjectivesMu.RUnlock()
	if !ds.PoolHasSynced() {
		return nil, errPoolNotSynced
	}
	return ds.pool, nil
}

func (ds *datastore) PoolHasSynced() bool {
	ds.poolAndObjectivesMu.RLock()
	defer ds.poolAndObjectivesMu.RUnlock()
	return ds.pool != nil
}

func (ds *datastore) PoolLabelsMatch(podLabels map[string]string) bool {
	ds.poolAndObjectivesMu.RLock()
	defer ds.poolAndObjectivesMu.RUnlock()
	if ds.pool == nil {
		return false
	}
	poolSelector := selectorFromInferencePoolSelector(ds.pool.Spec.Selector.MatchLabels)
	podSet := labels.Set(podLabels)
	return poolSelector.Matches(podSet)
}

func (ds *datastore) ObjectiveSet(infObjective *v1alpha2.InferenceObjective) {
	ds.poolAndObjectivesMu.Lock()
	defer ds.poolAndObjectivesMu.Unlock()
	// Set the objective.
	ds.objectives[infObjective.Name] = infObjective
}

func (ds *datastore) ObjectiveGet(objectiveName string) *v1alpha2.InferenceObjective {
	ds.poolAndObjectivesMu.RLock()
	defer ds.poolAndObjectivesMu.RUnlock()
	iObj, ok := ds.objectives[objectiveName]
	if !ok {
		return nil
	}
	return iObj
}

func (ds *datastore) ObjectiveDelete(namespacedName types.NamespacedName) {
	ds.poolAndObjectivesMu.Lock()
	defer ds.poolAndObjectivesMu.Unlock()
	delete(ds.objectives, namespacedName.Name)
}

func (ds *datastore) ObjectiveGetAll() []*v1alpha2.InferenceObjective {
	ds.poolAndObjectivesMu.RLock()
	defer ds.poolAndObjectivesMu.RUnlock()
	res := []*v1alpha2.InferenceObjective{}
	for _, v := range ds.objectives {
		res = append(res, v)
	}
	return res
}

// /// Pods/endpoints APIs ///
// TODO: add a flag for callers to specify the staleness threshold for metrics.
// ref: https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/1046#discussion_r2246351694
func (ds *datastore) PodList(predicate func(backendmetrics.PodMetrics) bool) []backendmetrics.PodMetrics {
	res := []backendmetrics.PodMetrics{}

	ds.pods.Range(func(k, v any) bool {
		pm := v.(backendmetrics.PodMetrics)
		if predicate(pm) {
			res = append(res, pm)
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
	if len(ds.pool.Spec.TargetPorts) == 1 {
		modelServerMetricsPort = int(ds.modelServerMetricsPort)
	}
	pods := []*datalayer.PodInfo{}
	for idx, port := range ds.pool.Spec.TargetPorts {
		metricsPort := modelServerMetricsPort
		if metricsPort == 0 {
			metricsPort = int(port.Number)
		}
		pods = append(pods,
			&datalayer.PodInfo{
				NamespacedName: types.NamespacedName{
					Name:      pod.Name + "-rank-" + strconv.Itoa(idx),
					Namespace: pod.Namespace,
				},
				PodName:     pod.Name,
				Address:     pod.Status.PodIP,
				Port:        strconv.Itoa(int(port.Number)),
				MetricsHost: net.JoinHostPort(pod.Status.PodIP, strconv.Itoa(metricsPort)),
				Labels:      labels,
			})
	}

	result := true
	for _, podInfo := range pods {
		var pm backendmetrics.PodMetrics
		existing, ok := ds.pods.Load(podInfo.NamespacedName)
		if !ok {
			pm = ds.epf.NewEndpoint(ds.parentCtx, podInfo, ds)
			ds.pods.Store(podInfo.NamespacedName, pm)
			result = false
		} else {
			pm = existing.(backendmetrics.PodMetrics)
		}
		// Update pod properties if anything changed.
		pm.UpdatePod(podInfo)
	}
	return result
}

func (ds *datastore) PodDelete(namespacedName types.NamespacedName) {
	v, ok := ds.pods.LoadAndDelete(namespacedName)
	if ok {
		ds.epf.ReleaseEndpoint(v.(backendmetrics.PodMetrics))
	}
}

func (ds *datastore) PodRemove(podName string) {
	ds.pods.Range(func(k, v any) bool {
		pm := v.(backendmetrics.PodMetrics)
		if pm.GetPod().PodName == podName {
			ds.PodDelete(pm.GetPod().NamespacedName)
		}
		return true
	})
}

func (ds *datastore) podResyncAll(ctx context.Context, reader client.Reader) error {
	logger := log.FromContext(ctx)
	podList := &corev1.PodList{}
	if err := reader.List(ctx, podList, &client.ListOptions{
		LabelSelector: selectorFromInferencePoolSelector(ds.pool.Spec.Selector.MatchLabels),
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
		pm := v.(backendmetrics.PodMetrics)
		if exist := activePods[pm.GetPod().PodName]; !exist {
			logger.V(logutil.VERBOSE).Info("Removing pod", "pod", pm.GetPod())
			ds.PodDelete(pm.GetPod().NamespacedName)
		}
		return true
	})

	return nil
}

func selectorFromInferencePoolSelector(selector map[v1.LabelKey]v1.LabelValue) labels.Selector {
	return labels.SelectorFromSet(stripLabelKeyAliasFromLabelMap(selector))
}

func stripLabelKeyAliasFromLabelMap(labels map[v1.LabelKey]v1.LabelValue) map[string]string {
	outMap := make(map[string]string)
	for k, v := range labels {
		outMap[string(k)] = string(v)
	}
	return outMap
}
