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
	"reflect"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	podutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/pod"
)

const (
	ModelNameIndexKey = "spec.modelName"
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
	PoolSet(ctx context.Context, client client.Client, pool *v1alpha2.InferencePool) error
	PoolGet() (*v1alpha2.InferencePool, error)
	PoolHasSynced() bool
	PoolLabelsMatch(podLabels map[string]string) bool

	// InferenceModel operations
	ModelSetIfOlder(infModel *v1alpha2.InferenceModel) bool
	ModelGet(modelName string) *v1alpha2.InferenceModel
	ModelDelete(namespacedName types.NamespacedName) *v1alpha2.InferenceModel
	ModelResync(ctx context.Context, ctrlClient client.Client, modelName string) (bool, error)
	ModelGetAll() []*v1alpha2.InferenceModel

	// PodMetrics operations
	// PodGetAll returns all pods and metrics, including fresh and stale.
	PodGetAll() []backendmetrics.PodMetrics
	// PodList lists pods matching the given predicate.
	PodList(predicate func(backendmetrics.PodMetrics) bool) []backendmetrics.PodMetrics
	PodUpdateOrAddIfNotExist(pod *corev1.Pod) bool
	PodDelete(namespacedName types.NamespacedName)

	// Clears the store state, happens when the pool gets deleted.
	Clear()
}

func NewDatastore(parentCtx context.Context, pmf *backendmetrics.PodMetricsFactory) Datastore {
	store := &datastore{
		parentCtx:       parentCtx,
		poolAndModelsMu: sync.RWMutex{},
		models:          make(map[string]*v1alpha2.InferenceModel),
		pods:            &sync.Map{},
		pmf:             pmf,
	}
	return store
}

type datastore struct {
	// parentCtx controls the lifecycle of the background metrics goroutines that spawn up by the datastore.
	parentCtx context.Context
	// poolAndModelsMu is used to synchronize access to pool and the models map.
	poolAndModelsMu sync.RWMutex
	pool            *v1alpha2.InferencePool
	// key: InferenceModel.Spec.ModelName, value: *InferenceModel
	models map[string]*v1alpha2.InferenceModel
	// key: types.NamespacedName, value: backendmetrics.PodMetrics
	pods *sync.Map
	pmf  *backendmetrics.PodMetricsFactory
}

func (ds *datastore) Clear() {
	ds.poolAndModelsMu.Lock()
	defer ds.poolAndModelsMu.Unlock()
	ds.pool = nil
	ds.models = make(map[string]*v1alpha2.InferenceModel)
	ds.pods.Clear()
}

// /// InferencePool APIs ///
func (ds *datastore) PoolSet(ctx context.Context, client client.Client, pool *v1alpha2.InferencePool) error {
	if pool == nil {
		ds.Clear()
		return nil
	}
	logger := log.FromContext(ctx)
	ds.poolAndModelsMu.Lock()
	defer ds.poolAndModelsMu.Unlock()

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
		if err := ds.podResyncAll(ctx, client); err != nil {
			return fmt.Errorf("failed to update pods according to the pool selector - %w", err)
		}
	}

	return nil
}

func (ds *datastore) PoolGet() (*v1alpha2.InferencePool, error) {
	ds.poolAndModelsMu.RLock()
	defer ds.poolAndModelsMu.RUnlock()
	if !ds.PoolHasSynced() {
		return nil, errPoolNotSynced
	}
	return ds.pool, nil
}

func (ds *datastore) PoolHasSynced() bool {
	ds.poolAndModelsMu.RLock()
	defer ds.poolAndModelsMu.RUnlock()
	return ds.pool != nil
}

func (ds *datastore) PoolLabelsMatch(podLabels map[string]string) bool {
	ds.poolAndModelsMu.RLock()
	defer ds.poolAndModelsMu.RUnlock()
	poolSelector := selectorFromInferencePoolSelector(ds.pool.Spec.Selector)
	podSet := labels.Set(podLabels)
	return poolSelector.Matches(podSet)
}

func (ds *datastore) ModelSetIfOlder(infModel *v1alpha2.InferenceModel) bool {
	ds.poolAndModelsMu.Lock()
	defer ds.poolAndModelsMu.Unlock()

	// Check first if the existing model is older.
	// One exception is if the incoming model object is the same, in which case, we should not
	// check for creation timestamp since that means the object was re-created, and so we should override.
	existing, exists := ds.models[infModel.Spec.ModelName]
	if exists {
		diffObj := infModel.Name != existing.Name || infModel.Namespace != existing.Namespace
		if diffObj && existing.ObjectMeta.CreationTimestamp.Before(&infModel.ObjectMeta.CreationTimestamp) {
			return false
		}
	}
	// Set the model.
	ds.models[infModel.Spec.ModelName] = infModel
	return true
}

func (ds *datastore) ModelResync(ctx context.Context, c client.Client, modelName string) (bool, error) {
	ds.poolAndModelsMu.Lock()
	defer ds.poolAndModelsMu.Unlock()

	var models v1alpha2.InferenceModelList
	if err := c.List(ctx, &models, client.MatchingFields{ModelNameIndexKey: modelName}, client.InNamespace(ds.pool.Namespace)); err != nil {
		return false, fmt.Errorf("listing models that match the modelName %s: %w", modelName, err)
	}
	if len(models.Items) == 0 {
		// No other instances of InferenceModels with this ModelName exists.
		return false, nil
	}

	var oldest *v1alpha2.InferenceModel
	for i := range models.Items {
		m := &models.Items[i]
		if m.Spec.ModelName != modelName || // The index should filter those out, but just in case!
			m.Spec.PoolRef.Name != v1alpha2.ObjectName(ds.pool.Name) || // We don't care about other pools, we could setup an index on this too!
			!m.DeletionTimestamp.IsZero() { // ignore objects marked for deletion
			continue
		}
		if oldest == nil || m.ObjectMeta.CreationTimestamp.Before(&oldest.ObjectMeta.CreationTimestamp) {
			oldest = m
		}
	}
	if oldest == nil {
		return false, nil
	}
	ds.models[modelName] = oldest
	return true, nil
}

func (ds *datastore) ModelGet(modelName string) *v1alpha2.InferenceModel {
	ds.poolAndModelsMu.RLock()
	defer ds.poolAndModelsMu.RUnlock()
	return ds.models[modelName]
}

func (ds *datastore) ModelDelete(namespacedName types.NamespacedName) *v1alpha2.InferenceModel {
	ds.poolAndModelsMu.Lock()
	defer ds.poolAndModelsMu.Unlock()
	for _, m := range ds.models {
		if m.Name == namespacedName.Name && m.Namespace == namespacedName.Namespace {
			delete(ds.models, m.Spec.ModelName)
			return m
		}
	}
	return nil
}

func (ds *datastore) ModelGetAll() []*v1alpha2.InferenceModel {
	ds.poolAndModelsMu.RLock()
	defer ds.poolAndModelsMu.RUnlock()
	res := []*v1alpha2.InferenceModel{}
	for _, v := range ds.models {
		res = append(res, v)
	}
	return res
}

// /// Pods/endpoints APIs ///

func (ds *datastore) PodGetAll() []backendmetrics.PodMetrics {
	return ds.PodList(func(backendmetrics.PodMetrics) bool { return true })
}

func (ds *datastore) PodList(predicate func(backendmetrics.PodMetrics) bool) []backendmetrics.PodMetrics {
	res := []backendmetrics.PodMetrics{}
	fn := func(k, v any) bool {
		pm := v.(backendmetrics.PodMetrics)
		if predicate(pm) {
			res = append(res, pm)
		}
		return true
	}
	ds.pods.Range(fn)
	return res
}

func (ds *datastore) PodUpdateOrAddIfNotExist(pod *corev1.Pod) bool {
	namespacedName := types.NamespacedName{
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}
	var pm backendmetrics.PodMetrics
	existing, ok := ds.pods.Load(namespacedName)
	if !ok {
		pm = ds.pmf.NewPodMetrics(ds.parentCtx, pod, ds)
		ds.pods.Store(namespacedName, pm)
	} else {
		pm = existing.(backendmetrics.PodMetrics)
	}
	// Update pod properties if anything changed.
	pm.UpdatePod(pod)
	return ok
}

func (ds *datastore) PodDelete(namespacedName types.NamespacedName) {
	v, ok := ds.pods.LoadAndDelete(namespacedName)
	if ok {
		pmr := v.(backendmetrics.PodMetrics)
		pmr.StopRefreshLoop()
	}
}

func (ds *datastore) podResyncAll(ctx context.Context, ctrlClient client.Client) error {
	logger := log.FromContext(ctx)
	podList := &corev1.PodList{}
	if err := ctrlClient.List(ctx, podList, &client.ListOptions{
		LabelSelector: selectorFromInferencePoolSelector(ds.pool.Spec.Selector),
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
		if ds.PodUpdateOrAddIfNotExist(&pod) {
			logger.V(logutil.DEFAULT).Info("Pod added", "name", namespacedName)
		} else {
			logger.V(logutil.DEFAULT).Info("Pod already exists", "name", namespacedName)
		}
	}

	// Remove pods that don't belong to the pool or not ready any more.
	deleteFn := func(k, v any) bool {
		pm := v.(backendmetrics.PodMetrics)
		if exist := activePods[pm.GetPod().NamespacedName.Name]; !exist {
			logger.V(logutil.VERBOSE).Info("Removing pod", "pod", pm.GetPod())
			ds.PodDelete(pm.GetPod().NamespacedName)
		}
		return true
	}
	ds.pods.Range(deleteFn)

	return nil
}

func selectorFromInferencePoolSelector(selector map[v1alpha2.LabelKey]v1alpha2.LabelValue) labels.Selector {
	return labels.SelectorFromSet(stripLabelKeyAliasFromLabelMap(selector))
}

func stripLabelKeyAliasFromLabelMap(labels map[v1alpha2.LabelKey]v1alpha2.LabelValue) map[string]string {
	outMap := make(map[string]string)
	for k, v := range labels {
		outMap[string(k)] = string(v)
	}
	return outMap
}
