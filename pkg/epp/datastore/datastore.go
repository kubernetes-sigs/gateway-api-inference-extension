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
	"math/rand"
	"sync"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
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
	PoolSet(pool *v1alpha2.InferencePool)
	PoolGet() (*v1alpha2.InferencePool, error)
	PoolHasSynced() bool
	PoolLabelsMatch(podLabels map[string]string) bool

	// InferenceModel operations
	ModelSetIfOlder(infModel *v1alpha2.InferenceModel) bool
	ModelGet(modelName string) (*v1alpha2.InferenceModel, bool)
	ModelDelete(namespacedName types.NamespacedName) (*v1alpha2.InferenceModel, bool)
	ModelResync(ctx context.Context, ctrlClient client.Client, modelName string) (bool, error)
	ModelGetAll() []*v1alpha2.InferenceModel

	// PodMetrics operations
	PodUpdateOrAddIfNotExist(pod *corev1.Pod) bool
	PodUpdateMetricsIfExist(namespacedName types.NamespacedName, m *Metrics) bool
	PodGet(namespacedName types.NamespacedName) (*PodMetrics, bool)
	PodDelete(namespacedName types.NamespacedName)
	PodResyncAll(ctx context.Context, ctrlClient client.Client)
	PodGetAll() []*PodMetrics
	PodDeleteAll() // This is only for testing.
	PodRange(f func(key, value any) bool)

	// Clears the store state, happens when the pool gets deleted.
	Clear()
}

func NewDatastore() Datastore {
	store := &datastore{
		poolAndModelsMu: sync.RWMutex{},
		models:          make(map[string]*v1alpha2.InferenceModel),
		pods:            &sync.Map{},
	}
	return store
}

// Used for test only
func NewFakeDatastore(pods []*PodMetrics, models []*v1alpha2.InferenceModel, pool *v1alpha2.InferencePool) Datastore {
	store := NewDatastore()

	for _, pod := range pods {
		// Making a copy since in tests we may use the same global PodMetric across tests.
		p := *pod
		store.(*datastore).pods.Store(pod.NamespacedName, &p)
	}

	for _, m := range models {
		store.ModelSetIfOlder(m)
	}

	if pool != nil {
		store.(*datastore).pool = pool
	}
	return store
}

type datastore struct {
	// poolAndModelsMu is used to synchronize access to pool and the models map.
	poolAndModelsMu sync.RWMutex
	pool            *v1alpha2.InferencePool
	// key: InferenceModel.Spec.ModelName, value: *InferenceModel
	models map[string]*v1alpha2.InferenceModel
	// key: types.NamespacedName, value: *PodMetrics
	pods *sync.Map
}

func (ds *datastore) Clear() {
	ds.poolAndModelsMu.Lock()
	defer ds.poolAndModelsMu.Unlock()
	ds.pool = nil
	ds.models = make(map[string]*v1alpha2.InferenceModel)
	ds.pods.Clear()
}

// /// InferencePool APIs ///
func (ds *datastore) PoolSet(pool *v1alpha2.InferencePool) {
	ds.poolAndModelsMu.Lock()
	defer ds.poolAndModelsMu.Unlock()
	ds.pool = pool
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

// /// InferenceModel APIs ///
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

func (ds *datastore) ModelGet(modelName string) (*v1alpha2.InferenceModel, bool) {
	ds.poolAndModelsMu.RLock()
	defer ds.poolAndModelsMu.RUnlock()
	m, exists := ds.models[modelName]
	return m, exists
}

func (ds *datastore) ModelDelete(namespacedName types.NamespacedName) (*v1alpha2.InferenceModel, bool) {
	ds.poolAndModelsMu.Lock()
	defer ds.poolAndModelsMu.Unlock()
	for _, m := range ds.models {
		if m.Name == namespacedName.Name && m.Namespace == namespacedName.Namespace {
			delete(ds.models, m.Spec.ModelName)
			return m, true
		}
	}
	return nil, false
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
func (ds *datastore) PodUpdateMetricsIfExist(namespacedName types.NamespacedName, m *Metrics) bool {
	if val, ok := ds.pods.Load(namespacedName); ok {
		existing := val.(*PodMetrics)
		existing.Metrics = *m
		return true
	}
	return false
}

func (ds *datastore) PodGet(namespacedName types.NamespacedName) (*PodMetrics, bool) {
	val, ok := ds.pods.Load(namespacedName)
	if ok {
		return val.(*PodMetrics), true
	}
	return nil, false
}

func (ds *datastore) PodGetAll() []*PodMetrics {
	res := []*PodMetrics{}
	fn := func(k, v any) bool {
		res = append(res, v.(*PodMetrics))
		return true
	}
	ds.pods.Range(fn)
	return res
}

func (ds *datastore) PodRange(f func(key, value any) bool) {
	ds.pods.Range(f)
}

func (ds *datastore) PodDelete(namespacedName types.NamespacedName) {
	ds.pods.Delete(namespacedName)
}

func (ds *datastore) PodUpdateOrAddIfNotExist(pod *corev1.Pod) bool {
	pool, _ := ds.PoolGet()
	new := &PodMetrics{
		Pod: Pod{
			NamespacedName: types.NamespacedName{
				Name:      pod.Name,
				Namespace: pod.Namespace,
			},
			Address:    pod.Status.PodIP,
			ScrapePath: "/metrics",
			ScrapePort: pool.Spec.TargetPortNumber,
		},
		Metrics: Metrics{
			ActiveModels: make(map[string]int),
		},
	}
	existing, ok := ds.pods.Load(new.NamespacedName)
	if !ok {
		ds.pods.Store(new.NamespacedName, new)
		return true
	}

	// Update pod properties if anything changed.
	existing.(*PodMetrics).Pod = new.Pod
	return false
}

func (ds *datastore) PodResyncAll(ctx context.Context, ctrlClient client.Client) {
	// Pool must exist to invoke this function.
	pool, _ := ds.PoolGet()
	podList := &corev1.PodList{}
	if err := ctrlClient.List(ctx, podList, &client.ListOptions{
		LabelSelector: selectorFromInferencePoolSelector(pool.Spec.Selector),
		Namespace:     pool.Namespace,
	}); err != nil {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(err, "Failed to list clients")
		return
	}

	activePods := make(map[string]bool)
	for _, pod := range podList.Items {
		if podIsReady(&pod) {
			activePods[pod.Name] = true
			ds.PodUpdateOrAddIfNotExist(&pod)
		}
	}

	// Remove pods that don't exist or not ready any more.
	deleteFn := func(k, v any) bool {
		pm := v.(*PodMetrics)
		if exist := activePods[pm.NamespacedName.Name]; !exist {
			ds.pods.Delete(pm.NamespacedName)
		}
		return true
	}
	ds.pods.Range(deleteFn)
}

func (ds *datastore) PodDeleteAll() {
	ds.pods.Clear()
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

func RandomWeightedDraw(logger logr.Logger, model *v1alpha2.InferenceModel, seed int64) string {
	var weights int32

	source := rand.NewSource(rand.Int63())
	if seed > 0 {
		source = rand.NewSource(seed)
	}
	r := rand.New(source)
	for _, model := range model.Spec.TargetModels {
		weights += *model.Weight
	}
	logger.V(logutil.TRACE).Info("Weights for model computed", "model", model.Name, "weights", weights)
	randomVal := r.Int31n(weights)
	for _, model := range model.Spec.TargetModels {
		if randomVal < *model.Weight {
			return model.Name
		}
		randomVal -= *model.Weight
	}
	return ""
}

func IsCritical(model *v1alpha2.InferenceModel) bool {
	if model.Spec.Criticality != nil && *model.Spec.Criticality == v1alpha2.Critical {
		return true
	}
	return false
}

// TODO: move out to share with pod_reconciler.go
func podIsReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			if condition.Status == corev1.ConditionTrue {
				return true
			}
			break
		}
	}
	return false
}
