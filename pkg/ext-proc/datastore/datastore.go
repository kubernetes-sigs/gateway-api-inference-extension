package datastore

import (
	"context"
	"errors"
	"math/rand"
	"sync"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

// The datastore is a local cache of relevant data for the given InferencePool (currently all pulled from k8s-api)
type Datastore interface {
	// InferencePool operations
	PoolSet(pool *v1alpha1.InferencePool)
	PoolGet() (*v1alpha1.InferencePool, error)
	PoolHasSynced() bool
	PoolLabelsMatch(podLabels map[string]string) bool

	// InferenceModel operations
	ModelSet(infModel *v1alpha1.InferenceModel)
	ModelGet(modelName string) (*v1alpha1.InferenceModel, bool)
	ModelDelete(modelName string)

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
		poolMu: sync.RWMutex{},
		models: &sync.Map{},
		pods:   &sync.Map{},
	}
	return store
}

// Used for test only
func NewFakeDatastore(pods, models *sync.Map, pool *v1alpha1.InferencePool) Datastore {
	store := NewDatastore()
	if pods != nil {
		store.(*datastore).pods = pods
	}
	if models != nil {
		store.(*datastore).models = models
	}
	if pool != nil {
		store.(*datastore).pool = pool
	}
	return store
}

type datastore struct {
	// poolMu is used to synchronize access to the inferencePool.
	poolMu sync.RWMutex
	pool   *v1alpha1.InferencePool
	models *sync.Map
	// key: types.NamespacedName, value: *PodMetrics
	pods *sync.Map
}

func (ds *datastore) Clear() {
	ds.poolMu.Lock()
	defer ds.poolMu.Unlock()
	ds.pool = nil
	ds.models.Clear()
	ds.pods.Clear()
}

// /// InferencePool APIs ///
func (ds *datastore) PoolSet(pool *v1alpha1.InferencePool) {
	ds.poolMu.Lock()
	defer ds.poolMu.Unlock()
	ds.pool = pool
}

func (ds *datastore) PoolGet() (*v1alpha1.InferencePool, error) {
	ds.poolMu.RLock()
	defer ds.poolMu.RUnlock()
	if !ds.PoolHasSynced() {
		return nil, errors.New("InferencePool is not initialized in data store")
	}
	return ds.pool, nil
}

func (ds *datastore) PoolHasSynced() bool {
	ds.poolMu.RLock()
	defer ds.poolMu.RUnlock()
	return ds.pool != nil
}

func (ds *datastore) PoolLabelsMatch(podLabels map[string]string) bool {
	poolSelector := selectorFromInferencePoolSelector(ds.pool.Spec.Selector)
	podSet := labels.Set(podLabels)
	return poolSelector.Matches(podSet)
}

// /// InferenceModel APIs ///
func (ds *datastore) ModelSet(infModel *v1alpha1.InferenceModel) {
	ds.models.Store(infModel.Spec.ModelName, infModel)
}

func (ds *datastore) ModelGet(modelName string) (*v1alpha1.InferenceModel, bool) {
	infModel, ok := ds.models.Load(modelName)
	if ok {
		return infModel.(*v1alpha1.InferenceModel), true
	}
	return nil, false
}

func (ds *datastore) ModelDelete(modelName string) {
	ds.models.Delete(modelName)
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

func selectorFromInferencePoolSelector(selector map[v1alpha1.LabelKey]v1alpha1.LabelValue) labels.Selector {
	return labels.SelectorFromSet(stripLabelKeyAliasFromLabelMap(selector))
}

func stripLabelKeyAliasFromLabelMap(labels map[v1alpha1.LabelKey]v1alpha1.LabelValue) map[string]string {
	outMap := make(map[string]string)
	for k, v := range labels {
		outMap[string(k)] = string(v)
	}
	return outMap
}

func RandomWeightedDraw(logger logr.Logger, model *v1alpha1.InferenceModel, seed int64) string {
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

func IsCritical(model *v1alpha1.InferenceModel) bool {
	if model.Spec.Criticality != nil && *model.Spec.Criticality == v1alpha1.Critical {
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
