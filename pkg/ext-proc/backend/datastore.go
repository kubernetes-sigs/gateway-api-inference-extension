package backend

import (
	"errors"
	"math/rand"
	"sync"

	"inference.networking.x-k8s.io/llm-instance-gateway/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

func NewK8sDataStore(options ...K8sDatastoreOption) *K8sDatastore {
	store := &K8sDatastore{
		poolMu:          sync.RWMutex{},
		InferenceModels: &sync.Map{},
		pods:            &sync.Map{},
	}
	for _, opt := range options {
		opt(store)
	}
	return store
}

// The datastore is a local cache of relevant data for the given InferencePool (currently all pulled from k8s-api)
type K8sDatastore struct {
	// poolMu is used to synchronize access to the inferencePool.
	poolMu          sync.RWMutex
	inferencePool   *v1alpha1.InferencePool
	InferenceModels *sync.Map
	pods            *sync.Map
}

type K8sDatastoreOption func(*K8sDatastore)

// WithPods can be used in tests to override the pods.
func WithPods(pods []*PodMetrics) K8sDatastoreOption {
	return func(store *K8sDatastore) {
		store.pods = &sync.Map{}
		for _, pod := range pods {
			store.pods.Store(pod.Pod, true)
		}
	}
}

func (ds *K8sDatastore) setInferencePool(pool *v1alpha1.InferencePool) {
	ds.poolMu.Lock()
	defer ds.poolMu.Unlock()
	ds.inferencePool = pool
}

func (ds *K8sDatastore) getInferencePool() (*v1alpha1.InferencePool, error) {
	ds.poolMu.RLock()
	defer ds.poolMu.RUnlock()
	if ds.inferencePool == nil {
		return nil, errors.New("InferencePool hasn't been initialized yet")
	}
	return ds.inferencePool, nil
}

func (ds *K8sDatastore) GetPodIPs() []string {
	var ips []string
	ds.pods.Range(func(name, pod any) bool {
		ips = append(ips, pod.(*corev1.Pod).Status.PodIP)
		return true
	})
	return ips
}

func (s *K8sDatastore) FetchModelData(modelName string) (returnModel *v1alpha1.InferenceModel) {
	infModel, ok := s.InferenceModels.Load(modelName)
	if ok {
		returnModel = infModel.(*v1alpha1.InferenceModel)
	}
	return
}

func RandomWeightedDraw(model *v1alpha1.InferenceModel, seed int64) string {
	var weights int32

	source := rand.NewSource(rand.Int63())
	if seed > 0 {
		source = rand.NewSource(seed)
	}
	r := rand.New(source)
	for _, model := range model.Spec.TargetModels {
		weights += model.Weight
	}
	klog.V(3).Infof("Weights for Model(%v) total to: %v", model.Name, weights)
	randomVal := r.Int31n(weights)
	for _, model := range model.Spec.TargetModels {
		if randomVal < model.Weight {
			return model.Name
		}
		randomVal -= model.Weight
	}
	return ""
}

func IsCritical(model *v1alpha1.InferenceModel) bool {
	if model.Spec.Criticality != nil && *model.Spec.Criticality == v1alpha1.Critical {
		return true
	}
	return false
}
