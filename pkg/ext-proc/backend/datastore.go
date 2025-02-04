package backend

import (
	"context"
	"errors"
	"math/rand"
	"sync"
	"time"

	"github.com/google/go-cmp/cmp"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	logutil "inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	informersv1 "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

func NewK8sDataStore(options ...K8sDatastoreOption) *K8sDatastore {
	store := &K8sDatastore{
		poolMu:          sync.RWMutex{},
		InferenceModels: &sync.Map{},
	}

	store.podListerFactory = store.createPodLister
	for _, opt := range options {
		opt(store)
	}
	return store
}

// The datastore is a local cache of relevant data for the given InferencePool (currently all pulled from k8s-api)
type K8sDatastore struct {
	client kubernetes.Interface
	// poolMu is used to synchronize access to the inferencePool.
	poolMu           sync.RWMutex
	inferencePool    *v1alpha1.InferencePool
	podListerFactory PodListerFactory
	podLister        *PodLister
	InferenceModels  *sync.Map
}

type K8sDatastoreOption func(*K8sDatastore)
type PodListerFactory func(*v1alpha1.InferencePool) *PodLister

// WithPods can be used in tests to override the pods.
func WithPodListerFactory(factory PodListerFactory) K8sDatastoreOption {
	return func(store *K8sDatastore) {
		store.podListerFactory = factory
	}
}

type PodLister struct {
	Lister         listersv1.PodLister
	sharedInformer informers.SharedInformerFactory
}

func (l *PodLister) listEverything() ([]*corev1.Pod, error) {
	return l.Lister.List(labels.Everything())

}

func (ds *K8sDatastore) SetClient(client kubernetes.Interface) {
	ds.client = client
}

func (ds *K8sDatastore) setInferencePool(pool *v1alpha1.InferencePool) {
	ds.poolMu.Lock()
	defer ds.poolMu.Unlock()

	if ds.inferencePool != nil && cmp.Equal(ds.inferencePool.Spec.Selector, pool.Spec.Selector) {
		// Pool updated, but the selector stayed the same, so no need to change the informer.
		ds.inferencePool = pool
		return
	}

	// New pool or selector updated.
	ds.inferencePool = pool

	if ds.podLister != nil && ds.podLister.sharedInformer != nil {
		// Shutdown the old informer async since this takes a few seconds.
		go func() {
			ds.podLister.sharedInformer.Shutdown()
		}()
	}

	if ds.podListerFactory != nil {
		// Create a new informer with the new selector.
		ds.podLister = ds.podListerFactory(ds.inferencePool)
		if ds.podLister != nil && ds.podLister.sharedInformer != nil {
			ctx := context.Background()
			ds.podLister.sharedInformer.Start(ctx.Done())
			ds.podLister.sharedInformer.WaitForCacheSync(ctx.Done())
		}
	}
}

func (ds *K8sDatastore) getInferencePool() (*v1alpha1.InferencePool, error) {
	ds.poolMu.RLock()
	defer ds.poolMu.RUnlock()
	if !ds.HasSynced() {
		return nil, errors.New("InferencePool is not initialized in data store")
	}
	return ds.inferencePool, nil
}

func (ds *K8sDatastore) createPodLister(pool *v1alpha1.InferencePool) *PodLister {
	if ds.client == nil {
		return nil
	}
	klog.V(logutil.DEFAULT).Infof("Creating informer for pool %v", pool.Name)
	selectorSet := make(map[string]string)
	for k, v := range pool.Spec.Selector {
		selectorSet[string(k)] = string(v)
	}

	newPodInformer := func(cs clientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
		informer := informersv1.NewFilteredPodInformer(cs, pool.Namespace, resyncPeriod, cache.Indexers{}, func(options *metav1.ListOptions) {
			options.LabelSelector = labels.SelectorFromSet(selectorSet).String()
		})
		err := informer.SetTransform(func(obj interface{}) (interface{}, error) {
			// Remove unnecessary fields to improve memory footprint.
			if accessor, err := meta.Accessor(obj); err == nil {
				if accessor.GetManagedFields() != nil {
					accessor.SetManagedFields(nil)
				}
			}
			return obj, nil
		})
		if err != nil {
			klog.Errorf("Failed to set pod transformer: %v", err)
		}
		return informer
	}
	// 0 means we disable resyncing, it is not really useful to resync every hour (the controller-runtime default),
	// if things go wrong in the watch, no one will wait for an hour for things to get fixed.
	// As precedence, kube-scheduler also disables this since it is expensive to list all pods from the api-server regularly.
	resyncPeriod := time.Duration(0)
	sharedInformer := informers.NewSharedInformerFactory(ds.client, resyncPeriod)
	sharedInformer.InformerFor(&v1.Pod{}, newPodInformer)

	return &PodLister{
		Lister:         sharedInformer.Core().V1().Pods().Lister(),
		sharedInformer: sharedInformer,
	}
}

func (ds *K8sDatastore) getPods() ([]*corev1.Pod, error) {
	ds.poolMu.RLock()
	defer ds.poolMu.RUnlock()
	if !ds.HasSynced() {
		return nil, errors.New("InferencePool is not initialized in datastore")
	}
	pods, err := ds.podLister.listEverything()
	if err != nil {
		return nil, err
	}
	return pods, nil
}

func (s *K8sDatastore) FetchModelData(modelName string) (returnModel *v1alpha1.InferenceModel) {
	infModel, ok := s.InferenceModels.Load(modelName)
	if ok {
		returnModel = infModel.(*v1alpha1.InferenceModel)
	}
	return
}

// HasSynced returns true if InferencePool is set in the data store.
func (ds *K8sDatastore) HasSynced() bool {
	ds.poolMu.RLock()
	defer ds.poolMu.RUnlock()
	return ds.inferencePool != nil
}

func RandomWeightedDraw(model *v1alpha1.InferenceModel, seed int64) string {
	var weights int32

	source := rand.NewSource(rand.Int63())
	if seed > 0 {
		source = rand.NewSource(seed)
	}
	r := rand.New(source)
	for _, model := range model.Spec.TargetModels {
		weights += *model.Weight
	}
	klog.V(logutil.VERBOSE).Infof("Weights for Model(%v) total to: %v", model.Name, weights)
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
