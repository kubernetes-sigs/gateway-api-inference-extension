package backend

import (
	"sync"
	"testing"

	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

var (
	service1 = &v1alpha1.InferenceModel{
		Spec: v1alpha1.InferenceModelSpec{
			ModelName: "fake model1",
			PoolRef:   v1alpha1.PoolObjectReference{Name: "test-pool"},
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-service",
		},
	}
	service1Modified = &v1alpha1.InferenceModel{
		Spec: v1alpha1.InferenceModelSpec{
			ModelName: "fake model1",
			PoolRef:   v1alpha1.PoolObjectReference{Name: "test-poolio"},
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-service",
		},
	}
	service2 = &v1alpha1.InferenceModel{
		Spec: v1alpha1.InferenceModelSpec{
			ModelName: "fake model",
			PoolRef:   v1alpha1.PoolObjectReference{Name: "test-pool"},
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-service-2",
		},
	}
)

func TestUpdateDatastore_InferenceModelReconciler(t *testing.T) {
	tests := []struct {
		name                string
		datastore           *K8sDatastore
		incomingService     *v1alpha1.InferenceModel
		wantInferenceModels *sync.Map
	}{
		{
			name: "No Services registered; valid, new service incoming.",
			datastore: &K8sDatastore{
				inferencePool: &v1alpha1.InferencePool{
					Spec: v1alpha1.InferencePoolSpec{
						Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{"app": "vllm"},
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:            "test-pool",
						ResourceVersion: "Old and boring",
					},
				},
				InferenceModels: &sync.Map{},
			},
			incomingService:     service1,
			wantInferenceModels: populateServiceMap(service1),
		},
		{
			name: "Removing existing service.",
			datastore: &K8sDatastore{
				inferencePool: &v1alpha1.InferencePool{
					Spec: v1alpha1.InferencePoolSpec{
						Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{"app": "vllm"},
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:            "test-pool",
						ResourceVersion: "Old and boring",
					},
				},
				InferenceModels: populateServiceMap(service1),
			},
			incomingService:     service1Modified,
			wantInferenceModels: populateServiceMap(),
		},
		{
			name: "Unrelated service, do nothing.",
			datastore: &K8sDatastore{
				inferencePool: &v1alpha1.InferencePool{
					Spec: v1alpha1.InferencePoolSpec{
						Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{"app": "vllm"},
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:            "test-pool",
						ResourceVersion: "Old and boring",
					},
				},
				InferenceModels: populateServiceMap(service1),
			},
			incomingService: &v1alpha1.InferenceModel{
				Spec: v1alpha1.InferenceModelSpec{
					ModelName: "fake model",
					PoolRef:   v1alpha1.PoolObjectReference{Name: "test-poolio"},
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "unrelated-service",
				},
			},
			wantInferenceModels: populateServiceMap(service1),
		},
		{
			name: "Add to existing",
			datastore: &K8sDatastore{
				inferencePool: &v1alpha1.InferencePool{
					Spec: v1alpha1.InferencePoolSpec{
						Selector: map[v1alpha1.LabelKey]v1alpha1.LabelValue{"app": "vllm"},
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:            "test-pool",
						ResourceVersion: "Old and boring",
					},
				},
				InferenceModels: populateServiceMap(service1),
			},
			incomingService:     service2,
			wantInferenceModels: populateServiceMap(service1, service2),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			InferenceModelReconciler := &InferenceModelReconciler{
				Datastore:          test.datastore,
				PoolNamespacedName: types.NamespacedName{Name: test.datastore.inferencePool.Name},
			}
			InferenceModelReconciler.updateDatastore(test.incomingService)

			if ok := mapsEqual(InferenceModelReconciler.Datastore.InferenceModels, test.wantInferenceModels); !ok {
				t.Error("Maps are not equal")
			}
		})
	}
}

func populateServiceMap(services ...*v1alpha1.InferenceModel) *sync.Map {
	returnVal := &sync.Map{}

	for _, service := range services {
		returnVal.Store(service.Spec.ModelName, service)
	}
	return returnVal
}
