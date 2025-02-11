package backend

import (
	"context"
	"sync"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

var (
	infModel1 = &v1alpha1.InferenceModel{
		Spec: v1alpha1.InferenceModelSpec{
			ModelName: "fake model1",
			PoolRef:   v1alpha1.PoolObjectReference{Name: "test-pool"},
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-service",
		},
	}
	infModel1Modified = &v1alpha1.InferenceModel{
		Spec: v1alpha1.InferenceModelSpec{
			ModelName: "fake model1",
			PoolRef:   v1alpha1.PoolObjectReference{Name: "test-poolio"},
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-service",
		},
	}
	infModel2 = &v1alpha1.InferenceModel{
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
			incomingService:     infModel1,
			wantInferenceModels: populateServiceMap(infModel1),
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
				InferenceModels: populateServiceMap(infModel1),
			},
			incomingService:     infModel1Modified,
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
				InferenceModels: populateServiceMap(infModel1),
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
			wantInferenceModels: populateServiceMap(infModel1),
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
				InferenceModels: populateServiceMap(infModel1),
			},
			incomingService:     infModel2,
			wantInferenceModels: populateServiceMap(infModel1, infModel2),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			reconciler := &InferenceModelReconciler{
				Datastore:          test.datastore,
				PoolNamespacedName: types.NamespacedName{Name: test.datastore.inferencePool.Name},
			}
			reconciler.updateDatastore(test.incomingService)

			if ok := mapsEqual(reconciler.Datastore.InferenceModels, test.wantInferenceModels); !ok {
				t.Error("Maps are not equal")
			}
		})
	}
}

func TestReconcile_ResourceNotFound(t *testing.T) {
	// Set up the scheme.
	scheme := runtime.NewScheme()
	_ = v1alpha1.AddToScheme(scheme)

	// Create a fake client with no InferenceModel objects.
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	// Create a minimal datastore.
	datastore := &K8sDatastore{
		InferenceModels: &sync.Map{},
		inferencePool: &v1alpha1.InferencePool{
			ObjectMeta: metav1.ObjectMeta{Name: "test-pool"},
		},
	}

	// Create the reconciler.
	reconciler := &InferenceModelReconciler{
		Client:             fakeClient,
		Scheme:             scheme,
		Record:             record.NewFakeRecorder(10),
		Datastore:          datastore,
		PoolNamespacedName: types.NamespacedName{Name: "test-pool"},
	}

	// Create a request for a non-existent resource.
	req := ctrl.Request{NamespacedName: types.NamespacedName{Name: "non-existent-model", Namespace: "default"}}

	// Call Reconcile.
	result, err := reconciler.Reconcile(context.Background(), req)
	if err != nil {
		t.Fatalf("expected no error when resource is not found, got %v", err)
	}

	// Check that no requeue is requested.
	if result.Requeue || result.RequeueAfter != 0 {
		t.Errorf("expected no requeue, got %+v", result)
	}
}

func TestReconcile_ResourceExists(t *testing.T) {
	// Set up the scheme.
	scheme := runtime.NewScheme()
	_ = v1alpha1.AddToScheme(scheme)

	// Create an InferenceModel object.
	existingModel := &v1alpha1.InferenceModel{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "existing-model",
			Namespace: "default",
		},
		Spec: v1alpha1.InferenceModelSpec{
			ModelName: "fake-model",
			PoolRef:   v1alpha1.PoolObjectReference{Name: "test-pool"},
		},
	}

	// Create a fake client with the existing model.
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(existingModel).Build()

	// Create a minimal datastore.
	datastore := &K8sDatastore{
		InferenceModels: &sync.Map{},
		inferencePool: &v1alpha1.InferencePool{
			ObjectMeta: metav1.ObjectMeta{Name: "test-pool"},
		},
	}

	// Create the reconciler.
	reconciler := &InferenceModelReconciler{
		Client:             fakeClient,
		Scheme:             scheme,
		Record:             record.NewFakeRecorder(10),
		Datastore:          datastore,
		PoolNamespacedName: types.NamespacedName{Name: "test-pool", Namespace: "default"},
	}

	// Create a request for the existing resource.
	req := ctrl.Request{NamespacedName: types.NamespacedName{Name: "existing-model", Namespace: "default"}}

	// Call Reconcile.
	result, err := reconciler.Reconcile(context.Background(), req)
	if err != nil {
		t.Fatalf("expected no error when resource exists, got %v", err)
	}

	// Check that no requeue is requested.
	if result.Requeue || result.RequeueAfter != 0 {
		t.Errorf("expected no requeue, got %+v", result)
	}

	// Verify that the datastore was updated.
	if _, ok := datastore.InferenceModels.Load(existingModel.Spec.ModelName); !ok {
		t.Errorf("expected datastore to contain model %q", existingModel.Spec.ModelName)
	}
}

func populateServiceMap(services ...*v1alpha1.InferenceModel) *sync.Map {
	returnVal := &sync.Map{}

	for _, service := range services {
		returnVal.Store(service.Spec.ModelName, service)
	}
	return returnVal
}

func mapsEqual(map1, map2 *sync.Map) bool {
	equal := true

	map1.Range(func(k, v any) bool {
		if _, ok := map2.Load(k); !ok {
			equal = false
			return false
		}
		return true
	})
	map2.Range(func(k, v any) bool {
		if _, ok := map1.Load(k); !ok {
			equal = false
			return false
		}
		return true
	})

	return equal
}
