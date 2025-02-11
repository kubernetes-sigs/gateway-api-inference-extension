package backend

import (
	"context"

	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	logutil "inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type InferenceModelReconciler struct {
	client.Client
	Scheme             *runtime.Scheme
	Record             record.EventRecorder
	Datastore          *K8sDatastore
	PoolNamespacedName types.NamespacedName
}

func (c *InferenceModelReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	if req.Namespace != c.PoolNamespacedName.Namespace {
		return ctrl.Result{}, nil
	}
	klog.V(1).Infof("Reconciling InferenceModel %v", req.NamespacedName)

	infModel := &v1alpha1.InferenceModel{}
	if err := c.Get(ctx, req.NamespacedName, infModel); err != nil {
		if errors.IsNotFound(err) {
			klog.V(1).Infof("InferenceModel %v not found. Removing from datastore since object must be deleted", req.NamespacedName)
			c.Datastore.InferenceModels.Delete(infModel.Spec.ModelName)
			return ctrl.Result{}, nil
		}
		klog.Error(err, "Unable to get InferenceModel")
		return ctrl.Result{}, err
	} else if !infModel.DeletionTimestamp.IsZero() {
		klog.V(1).Infof("InferenceModel %v is marked for deletion. Removing from datastore", req.NamespacedName)
		c.Datastore.InferenceModels.Delete(infModel.Spec.ModelName)
		return ctrl.Result{}, nil
	}

	c.updateDatastore(infModel)
	return ctrl.Result{}, nil
}

func (c *InferenceModelReconciler) updateDatastore(infModel *v1alpha1.InferenceModel) {
	if infModel.Spec.PoolRef.Name == c.PoolNamespacedName.Name {
		klog.V(1).Infof("Incoming pool ref %v, server pool name: %v", infModel.Spec.PoolRef, c.PoolNamespacedName.Name)
		klog.V(1).Infof("Adding/Updating InferenceModel: %v", infModel.Spec.ModelName)
		c.Datastore.InferenceModels.Store(infModel.Spec.ModelName, infModel)
		return
	}
	klog.V(logutil.DEFAULT).Infof("Removing/Not adding InferenceModel: %v", infModel.Spec.ModelName)
	// If we get here. The model is not relevant to this pool, remove.
	c.Datastore.InferenceModels.Delete(infModel.Spec.ModelName)
}

func (c *InferenceModelReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.InferenceModel{}).
		Complete(c)
}
