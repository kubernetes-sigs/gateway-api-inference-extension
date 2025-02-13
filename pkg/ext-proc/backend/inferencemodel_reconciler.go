package backend

import (
	"context"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
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

	klogV := klog.V(logutil.DEFAULT)
	klogV.InfoS("Reconciling InferenceModel", "name", req.NamespacedName)

	infModel := &v1alpha1.InferenceModel{}
	if err := c.Get(ctx, req.NamespacedName, infModel); err != nil {
		if errors.IsNotFound(err) {
			klogV.InfoS("InferenceModel not found. Removing from datastore since object must be deleted", "name", req.NamespacedName)
			c.Datastore.InferenceModels.Delete(infModel.Spec.ModelName)
			return ctrl.Result{}, nil
		}
		klogV.ErrorS(err, "Unable to get InferenceModel", "name", req.NamespacedName)
		return ctrl.Result{}, err
	} else if !infModel.DeletionTimestamp.IsZero() {
		klogV.InfoS("InferenceModel is marked for deletion. Removing from datastore", "name", req.NamespacedName)
		c.Datastore.InferenceModels.Delete(infModel.Spec.ModelName)
		return ctrl.Result{}, nil
	}

	c.updateDatastore(infModel)
	return ctrl.Result{}, nil
}

func (c *InferenceModelReconciler) updateDatastore(infModel *v1alpha1.InferenceModel) {
	klogV := klog.V(logutil.DEFAULT)

	if infModel.Spec.PoolRef.Name == c.PoolNamespacedName.Name {
		klogV.InfoS("Updating datastore", "poolRef", infModel.Spec.PoolRef, "serverPoolName", c.PoolNamespacedName)
		klogV.InfoS("Adding/Updating InferenceModel", "modelName", infModel.Spec.ModelName)
		c.Datastore.InferenceModels.Store(infModel.Spec.ModelName, infModel)
		return
	}
	klogV.InfoS("Removing/Not adding InferenceModel", "modelName", infModel.Spec.ModelName)
	// If we get here. The model is not relevant to this pool, remove.
	c.Datastore.InferenceModels.Delete(infModel.Spec.ModelName)
}

func (c *InferenceModelReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.InferenceModel{}).
		Complete(c)
}
