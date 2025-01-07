package backend

import (
	"context"

	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type InferenceModelReconciler struct {
	client.Client
	Scheme         *runtime.Scheme
	Record         record.EventRecorder
	Datastore      *K8sDatastore
	ServerPoolName string
	Namespace      string
}

func (c *InferenceModelReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	if req.Namespace != c.Namespace {
		return ctrl.Result{}, nil
	}
	klog.V(1).Info("reconciling InferenceModel", req.NamespacedName)

	service := &v1alpha1.InferenceModel{}
	if err := c.Get(ctx, req.NamespacedName, service); err != nil {
		klog.Error(err, "unable to get InferencePool")
		return ctrl.Result{}, err
	}

	c.updateDatastore(service)
	return ctrl.Result{}, nil
}

func (c *InferenceModelReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.InferenceModel{}).
		Complete(c)
}

func (c *InferenceModelReconciler) updateDatastore(infModel *v1alpha1.InferenceModel) {
	if infModel.Spec.PoolRef.Name == c.ServerPoolName {
		klog.V(1).Infof("Incoming pool ref %v, server pool name: %v", infModel.Spec.PoolRef, c.ServerPoolName)
		klog.V(1).Infof("Adding/Updating inference model: %v", infModel.Spec.ModelName)
		c.Datastore.InferenceModels.Store(infModel.Spec.ModelName, infModel)
		return
	}
	klog.V(2).Infof("Removing/Not adding inference model: %v", infModel.Spec.ModelName)
	// If we get here. The model is not relevant to this pool, remove.
	c.Datastore.InferenceModels.Delete(infModel.Spec.ModelName)
}
