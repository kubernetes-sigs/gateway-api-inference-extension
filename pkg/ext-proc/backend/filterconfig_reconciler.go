package backend

import (
	"context"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

type FilterConfigReconciler struct {
	client.Client
	Datastore *K8sDatastore
}

func (c *FilterConfigReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	cm := &corev1.ConfigMap{}
	if err := c.Get(ctx, req.NamespacedName, cm); err != nil {
		if client.IgnoreNotFound(err) != nil {
			klog.Errorf("unable to get ConfigMap, err: %v", err)
			return ctrl.Result{}, err
		}
		c.Datastore.poolMu.Lock()
		defer c.Datastore.poolMu.Unlock()
		klog.V(1).Info("filter config deleted, reset filter config")
		c.Datastore.filterConfigMap = nil
		return ctrl.Result{}, nil
	}

	c.Datastore.poolMu.Lock()
	defer c.Datastore.poolMu.Unlock()

	if cm.DeletionTimestamp != nil {
		klog.V(1).Info("filter config deleting, reset filter config")
		c.Datastore.filterConfigMap = nil
		return ctrl.Result{}, nil
	}

	klog.V(1).Infof("update filter config to: %++v", cm.Data)
	c.Datastore.filterConfigMap = cm.DeepCopy()
	return ctrl.Result{}, nil
}

func (c *FilterConfigReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&corev1.ConfigMap{}).
		WithEventFilter(predicate.NewPredicateFuncs(func(object client.Object) bool {
			return object.GetName() == "filter-config" && object.GetNamespace() == "default"
		})).
		Complete(c)
}
