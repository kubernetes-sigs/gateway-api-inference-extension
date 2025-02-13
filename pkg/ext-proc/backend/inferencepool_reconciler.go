package backend

import (
	"context"
	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	klog "k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

// InferencePoolReconciler utilizes the controller runtime to reconcile Instance Gateway resources
// This implementation is just used for reading & maintaining data sync. The Gateway implementation
// will have the proper controller that will create/manage objects on behalf of the server pool.
type InferencePoolReconciler struct {
	client.Client
	Scheme             *runtime.Scheme
	Record             record.EventRecorder
	PoolNamespacedName types.NamespacedName
	Datastore          *K8sDatastore
}

func (c *InferencePoolReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	if req.NamespacedName.Name != c.PoolNamespacedName.Name || req.NamespacedName.Namespace != c.PoolNamespacedName.Namespace {
		return ctrl.Result{}, nil
	}
	klogV := klog.V(logutil.DEFAULT)
	klogV.InfoS("Reconciling InferencePool", "name", req.NamespacedName)

	serverPool := &v1alpha1.InferencePool{}
	if err := c.Get(ctx, req.NamespacedName, serverPool); err != nil {
		klogV.ErrorS(err, "Unable to get InferencePool", "name", req.NamespacedName)
		return ctrl.Result{}, err
	}
	if c.Datastore.inferencePool == nil || !reflect.DeepEqual(serverPool.Spec.Selector, c.Datastore.inferencePool.Spec.Selector) {
		c.updateDatastore(serverPool)
		c.Datastore.flushPodsAndRefetch(ctx, c.Client, serverPool)
	} else {
		c.updateDatastore(serverPool)
	}

	return ctrl.Result{}, nil
}

func (c *InferencePoolReconciler) updateDatastore(serverPool *v1alpha1.InferencePool) {
	pool, _ := c.Datastore.getInferencePool()
	if pool == nil ||
		serverPool.ObjectMeta.ResourceVersion != pool.ObjectMeta.ResourceVersion {
		klog.V(logutil.DEFAULT).InfoS("Updating inference pool", "target", klog.KMetadata(&serverPool.ObjectMeta))
		c.Datastore.setInferencePool(serverPool)
	}
}

func (c *InferencePoolReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.InferencePool{}).
		Complete(c)
}
