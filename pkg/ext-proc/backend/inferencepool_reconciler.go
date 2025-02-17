package backend

import (
	"context"
	"reflect"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	klog "k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
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
	Datastore          Datastore
}

func (c *InferencePoolReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	if req.NamespacedName.Name != c.PoolNamespacedName.Name || req.NamespacedName.Namespace != c.PoolNamespacedName.Namespace {
		return ctrl.Result{}, nil
	}

	logger := log.FromContext(ctx)
	loggerDefault := logger.V(logutil.DEFAULT)
	loggerDefault.Info("Reconciling InferencePool", "name", req.NamespacedName)

	serverPool := &v1alpha1.InferencePool{}

	if err := c.Get(ctx, req.NamespacedName, serverPool); err != nil {
		if errors.IsNotFound(err) {
			loggerDefault.Info("InferencePool not found. Clearing the datastore", "name", req.NamespacedName)
			c.Datastore.Clear()
			return ctrl.Result{}, nil
		}
		loggerDefault.Error(err, "Unable to get InferencePool", "name", req.NamespacedName)
		return ctrl.Result{}, err
	} else if !serverPool.DeletionTimestamp.IsZero() {
		loggerDefault.Info("InferencePool is marked for deletion. Clearing the datastore", "name", req.NamespacedName)
		c.Datastore.Clear()
		return ctrl.Result{}, nil
	}

	c.updateDatastore(ctx, serverPool)

	return ctrl.Result{}, nil
}

func (c *InferencePoolReconciler) updateDatastore(ctx context.Context, newPool *v1alpha1.InferencePool) {
	logger := log.FromContext(ctx)
	oldPool, _ := c.Datastore.PoolGet()
	c.Datastore.PoolSet(newPool)
	if oldPool == nil || !reflect.DeepEqual(newPool.Spec.Selector, oldPool.Spec.Selector) {
		logger.V(logutil.DEFAULT).Info("Updating inference pool endpoints", "target", klog.KMetadata(&newPool.ObjectMeta))
		c.Datastore.PodFlushAll(ctx, c.Client)
	}
}

func (c *InferencePoolReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.InferencePool{}).
		Complete(c)
}
