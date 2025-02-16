package backend

import (
	"context"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

type InferenceModelReconciler struct {
	client.Client
	Scheme             *runtime.Scheme
	Record             record.EventRecorder
	Datastore          Datastore
	PoolNamespacedName types.NamespacedName
}

func (c *InferenceModelReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	if req.Namespace != c.PoolNamespacedName.Namespace {
		return ctrl.Result{}, nil
	}

	logger := log.FromContext(ctx)
	loggerDefault := logger.V(logutil.DEFAULT)
	loggerDefault.Info("Reconciling InferenceModel", "name", req.NamespacedName)

	infModel := &v1alpha1.InferenceModel{}
	if err := c.Get(ctx, req.NamespacedName, infModel); err != nil {
		if errors.IsNotFound(err) {
			loggerDefault.Info("InferenceModel not found. Removing from datastore since object must be deleted", "name", req.NamespacedName)
			c.Datastore.ModelDelete(infModel.Spec.ModelName)
			return ctrl.Result{}, nil
		}
		loggerDefault.Error(err, "Unable to get InferenceModel", "name", req.NamespacedName)
		return ctrl.Result{}, err
	} else if !infModel.DeletionTimestamp.IsZero() {
		loggerDefault.Info("InferenceModel is marked for deletion. Removing from datastore", "name", req.NamespacedName)
		c.Datastore.ModelDelete(infModel.Spec.ModelName)
		return ctrl.Result{}, nil
	}

	c.updateDatastore(logger, infModel)
	return ctrl.Result{}, nil
}

func (c *InferenceModelReconciler) updateDatastore(logger logr.Logger, infModel *v1alpha1.InferenceModel) {
	loggerDefault := logger.V(logutil.DEFAULT)

	if infModel.Spec.PoolRef.Name == c.PoolNamespacedName.Name {
		loggerDefault.Info("Updating datastore", "poolRef", infModel.Spec.PoolRef, "serverPoolName", c.PoolNamespacedName)
		loggerDefault.Info("Adding/Updating InferenceModel", "modelName", infModel.Spec.ModelName)
		c.Datastore.ModelSet(infModel)
		return
	}
	loggerDefault.Info("Removing/Not adding InferenceModel", "modelName", infModel.Spec.ModelName)
	// If we get here. The model is not relevant to this pool, remove.
	c.Datastore.ModelDelete(infModel.Spec.ModelName)
}

func (c *InferenceModelReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.InferenceModel{}).
		Complete(c)
}
