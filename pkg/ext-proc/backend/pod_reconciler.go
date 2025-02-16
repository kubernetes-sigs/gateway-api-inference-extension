package backend

import (
	"context"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

type PodReconciler struct {
	client.Client
	Datastore Datastore
	Scheme    *runtime.Scheme
	Record    record.EventRecorder
}

func (c *PodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	inferencePool, err := c.Datastore.PoolGet()
	if err != nil {
		logger.V(logutil.TRACE).Info("Skipping reconciling Pod because the InferencePool is not available yet", "error", err)
		// When the inferencePool is initialized it lists the appropriate pods and populates the datastore, so no need to requeue.
		return ctrl.Result{}, nil
	} else if inferencePool.Namespace != req.Namespace {
		return ctrl.Result{}, nil
	}

	logger.V(logutil.VERBOSE).Info("Pod being reconciled", "name", req.NamespacedName)

	pod := &corev1.Pod{}
	if err := c.Get(ctx, req.NamespacedName, pod); err != nil {
		if apierrors.IsNotFound(err) {
			c.Datastore.PodDelete(req.NamespacedName)
			return ctrl.Result{}, nil
		}
		logger.V(logutil.DEFAULT).Error(err, "Unable to get pod", "name", req.NamespacedName)
		return ctrl.Result{}, err
	}

	c.updateDatastore(logger, pod)
	return ctrl.Result{}, nil
}

func (c *PodReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&corev1.Pod{}).
		Complete(c)
}

func (c *PodReconciler) updateDatastore(logger logr.Logger, pod *corev1.Pod) {
	namespacedName := types.NamespacedName{Name: pod.Name, Namespace: pod.Namespace}
	if !pod.DeletionTimestamp.IsZero() || !c.Datastore.PoolLabelsMatch(pod.Labels) || !podIsReady(pod) {
		logger.V(logutil.DEFAULT).Info("Pod removed or not added", "name", namespacedName)
		c.Datastore.PodDelete(namespacedName)
	} else {
		if c.Datastore.PodAddIfNotExist(pod) {
			logger.V(logutil.DEFAULT).Info("Pod added", "name", namespacedName)
		} else {
			logger.V(logutil.DEFAULT).Info("Pod already exists", "name", namespacedName)
		}
	}
}

func podIsReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			if condition.Status == corev1.ConditionTrue {
				return true
			}
			break
		}
	}
	return false
}
