package backend

import (
	"context"
	"strconv"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

type PodReconciler struct {
	client.Client
	Datastore *K8sDatastore
	Scheme    *runtime.Scheme
	Record    record.EventRecorder
}

func (c *PodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	inferencePool, err := c.Datastore.getInferencePool()
	if err != nil {
		logger.V(logutil.DEFAULT).Info("Skipping reconciling Pod because the InferencePool is not available yet", "error", err)
		// When the inferencePool is initialized it lists the appropriate pods and populates the datastore, so no need to requeue.
		return ctrl.Result{}, nil
	} else if inferencePool.Namespace != req.Namespace {
		return ctrl.Result{}, nil
	}

	logger.V(logutil.VERBOSE).Info("Pod being reconciled", "name", req.NamespacedName)

	pod := &corev1.Pod{}
	if err := c.Get(ctx, req.NamespacedName, pod); err != nil {
		logger.V(logutil.DEFAULT).Error(err, "Unabled to get pod", "name", req.NamespacedName)
		if apierrors.IsNotFound(err) {
			c.Datastore.pods.Delete(pod)
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	c.updateDatastore(pod, inferencePool)

	return ctrl.Result{}, nil
}

func (c *PodReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&corev1.Pod{}).
		Complete(c)
}

func (c *PodReconciler) updateDatastore(k8sPod *corev1.Pod, inferencePool *v1alpha1.InferencePool) {
	pod := Pod{
		Name:    k8sPod.Name,
		Address: k8sPod.Status.PodIP + ":" + strconv.Itoa(int(inferencePool.Spec.TargetPortNumber)),
	}
	if !k8sPod.DeletionTimestamp.IsZero() || !c.Datastore.LabelsMatch(k8sPod.ObjectMeta.Labels) || !podIsReady(k8sPod) {
		c.Datastore.pods.Delete(pod)
	} else {
		c.Datastore.pods.Store(pod, true)
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
