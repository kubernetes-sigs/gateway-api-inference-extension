package backend

import (
	"context"
	"strconv"
	"time"

	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	logutil "inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type PodReconciler struct {
	client.Client
	Datastore *K8sDatastore
	Scheme    *runtime.Scheme
	Record    record.EventRecorder
}

func (c *PodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	inferencePool, err := c.Datastore.getInferencePool()
	if err != nil {
		klog.V(logutil.DEFAULT).Infof("Skipping reconciling EndpointSlice because the InferencePool is not available yet: %v", err)
		return ctrl.Result{Requeue: true, RequeueAfter: time.Second}, nil
	}

	klog.V(logutil.VERBOSE).Info("reconciling Pod", req.NamespacedName)

	pod := &corev1.Pod{}
	if err := c.Get(ctx, req.NamespacedName, pod); err != nil {
		klog.Error(err, "unable to get InferencePool")
		return ctrl.Result{}, err
	}

	if !podIsReady(pod) {
		return ctrl.Result{RequeueAfter: time.Second * 5}, nil
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
	if !c.Datastore.LabelsMatch(k8sPod.ObjectMeta.Labels) {
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
