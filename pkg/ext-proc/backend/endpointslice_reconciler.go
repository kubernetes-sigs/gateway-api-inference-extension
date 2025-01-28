package backend

import (
	"context"
	"strconv"

	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	logutil "inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	klog "k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

var (
	serviceOwnerLabel = "kubernetes.io/service-name"
)

type EndpointSliceReconciler struct {
	client.Client
	Scheme      *runtime.Scheme
	Record      record.EventRecorder
	ServiceName string
	Zone        string
	Datastore   *K8sDatastore
}

func (c *EndpointSliceReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	klog.V(logutil.DEFAULT).Info("Reconciling EndpointSlice ", req.NamespacedName)

	endpointSlice := &discoveryv1.EndpointSlice{}
	if err := c.Get(ctx, req.NamespacedName, endpointSlice); err != nil {
		klog.Errorf("Unable to get EndpointSlice: %v", err)
		return ctrl.Result{}, err
	}
	inferencePool, err := c.Datastore.getInferencePool()
	if err != nil {
		return ctrl.Result{}, err
	}
	c.updateDatastore(endpointSlice, inferencePool)

	return ctrl.Result{}, nil
}

// TODO: Support multiple endpointslices for a single service
func (c *EndpointSliceReconciler) updateDatastore(
	slice *discoveryv1.EndpointSlice,
	inferencePool *v1alpha1.InferencePool) {
	podMap := make(map[Pod]bool)

	for _, endpoint := range slice.Endpoints {
		klog.V(logutil.DEFAULT).Infof("Zone: %v \n endpoint: %+v \n", c.Zone, endpoint)
		if c.validPod(endpoint) {
			pod := Pod{
				Name:    endpoint.TargetRef.Name,
				Address: endpoint.Addresses[0] + ":" + strconv.Itoa(int(inferencePool.Spec.TargetPortNumber)),
			}
			podMap[pod] = true
			klog.V(logutil.DEFAULT).Infof("Storing pod %v", pod)
			c.Datastore.pods.Store(pod, true)
		}
	}

	removeOldPods := func(k, v any) bool {
		pod, ok := k.(Pod)
		if !ok {
			klog.Errorf("Unable to cast key to Pod: %v", k)
			return false
		}
		if _, ok := podMap[pod]; !ok {
			klog.V(logutil.DEFAULT).Infof("Removing pod %v", pod)
			c.Datastore.pods.Delete(pod)
		}
		return true
	}
	c.Datastore.pods.Range(removeOldPods)
}

func (c *EndpointSliceReconciler) SetupWithManager(mgr ctrl.Manager) error {
	inferencePoolAvailable := func(object client.Object) bool {
		_, err := c.Datastore.getInferencePool()
		if err != nil {
			klog.V(logutil.DEFAULT).Infof("Skipping reconciling EndpointSlice because the InferencePool is not available yet: %v", err)
		}
		return err == nil
	}

	ownsEndPointSlice := func(object client.Object) bool {
		// Check if the object is an EndpointSlice
		endpointSlice, ok := object.(*discoveryv1.EndpointSlice)
		if !ok {
			return false
		}

		gotLabel := endpointSlice.ObjectMeta.Labels[serviceOwnerLabel]
		wantLabel := c.ServiceName
		if gotLabel != wantLabel {
			namesapcedName := endpointSlice.ObjectMeta.Namespace + "/" + endpointSlice.ObjectMeta.Name
			klog.V(logutil.DEFAULT).Infof("Skipping EndpointSlice %v because its service owner label %v doesn't match the pool service name %v", namesapcedName, gotLabel, wantLabel)
		}
		return gotLabel == wantLabel
	}

	return ctrl.NewControllerManagedBy(mgr).
		For(&discoveryv1.EndpointSlice{},
			builder.WithPredicates(predicate.NewPredicateFuncs(inferencePoolAvailable),
				predicate.NewPredicateFuncs(ownsEndPointSlice))).
		Complete(c)
}

func (c *EndpointSliceReconciler) validPod(endpoint discoveryv1.Endpoint) bool {
	validZone := c.Zone == "" || c.Zone != "" && *endpoint.Zone == c.Zone
	return validZone && *endpoint.Conditions.Ready

}
