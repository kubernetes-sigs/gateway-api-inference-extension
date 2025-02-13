package backend

import (
	"context"

	klog "k8s.io/klog/v2"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

type FakePodMetricsClient struct {
	Err map[Pod]error
	Res map[Pod]*PodMetrics
}

func (f *FakePodMetricsClient) FetchMetrics(ctx context.Context, pod Pod, existing *PodMetrics) (*PodMetrics, error) {
	if err, ok := f.Err[pod]; ok {
		return nil, err
	}
	klog.V(logutil.VERBOSE).InfoS("Fetching metrics for pod", "pod", pod, "existing", existing, "new", f.Res[pod])
	return f.Res[pod], nil
}

type FakeDataStore struct {
	Res map[string]*v1alpha1.InferenceModel
}

func (fds *FakeDataStore) FetchModelData(modelName string) (returnModel *v1alpha1.InferenceModel) {
	return fds.Res[modelName]
}
