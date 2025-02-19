package backend

import (
	"context"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

type FakePodMetricsClient struct {
	Err map[types.NamespacedName]error
	Res map[types.NamespacedName]*datastore.PodMetrics
}

func (f *FakePodMetricsClient) FetchMetrics(ctx context.Context, existing *datastore.PodMetrics) (*datastore.PodMetrics, error) {
	if err, ok := f.Err[existing.NamespacedName]; ok {
		return nil, err
	}
	log.FromContext(ctx).V(logutil.VERBOSE).Info("Fetching metrics for pod", "existing", existing, "new", f.Res[existing.NamespacedName])
	return f.Res[existing.NamespacedName], nil
}

type FakeDataStore struct {
	Res map[string]*v1alpha1.InferenceModel
}

func (fds *FakeDataStore) FetchModelData(modelName string) (returnModel *v1alpha1.InferenceModel) {
	return fds.Res[modelName]
}
