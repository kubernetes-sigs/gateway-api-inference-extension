package scheduling

import (
	"encoding/json"
	"fmt"

	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	corev1 "k8s.io/api/core/v1"
	klog "k8s.io/klog/v2"
)

type FilterOrchestrator interface {
	Orchestrate() FilterChain
}

func NewFilterOrchestrator(datastore *backend.K8sDatastore) *FilterOrchestratorImpl {
	return &FilterOrchestratorImpl{
		datastore: datastore,
	}
}

type FilterOrchestratorImpl struct {
	datastore    *backend.K8sDatastore
	lastUpdated  string
	storedFilter *filterChainImpl
}

var _ FilterOrchestrator = &FilterOrchestratorImpl{}

type FilterOrchestration struct {
	Name                   string               `json:"name"`
	NextOnSuccess          *FilterOrchestration `json:"nextOnSuccess,omitempty"`
	NextOnFailure          *FilterOrchestration `json:"nextOnFailure,omitempty"`
	NextOnSuccessOrFailure *FilterOrchestration `json:"nextOnSuccessOrFailure,omitempty"`
	FilterOption           *FilterOption        `json:"filterOption,omitempty"`
}

func (o *FilterOrchestratorImpl) Orchestrate() FilterChain {
	if o == nil {
		return defaultFilter
	}

	cm := o.datastore.GetFilterConfigMap()
	if cm == nil {
		return defaultFilter
	}

	if o.lastUpdated == lastUpdatedKey(cm) {
		return o.storedFilter
	}

	o.lastUpdated = lastUpdatedKey(cm)

	f := &FilterOrchestration{}
	if err := json.Unmarshal([]byte(cm.Data["filter"]), f); err != nil {
		o.storedFilter = defaultFilter
		klog.Errorf("error unmarshalling filter config: %v", err)
		return defaultFilter
	}

	filter, err := o.orchestrate(f)
	if err != nil {
		klog.Errorf("error orchestrating filters: %v", err)
		filter = defaultFilter
	}

	klog.V(1).Infof("filter orchestrated")
	o.storedFilter = filter
	return filter
}

func (o *FilterOrchestratorImpl) orchestrate(fo *FilterOrchestration) (*filterChainImpl, error) {
	if fo == nil {
		return nil, nil
	}

	fg, ok := filterMap[fo.Name]
	if !ok {
		return nil, fmt.Errorf("unknown filter %s", fo.Name)
	}

	if err := fg.Validate(fo.FilterOption); err != nil {
		return nil, err
	}

	filter := &filterChainImpl{
		filter: fg.Get(fo.FilterOption),
		name:   fg.Name(),
	}

	nextOnSuccess, err := o.orchestrate(fo.NextOnSuccess)
	if err != nil {
		return nil, err
	}
	nextOnFailure, err := o.orchestrate(fo.NextOnFailure)
	if err != nil {
		return nil, err
	}
	nextOnSuccessOrFailure, err := o.orchestrate(fo.NextOnSuccessOrFailure)
	if err != nil {
		return nil, err
	}
	filter.nextOnFailure = nextOnFailure
	filter.nextOnSuccess = nextOnSuccess
	filter.nextOnSuccessOrFailure = nextOnSuccessOrFailure
	return filter, nil
}

func lastUpdatedKey(cm *corev1.ConfigMap) string {
	return string(cm.UID) + cm.ResourceVersion
}
