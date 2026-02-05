/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	"encoding/json"
	"fmt"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +kubebuilder:object:root=true

// EndpointPickerConfig is the Schema for the endpointpickerconfigs API
type EndpointPickerConfig struct {
	metav1.TypeMeta `json:",inline"`

	// +optional
	// FeatureGates is a set of flags that enable various experimental features with the EPP.
	// If omitted non of these experimental features will be enabled.
	FeatureGates FeatureGates `json:"featureGates,omitempty"`

	// +required
	// +kubebuilder:validation:Required
	// Plugins is the list of plugins that will be instantiated.
	Plugins []PluginSpec `json:"plugins"`

	// +required
	// +kubebuilder:validation:Required
	// SchedulingProfiles is the list of named SchedulingProfiles
	// that will be created.
	SchedulingProfiles []SchedulingProfile `json:"schedulingProfiles"`

	// +optional
	// SaturationDetector when present specifies the configuration of the
	// Saturation detector. If not present, default values are used.
	SaturationDetector *SaturationDetector `json:"saturationDetector,omitempty"`

	// +optional
	// Data configures the DataLayer. It is required if the new DataLayer is enabled.
	Data *DataLayerConfig `json:"data"`

	// +optional
	// FlowControl configures the Flow Control layer.
	// This configuration is only respected if the "flowControl" FeatureGate is enabled.
	FlowControl *FlowControlConfig `json:"flowControl,omitempty"`

	// +optional
	// RequestControl configures the request control stage of the EPP pipeline.
	RequestControl *RequestControlConfig `json:"requestControl,omitempty"`
}

func (cfg EndpointPickerConfig) String() string {
	return fmt.Sprintf(
		"{FeatureGates: %v, Plugins: %v, SchedulingProfiles: %v, Data: %v, SaturationDetector: %v, FlowControl: %v, RequestControl: %v}",
		cfg.FeatureGates,
		cfg.Plugins,
		cfg.SchedulingProfiles,
		cfg.Data,
		cfg.SaturationDetector,
		cfg.FlowControl,
		cfg.RequestControl,
	)
}

// PluginSpec contains the information that describes a plugin that
// will be instantiated.
type PluginSpec struct {
	// +optional
	// Name provides a name for plugin entries to reference. If
	// omitted, the value of the Plugin's Type field will be used.
	Name string `json:"name"`

	// +required
	// +kubebuilder:validation:Required
	// Type specifies the plugin type to be instantiated.
	Type string `json:"type"`

	// +optional
	// Parameters are the set of parameters to be passed to the plugin's
	// factory function. The factory function is responsible
	// to parse the parameters.
	Parameters json.RawMessage `json:"parameters"`
}

func (ps PluginSpec) String() string {
	var parameters string
	if ps.Parameters != nil {
		parameters = fmt.Sprintf(", Parameters: %s", ps.Parameters)
	}
	return fmt.Sprintf("{%s/%s%s}", ps.Name, ps.Type, parameters)
}

// SchedulingProfile contains the information to create a SchedulingProfile
// entry to be used by the scheduler.
type SchedulingProfile struct {
	// +kubebuilder:validation:Required
	// Name specifies the name of this SchedulingProfile
	Name string `json:"name"`

	// +required
	// +kubebuilder:validation:Required
	// Plugins is the list of plugins for this SchedulingProfile. They are assigned
	// to the appropriate "slots" based on their type.
	Plugins []SchedulingPlugin `json:"plugins"`
}

func (sp SchedulingProfile) String() string {
	return fmt.Sprintf("{Name: %s, Plugins: %v}", sp.Name, sp.Plugins)
}

// SchedulingPlugin describes a plugin that will be associated with a
// SchedulingProfile entry.
type SchedulingPlugin struct {
	// +required
	// +kubebuilder:validation:Required
	// PluginRef specifies a partiular Plugin instance to be associated with
	// this SchedulingProfile. The reference is to the name of an
	// entry of the Plugins defined in the configuration's Plugins
	// section
	PluginRef string `json:"pluginRef"`

	// +optional
	// Weight is the weight fo be used if this plugin is a Scorer.
	Weight *float64 `json:"weight"`
}

func (sp SchedulingPlugin) String() string {
	var weight string
	if sp.Weight != nil {
		weight = fmt.Sprintf(", Weight: %f", *sp.Weight)
	}
	return fmt.Sprintf("{PluginRef: %s%s}", sp.PluginRef, weight)
}

// FeatureGates is a set of flags that enable various experimental features with the EPP
type FeatureGates []string

func (fg FeatureGates) String() string {
	if fg == nil {
		return "{}"
	}

	builder := strings.Builder{}
	for _, gate := range fg {
		builder.WriteString(gate + ",")
	}

	result := builder.String()
	if len(result) > 0 {
		result = result[:len(result)-1]
	}
	return "{" + result + "}"
}

// SaturationDetector
type SaturationDetector struct {
	// +optional
	// QueueDepthThreshold defines the backend waiting queue size above which a
	// pod is considered to have insufficient capacity for new requests.
	QueueDepthThreshold int `json:"queueDepthThreshold,omitempty"`

	// +optional
	// KVCacheUtilThreshold defines the KV cache utilization (0.0 to 1.0) above
	// which a pod is considered to have insufficient capacity.
	KVCacheUtilThreshold float64 `json:"kvCacheUtilThreshold,omitempty"`

	// +optional
	// MetricsStalenessThreshold defines how old a pod's metrics can be.
	// If a pod's metrics are older than this, it might be excluded from
	// "good capacity" considerations or treated as having no capacity for
	// safety.
	MetricsStalenessThreshold metav1.Duration `json:"metricsStalenessThreshold,omitempty"`
}

// RequestControlConfig configures the request control stage.
type RequestControlConfig struct {
	// +optional
	// PrepareDataTimeout defines the timeout for PrepareData plugins.
	// If omitted, defaults to 400ms.
	PrepareDataTimeout *metav1.Duration `json:"prepareDataTimeout,omitempty"`
}

func (rc *RequestControlConfig) String() string {
	if rc == nil {
		return "{}"
	}
	if rc.PrepareDataTimeout == nil {
		return "{PrepareDataTimeout: default(400ms)}"
	}
	return "{PrepareDataTimeout: " + rc.PrepareDataTimeout.String() + "}"
}

func (sd *SaturationDetector) String() string {
	result := ""
	if sd != nil {
		if sd.QueueDepthThreshold != 0 {
			result += fmt.Sprintf("QueueDepthThreshold: %d", sd.QueueDepthThreshold)
		}
		if sd.KVCacheUtilThreshold != 0.0 {
			if len(result) != 0 {
				result += ", "
			}
			result += fmt.Sprintf("KVCacheUtilThreshold: %g", sd.KVCacheUtilThreshold)
		}
		if sd.MetricsStalenessThreshold.Duration != 0.0 {
			if len(result) != 0 {
				result += ", "
			}
			result += fmt.Sprintf("MetricsStalenessThreshold: %s", sd.MetricsStalenessThreshold)
		}
	}
	return "{" + result + "}"
}

// DataLayerConfig contains the configuration of the DataLayer feature
type DataLayerConfig struct {
	// +required
	// +kubebuilder:validation:Required
	// Sources is the list of sources to define to the DataLayer
	Sources []DataLayerSource `json:"sources"`
}

func (dlc DataLayerConfig) String() string {
	return fmt.Sprintf("{Sources: %v}", dlc.Sources)
}

// DataLayerSource contains the configuration of a DataSource of the DataLayer feature
type DataLayerSource struct {
	// +required
	// +kubebuilder:validation:Required
	// PluginRef specifies a partiular Plugin instance to be associated with
	// this Source. The reference is to the name of an entry of the Plugins
	// defined in the configuration's Plugins section
	PluginRef string `json:"pluginRef"`

	// +required
	// +kubebuilder:validation:Required
	// Extractors specifies the list of Plugin instances to be associated with
	// this Source. The entries are references to the names of entries of the Plugins
	// defined in the configuration's Plugins section
	Extractors []DataLayerExtractor `json:"extractors"`
}

func (dls DataLayerSource) String() string {
	return fmt.Sprintf("{PluginRef: %s, Extractors: %v}", dls.PluginRef, dls.Extractors)
}

// DataLayerExtractor contains the configuration of an Extractor of the DataLayer feature
type DataLayerExtractor struct {
	// +required
	// +kubebuilder:validation:Required
	// PluginRef specifies a partiular Plugin instance to be associated with
	// this Extractor. The reference is to the name of an entry of the Plugins
	// defined in the configuration's Plugins section
	PluginRef string `json:"pluginRef"`
}

func (dle DataLayerExtractor) String() string {
	return "{PluginRef: " + dle.PluginRef + "}"
}

// FlowControlConfig configures the Flow Control layer.
type FlowControlConfig struct {
	// +optional
	// MaxBytes defines the global capacity limit for the Flow Control system.
	// It represents the maximum aggregate byte size of all active requests across all priority
	// levels. If this limit is exceeded, new requests will be rejected even if their specific
	// priority band has capacity.
	// If 0 or omitted, no global limit is enforced (unlimited).
	// Default: 0 (unlimited).
	MaxBytes *int64 `json:"maxBytes,omitempty"`

	// +optional
	// DefaultRequestTTL serves as a fallback timeout for requests that do not specify their own
	// deadline.
	// It ensures that requests do not hang indefinitely in the queue.
	// If 0 or omitted, it defaults to the client context deadline, meaning requests may wait
	// indefinitely unless cancelled by the client.
	DefaultRequestTTL *metav1.Duration `json:"defaultRequestTTL,omitempty"`

	// +optional
	// DefaultPriorityBand allows you to define a template for handling traffic with priority levels
	// that are not explicitly configured in `PriorityBands`.
	// This ensures that unforeseen traffic classes are still managed (e.g., given a default capacity
	// limit) rather than being rejected or treated arbitrarily.
	// If not specified, a system-default template is used to dynamically provision bands for new
	// priority levels. This template cascades to the standard `PriorityBandConfig` defaults.
	DefaultPriorityBand *PriorityBandConfig `json:"defaultPriorityBand,omitempty"`

	// PriorityBands allows you to explicitly define policies (like capacity limits) for specific
	// priority levels. Traffic matching these priorities will be handled according to these rules.
	// If a priority band is not specified, it uses specific defaults.
	PriorityBands []PriorityBandConfig `json:"priorityBands,omitempty"`
}

func (fcc *FlowControlConfig) String() string {
	return fmt.Sprintf("{MaxBytes: %v, DefaultPriorityBand: %v, PriorityBands: %v}",
		fcc.MaxBytes, fcc.DefaultPriorityBand, fcc.PriorityBands)
}

// PriorityBandConfig configures a single priority band.
type PriorityBandConfig struct {
	// Priority is the integer priority level for this band.
	// Higher values indicate higher priority.
	Priority int `json:"priority"`

	// +optional
	// MaxBytes is the maximum number of bytes allowed for this priority band.
	// If 0 or omitted, the system default is used.
	// Default: 1 GB.
	MaxBytes *int64 `json:"maxBytes,omitempty"`

	// +optional
	// FairnessPolicyRef specifies the name of the policy that governs flow selection.
	// If omitted, the system default ("global-strict-fairness-policy") is used.
	FairnessPolicyRef string `json:"fairnessPolicyRef,omitempty"`

	// +optional
	// OrderingPolicyRef specifies the name of the policy that governs request selection within a flow.
	// If omitted, the system default ("fcfs-ordering-policy") is used.
	OrderingPolicyRef string `json:"orderingPolicyRef,omitempty"`
}

func (pbc PriorityBandConfig) String() string {
	return fmt.Sprintf("{Priority: %d, MaxBytes: %v, FairnessPolicyRef: %s, OrderingPolicyRef: %s}",
		pbc.Priority, pbc.MaxBytes, pbc.FairnessPolicyRef, pbc.OrderingPolicyRef)
}
