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

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const nilString = "<nil>"

// +kubebuilder:object:root=true

// EndpointPickerConfig is the Schema for the endpointpickerconfigs API
type EndpointPickerConfig struct {
	metav1.TypeMeta `json:",inline"`

	// +optional
	// FeatureGates is a set of flags that enable various experimental features with the EPP.
	// If omitted none of these experimental features will be enabled.
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
	// Data configures the DataLayer. It is required if the new DataLayer is enabled.
	Data *DataLayerConfig `json:"data"`

	// +optional
	// FlowControl configures the Flow Control layer.
	// This configuration is only respected if the "flowControl" FeatureGate is enabled.
	FlowControl *FlowControlConfig `json:"flowControl,omitempty"`

	// +optional
	// Parser specifies the parsing logic used by the EPP to process protocol messages.
	// If unspecified, default parsing behavior will be applied.
	Parser *ParserConfig `json:"parser,omitempty"`
}

func (cfg EndpointPickerConfig) String() string {
	var parts []string
	if len(cfg.FeatureGates) > 0 {
		parts = append(parts, fmt.Sprintf("FeatureGates: %s", cfg.FeatureGates))
	}
	if len(cfg.Plugins) > 0 {
		parts = append(parts, fmt.Sprintf("Plugins: %v", cfg.Plugins))
	}
	if len(cfg.SchedulingProfiles) > 0 {
		parts = append(parts, fmt.Sprintf("SchedulingProfiles: %v", cfg.SchedulingProfiles))
	}
	if cfg.Data != nil {
		parts = append(parts, fmt.Sprintf("Data: %v", cfg.Data))
	}
	if cfg.FlowControl != nil {
		parts = append(parts, fmt.Sprintf("FlowControl: %v", cfg.FlowControl))
	}
	if cfg.Parser != nil {
		parts = append(parts, fmt.Sprintf("Parser: %v", cfg.Parser))
	}
	return "{" + strings.Join(parts, ", ") + "}"
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
	var parts []string
	if ps.Name != "" {
		parts = append(parts, "Name: "+ps.Name)
	}
	parts = append(parts, "Type: "+ps.Type)
	if len(ps.Parameters) > 0 {
		parts = append(parts, "Parameters: "+string(ps.Parameters))
	}
	return "{" + strings.Join(parts, ", ") + "}"
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
	var parts []string
	parts = append(parts, "Name: "+sp.Name)
	if len(sp.Plugins) > 0 {
		parts = append(parts, fmt.Sprintf("Plugins: %v", sp.Plugins))
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

// SchedulingPlugin describes a plugin that will be associated with a
// SchedulingProfile entry.
type SchedulingPlugin struct {
	// +required
	// +kubebuilder:validation:Required
	// PluginRef specifies a particular Plugin instance to be associated with
	// this SchedulingProfile. The reference is to the name of an
	// entry of the Plugins defined in the configuration's Plugins
	// section
	PluginRef string `json:"pluginRef"`

	// +optional
	// Weight is the weight to be used if this plugin is a Scorer.
	Weight *float64 `json:"weight"`
}

func (sp SchedulingPlugin) String() string {
	var parts []string
	parts = append(parts, "PluginRef: "+sp.PluginRef)
	if sp.Weight != nil {
		parts = append(parts, fmt.Sprintf("Weight: %.2f", *sp.Weight))
	}
	return "{" + strings.Join(parts, ", ") + "}"
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

// DataLayerConfig contains the configuration of the DataLayer feature
type DataLayerConfig struct {
	// +required
	// +kubebuilder:validation:Required
	// Sources is the list of sources to define to the DataLayer
	Sources []DataLayerSource `json:"sources"`
}

func (dlc *DataLayerConfig) String() string {
	if dlc == nil {
		return nilString
	}
	return fmt.Sprintf("{Sources: %v}", dlc.Sources)
}

// DataLayerSource contains the configuration of a DataSource of the DataLayer feature
type DataLayerSource struct {
	// +required
	// +kubebuilder:validation:Required
	// PluginRef specifies a particular Plugin instance to be associated with
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
	var parts []string
	parts = append(parts, "PluginRef: "+dls.PluginRef)
	if len(dls.Extractors) > 0 {
		parts = append(parts, fmt.Sprintf("Extractors: %v", dls.Extractors))
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

// DataLayerExtractor contains the configuration of an Extractor of the DataLayer feature
type DataLayerExtractor struct {
	// +required
	// +kubebuilder:validation:Required
	// PluginRef specifies a particular Plugin instance to be associated with
	// this Extractor. The reference is to the name of an entry of the Plugins
	// defined in the configuration's Plugins section
	PluginRef string `json:"pluginRef"`
}

func (dle DataLayerExtractor) String() string {
	return fmt.Sprintf("{PluginRef: %s}", dle.PluginRef)
}

func (pc *ParserConfig) String() string {
	if pc == nil {
		return nilString
	}
	return fmt.Sprintf("{PluginRef: %s}", pc.PluginRef)
}

// ParserConfig contains the configuration for a parser.
type ParserConfig struct {
	// +required
	// +kubebuilder:validation:Required
	// PluginRef specifies a particular Plugin instance to be associated with
	// this Parser. The reference is to the name of an entry of the Plugins
	// defined in the configuration's Plugins section
	// Default: openai-parser
	PluginRef string `json:"pluginRef"`
}

// FlowControlConfig configures the Flow Control layer.
type FlowControlConfig struct {
	// +optional
	// MaxBytes is the global maximum number of bytes allowed across all priority levels.
	// If exceeded, new requests will be rejected even if their specific priority band has capacity.
	// Accepts standard Kubernetes resource quantities (e.g., "1Gi", "500M").
	// If not specified, no global limits are enforced.
	MaxBytes *resource.Quantity `json:"maxBytes,omitempty"`

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

	// +optional
	// SaturationDetectorRef specifies the name of the plugin instance to use for saturation detection.
	// The reference is to the name of an entry of the Plugins defined in the configuration's Plugins section.
	// If unspecified, "utilization-detector" is used by default.
	SaturationDetectorRef string `json:"saturationDetectorRef,omitempty"`
}

func (fcc *FlowControlConfig) String() string {
	if fcc == nil {
		return nilString
	}

	var parts []string

	if fcc.SaturationDetectorRef != "" {
		parts = append(parts, "SaturationDetectorRef: "+fcc.SaturationDetectorRef)
	}

	if fcc.MaxBytes != nil {
		parts = append(parts, "MaxBytes: "+fcc.MaxBytes.String())
	} else {
		parts = append(parts, "MaxBytes: unlimited")
	}

	if fcc.DefaultRequestTTL != nil {
		parts = append(parts, fmt.Sprintf("DefaultRequestTTL: %s", fcc.DefaultRequestTTL.Duration))
	}

	if fcc.DefaultPriorityBand != nil {
		parts = append(parts, fmt.Sprintf("DefaultPriorityBand: %v", fcc.DefaultPriorityBand))
	}

	if len(fcc.PriorityBands) > 0 {
		parts = append(parts, fmt.Sprintf("PriorityBands: %v", fcc.PriorityBands))
	}

	return "{" + strings.Join(parts, ", ") + "}"
}

// PriorityBandConfig configures a single priority band.
type PriorityBandConfig struct {
	// Priority is the integer priority level for this band.
	// Higher values indicate higher priority.
	Priority int `json:"priority"`

	// +optional
	// MaxBytes is the maximum number of bytes allowed for this priority band.
	// If exceeded, new requests at this priority will be shed.
	// Accepts standard Kubernetes resource quantities (e.g., "1Gi", "500M").
	// If not specified, the system default is used (e.g., 1 GB).
	MaxBytes *resource.Quantity `json:"maxBytes,omitempty"`

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
	var parts []string
	parts = append(parts, fmt.Sprintf("Priority: %d", pbc.Priority))

	if pbc.MaxBytes != nil {
		parts = append(parts, fmt.Sprintf("MaxBytes: %d", pbc.MaxBytes.Value()))
	}

	if pbc.FairnessPolicyRef != "" {
		parts = append(parts, "FairnessPolicyRef: "+pbc.FairnessPolicyRef)
	}

	if pbc.OrderingPolicyRef != "" {
		parts = append(parts, "OrderingPolicyRef: "+pbc.OrderingPolicyRef)
	}

	return "{" + strings.Join(parts, ", ") + "}"
}
