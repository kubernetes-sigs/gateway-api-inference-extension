/*
Copyright 2026 The Kubernetes Authors.

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

package plugins

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	BodyFieldToHeaderPluginType = "body-field-to-header"
)

// compile-time type validation
var _ framework.RequestProcessor = &BodyFieldToHeaderPlugin{}

// BodyFieldToHeaderConfig defines the JSON configuration structure for the plugin.
type BodyFieldToHeaderConfig struct {
	// FieldName is the name of the body field to extract
	FieldName string `json:"field_name"`
	// HeaderName is the name of the header to set
	HeaderName string `json:"header_name"`
}

// BodyFieldToHeaderPluginFactory defines the factory function for NewBodyFieldToHeaderPlugin.
func BodyFieldToHeaderPluginFactory(name string, rawParameters json.RawMessage) (framework.BBRPlugin, error) {
	var config BodyFieldToHeaderConfig

	if len(rawParameters) > 0 {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", BodyFieldToHeaderPluginType, err)
		}
	}

	plugin, err := NewBodyFieldToHeaderPlugin(config.FieldName, config.HeaderName)
	if err != nil {
		return nil, fmt.Errorf("failed to create '%s' plugin - %w", BodyFieldToHeaderPluginType, err)
	}

	return plugin.WithName(name), nil
}

// NewBodyFieldToHeaderPlugin initializes a new BodyFieldToHeaderPlugin and returns its pointer.
func NewBodyFieldToHeaderPlugin(fieldName, headerName string) (*BodyFieldToHeaderPlugin, error) {
	if fieldName == "" {
		return nil, errors.New("body fieldName is required in BodyFieldToHeader plugin")
	}

	if headerName == "" {
		return nil, errors.New("headerName is required in BodyFieldToHeader plugin")
	}

	return &BodyFieldToHeaderPlugin{
		typedName: plugin.TypedName{
			Type: BodyFieldToHeaderPluginType,
			Name: BodyFieldToHeaderPluginType,
		},
		fieldName:  fieldName,
		headerName: headerName,
	}, nil
}

// BodyFieldToHeaderPlugin extracts value from a given body field and sets it as HTTP header.
type BodyFieldToHeaderPlugin struct {
	typedName  plugin.TypedName
	fieldName  string
	headerName string
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *BodyFieldToHeaderPlugin) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the name of the plugin instance.
func (p *BodyFieldToHeaderPlugin) WithName(name string) *BodyFieldToHeaderPlugin {
	p.typedName.Name = name
	return p
}

// ProcessRequest extracts value from a given body field and sets it as HTTP header.
func (p *BodyFieldToHeaderPlugin) ProcessRequest(ctx context.Context, _ *framework.CycleState, request *framework.InferenceRequest) error {
	if request == nil || request.Headers == nil || request.Body == nil {
		return nil // this shouldn't happen
	}

	// extract raw field value from body
	rawFieldValue, exists := request.Body[p.fieldName]
	if !exists {
		metrics.RecordBodyFieldNotFound(p.fieldName)
		return fmt.Errorf("field '%s' not found in request body", p.fieldName)
	}

	fieldStr := fmt.Sprintf("%v", rawFieldValue) // convert any type to string
	if fieldStr == "" {
		metrics.RecordBodyFieldEmpty(p.fieldName)
		return fmt.Errorf("field '%s' is empty and couldn't be processed", p.fieldName)
	}

	log.FromContext(ctx).V(logutil.VERBOSE).Info("parsed field from body", "field", p.fieldName, "value", fieldStr)
	request.SetHeader(p.headerName, fieldStr)

	return nil
}
