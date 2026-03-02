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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	BodyFieldToHeaderPluginType = "body-field-to-header"
	deafultHeaderPrefix         = "X-Gateway-"
)

// compile-time type validation
var _ framework.PayloadProcessor = &BodyFieldToHeaderPlugin{}

// BodyFieldToHeaderPlugin extracts value from a given body field and sets it as HTTP header.
type BodyFieldToHeaderPlugin struct {
	typedName  plugin.TypedName
	fieldName  string
	headerName string
}

// BodyFieldToHeaderConfig defines the JSON configuration structure for the plugin.
type BodyFieldToHeaderConfig struct {
	// FieldName is the name of the body field to extract
	FieldName string `json:"field_name"`
	// HeaderName is the name of the header to set
	HeaderName string `json:"header_name"`
}

// BodyFieldToHeaderPluginFactory creates a new BodyFieldToHeaderPlugin instance from JSON configuration.
func BodyFieldToHeaderPluginFactory(name string, rawParameters json.RawMessage) (framework.PayloadProcessor, error) {
	var config BodyFieldToHeaderConfig

	if len(rawParameters) > 0 {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal BodyFieldToHeader plugin configuration: %w", err)
		}
	}

	if config.FieldName == "" {
		return nil, errors.New("BodyFieldToHeader plugin requires field_name in configuration")
	}

	if config.HeaderName == "" {
		// Create a default header name based on the field name
		config.HeaderName = deafultHeaderPrefix + config.FieldName
	}

	return NewBodyFieldToHeaderPlugin(config.FieldName, config.HeaderName).WithName(name), nil
}

// NewBodyFieldToHeaderPlugin creates a new BodyFieldToHeaderPlugin with the given configuration.
func NewBodyFieldToHeaderPlugin(fieldName, headerName string) *BodyFieldToHeaderPlugin {
	return &BodyFieldToHeaderPlugin{
		typedName: plugin.TypedName{
			Type: BodyFieldToHeaderPluginType,
			Name: BodyFieldToHeaderPluginType,
		},
		fieldName:  fieldName,
		headerName: headerName,
	}
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

// Execute extracts value from a given body field and sets it as HTTP header.
func (p *BodyFieldToHeaderPlugin) Execute(ctx context.Context, headers map[string]string, body map[string]any) (map[string]string, map[string]any, error) {
	if headers == nil {
		return headers, body, errors.New("headers map is nil")
	}
	if body == nil {
		return headers, body, errors.New("body map is nil")
	}

	// Extract field value from body
	fieldValue, exists := body[p.fieldName]
	if !exists {
		// Field doesn't exist in body, return headers map unchanged
		return headers, body, nil
	}

	headers[p.headerName] = fmt.Sprintf("%v", fieldValue)

	return headers, body, nil
}
