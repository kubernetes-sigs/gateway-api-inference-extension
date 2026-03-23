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

package basemodelextractor

import (
	"context"
	"encoding/json"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	BaseModelToHeaderPluginType = "base-model-to-header"
	// These constants must match the header names defined in pkg/bbr/handlers/server.go
	baseModelHeader = "X-Gateway-Base-Model-Name"
	modelField      = "model"
)

// compile-time type validation
var _ framework.RequestProcessor = &BaseModelToHeaderPlugin{}

type BaseModelToHeaderPlugin struct {
	typedName     plugin.TypedName
	AdaptersStore AdaptersStore
}

// BaseModelToHeaderPluginFactory defines the factory function for BaseModelToHeaderPlugin
func BaseModelToHeaderPluginFactory(name string, _ json.RawMessage, handle framework.Handle) (framework.BBRPlugin, error) {
	plugin, err := NewBaseModelToHeaderPlugin(handle.ReconcilerBuilder, handle.ClientReader())
	if err != nil {
		return nil, fmt.Errorf("failed to create plugin '%s' - %w", BaseModelToHeaderPluginType, err)
	}

	return plugin.WithName(name), nil
}

// NewBaseModelToHeaderPlugin returns a *BaseModelToHeaderPlugin with an initialized AdaptersStore.
func NewBaseModelToHeaderPlugin(reconcilerBuilder func() *builder.Builder, clientReader client.Reader) (*BaseModelToHeaderPlugin, error) {
	reconcilerBuidler := reconcilerBuilder()
	adaptersStore := NewAdaptersStore()
	configMapReconciler := &ConfigMapReconciler{
		Reader:        clientReader,
		AdaptersStore: adaptersStore,
	}

	if err := reconcilerBuidler.For(&corev1.ConfigMap{}).Complete(configMapReconciler); err != nil {
		return nil, fmt.Errorf("failed to register configmap reconciler for plugin '%s' - %w", BaseModelToHeaderPluginType, err)
	}

	return &BaseModelToHeaderPlugin{
		typedName:     plugin.TypedName{Type: BaseModelToHeaderPluginType, Name: BaseModelToHeaderPluginType},
		AdaptersStore: adaptersStore,
	}, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *BaseModelToHeaderPlugin) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the name of the BBR plugin instance.
func (p *BaseModelToHeaderPlugin) WithName(name string) *BaseModelToHeaderPlugin {
	p.typedName.Name = name
	return p
}

// ProcessRequest sets base model name on the header
func (p *BaseModelToHeaderPlugin) ProcessRequest(ctx context.Context, _ *framework.CycleState, request *framework.InferenceRequest) error {
	if request == nil || request.Headers == nil || request.Body == nil {
		return nil // this shouldn't happen
	}

	// extract raw field value from body
	rawFieldValue, exists := request.Body[modelField]
	if !exists {
		// If model field is not present, set empty base model header
		request.SetHeader(baseModelHeader, "")
		log.FromContext(ctx).V(logutil.VERBOSE).Info("model field not found, setting empty base model header")
		return nil
	}

	targetModel := fmt.Sprintf("%v", rawFieldValue) // convert any type to string

	// Look up base model using configured AdaptersStore
	// If baseModel is empty, it means the model is neither a LoRA adapter nor a registered base model
	baseModel := p.AdaptersStore.getBaseModel(targetModel)

	// Set model headers for routing (empty string is valid)
	request.SetHeader(baseModelHeader, baseModel)
	log.FromContext(ctx).V(logutil.VERBOSE).Info("updated base model header based on the request target model", "targetModel", targetModel, "baseModel", baseModel)
	return nil
}
