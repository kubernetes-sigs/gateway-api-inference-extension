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
	adaptersStore adaptersStore
}

// BaseModelToHeaderPluginFactory defines the factory function for BaseModelToHeaderPlugin
func BaseModelToHeaderPluginFactory(name string, _ json.RawMessage) (framework.BBRPlugin, error) {
	return NewBaseModelToHeaderPlugin().WithName(name), nil
}

// NewBaseModelToHeaderPlugin returns a concrete *BaseModelToHeaderPlugin with an initialized adaptersStore.
func NewBaseModelToHeaderPlugin() *BaseModelToHeaderPlugin {
	return &BaseModelToHeaderPlugin{
		typedName:     plugin.TypedName{Type: BaseModelToHeaderPluginType, Name: BaseModelToHeaderPluginType},
		adaptersStore: newAdaptersStore(),
	}
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

	// Look up base model using configured adaptersStore
	// If baseModel is empty, it means the model is neither a LoRA adapter nor a registered base model
	baseModel := p.adaptersStore.getBaseModel(targetModel)

	// Set model headers for routing (empty string is valid)
	request.SetHeader(baseModelHeader, baseModel)
	log.FromContext(ctx).V(logutil.VERBOSE).Info("updated base model header based on the request target model", "targetModel", targetModel, "baseModel", baseModel)
	return nil
}

// GetReconciler returns a configMapReconciler that can be registered with a controller manager.
func (p *BaseModelToHeaderPlugin) GetReconciler() *configMapReconciler {
	return &configMapReconciler{
		adaptersStore: p.adaptersStore,
	}
}
