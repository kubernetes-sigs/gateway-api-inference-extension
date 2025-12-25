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

package bbrplugins

import (
	"encoding/json"

	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

// ------------------------------------  INTERFACES ---------------------------------------------------------------
// Interfaces are defined in "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework/plugins/interfaces.go"
// ----------------------------------------------------------------------------------------------------------------

// ------------------------------------ DEFAULT PLUGIN IMPLEMENTATION ----------------------------------------------

// defaultMetaDataExtractor implements the MetadataExtractor interface and extracts only the mmodel name AS-IS
type defaultMetaDataExtractor struct {
	typedName           plugins.TypedName
	requiresFullParsing bool //this field will be used to determine whether shared struct should be created in this chain
}

// NewSimpleModelExtractor is a factory that constructs SimpleModelExtractor plugin
// A developer who wishes to create her own implementation, will implement the BBRPlugin interface and
// use Registry and PluginsChain to register and execute the plugin (together with other plugins in a chain)
func NewDefaultMetaDataExtractor() BBRPlugin {
	return &defaultMetaDataExtractor{
		typedName: plugins.TypedName{
			Type: DefaultPluginType,
			Name: "simple-model-extractor",
		},
		requiresFullParsing: false,
	}
}

func (s *defaultMetaDataExtractor) RequiresFullParsing() bool {
	return s.requiresFullParsing
}

func (s *defaultMetaDataExtractor) TypedName() plugins.TypedName {
	return s.typedName
}

// Execute extracts the "model" from the JSON request body and sets X-Gateway-Model-Name header.
// This implementation intentionally ignores metaDataKeys and does not mutate the body.
// It expects the request body to be a JSON object containing a "model" field.
// A nil for metaDataKeysToHeaders map SHOULD be specified by a caller for clarity
// The metaDataKeysToHeaders is explicitly ignored in this implementation
// This implementation is simply refactoring of the default BBR implementation to work with the pluggable framework
func (s *defaultMetaDataExtractor) Execute(requestBodyBytes []byte) (
	headers map[string]string,
	mutatedBodyBytes []byte,
	err error) {

	type RequestBody struct {
		Model string `json:"model"`
	}

	h := make(map[string]string)

	var requestBody RequestBody

	if err := json.Unmarshal(requestBodyBytes, &requestBody); err != nil {
		// return original body on decode failure
		return nil, requestBodyBytes, err
	}

	if requestBody.Model == "" {
		return nil, requestBodyBytes, fmt.Errorf("missing required field: model")
	}

	// ModelHeader is a constant defined in ./pkg/bbr/plugins/interfaces
	h[ModelHeader] = requestBody.Model

	// Body is not mutated in this implementation hence returning original requestBodyBytes. This is intentional.
	return h, requestBodyBytes, nil
}

func (s *defaultMetaDataExtractor) String() string {
	return fmt.Sprintf(("BBRPlugin{%v/requiresFullParsing=%v}"), s.TypedName(), s.requiresFullParsing)
}
