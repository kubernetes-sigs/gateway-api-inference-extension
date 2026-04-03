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

package loader

import (
	"fmt"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

var scheme = runtime.NewScheme()

func init() {
	utilruntime.Must(configapi.Install(scheme))
}

// LoadRawConfig parses the raw configuration bytes and applies initial defaults.
// It does not instantiate plugins.
// If configBytes is empty, a default configuration is returned.
func LoadRawConfig(configBytes []byte, logger logr.Logger) (*configapi.BodyBasedRoutingConfig, error) {
	var rawConfig *configapi.BodyBasedRoutingConfig
	var err error
	if len(configBytes) != 0 {
		rawConfig, err = decodeRawConfig(configBytes)
		if err != nil {
			return nil, err
		}
		logger.Info("Loaded raw configuration", "config", rawConfig.String())
	} else {
		logger.Info("A configuration wasn't specified. A default one is being used.")
		rawConfig = loadDefaultConfig()
		logger.Info("Default raw configuration used", "config", rawConfig.String())
	}

	applyStaticDefaults(rawConfig)

	if err := validateConfig(rawConfig); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	return rawConfig, nil
}

func decodeRawConfig(configBytes []byte) (*configapi.BodyBasedRoutingConfig, error) {
	cfg := &configapi.BodyBasedRoutingConfig{}
	codecs := serializer.NewCodecFactory(scheme, serializer.EnableStrict)
	if err := runtime.DecodeInto(codecs.UniversalDecoder(), configBytes, cfg); err != nil {
		return nil, fmt.Errorf("failed to decode configuration JSON/YAML: %w", err)
	}
	return cfg, nil
}
