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

package metrics

import (
	"encoding/json"
	"errors"
	"fmt"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	MetricsExtractorType = "model-server-protocol-metrics"
)

// Configuration parameters for metrics data source and extractor.
type (
	// engineConfigParams holds metric specifications for a specific engine type.
	engineConfigParams struct {
		// Name is the engine type identifier.
		Name string `json:"name"`
		// QueuedRequestsSpec defines the metric specification string for retrieving queued request count.
		QueuedRequestsSpec string `json:"queuedRequestsSpec"`
		// RunningRequestsSpec defines the metric specification string for retrieving running requests count.
		RunningRequestsSpec string `json:"runningRequestsSpec"`
		// KVUsageSpec defines the metric specification string for retrieving KV cache usage.
		KVUsageSpec string `json:"kvUsageSpec"`
		// LoRASpec defines the metric specification string for retrieving LoRA availability.
		LoRASpec string `json:"loraSpec"`
		// CacheInfoSpec defines the metrics specification string for retrieving KV cache configuration.
		CacheInfoSpec string `json:"cacheInfoSpec"`
	}

	// Extractor configuration parameters
	modelServerExtractorParams struct {
		// EngineLabelKey is the Pod label key used to identify the engine type.
		// Defaults to "inference.networking.k8s.io/engine-type".
		EngineLabelKey string `json:"engineLabelKey"`
		// DefaultEngine specifies which engine to use as the default for unlabeled Pods.
		// Can be any engine name from EngineConfigs. Defaults to "vllm".
		DefaultEngine string `json:"defaultEngine"`
		// EngineConfigs defines metric specifications for specific engine types.
		// Built-in vLLM and SGLang configs are automatically appended if not explicitly defined.
		EngineConfigs []engineConfigParams `json:"engineConfigs"`
	}
)

// Default engine configurations for vLLM and SGLang.
var defaultEngineConfigs = []engineConfigParams{
	{
		Name:                "vllm",
		QueuedRequestsSpec:  "vllm:num_requests_waiting",
		RunningRequestsSpec: "vllm:num_requests_running",
		KVUsageSpec:         "vllm:kv_cache_usage_perc",
		LoRASpec:            "vllm:lora_requests_info",
		CacheInfoSpec:       "vllm:cache_config_info",
	},
	{
		Name:                "sglang",
		QueuedRequestsSpec:  "sglang:num_queue_reqs",
		RunningRequestsSpec: "sglang:num_running_reqs",
		KVUsageSpec:         "sglang:token_usage",
		LoRASpec:            "",
		CacheInfoSpec:       "",
	},
}

// defaultEngineName is the default engine used when defaultEngine is not specified.
const defaultEngineName = "vllm"

// ModelServerExtractorFactory is a factory function used to instantiate data layer's metrics
// Extractor plugins specified in a configuration.
func ModelServerExtractorFactory(name string, parameters json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := defaultExtractorConfigParams()

	if parameters != nil { // overlay the defaults with configured values
		if err := json.Unmarshal(parameters, cfg); err != nil {
			return nil, err
		}
	}

	// Use defaultEngineName if defaultEngine is not specified
	if cfg.DefaultEngine == "" {
		cfg.DefaultEngine = defaultEngineName
	}

	// Use DefaultEngineTypeLabelKey if engineLabelKey is not specified
	if cfg.EngineLabelKey == "" {
		cfg.EngineLabelKey = DefaultEngineTypeLabelKey
	}

	// Append default engine configs (vllm, sglang) if not explicitly defined by user
	userDefinedEngines := make(map[string]bool)
	for _, ec := range cfg.EngineConfigs {
		userDefinedEngines[ec.Name] = true
	}
	for _, defaultCfg := range defaultEngineConfigs {
		if !userDefinedEngines[defaultCfg.Name] {
			cfg.EngineConfigs = append(cfg.EngineConfigs, defaultCfg)
		}
	}

	registry := NewMappingRegistry()

	// Validate and register engine configurations
	var defaultMapping *Mapping
	for _, engineConfig := range cfg.EngineConfigs {
		if engineConfig.Name == "" {
			return nil, errors.New("engine config name cannot be empty")
		}
		if engineConfig.Name == DefaultEngineType {
			return nil, fmt.Errorf("engine config name cannot be %q (reserved)", DefaultEngineType)
		}

		mapping, err := NewMapping(
			engineConfig.QueuedRequestsSpec,
			engineConfig.RunningRequestsSpec,
			engineConfig.KVUsageSpec,
			engineConfig.LoRASpec,
			engineConfig.CacheInfoSpec,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create mapping for engine %q: %w", engineConfig.Name, err)
		}

		// Register by engine name
		if err := registry.Register(engineConfig.Name, mapping); err != nil {
			return nil, fmt.Errorf("failed to register engine mapping for %q: %w", engineConfig.Name, err)
		}

		// Track the default engine mapping
		if engineConfig.Name == cfg.DefaultEngine {
			defaultMapping = mapping
		}
	}

	// Validate and register the default engine
	if defaultMapping == nil {
		return nil, fmt.Errorf("defaultEngine %q not found in engineConfigs", cfg.DefaultEngine)
	}
	if err := registry.Register(DefaultEngineType, defaultMapping); err != nil {
		return nil, fmt.Errorf("failed to register default mapping: %w", err)
	}

	extractor, err := NewModelServerExtractor(registry, cfg.EngineLabelKey)
	if err != nil {
		return nil, err
	}
	extractor.typedName.Name = name
	return extractor, nil
}

func defaultExtractorConfigParams() *modelServerExtractorParams {
	return &modelServerExtractorParams{
		EngineLabelKey: DefaultEngineTypeLabelKey,
	}
}
