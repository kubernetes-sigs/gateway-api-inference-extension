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

package metrics

import (
	"errors"
	"fmt"
	"strings"
)

// Mapping holds specifications for the well-known metrics defined
// in the Model Server Protocol.
type Mapping struct {
	TotalQueuedRequests  *Spec
	TotalRunningRequests *Spec
	KVCacheUtilization   *Spec
	LoraRequestInfo      *LoRASpec
	// CacheInfo is used for info-style gauge metrics where block_size and
	// num_gpu_blocks are exposed as label values (e.g. vLLM, trtllm-serve, SGLang).
	CacheInfo *Spec
	// CacheBlockSizeLabel and CacheNumBlocksLabel allow engines to use different
	// label names for the CacheInfo metric. If empty, defaults to "block_size"
	// and "num_gpu_blocks".
	CacheBlockSizeLabel string
	CacheNumBlocksLabel string
	// CacheBlockSize and CacheNumBlocks are used for engines that expose cache
	// config as separate gauge values rather than labels on an info metric
	// (e.g. Triton TRT-LLM).
	CacheBlockSize *Spec
	CacheNumBlocks *Spec
}

// MappingConfig holds the string-based configuration used to build a Mapping.
type MappingConfig struct {
	Queue               string
	Running             string
	KVUsage             string
	Lora                string
	CacheInfo           string
	CacheBlockSizeLabel string
	CacheNumBlocksLabel string
	CacheBlockSize      string
	CacheNumBlocks      string
}

// String returns a human-readable representation of the Mapping, listing which specs are disabled (nil).
func (m *Mapping) String() string {
	specs := []struct {
		name string
		val  any
	}{
		{"queue", m.TotalQueuedRequests},
		{"running", m.TotalRunningRequests},
		{"kv", m.KVCacheUtilization},
		{"lora", m.LoraRequestInfo},
		{"cacheInfo", m.CacheInfo},
	}
	var disabled []string
	for _, s := range specs {
		if s.val == nil {
			disabled = append(disabled, s.name)
		}
	}
	if len(disabled) == 0 {
		return "Mapping{all specs enabled}"
	}
	return fmt.Sprintf("Mapping{disabled: [%s]}", strings.Join(disabled, ", "))
}

// NewMapping creates a metrics.Mapping from the input specification strings.
func NewMapping(queue, running, kvusage, lora, cacheInfo string) (*Mapping, error) {
	return NewMappingFromConfig(MappingConfig{
		Queue:     queue,
		Running:   running,
		KVUsage:   kvusage,
		Lora:      lora,
		CacheInfo: cacheInfo,
	})
}

// NewMappingFromConfig creates a metrics.Mapping from a MappingConfig.
func NewMappingFromConfig(cfg MappingConfig) (*Mapping, error) {
	var errs []error

	queueSpec, err := parseStringToSpec(cfg.Queue)
	if err != nil {
		errs = append(errs, err)
	}
	runningSpec, err := parseStringToSpec(cfg.Running)
	if err != nil {
		errs = append(errs, err)
	}
	kvusageSpec, err := parseStringToSpec(cfg.KVUsage)
	if err != nil {
		errs = append(errs, err)
	}
	loraSpec, err := parseStringToLoRASpec(cfg.Lora)
	if err != nil {
		errs = append(errs, err)
	}
	cacheInfoSpec, err := parseStringToSpec(cfg.CacheInfo)
	if err != nil {
		errs = append(errs, err)
	}
	cacheBlockSizeSpec, err := parseStringToSpec(cfg.CacheBlockSize)
	if err != nil {
		errs = append(errs, err)
	}
	cacheNumBlocksSpec, err := parseStringToSpec(cfg.CacheNumBlocks)
	if err != nil {
		errs = append(errs, err)
	}

	if len(errs) != 0 {
		return nil, errors.Join(errs...)
	}
	return &Mapping{
		TotalQueuedRequests:  queueSpec,
		TotalRunningRequests: runningSpec,
		KVCacheUtilization:   kvusageSpec,
		LoraRequestInfo:      loraSpec,
		CacheInfo:            cacheInfoSpec,
		CacheBlockSizeLabel:  cfg.CacheBlockSizeLabel,
		CacheNumBlocksLabel:  cfg.CacheNumBlocksLabel,
		CacheBlockSize:       cacheBlockSizeSpec,
		CacheNumBlocks:       cacheNumBlocksSpec,
	}, nil
}
