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
)

// Mapping holds specifications for the well-known metrics defined
// in the Model Server Protocol.
type Mapping struct {
	TotalQueuedRequests *Spec
	KVCacheUtilization  *Spec
	LoraRequestInfo     *LoRASpec
	CacheInfo           *Spec
}

// NewMapping creates a metrics.Mapping from the input specification strings.
func NewMapping(queue, kvusage, lora, cacheInfo string) (*Mapping, error) {
	var errs []error

	queueSpec, err := parseStringToSpec(queue)
	if err != nil {
		errs = append(errs, err)
	}
	kvusageSpec, err := parseStringToSpec(kvusage)
	if err != nil {
		errs = append(errs, err)
	}
	loraSpec, err := parseStringToLoRASpec(lora)
	if err != nil {
		errs = append(errs, err)
	}

	cacheInfoSpec, err := parseStringToSpec(cacheInfo)
	if err != nil {
		errs = append(errs, err)
	}

	if len(errs) != 0 {
		return nil, errors.Join(errs...)
	}
	return &Mapping{
		TotalQueuedRequests: queueSpec,
		KVCacheUtilization:  kvusageSpec,
		LoraRequestInfo:     loraSpec,
		CacheInfo:           cacheInfoSpec,
	}, nil
}
