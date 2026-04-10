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
// Package concurrency implements a synchronous saturation detector and scheduling filter for LLM
// routing. It consumes in-flight requests and tokens data from the Endpoint's AttributeMap
// to provide instantaneous backpressure and protect endpoints from sudden traffic bursts.
//
// For detailed architectural trade-offs and configuration, see the package README.
package concurrency

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrconcurrency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

const (
	ConcurrencyDetectorType = "concurrency-detector"
)

// ConcurrencyDetectorFactory instantiates the detector plugin using the provided JSON parameters.
func ConcurrencyDetectorFactory(
	name string,
	params json.RawMessage,
	handle fwkplugin.Handle,
) (fwkplugin.Plugin, error) {
	var apiCfg apiConfig
	if len(params) > 0 {
		if err := json.Unmarshal(params, &apiCfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal concurrency detector config: %w", err)
		}
	}
	cfg, err := buildConfig(&apiCfg)
	if err != nil {
		return nil, err
	}
	return newDetector(name, *cfg, log.FromContext(handle.Context())), nil
}

var (
	_ framework.Filter               = &detector{}
	_ flowcontrol.SaturationDetector = &detector{}
)

// detector implements a saturation detector and scheduling filter based on active request concurrency.
type detector struct {
	config    config
	typedName fwkplugin.TypedName
}

// newDetector creates a new instance of the Concurrency Detector.
func newDetector(name string, cfg config, logger logr.Logger) *detector {
	typedName := fwkplugin.TypedName{
		Type: ConcurrencyDetectorType,
		Name: name,
	}

	pluginLogger := logger.WithName(typedName.String())
	pluginLogger.V(logutil.DEFAULT).Info("Creating new ConcurrencyDetector",
		"mode", cfg.mode,
		"maxConcurrency", cfg.maxConcurrency,
		"maxTokenConcurrency", cfg.maxTokenConcurrency,
		"headroom", cfg.headroom)

	if cfg.headroom > 1.0 {
		pluginLogger.Info("Unusually high headroom configured; verify value is a fraction, not a percentage",
			"headroom", cfg.headroom,
			"effectiveBurst", fmt.Sprintf("%.0f%%", cfg.headroom*100))
	}

	return &detector{
		config:    cfg,
		typedName: typedName,
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (d *detector) TypedName() fwkplugin.TypedName {
	return d.typedName
}

func (d *detector) getLoad(m datalayer.AttributeMap) *attrconcurrency.InFlightLoad {
	if val, ok := m.Get(attrconcurrency.InFlightLoadKey); ok {
		if load, ok := val.(*attrconcurrency.InFlightLoad); ok {
			return load
		}
	}
	return &attrconcurrency.InFlightLoad{}
}

// Saturation calculates the saturation level of the pool.
//
// It returns an aggregate saturation signal where:
//
//	Saturation = Total Inflight Requests / Total MaxConcurrency Capacity.
func (d *detector) Saturation(_ context.Context, endpoints []datalayer.Endpoint) float64 {
	var totalInflight, totalCapacity int64
	for _, e := range endpoints {
		if e.GetMetadata() == nil {
			continue
		}

		load := d.getLoad(e.GetAttributes())

		if d.config.mode == modeTokens {
			totalInflight += load.Tokens
			totalCapacity += d.config.maxTokenConcurrency
		} else {
			totalInflight += load.Requests
			totalCapacity += d.config.maxConcurrency
		}
	}

	if totalCapacity == 0 {
		return 1.0
	}

	return float64(totalInflight) / float64(totalCapacity)
}

// Filter blocks traffic to specific endpoints that are physically saturated or exceeding their safety limits.
//
// It applies a relaxed limit (MaxConcurrency * (1 + Headroom)) to allow for scheduling flexibility and burst tolerance.
func (d *detector) Filter(
	_ context.Context,
	_ *framework.CycleState,
	_ *framework.InferenceRequest,
	endpoints []framework.Endpoint,
) []framework.Endpoint {
	// Pre-allocate assuming most endpoints will pass the filter to minimize allocations.
	filtered := make([]framework.Endpoint, 0, len(endpoints))

	var limit int64
	if d.config.mode == modeTokens {
		limit = int64(float64(d.config.maxTokenConcurrency) * (1.0 + d.config.headroom))
	} else {
		limit = int64(float64(d.config.maxConcurrency) * (1.0 + d.config.headroom))
	}

	for _, e := range endpoints {
		load := d.getLoad(e)

		if d.config.mode == modeTokens {
			if load.Tokens < limit {
				filtered = append(filtered, e)
			}
		} else {
			if load.Requests < limit {
				filtered = append(filtered, e)
			}
		}
	}
	return filtered
}
