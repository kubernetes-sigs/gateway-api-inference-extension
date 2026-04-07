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

package latencyslo

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrlatency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
)

const (
	LatencyAdmissionPluginType = "latency-slo-admitter"

	ttftSLOHeaderKey = "x-slo-ttft-ms"
	tpotSLOHeaderKey = "x-slo-tpot-ms"
)

// compile-time validation
var _ requestcontrol.AdmissionPlugin = &LatencyAdmission{}

// LatencyAdmissionConfig holds configuration for the latency admission plugin.
type LatencyAdmissionConfig struct{}

var LatencyAdmissionDefaultConfig = LatencyAdmissionConfig{}

// LatencyAdmission rejects sheddable requests when no endpoint can meet SLO constraints.
// It reads latency predictions from endpoint attributes (published by the data provider)
// and makes an independent admission decision.
type LatencyAdmission struct {
	typedName fwkplugin.TypedName
	config    LatencyAdmissionConfig
}

// LatencyAdmissionFactory creates a new LatencyAdmission plugin instance.
func LatencyAdmissionFactory(name string, rawParameters json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := LatencyAdmissionDefaultConfig
	if len(rawParameters) > 0 {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config for LatencyAdmission: %w", err)
		}
	}
	return NewLatencyAdmission(config).WithName(name), nil
}

// NewLatencyAdmission creates a new LatencyAdmission plugin.
func NewLatencyAdmission(config LatencyAdmissionConfig) *LatencyAdmission {
	return &LatencyAdmission{
		typedName: fwkplugin.TypedName{Type: LatencyAdmissionPluginType, Name: LatencyAdmissionPluginType},
		config:    config,
	}
}

func (p *LatencyAdmission) WithName(name string) *LatencyAdmission {
	p.typedName.Name = name
	return p
}

func (p *LatencyAdmission) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Consumes declares that this plugin reads latency prediction data from endpoints.
func (p *LatencyAdmission) Consumes() map[string]any {
	return map[string]any{
		attrlatency.LatencyPredictionInfoKey: attrlatency.LatencyPredictionInfo{},
	}
}

// AdmitRequest rejects sheddable requests if no endpoint can serve them within SLO.
//
// Reject only when ALL of:
//   - No endpoint has a valid prediction (all violate SLO)
//   - No endpoint is idle (all have running requests)
//   - No cold pod exists (predictions are reliable)
func (p *LatencyAdmission) AdmitRequest(ctx context.Context, request *schedulingtypes.InferenceRequest, endpoints []schedulingtypes.Endpoint) error {
	logger := log.FromContext(ctx)
	if request == nil {
		return nil
	}

	// Only reject sheddable requests (negative priority).
	if request.Objectives.Priority >= 0 {
		return nil
	}

	// Check if SLOs are set — if not, we can't determine validity, so admit.
	ttftSLO := parseFloatHeaderValue(request.Headers[ttftSLOHeaderKey])
	tpotSLO := parseFloatHeaderValue(request.Headers[tpotSLOHeaderKey])
	hasSLO := ttftSLO > 0 || tpotSLO > 0
	if !hasSLO {
		return nil
	}

	hasValid := false
	hasCold := false
	hasIdle := false
	hasPredictions := false

	for _, endpoint := range endpoints {
		metrics := endpoint.GetMetrics()

		// Cold pod: KV cache < 2% — predictions may be unreliable, don't reject.
		if metrics.KVCacheUsagePercent < 0.02 {
			hasCold = true
		}

		// Idle pod: no running requests — likely can serve the request.
		if metrics.RunningRequestsSize == 0 {
			hasIdle = true
		}

		// Valid prediction: both TTFT and TPOT within SLO.
		if latencyInfoRaw, ok := endpoint.Get(attrlatency.LatencyPredictionInfoKey); ok {
			hasPredictions = true
			latencyInfo := latencyInfoRaw.(*attrlatency.LatencyPredictionInfo)
			if latencyInfo.IsValid() {
				hasValid = true
			}
		}
	}

	// If no predictions are available, fail-open.
	if !hasPredictions {
		return nil
	}

	if !hasValid && !hasIdle && !hasCold {
		logger.V(logutil.DEBUG).Info("LatencyAdmission: rejecting sheddable request, no valid endpoint available",
			"endpoints", len(endpoints))
		return errors.New("no valid endpoint available to serve the request")
	}

	return nil
}

func parseFloatHeaderValue(s string) float64 {
	if s == "" {
		return 0
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return v
}
