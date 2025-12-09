/*
© 2025 The Kubernetes Authors.
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

// Package requestcontrol contains helpers for P/D disaggregation SLO tracking.
package requestcontrol

import (
	"context"
	"os"
	"strconv"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/latencypredictorasync"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// Configuration constants for P/D SLO tracking

// EnablePDSLOTracking enables/disables P/D SLO tracking
var EnablePDSLOTracking = func() bool {
	if value, exists := os.LookupEnv("ENABLE_PD_SLO_TRACKING"); exists {
		if parsedValue, err := strconv.ParseBool(value); err == nil {
			return parsedValue
		}
	}
	return true // default: enabled
}()

// TTFTPrefillBudgetRatio is the fraction of TTFT SLO allocated to prefill phase
// The remainder (1 - ratio) is allocated to KV transfer
var TTFTPrefillBudgetRatio = func() float64 {
	if value, exists := os.LookupEnv("TTFT_PREFILL_BUDGET_RATIO"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue >= 0 && parsedValue <= 1 {
			return parsedValue
		}
	}
	return 0.7 // default: 70% for prefill, 30% for KV transfer
}()

// KVTransferOverheadMs is the fixed overhead for KV transfer setup (ms)
var KVTransferOverheadMs = func() float64 {
	if value, exists := os.LookupEnv("KV_TRANSFER_OVERHEAD_MS"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue >= 0 {
			return parsedValue
		}
	}
	return 5.0 // default: 5ms overhead
}()

// KVTransferBandwidthGbps is the estimated network bandwidth for KV transfer (Gbps)
var KVTransferBandwidthGbps = func() float64 {
	if value, exists := os.LookupEnv("KV_TRANSFER_BANDWIDTH_GBPS"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue > 0 {
			return parsedValue
		}
	}
	return 100.0 // default: 100 Gbps (RDMA/RoCE)
}()

// PDSLOBufferFactor is the safety margin for P/D SLO budgets
var PDSLOBufferFactor = func() float64 {
	if value, exists := os.LookupEnv("PD_SLO_BUFFER_FACTOR"); exists {
		if parsedValue, err := strconv.ParseFloat(value, 64); err == nil && parsedValue >= 0 {
			return parsedValue
		}
	}
	return 0.9 // default: 90% of budget (10% safety margin)
}()

// PDPredictionRequest extends the standard prediction request with P/D specific fields
type PDPredictionRequest struct {
	latencypredictor.PredictionRequest
	PodRole              string  // "prefill" | "decode"
	KVCacheSizeMB        float64 // Size of KV cache to transfer
	NetworkBandwidthGbps float64 // Network bandwidth
	IsDisaggregated      bool    // True if using P/D disaggregation
}

// PDPredictionResponse contains predictions for all P/D phases
type PDPredictionResponse struct {
	PrefillTTFT    float64 // Prefill phase latency (ms)
	KVTransferMs   float64 // KV transfer latency (ms)
	DecodeTPOT     float64 // Decode phase TPOT (ms)
	TotalTTFT      float64 // prefill_ttft + kv_transfer_ms
	PredictionTime time.Time
}

// AllocatePDSLOBudgets allocates TTFT and TPOT SLO budgets across P/D phases
func AllocatePDSLOBudgets(ctx context.Context, reqCtx *handlers.RequestContext, ttftSLO, tpotSLO float64) {
	logger := log.FromContext(ctx)

	if !reqCtx.PDMode {
		logger.V(logutil.DEBUG).Info("P/D mode not enabled, skipping budget allocation")
		return
	}

	// Allocate TTFT budget: prefill gets TTFTPrefillBudgetRatio, KV transfer gets remainder
	reqCtx.PrefillTTFTBudget = ttftSLO * TTFTPrefillBudgetRatio * PDSLOBufferFactor
	reqCtx.KVTransferBudget = ttftSLO * (1 - TTFTPrefillBudgetRatio) * PDSLOBufferFactor

	// TPOT budget goes entirely to decode phase
	reqCtx.DecodeTPOTBudget = tpotSLO * PDSLOBufferFactor

	// Initialize remaining budgets
	reqCtx.RemainingTTFTBudget = ttftSLO
	reqCtx.RemainingTPOTBudget = tpotSLO

	// Initialize phase start times
	// Prefill starts immediately when budget is allocated
	reqCtx.PrefillStartTime = time.Now()
	// KV transfer will start when prefill completes (set in UpdatePrefillPhase)
	// Decode starts after KV transfer (set in UpdateKVTransferPhase)

	logger.V(logutil.DEBUG).Info("Allocated P/D SLO budgets",
		"ttft_slo", ttftSLO,
		"tpot_slo", tpotSLO,
		"prefill_budget", reqCtx.PrefillTTFTBudget,
		"kv_transfer_budget", reqCtx.KVTransferBudget,
		"decode_tpot_budget", reqCtx.DecodeTPOTBudget,
		"prefill_ratio", TTFTPrefillBudgetRatio,
		"buffer_factor", PDSLOBufferFactor)
}

// UpdatePrefillPhase records prefill phase completion and updates budgets
func UpdatePrefillPhase(ctx context.Context, reqCtx *handlers.RequestContext) {
	logger := log.FromContext(ctx)

	if !reqCtx.PDMode || reqCtx.PrefillStartTime.IsZero() {
		return
	}

	reqCtx.PrefillEndTime = time.Now()
	reqCtx.ActualPrefillLatency = float64(reqCtx.PrefillEndTime.Sub(reqCtx.PrefillStartTime).Milliseconds())

	// KV transfer starts immediately after prefill completes
	reqCtx.KVTransferStartTime = reqCtx.PrefillEndTime

	// Update remaining TTFT budget
	reqCtx.RemainingTTFTBudget -= reqCtx.ActualPrefillLatency

	// Check if prefill violated its budget
	if reqCtx.ActualPrefillLatency > reqCtx.PrefillTTFTBudget {
		reqCtx.PDSLOViolation = true
		reqCtx.PDSLOViolationPhase = "prefill"
		logger.Info("Prefill phase violated SLO budget",
			"actual", reqCtx.ActualPrefillLatency,
			"budget", reqCtx.PrefillTTFTBudget,
			"overage", reqCtx.ActualPrefillLatency-reqCtx.PrefillTTFTBudget)
	}

	logger.V(logutil.DEBUG).Info("Prefill phase completed",
		"latency_ms", reqCtx.ActualPrefillLatency,
		"budget_ms", reqCtx.PrefillTTFTBudget,
		"remaining_ttft_budget", reqCtx.RemainingTTFTBudget,
		"slo_violated", reqCtx.PDSLOViolation)
}

// UpdateKVTransferPhase records KV transfer completion and updates budgets
func UpdateKVTransferPhase(ctx context.Context, reqCtx *handlers.RequestContext) {
	logger := log.FromContext(ctx)

	if !reqCtx.PDMode || reqCtx.KVTransferStartTime.IsZero() {
		return
	}

	reqCtx.KVTransferEndTime = time.Now()
	reqCtx.ActualKVTransferLatency = float64(reqCtx.KVTransferEndTime.Sub(reqCtx.KVTransferStartTime).Milliseconds())

	// Decode starts immediately after KV transfer completes
	reqCtx.DecodeStartTime = reqCtx.KVTransferEndTime

	// Update remaining TTFT budget
	reqCtx.RemainingTTFTBudget -= reqCtx.ActualKVTransferLatency

	// Check if KV transfer violated its budget
	if reqCtx.ActualKVTransferLatency > reqCtx.KVTransferBudget {
		reqCtx.PDSLOViolation = true
		reqCtx.PDSLOViolationPhase = "kv_transfer"
		logger.Info("KV transfer phase violated SLO budget",
			"actual", reqCtx.ActualKVTransferLatency,
			"budget", reqCtx.KVTransferBudget,
			"overage", reqCtx.ActualKVTransferLatency-reqCtx.KVTransferBudget)
	}

	logger.V(logutil.DEBUG).Info("KV transfer phase completed",
		"latency_ms", reqCtx.ActualKVTransferLatency,
		"budget_ms", reqCtx.KVTransferBudget,
		"remaining_ttft_budget", reqCtx.RemainingTTFTBudget,
		"slo_violated", reqCtx.PDSLOViolation)
}

// UpdateDecodePhase records decode phase TPOT and checks against budget
func UpdateDecodePhase(ctx context.Context, reqCtx *handlers.RequestContext, tpot float64) {
	logger := log.FromContext(ctx)

	if !reqCtx.PDMode {
		return
	}

	reqCtx.ActualDecodeTPOT = tpot
	reqCtx.RemainingTPOTBudget -= tpot

	// Check if decode violated its budget
	if reqCtx.ActualDecodeTPOT > reqCtx.DecodeTPOTBudget {
		reqCtx.PDSLOViolation = true
		reqCtx.PDSLOViolationPhase = "decode"
		logger.Info("Decode phase violated SLO budget",
			"actual_tpot", reqCtx.ActualDecodeTPOT,
			"budget", reqCtx.DecodeTPOTBudget,
			"overage", reqCtx.ActualDecodeTPOT-reqCtx.DecodeTPOTBudget)
	}

	logger.V(logutil.DEBUG).Info("Decode phase TPOT recorded",
		"tpot_ms", reqCtx.ActualDecodeTPOT,
		"budget_ms", reqCtx.DecodeTPOTBudget,
		"remaining_tpot_budget", reqCtx.RemainingTPOTBudget,
		"slo_violated", reqCtx.PDSLOViolation)
}

// EstimateKVTransferLatency estimates KV transfer latency based on cache size and bandwidth
func EstimateKVTransferLatency(kvCacheSizeMB float64, bandwidthGbps float64) float64 {
	if bandwidthGbps <= 0 {
		bandwidthGbps = KVTransferBandwidthGbps
	}

	// Convert bandwidth from Gbps to MB/s (megabytes per second)
	// 1 Gbps = 1,000,000,000 bits/s = 125,000,000 bytes/s = 125 MB/s
	bandwidthMBPerSec := bandwidthGbps * 125.0

	// Transfer time = size / bandwidth (in seconds), convert to ms
	transferTimeMs := (kvCacheSizeMB / bandwidthMBPerSec) * 1000.0

	// Add fixed overhead
	return transferTimeMs + KVTransferOverheadMs
}

// PredictPDPhases predicts latency for all P/D phases
// This is a placeholder - actual implementation will call the extended latency predictor
func PredictPDPhases(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	prefillMetrics, decodeMetrics interface{},
	prompt string,
	kvCacheSizeMB float64,
) (*PDPredictionResponse, error) {
	logger := log.FromContext(ctx)

	// For now, use the existing predictor for prefill and decode
	// TODO: Extend the actual latency predictor to support P/D phase predictions

	response := &PDPredictionResponse{
		PredictionTime: time.Now(),
	}

	// Estimate KV transfer latency
	response.KVTransferMs = EstimateKVTransferLatency(kvCacheSizeMB, KVTransferBandwidthGbps)

	// Use existing predictor for prefill TTFT (will be enhanced later)
	// For now, return placeholder values
	response.PrefillTTFT = 0
	response.DecodeTPOT = 0
	response.TotalTTFT = response.PrefillTTFT + response.KVTransferMs

	logger.V(logutil.DEBUG).Info("Predicted P/D phases",
		"prefill_ttft", response.PrefillTTFT,
		"kv_transfer_ms", response.KVTransferMs,
		"decode_tpot", response.DecodeTPOT,
		"total_ttft", response.TotalTTFT)

	return response, nil
}

// GetPDSLOMetrics returns end-to-end P/D SLO compliance metrics
func GetPDSLOMetrics(reqCtx *handlers.RequestContext) map[string]interface{} {
	if !reqCtx.PDMode {
		return nil
	}

	return map[string]interface{}{
		"pd_mode":                     reqCtx.PDMode,
		"prefill_pod":                 reqCtx.PrefillPodName,
		"decode_pod":                  reqCtx.DecodePodName,
		"prefill_latency_ms":          reqCtx.ActualPrefillLatency,
		"kv_transfer_latency_ms":      reqCtx.ActualKVTransferLatency,
		"decode_tpot_ms":              reqCtx.ActualDecodeTPOT,
		"prefill_budget_ms":           reqCtx.PrefillTTFTBudget,
		"kv_transfer_budget_ms":       reqCtx.KVTransferBudget,
		"decode_tpot_budget_ms":       reqCtx.DecodeTPOTBudget,
		"remaining_ttft_budget_ms":    reqCtx.RemainingTTFTBudget,
		"remaining_tpot_budget_ms":    reqCtx.RemainingTPOTBudget,
		"slo_violation":               reqCtx.PDSLOViolation,
		"slo_violation_phase":         reqCtx.PDSLOViolationPhase,
		"total_ttft_ms":               reqCtx.ActualPrefillLatency + reqCtx.ActualKVTransferLatency,
		"predicted_prefill_ttft_ms":   reqCtx.PredictedPrefillTTFT,
		"predicted_kv_transfer_ms":    reqCtx.PredictedKVTransferMs,
		"predicted_decode_tpot_ms":    reqCtx.PredictedDecodeTPOT,
		"prefill_prediction_error_ms": reqCtx.ActualPrefillLatency - reqCtx.PredictedPrefillTTFT,
		"kv_prediction_error_ms":      reqCtx.ActualKVTransferLatency - reqCtx.PredictedKVTransferMs,
		"decode_prediction_error_ms":  reqCtx.ActualDecodeTPOT - reqCtx.PredictedDecodeTPOT,
	}
}
