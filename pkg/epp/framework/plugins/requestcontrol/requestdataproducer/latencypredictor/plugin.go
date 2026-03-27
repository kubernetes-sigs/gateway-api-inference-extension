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

package latencypredictor

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"k8s.io/apimachinery/pkg/types"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

const (
	// LatencyDataProviderPluginType is the plugin type for the latency predictor.
	// It trains XGBoost models via the sidecar and generates predictions for scoring.
	LatencyDataProviderPluginType = "latency-predictor-producer"

	// TTFTSLOHeaderKey is the header key for the TTFT SLO.
	TTFTSLOHeaderKey = "x-slo-ttft-ms"
	// TPOTSLOHeaderKey is the header key for the TPOT SLO.
	TPOTSLOHeaderKey = "x-slo-tpot-ms"

	// Experimental_DefaultPrefillProfile is the default profile name for prefill pods in disaggregated serving.
	Experimental_DefaultPrefillProfile = "prefill"
)

// PredictedLatency is the latency data provider plugin. It handles:
//   - PrepareRequestData: bulk predictions via the latency predictor sidecar
//   - PreRequest: dispatch-time bookkeeping (token counters, request queues)
//   - ResponseHeader/ResponseBody: training data collection (TTFT/TPOT)
//   - Produces/Consumes: endpoint attribute declarations
//
// Scoring, picking, and admission are handled by separate sub-plugins:
// LatencyScorer, AffinityWeightedPicker, and LatencyAdmission.
type PredictedLatency struct {
	typedName             plugin.TypedName
	latencypredictor      latencypredictor.PredictorInterface
	runningRequestLists   sync.Map                                      // Key: types.NamespacedName, Value: *requestPriorityQueue
	sloContextStore       *ttlcache.Cache[string, *predictedLatencyCtx] // TTL cache for request contexts
	config                Config
	prefillTokensInFlight sync.Map // Key: pod NamespacedName.String(), Value: *atomic.Int64
}

// podCounter returns the atomic counter for the given pod key, creating it if necessary.
func (t *PredictedLatency) podCounter(m *sync.Map, key string) *atomic.Int64 {
	v, _ := m.LoadOrStore(key, new(atomic.Int64))
	return v.(*atomic.Int64)
}

type Config struct {
	SamplingMean                       float64       `json:"samplingMean,omitempty"`
	MaxDecodeTokenSamplesForPrediction int           `json:"maxDecodeTokenSamplesForPrediction,omitempty"`
	SLOBufferFactor                    float64       `json:"sloBufferFactor,omitempty"`
	ContextTTL                         time.Duration `json:"contextTTL,omitempty"`
	StreamingMode                      bool          `json:"streamingMode,omitempty"`
	EndpointRoleLabel                  string        `json:"endpointRoleLabel,omitempty"`
	// PredictInPrepareData controls whether bulk predictions are generated during
	// PrepareRequestData. Set to false to disable predictions (training-only mode).
	// When false, the predictor still collects training data but does not call the
	// sidecar for predictions. Default: true.
	PredictInPrepareData bool `json:"predictInPrepareData,omitempty"`
}

var DefaultConfig = Config{
	SamplingMean:                       1000,
	MaxDecodeTokenSamplesForPrediction: 0,
	SLOBufferFactor:                    1,
	ContextTTL:                         5 * time.Minute,
	StreamingMode:                      false,
	PredictInPrepareData:               true,
}

func PredictedLatencyFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := DefaultConfig
	if len(rawParameters) > 0 {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config for PredictedLatency: %w", err)
		}
	}

	if err := parameters.validate(); err != nil {
		return nil, fmt.Errorf("invalid PredictedLatency config: %w", err)
	}

	predictor, err := startPredictor(handle)
	if err != nil {
		return nil, fmt.Errorf("failed to start latency predictor: %w", err)
	}

	return NewPredictedLatency(parameters, predictor).WithName(name), nil
}

func (c *Config) validate() error {
	var errs []error

	if c.SamplingMean <= 0 {
		errs = append(errs, fmt.Errorf("samplingMean must be > 0, got %f", c.SamplingMean))
	}

	if c.MaxDecodeTokenSamplesForPrediction < 0 {
		errs = append(errs, fmt.Errorf("maxDecodeTokenSamplesForPrediction must be >= 0, got %d", c.MaxDecodeTokenSamplesForPrediction))
	}

	if c.SLOBufferFactor <= 0 {
		errs = append(errs, fmt.Errorf("sloBufferFactor must be > 0, got %f", c.SLOBufferFactor))
	}

	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return nil
}

func NewPredictedLatency(config Config, predictor latencypredictor.PredictorInterface) *PredictedLatency {
	predictedLatency := &PredictedLatency{
		typedName:        plugin.TypedName{Type: LatencyDataProviderPluginType, Name: LatencyDataProviderPluginType},
		latencypredictor: predictor,
		config:           config,
	}

	predictedLatency.sloContextStore = ttlcache.New(
		ttlcache.WithTTL[string, *predictedLatencyCtx](config.ContextTTL),
	)

	predictedLatency.sloContextStore.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, *predictedLatencyCtx]) {
		if reason != ttlcache.EvictionReasonExpired {
			return
		}
		plCtx := item.Value()
		predictedLatency.removeRequestFromQueue(item.Key(), plCtx)
		if plCtx.prefillTokensAtDispatch > 0 || plCtx.prefillTokensAtDispatchOnPrefill > 0 {
			if plCtx.prefillTargetMetadata != nil && plCtx.ttft == 0 {
				prefillPodKey := plCtx.prefillTargetMetadata.NamespacedName.String()
				if predictedLatency.podCounter(&predictedLatency.prefillTokensInFlight, prefillPodKey).Add(-int64(plCtx.inputTokenCount)) == 0 {
					predictedLatency.prefillTokensInFlight.Delete(prefillPodKey)
				}
			}
			if plCtx.targetMetadata != nil {
				decodePodKey := plCtx.targetMetadata.NamespacedName.String()
				if predictedLatency.podCounter(&predictedLatency.prefillTokensInFlight, decodePodKey).Add(-int64(plCtx.inputTokenCount)) == 0 {
					predictedLatency.prefillTokensInFlight.Delete(decodePodKey)
				}
			}
		}
	})

	go predictedLatency.sloContextStore.Start()
	return predictedLatency
}

func startPredictor(handle plugin.Handle) (latencypredictor.PredictorInterface, error) {
	predictor := latencypredictor.New(latencypredictor.ConfigFromEnv(), ctrl.Log.WithName("latency-predictor-producer"))
	if err := predictor.Start(handle.Context()); err != nil {
		return nil, fmt.Errorf("failed to start latency predictor: %w", err)
	}

	go func() {
		<-handle.Context().Done()
		stopCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		predictor.Stop(stopCtx)
	}()
	return predictor, nil
}

func (s *PredictedLatency) TypedName() plugin.TypedName {
	return s.typedName
}

func (s *PredictedLatency) WithName(name string) *PredictedLatency {
	s.typedName.Name = name
	return s
}

func (t *PredictedLatency) getOrMakePredictedLatencyContextForRequest(request *framework.LLMRequest) *predictedLatencyCtx {
	sloCtx, err := t.getPredictedLatencyContextForRequest(request)
	if err != nil {
		sloCtx = newPredictedLatencyContext(request)
	}
	return sloCtx
}

// --- Per-request context ---

// predictedLatencyCtx holds per-request state for latency prediction and training.
type predictedLatencyCtx struct {
	schedulingRequest         framework.LLMRequest
	targetMetadata            *fwkdl.EndpointMetadata
	prefillTargetMetadata     *fwkdl.EndpointMetadata
	schedulingResult          *framework.SchedulingResult
	lastSeenMetrics           map[string]*fwkdl.Metrics
	lastTokenTimestamp        time.Time
	requestReceivedTimestamp  time.Time
	generatedTokenCount       int
	incomingModelName         string
	ttft                      float64
	predictedTTFT             float64
	avgTPOT                   float64
	avgPredictedTPOT          float64
	decodeTokenSampler        *decodeTokenSampler
	tpotObservations          []float64
	predictedTPOTObservations []float64

	promptText      string
	inputTokenCount int

	prefixCacheScoresForEndpoints map[string]float64

	ttftSLO    float64
	avgTPOTSLO float64

	predictionsForScheduling map[string]endpointPredictionResult

	prefillTokensAtDispatch          int64
	prefillTokensAtDispatchOnPrefill int64
	decodeTokensAtDispatch           int64
}

func newPredictedLatencyContext(request *framework.LLMRequest) *predictedLatencyCtx {
	var promptText string
	if request.Body != nil {
		promptText = request.Body.PromptText()
	}
	return &predictedLatencyCtx{
		schedulingRequest:             *request,
		promptText:                    promptText,
		inputTokenCount:               len(strings.Fields(promptText)),
		lastSeenMetrics:               make(map[string]*fwkdl.Metrics),
		prefixCacheScoresForEndpoints: make(map[string]float64),
		predictionsForScheduling:      make(map[string]endpointPredictionResult),
	}
}

func (s *PredictedLatency) getPredictedLatencyContextForRequest(request *framework.LLMRequest) (*predictedLatencyCtx, error) {
	id := request.Headers[reqcommon.RequestIdHeaderKey]
	if item := s.sloContextStore.Get(id); item != nil {
		return item.Value(), nil
	}
	return nil, fmt.Errorf("SLO context not found for request ID: %s", id)
}

func (s *PredictedLatency) setPredictedLatencyContextForRequest(request *framework.LLMRequest, ctx *predictedLatencyCtx) {
	id := request.Headers[reqcommon.RequestIdHeaderKey]
	s.sloContextStore.Set(id, ctx, ttlcache.DefaultTTL)
}

func (s *PredictedLatency) deletePredictedLatencyContextForRequest(request *framework.LLMRequest) {
	id := request.Headers[reqcommon.RequestIdHeaderKey]
	s.sloContextStore.Delete(id)
}

// --- Header parsing ---

// parseFloatHeader retrieves a header by name, parses it as a float64,
// and returns the value or an error if the header is missing or invalid.
func parseFloatHeader(request framework.LLMRequest, headerName string) (float64, error) {
	headerValue, ok := request.Headers[headerName]
	if !ok {
		return 0, nil
	}
	parsedFloat, err := strconv.ParseFloat(headerValue, 64)
	if err != nil {
		return 0, errcommon.Error{
			Code: errcommon.BadRequest,
			Msg:  headerName + " must be a float",
		}
	}
	return parsedFloat, nil
}

func (s *PredictedLatency) parseSLOHeaders(ctx context.Context, request *framework.LLMRequest, predictedLatencyCtx *predictedLatencyCtx) {
	logger := log.FromContext(ctx)
	var err error

	predictedLatencyCtx.ttftSLO, err = parseFloatHeader(*request, TTFTSLOHeaderKey)
	if err != nil {
		logger.V(logutil.DEBUG).Error(errcommon.Error{Code: errcommon.BadRequest, Msg: fmt.Sprintf("%v must be a float: %v", TTFTSLOHeaderKey, err)}, "PredictedLatency: Error parsing TTFT SLO from header")
	}

	predictedLatencyCtx.avgTPOTSLO, err = parseFloatHeader(*request, TPOTSLOHeaderKey)
	if err != nil {
		logger.V(logutil.DEBUG).Error(errcommon.Error{Code: errcommon.BadRequest, Msg: fmt.Sprintf("%v must be a float: %v", TPOTSLOHeaderKey, err)}, "PredictedLatency: Error parsing TPOT SLO from header")
	}
}

// --- Running request queue helpers ---

func (s *PredictedLatency) getEndpointMinTPOTSLO(endpoint framework.Endpoint) float64 {
	endpointName := endpoint.GetMetadata().NamespacedName
	if runningReqs := s.getRunningRequestList(endpointName); runningReqs != nil && runningReqs.GetSize() > 0 {
		if min := runningReqs.Peek(); min != nil {
			return min.tpot
		}
	}
	return 0
}

func (s *PredictedLatency) getEndpointRunningRequestCount(endpoint framework.Endpoint) int {
	endpointName := endpoint.GetMetadata().NamespacedName
	if runningReqs := s.getRunningRequestList(endpointName); runningReqs != nil {
		return runningReqs.GetSize()
	}
	return 0
}

func (s *PredictedLatency) getRunningRequestList(endpointName types.NamespacedName) *requestPriorityQueue {
	if value, ok := s.runningRequestLists.Load(endpointName); ok {
		return value.(*requestPriorityQueue)
	}
	return nil
}

func (s *PredictedLatency) removeRequestFromEndpoint(endpointName types.NamespacedName, requestID string) {
	if queue := s.getRunningRequestList(endpointName); queue != nil {
		queue.Remove(requestID)
		if queue.GetSize() == 0 {
			s.runningRequestLists.Delete(endpointName)
		}
	}
}

func (s *PredictedLatency) removeRequestFromQueue(requestID string, ctx *predictedLatencyCtx) {
	if ctx == nil || ctx.targetMetadata == nil {
		return
	}
	endpointName := types.NamespacedName{
		Name:      ctx.targetMetadata.NamespacedName.Name,
		Namespace: ctx.targetMetadata.NamespacedName.Namespace,
	}
	s.removeRequestFromEndpoint(endpointName, requestID)
}
