// Package latencypredictorasync provides a Go client for the Python-based
// latency prediction service with asynchronous batching and cached metrics.
package latencypredictorasync

import (
	"context"
	"fmt"
	"math/rand"
	"net/http"
	"sync"
	"time"

	"github.com/go-logr/logr"
)

// --- Predictor Client ---

type Predictor struct {
	config     *Config
	httpClient *http.Client
	logger     logr.Logger
	rng        *rand.Rand

	metricsMu     sync.RWMutex
	cachedMetrics *MetricsResponse
	modelInfo     *ModelInfo
	serverStatus  *ServerStatusResponse

	xgboostMu sync.RWMutex

	bufferMu sync.Mutex
	pending  []TrainingEntry

	wg   sync.WaitGroup
	done chan struct{}
}

func New(config *Config, logger logr.Logger) *Predictor {
	if config == nil {
		config = ConfigFromEnv()
	}
	p := &Predictor{
		config:     config,
		httpClient: &http.Client{Timeout: config.HTTPTimeout},
		logger:     logger.WithName("latency-predictor-client"),
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
		done:       make(chan struct{}),
	}
	go p.backgroundLoop()
	return p
}

// Start initializes the predictor by fetching server status and model info.
func (p *Predictor) Start(ctx context.Context) error {
	// Get initial server status
	if err := p.refreshServerStatus(ctx); err != nil {
		return fmt.Errorf("failed to get initial server status: %v", err)
	}

	// Get initial model info if training server is available
	if err := p.refreshModelInfo(ctx); err != nil {
		return fmt.Errorf("failed to get initial model info: %v", err)
	}

	p.logger.Info("Latency predictor async client started.",
		"training_url", p.config.TrainingURL,
		"prediction_urls", p.config.PredictionURLs,
		"max_sample_size", p.config.MaxSampleSize,
		"flush_interval", p.config.FlushInterval,
		"use_native_xgboost", p.config.UseNativeXGBoost,
		"max_bulk_size", p.config.MaxBulkSize)
	return nil
}

// Stop stops background work, then does a final flush/refresh.
func (p *Predictor) Stop() {
	close(p.done)
	p.wg.Wait() // Wait for the background loop to finish
	// final flush & refresh
	p.flushTraining()
	p.refreshMetrics()
	p.logger.Info("Latency predictor async client stopped.")
}

// backgroundLoop runs flush & refresh at configured intervals.
func (p *Predictor) backgroundLoop() {
	p.wg.Add(1)
	defer p.wg.Done()
	flushTicker := time.NewTicker(p.config.FlushInterval)
	metricsTicker := time.NewTicker(p.config.MetricsRefreshInterval)
	defer flushTicker.Stop()
	defer metricsTicker.Stop()

	for {
		select {
		case <-flushTicker.C:
			p.flushTraining()
		case <-metricsTicker.C:
			p.refreshMetrics()
			// Also refresh server status periodically
			ctx, cancel := context.WithTimeout(context.Background(), p.config.HTTPTimeout)
			if err := p.refreshServerStatus(ctx); err != nil {
				p.logger.Error(err, "failed to refresh server status during background refresh")
			}
			cancel()
		case <-p.done:
			return
		}
	}
}

// GetXGBoostTrees returns the cached XGBoost tree data. It does not fetch new data.
func (p *Predictor) GetXGBoostTrees(ctx context.Context) (*XGBoostTrees, error) {
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()
	if p.cachedMetrics == nil || p.cachedMetrics.XGBoostTrees == nil {
		return nil, fmt.Errorf("no cached XGBoost trees available")
	}
	return p.cachedMetrics.XGBoostTrees, nil
}

// GetModelInfo fetches the latest model info from the training server.
func (p *Predictor) GetModelInfo(ctx context.Context) (*ModelInfo, error) {
	if err := p.refreshModelInfo(ctx); err != nil {
		return nil, err
	}
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()

	return p.modelInfo, nil
}

// GetCachedMetrics returns the last metrics fetched. The bool indicates if a value is cached.
func (p *Predictor) GetCachedMetrics() (*MetricsResponse, bool) {
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()
	if p.cachedMetrics == nil {
		return nil, false
	}
	return p.cachedMetrics, true
}

// IsXGBoostReady returns true if native XGBoost models are loaded and ready.
func (p *Predictor) IsXGBoostReady() bool {
	p.xgboostMu.RLock()
	defer p.xgboostMu.RUnlock()
	return p.modelInfo != nil && p.modelInfo.ModelType == xgBoostModelType
}

// IsLightGBMReady returns true if LightGBM models are available via HTTP.
func (p *Predictor) IsLightGBMReady() bool {
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()
	return p.modelInfo != nil && p.modelInfo.ModelType == gbmModelType && len(p.config.PredictionURLs) > 0
}

// IsBayesianRidgeReady returns true if Bayesian Ridge coefficients are cached.
func (p *Predictor) IsBayesianRidgeReady() bool {
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()
	return p.cachedMetrics != nil && p.cachedMetrics.Coefficients != nil
}

// GetCurrentModelType returns the current model type from cached server status or model info.
func (p *Predictor) GetCurrentModelType() string {
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()

	// Prefer server status if available
	if p.serverStatus != nil {
		return p.serverStatus.ModelType
	}

	if p.modelInfo == nil {
		return ""
	}
	return p.modelInfo.ModelType
}

// GetCurrentQuantile returns the current quantile from server status or defaults to 0.9
func (p *Predictor) GetCurrentQuantile() float64 {
	p.metricsMu.RLock()
	defer p.metricsMu.RUnlock()

	// Prefer server status if available
	if p.serverStatus != nil && p.serverStatus.Quantile > 0 {
		return p.serverStatus.Quantile
	}

	if p.modelInfo != nil && p.modelInfo.Quantile > 0 {
		return p.modelInfo.Quantile
	}

	return 0.9 // Default quantile
}

// IsReady returns true if a prediction method is ready based on the current model type.
func (p *Predictor) IsReady() bool {
	switch p.GetCurrentModelType() {
	case bayesianRidgeModelType:
		return p.IsBayesianRidgeReady()
	case xgBoostModelType:
		// Ready if we have prediction URLs for HTTP calls
		return len(p.config.PredictionURLs) > 0
	case gbmModelType:
		// Ready if we have prediction URLs for HTTP calls
		return p.IsLightGBMReady()
	default:
		return false
	}
}

// GetPredictionURLs returns the list of configured prediction URLs for debugging/monitoring.
func (p *Predictor) GetPredictionURLs() []string {
	return p.config.PredictionURLs
}

// GetTrainingURL returns the configured training URL for debugging/monitoring.
func (p *Predictor) GetTrainingURL() string {
	return p.config.TrainingURL
}
