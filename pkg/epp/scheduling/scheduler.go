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

// Package scheduling implements request scheduling algorithms.
package scheduling

import (
	"context"
	"fmt"
	"math/rand"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	envutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/env"
)

// Config holds all the configuration values for the scheduler
type Config struct {
	KVCacheThreshold       float64
	QueueThresholdCritical int
	QueueingThresholdLoRA  int
	LoraAffinityThreshold  float64
}

var (
	// Default values to use if environment variables are not set
	defaultKVCacheThreshold       = 0.8
	defaultQueueThresholdCritical = 5
	defaultQueueingThresholdLoRA  = 128
	defaultLoraAffinityThreshold  = 0.999
)

// LoadConfig loads configuration from environment variables
func LoadConfig() Config {
	// Use a default logger for initial configuration loading
	baseLogger := log.Log.WithName("scheduling-config")

	config := Config{
		KVCacheThreshold:       envutil.GetEnvFloat("KV_CACHE_THRESHOLD", defaultKVCacheThreshold, baseLogger),
		QueueThresholdCritical: envutil.GetEnvInt("QUEUE_THRESHOLD_CRITICAL", defaultQueueThresholdCritical, baseLogger),
		QueueingThresholdLoRA:  envutil.GetEnvInt("QUEUING_THRESHOLD_LORA", defaultQueueingThresholdLoRA, baseLogger),
		LoraAffinityThreshold:  envutil.GetEnvFloat("LORA_AFFINITY_THRESHOLD", defaultLoraAffinityThreshold, baseLogger),
	}

	baseLogger.V(logutil.DEFAULT).Info("Scheduler configuration loaded",
		"kvCacheThreshold", config.KVCacheThreshold,
		"queueThresholdCritical", config.QueueThresholdCritical,
		"queueingThresholdLoRA", config.QueueingThresholdLoRA,
		"loraAffinityThreshold", config.LoraAffinityThreshold)

	return config
}

var config = LoadConfig()

var (
	defaultFilter = &filter{
		name:          "critical request",
		filter:        toFilterFunc(criticalRequestPredicate),
		nextOnSuccess: lowLatencyFilter,
		nextOnFailure: sheddableRequestFilter,
	}

	// queueLoRAAndKVCacheFilter applied least queue -> low cost lora ->  least KV Cache filter
	queueLoRAAndKVCacheFilter = &filter{
		name:   "least queuing",
		filter: leastQueuingFilterFunc,
		nextOnSuccessOrFailure: &filter{
			name:   "low cost LoRA",
			filter: loRASoftAffinityFilter,
			nextOnSuccessOrFailure: &filter{
				name:   "least KV cache percent",
				filter: leastKVCacheFilterFunc,
			},
		},
	}

	// queueAndKVCacheFilter applies least queue followed by least KV Cache filter
	queueAndKVCacheFilter = &filter{
		name:   "least queuing",
		filter: leastQueuingFilterFunc,
		nextOnSuccessOrFailure: &filter{
			name:   "least KV cache percent",
			filter: leastKVCacheFilterFunc,
		},
	}

	lowLatencyFilter = &filter{
		name:   "low queueing filter",
		filter: toFilterFunc((lowQueueingPodPredicate)),
		nextOnSuccess: &filter{
			name:                   "affinity LoRA",
			filter:                 loRASoftAffinityFilter,
			nextOnSuccessOrFailure: queueAndKVCacheFilter,
		},
		nextOnFailure: queueLoRAAndKVCacheFilter,
	}

	sheddableRequestFilter = &filter{
		// When there is at least one model server that's not queuing requests, and still has KV
		// cache below a certain threshold, we consider this model server has capacity to handle
		// a sheddable request without impacting critical requests.
		name:          "has capacity for sheddable requests",
		filter:        toFilterFunc(noQueueAndLessThanKVCacheThresholdPredicate(config.QueueThresholdCritical, config.KVCacheThreshold)),
		nextOnSuccess: queueLoRAAndKVCacheFilter,
		// If all pods are queuing or running above the KVCache threshold, we drop the sheddable
		// request to make room for critical requests.
		nextOnFailure: &filter{
			name: "drop request",
			filter: func(logger logr.Logger, req *LLMRequest, pods []backendmetrics.PodMetrics) ([]backendmetrics.PodMetrics, error) {
				logger.V(logutil.DEFAULT).Info("Request dropped", "request", req)
				return []backendmetrics.PodMetrics{}, errutil.Error{
					Code: errutil.InferencePoolResourceExhausted, Msg: "dropping request due to limited backend resources",
				}
			},
		},
	}
)



func NewScheduler(datastore datastore.Datastore) *Scheduler {
	return &Scheduler{
		datastore: datastore,
		filter:    defaultFilter,
	}
}

type Scheduler struct {
	datastore datastore.Datastore
	filter    Filter
}

// Schedule finds the target pod based on metrics and the requested lora adapter.
func (s *Scheduler) Schedule(ctx context.Context, req *LLMRequest) (targetPod backendmetrics.PodMetrics, err error) {
	logger := log.FromContext(ctx).WithValues("request", req)

	// Log current configuration values for debugging purposes.
	logger.V(logutil.TRACE).Info("Scheduler configuration",
		"KVCacheThreshold", config.KVCacheThreshold,
		"QueueThresholdCritical", config.QueueThresholdCritical,
		"QueueingThresholdLoRA", config.QueueingThresholdLoRA,
		"LoraAffinityThreshold", config.LoraAffinityThreshold,
	)

	podMetrics := s.datastore.PodGetAll()
	logger.V(logutil.VERBOSE).Info(fmt.Sprintf("Scheduling a request. Metrics: %+v", podMetrics))

	pods, err := s.filter.Filter(logger, req, podMetrics)
	if err != nil || len(pods) == 0 {
		return nil, fmt.Errorf("failed to apply filter, resulted %v pods, this should never happen: %w", len(pods), err)
	}
	logger.V(logutil.VERBOSE).Info(fmt.Sprintf("Selecting a random pod from %d candidates: %+v", len(pods), pods))
	i := rand.Intn(len(pods))
	return pods[i], nil
}
