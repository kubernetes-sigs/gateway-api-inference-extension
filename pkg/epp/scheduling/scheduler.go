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
)

const (
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/16) Make this configurable.
	kvCacheThreshold = 0.8
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/16) Make this configurable.
	queueThresholdCritical = 5
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/16) Make this configurable.
	// the threshold for queued requests to be considered low below which we can prioritize LoRA affinity.
	// The value of 128 is arrived heuristicically based on experiments.
	queueingThresholdLoRA = 128
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/16) Make this configurable.
	// loraAffinityThreshold indicates the probability with which we prefer a pod with LoRA affinity over a pod without but having room to fit more LoRA adapters.
	loraAffinityThreshold = 0.999
)

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
		filter:        toFilterFunc(noQueueAndLessThanKVCacheThresholdPredicate(queueThresholdCritical, kvCacheThreshold)),
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
	podMetrics := s.datastore.PodGetAll()
	logger.V(logutil.VERBOSE).Info("Scheduling a request", "metrics", podMetrics)
	pods, err := s.filter.Filter(logger, req, podMetrics)
	if err != nil || len(pods) == 0 {
		return nil, fmt.Errorf(
			"failed to apply filter, resulted %v pods, this should never happen: %w", len(pods), err)
	}
	logger.V(logutil.VERBOSE).Info("Selecting a random pod from the candidates", "candidatePods", pods)
	i := rand.Intn(len(pods))
	return pods[i], nil
}
