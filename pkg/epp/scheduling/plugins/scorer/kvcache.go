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

package scorer

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

type KVCacheScorer struct{}

func (ss *KVCacheScorer) Name() string {
	return "kv-cache"
}

func (ss *KVCacheScorer) Score(ctx *types.SchedulingContext, pods []types.Pod) map[types.Pod]float64 {
	scoreFunc := ToScoreFunc(scorePerPod)
	return scoreFunc(ctx, pods)
}

func scorePerPod(ctx *types.SchedulingContext, pod types.Pod) float64 {
	return 1 - pod.GetMetrics().KVCacheUsagePercent
}
