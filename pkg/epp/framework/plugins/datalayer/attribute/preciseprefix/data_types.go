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

package preciseprefix

import (
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	PrecisePrefixCacheMatchInfoKey = "PrecisePrefixCacheMatchInfoKey"
)

// PrecisePrefixCacheMatchInfo extends PrefixCacheMatchInfo with weighted scoring information
// from the llm-d-inference-scheduler's LongestPrefixScorer.
// It inherits matchBlocks, totalBlocks, and blockSizeTokens from PrefixCacheMatchInfo,
// and adds weightedScore which accounts for device tier weights (e.g., HBM=1.0, SSD=0.5).
type PrecisePrefixCacheMatchInfo struct {
	*prefix.PrefixCacheMatchInfo
	// weightedScore is the weighted sum of consecutive matched blocks from the start.
	// The LongestPrefixScorer calculates this as: sum(weight_i) for i=0 to N-1,
	// where N is the number of consecutive blocks matched from block 0,
	// and weight_i is based on the device tier (e.g., HBM=1.0, SSD=0.5).
	// The sequence breaks as soon as a block is not found, ensuring consecutive prefix matching.
	weightedScore float64
}

func NewPrecisePrefixCacheMatchInfo(matchBlocks int, totalBlocks int, blockSizeTokens int, weightedScore float64) *PrecisePrefixCacheMatchInfo {
	return &PrecisePrefixCacheMatchInfo{
		PrefixCacheMatchInfo: prefix.NewPrefixCacheMatchInfo(matchBlocks, totalBlocks, blockSizeTokens),
		weightedScore:        weightedScore,
	}
}

func (p *PrecisePrefixCacheMatchInfo) WeightedScore() float64 {
	return p.weightedScore
}

func (p *PrecisePrefixCacheMatchInfo) Clone() fwkdl.Cloneable {
	return &PrecisePrefixCacheMatchInfo{
		PrefixCacheMatchInfo: p.PrefixCacheMatchInfo.Clone().(*prefix.PrefixCacheMatchInfo),
		weightedScore:        p.weightedScore,
	}
}
