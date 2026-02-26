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

package prefix

import (
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

const (
	PrefixCacheMatchInfoKey = "PrefixCacheMatchInfoKey"
)

type PrefixCacheMatchInfo struct {
	// matched prefix length in blocks
	matchBlocks int
	// total length in blocks
	totalBlocks int
	// block length in tokens
	blockSizeTokens int
}

func NewPrefixCacheMatchInfo(matchBlocks int, totalBlocks int, blockSizeTokens int) *PrefixCacheMatchInfo {
	return &PrefixCacheMatchInfo{
		matchBlocks:     matchBlocks,
		totalBlocks:     totalBlocks,
		blockSizeTokens: blockSizeTokens,
	}
}

func (p *PrefixCacheMatchInfo) MatchBlocks() int {
	return p.matchBlocks
}

func (p *PrefixCacheMatchInfo) TotalBlocks() int {
	return p.totalBlocks
}

func (p *PrefixCacheMatchInfo) BlockSizeTokens() int {
	return p.blockSizeTokens
}

func (p *PrefixCacheMatchInfo) Clone() fwkdl.Cloneable {
	return &PrefixCacheMatchInfo{
		matchBlocks:     p.matchBlocks,
		totalBlocks:     p.totalBlocks,
		blockSizeTokens: p.blockSizeTokens,
	}
}
