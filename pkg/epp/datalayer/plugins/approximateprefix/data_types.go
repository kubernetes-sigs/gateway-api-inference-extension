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

package approximateprefix

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

const (
	PrefixCacheMatchInfoKey = "PrefixCacheMatchInfoKey"
)

type PrefixCacheMatchInfo struct {
	matchLength int
	totalBlocks int
}

func NewPrefixCacheMatchInfo(matchLen int, blockHashLen int) *PrefixCacheMatchInfo {
	return &PrefixCacheMatchInfo{
		matchLength: matchLen,
		totalBlocks: blockHashLen,
	}
}

func (p *PrefixCacheMatchInfo) MatchLength() int {
	return p.matchLength
}

func (p *PrefixCacheMatchInfo) TotalLength() int {
	return p.totalBlocks
}

func (p *PrefixCacheMatchInfo) Clone() datalayer.Cloneable {
	return &PrefixCacheMatchInfo{
		matchLength: p.matchLength,
		totalBlocks: p.totalBlocks,
	}
}
