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

package framework

// NewWeightedScorer initializes a new WeightedScorer and returns its pointer.
func NewWeightedScorer(scorer Scorer, weight int) *WeightedScorer {
	return &WeightedScorer{
		Scorer: scorer,
		weight: weight,
	}
}

// WeightedScorer is a struct that encapsulates a scorer with its weight.
type WeightedScorer struct {
	Scorer
	weight int
}

// Weight returns the weight of the scorer.
func (s *WeightedScorer) Weight() int {
	return s.weight
}
