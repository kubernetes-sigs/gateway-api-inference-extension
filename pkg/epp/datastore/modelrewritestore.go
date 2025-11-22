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

package datastore

import (
	"sort"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
)

// ModelRewriteStore encapsulates the logic for storing and retrieving
// InferenceModelRewrite rules, handling precedence correctly. This struct is not
// thread-safe; concurrency must be managed by its consumer.
type ModelRewriteStore struct {
	genericRules           []*rewriteRuleWithMetadata
	rulesByExactModelMatch map[string][]*rewriteRuleWithMetadata
	allReWrites            map[types.NamespacedName]*v1alpha2.InferenceModelRewrite
}

func NewModelRewriteStore() *ModelRewriteStore {
	return &ModelRewriteStore{
		genericRules:           []*rewriteRuleWithMetadata{},
		rulesByExactModelMatch: map[string][]*rewriteRuleWithMetadata{},
		allReWrites:            map[types.NamespacedName]*v1alpha2.InferenceModelRewrite{},
	}
}

// Set adds or updates an InferenceModelRewrite in the store. It deconstructs the
// object into individual rules and stores them in the appropriate data structures,
// ensuring they remain sorted by precedence.
func (ms *ModelRewriteStore) Set(infModelRewrite *v1alpha2.InferenceModelRewrite) {
	nn := getNN(infModelRewrite)

	// If the rewrite object already exists, remove its old rules before adding new ones.
	if _, ok := ms.allReWrites[nn]; ok {
		ms.deleteInternal(nn)
	}
	ms.allReWrites[nn] = infModelRewrite

	for i := range infModelRewrite.Spec.Rules {
		ruleWithMetadata := newRuleWithMetadata(infModelRewrite, i)
		if ruleWithMetadata == nil {
			continue
		}

		if ruleWithMetadata.isGeneric() {
			ms.genericRules = append(ms.genericRules, ruleWithMetadata)
		} else {
			for model := range ruleWithMetadata.exactModels() {
				ms.rulesByExactModelMatch[model] = append(ms.rulesByExactModelMatch[model], ruleWithMetadata)
			}
		}
	}

	// Sort all rule lists by timestamp to maintain precedence.
	sort.Slice(ms.genericRules, func(i, j int) bool {
		return ms.genericRules[i].createTimestamp.Before(ms.genericRules[j].createTimestamp)
	})

	for model := range ms.rulesByExactModelMatch {
		sort.Slice(ms.rulesByExactModelMatch[model], func(i, j int) bool {
			return ms.rulesByExactModelMatch[model][i].createTimestamp.Before(ms.rulesByExactModelMatch[model][j].createTimestamp)
		})
	}
}

// Delete removes an InferenceModelRewrite and all its associated rules from the store.
func (ms *ModelRewriteStore) Delete(nn types.NamespacedName) {
	ms.deleteInternal(nn)
}

// deleteInternal is the non-locking implementation for deleting a rewrite.
func (ms *ModelRewriteStore) deleteInternal(nn types.NamespacedName) {
	if _, ok := ms.allReWrites[nn]; !ok {
		return
	}
	delete(ms.allReWrites, nn)

	// Filter out the generic rules associated with the deleted rewrite.
	newGenericRules := make([]*rewriteRuleWithMetadata, 0, len(ms.genericRules))
	for _, ruleWithMd := range ms.genericRules {
		if ruleWithMd.parentNN() != nn {
			newGenericRules = append(newGenericRules, ruleWithMd)
		}
	}
	ms.genericRules = newGenericRules

	// Filter out the exact-match rules associated with the deleted rewrite.
	for modelName, rulesWithMd := range ms.rulesByExactModelMatch {
		newRules := make([]*rewriteRuleWithMetadata, 0, len(rulesWithMd))
		for _, r := range rulesWithMd {
			if r.parentNN() != nn {
				newRules = append(newRules, r)
			}
		}

		if len(newRules) == 0 {
			delete(ms.rulesByExactModelMatch, modelName)
		} else {
			ms.rulesByExactModelMatch[modelName] = newRules
		}
	}
}

// GetRule returns the single, highest-precedence rule for a given model name.
// It prioritizes exact matches over generic ones, and among those, the oldest rule wins.
func (ms *ModelRewriteStore) GetRule(modelName string) *v1alpha2.InferenceModelRewriteRule {
	// Exact matches have the highest precedence.
	if rulesWithMd, ok := ms.rulesByExactModelMatch[modelName]; ok && len(rulesWithMd) > 0 {
		return &rulesWithMd[0].rule // The list is pre-sorted, so the first element is the oldest.
	}

	// If no exact match, fall back to the oldest generic rule.
	if len(ms.genericRules) > 0 {
		return &ms.genericRules[0].rule // The list is pre-sorted.
	}
	return nil
}

// GetAll returns a slice of all InferenceModelRewrite objects currently in the store.
func (ms *ModelRewriteStore) GetAll() []*v1alpha2.InferenceModelRewrite {
	rewrites := make([]*v1alpha2.InferenceModelRewrite, 0, len(ms.allReWrites))
	for _, rewrite := range ms.allReWrites {
		rewrites = append(rewrites, rewrite)
	}
	return rewrites
}

func getNN(infModelRewrite *v1alpha2.InferenceModelRewrite) types.NamespacedName {
	return types.NamespacedName{
		Namespace: infModelRewrite.Namespace,
		Name:      infModelRewrite.Name,
	}
}

// rewriteRuleWithMetadata decorates a rule with metadata from its parent object
// to be used in precedence sorting.
type rewriteRuleWithMetadata struct {
	rule            v1alpha2.InferenceModelRewriteRule
	createTimestamp time.Time
	parentRewriteNN types.NamespacedName
}

func newRuleWithMetadata(infModelRewrite *v1alpha2.InferenceModelRewrite, ruleIdx int) *rewriteRuleWithMetadata {
	if ruleIdx >= len(infModelRewrite.Spec.Rules) {
		return nil
	}
	return &rewriteRuleWithMetadata{
		rule:            infModelRewrite.Spec.Rules[ruleIdx],
		createTimestamp: infModelRewrite.CreationTimestamp.Time,
		parentRewriteNN: getNN(infModelRewrite),
	}
}

func (rr rewriteRuleWithMetadata) isGeneric() bool {
	return len(rr.rule.Matches) == 0
}

func (rr rewriteRuleWithMetadata) exactModels() map[string]bool {
	modelSet := map[string]bool{}
	for _, match := range rr.rule.Matches {
		if match.Model != nil {
			modelSet[match.Model.Value] = true
		}
	}
	return modelSet
}

func (rr rewriteRuleWithMetadata) parentNN() types.NamespacedName {
	return rr.parentRewriteNN
}
