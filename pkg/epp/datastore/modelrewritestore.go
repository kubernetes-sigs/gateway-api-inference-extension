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

// modelRewriteStore encapsulates the logic for storing and retrieving
// InferenceModelRewrite rules, handling precedence correctly. This struct is not
// thread-safe; concurrency must be managed by its consumer.
type modelRewriteStore struct {
	genericRules           []*rewriteRuleWithMetadata
	rulesByExactModelMatch map[string][]*rewriteRuleWithMetadata
	allReWrites            map[string]*v1alpha2.InferenceModelRewrite
}

func newModelRewriteStore() *modelRewriteStore {
	return &modelRewriteStore{
		genericRules:           []*rewriteRuleWithMetadata{},
		rulesByExactModelMatch: map[string][]*rewriteRuleWithMetadata{},      // Key is the exact model name.
		allReWrites:            map[string]*v1alpha2.InferenceModelRewrite{}, // Key is the rewrites name.
	}
}

// set adds or updates an InferenceModelRewrite in the store. It deconstructs the
// object into individual rules and stores them in the appropriate data structures,
// ensuring they remain sorted by precedence.
func (ms *modelRewriteStore) set(infModelRewrite *v1alpha2.InferenceModelRewrite) {
	name := infModelRewrite.Name
	// If the rewrite object already exists, remove its old rules before adding new ones.
	if _, ok := ms.allReWrites[name]; ok {
		ms.deleteInternal(infModelRewrite.Name)
	}
	ms.allReWrites[name] = infModelRewrite

	for i := range infModelRewrite.Spec.Rules {
		ruleWithMetadata := &rewriteRuleWithMetadata{
			rule:              infModelRewrite.Spec.Rules[i],
			createTimestamp:   infModelRewrite.CreationTimestamp.Time,
			parentRewriteName: name,
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

// delete removes an InferenceModelRewrite and all its associated rules from the store.
func (ms *modelRewriteStore) delete(nn types.NamespacedName) {
	ms.deleteInternal(nn.Name)
}

// deleteInternal is the non-locking implementation for deleting a rewrite.
func (ms *modelRewriteStore) deleteInternal(n string) {
	if _, ok := ms.allReWrites[n]; !ok {
		return
	}
	delete(ms.allReWrites, n)

	// Filter out the generic rules associated with the deleted rewrite.
	newGenericRules := make([]*rewriteRuleWithMetadata, 0, len(ms.genericRules))
	for _, ruleWithMd := range ms.genericRules {
		if ruleWithMd.parentName() != n {
			newGenericRules = append(newGenericRules, ruleWithMd)
		}
	}
	ms.genericRules = newGenericRules

	// Filter out the exact-match rules associated with the deleted rewrite.
	for modelName, rulesWithMd := range ms.rulesByExactModelMatch {
		newRules := make([]*rewriteRuleWithMetadata, 0, len(rulesWithMd))
		for _, r := range rulesWithMd {
			if r.parentName() != n {
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

// getRule returns the single, highest-precedence rule for a given model name.
// It prioritizes exact matches over generic ones, and among those, the oldest rule wins.
// It also returns the name of the InferenceModelRewrite resource that provided the rule.
func (ms *modelRewriteStore) getRule(modelName string) (*v1alpha2.InferenceModelRewriteRule, string) {
	// Exact matches have the highest precedence.
	if rulesWithMd, ok := ms.rulesByExactModelMatch[modelName]; ok && len(rulesWithMd) > 0 {
		return &rulesWithMd[0].rule, rulesWithMd[0].parentName() // The list is pre-sorted, so the first element is the oldest.
	}

	// If no exact match, fall back to the oldest generic rule.
	if len(ms.genericRules) > 0 {
		return &ms.genericRules[0].rule, ms.genericRules[0].parentName() // The list is pre-sorted.
	}
	return nil, ""
}

// getAll returns a slice of all InferenceModelRewrite objects currently in the store.
func (ms *modelRewriteStore) getAll() []*v1alpha2.InferenceModelRewrite {
	rewrites := make([]*v1alpha2.InferenceModelRewrite, 0, len(ms.allReWrites))
	for _, rewrite := range ms.allReWrites {
		rewrites = append(rewrites, rewrite)
	}
	return rewrites
}

// rewriteRuleWithMetadata decorates a rule with metadata from its parent object
// to be used in precedence sorting.
type rewriteRuleWithMetadata struct {
	rule              v1alpha2.InferenceModelRewriteRule
	createTimestamp   time.Time
	parentRewriteName string
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

func (rr rewriteRuleWithMetadata) parentName() string {
	return rr.parentRewriteName
}
