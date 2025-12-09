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
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
)

func TestModelRewriteStore(t *testing.T) {
	now := time.Now()
	oneMinuteAgo := now.Add(-1 * time.Minute)

	// Define common rules with generic names
	ruleModel1V1 := v1alpha2.InferenceModelRewriteRule{
		Matches: []v1alpha2.Match{{Model: &v1alpha2.ModelMatch{Value: "model1"}}},
		Targets: []v1alpha2.TargetModel{{ModelRewrite: "model1-v1"}},
	}
	ruleModel1V2 := v1alpha2.InferenceModelRewriteRule{
		Matches: []v1alpha2.Match{{Model: &v1alpha2.ModelMatch{Value: "model1"}}},
		Targets: []v1alpha2.TargetModel{{ModelRewrite: "model1-v2"}},
	}
	ruleGeneric := v1alpha2.InferenceModelRewriteRule{
		Targets: []v1alpha2.TargetModel{{ModelRewrite: "generic-fallback"}},
	}

	// Define rewrite objects using plain structs
	rewriteOld := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{Name: "rewrite-old", Namespace: "default", CreationTimestamp: metav1.NewTime(oneMinuteAgo)},
		Spec:       v1alpha2.InferenceModelRewriteSpec{Rules: []v1alpha2.InferenceModelRewriteRule{ruleModel1V1}},
	}
	rewriteNew := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{Name: "rewrite-new", Namespace: "default", CreationTimestamp: metav1.NewTime(now)},
		Spec:       v1alpha2.InferenceModelRewriteSpec{Rules: []v1alpha2.InferenceModelRewriteRule{ruleModel1V2}},
	}
	rewriteGenericOld := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{Name: "rewrite-generic-old", Namespace: "default", CreationTimestamp: metav1.NewTime(oneMinuteAgo)},
		Spec:       v1alpha2.InferenceModelRewriteSpec{Rules: []v1alpha2.InferenceModelRewriteRule{ruleGeneric}},
	}
	rewriteGenericNew := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{Name: "rewrite-generic-new", Namespace: "default", CreationTimestamp: metav1.NewTime(now)},
		Spec:       v1alpha2.InferenceModelRewriteSpec{Rules: []v1alpha2.InferenceModelRewriteRule{{Targets: []v1alpha2.TargetModel{{ModelRewrite: "new-generic"}}}}},
	}
	rewriteUpdated := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{Name: "rewrite-old", Namespace: "default", CreationTimestamp: metav1.NewTime(now)}, // Same name as rewriteOld
		Spec:       v1alpha2.InferenceModelRewriteSpec{Rules: []v1alpha2.InferenceModelRewriteRule{ruleModel1V2}},
	}

	tests := []struct {
		name         string
		initialState []*v1alpha2.InferenceModelRewrite
		op           func(store *modelRewriteStore)
		modelToGet   string
		wantRule     *v1alpha2.InferenceModelRewriteRule
		wantName     string
		wantGetAll   []*v1alpha2.InferenceModelRewrite
	}{
		{
			name:         "Simple exact match",
			initialState: []*v1alpha2.InferenceModelRewrite{rewriteOld},
			modelToGet:   "model1",
			wantRule:     &ruleModel1V1,
			wantName:     rewriteOld.Name,
			wantGetAll:   []*v1alpha2.InferenceModelRewrite{rewriteOld},
		},
		{
			name:         "Simple generic match",
			initialState: []*v1alpha2.InferenceModelRewrite{rewriteGenericOld},
			modelToGet:   "model2", // A different model to test generic fallback
			wantRule:     &ruleGeneric,
			wantName:     rewriteGenericOld.Name,
			wantGetAll:   []*v1alpha2.InferenceModelRewrite{rewriteGenericOld},
		},
		{
			name:         "No match",
			initialState: []*v1alpha2.InferenceModelRewrite{rewriteOld},
			modelToGet:   "model2",
			wantRule:     nil,
			wantName:     "",
			wantGetAll:   []*v1alpha2.InferenceModelRewrite{rewriteOld},
		},
		{
			name:         "Precedence: Exact match wins over generic",
			initialState: []*v1alpha2.InferenceModelRewrite{rewriteOld, rewriteGenericOld},
			modelToGet:   "model1",
			wantRule:     &ruleModel1V1,
			wantName:     rewriteOld.Name,
			wantGetAll:   []*v1alpha2.InferenceModelRewrite{rewriteOld, rewriteGenericOld},
		},
		{
			name:         "Precedence: Fallback to generic when no exact match",
			initialState: []*v1alpha2.InferenceModelRewrite{rewriteOld, rewriteGenericOld},
			modelToGet:   "model2",
			wantRule:     &ruleGeneric,
			wantName:     rewriteGenericOld.Name,
			wantGetAll:   []*v1alpha2.InferenceModelRewrite{rewriteOld, rewriteGenericOld},
		},
		{
			name:         "Precedence: Oldest exact match wins",
			initialState: []*v1alpha2.InferenceModelRewrite{rewriteNew, rewriteOld},
			modelToGet:   "model1",
			wantRule:     &ruleModel1V1,
			wantName:     rewriteOld.Name,
			wantGetAll:   []*v1alpha2.InferenceModelRewrite{rewriteNew, rewriteOld},
		},
		{
			name:         "Precedence: Oldest generic match wins",
			initialState: []*v1alpha2.InferenceModelRewrite{rewriteGenericNew, rewriteGenericOld},
			modelToGet:   "any-model",
			wantRule:     &ruleGeneric,
			wantName:     rewriteGenericOld.Name,
			wantGetAll:   []*v1alpha2.InferenceModelRewrite{rewriteGenericNew, rewriteGenericOld},
		},
		{
			name:         "Delete: successfully deletes a rewrite",
			initialState: []*v1alpha2.InferenceModelRewrite{rewriteOld, rewriteGenericOld},
			op: func(store *modelRewriteStore) {
				store.delete(types.NamespacedName{Namespace: rewriteOld.Namespace, Name: rewriteOld.Name})
			},
			modelToGet: "model1",
			wantRule:   &ruleGeneric, // Falls back to generic
			wantName:   rewriteGenericOld.Name,
			wantGetAll: []*v1alpha2.InferenceModelRewrite{rewriteGenericOld},
		},
		{
			name:         "Delete: non-existent rewrite does nothing",
			initialState: []*v1alpha2.InferenceModelRewrite{rewriteOld},
			op: func(store *modelRewriteStore) {
				store.delete(types.NamespacedName{Name: "non-existent", Namespace: "default"})
			},
			modelToGet: "model1",
			wantRule:   &ruleModel1V1,
			wantName:   rewriteOld.Name,
			wantGetAll: []*v1alpha2.InferenceModelRewrite{rewriteOld},
		},
		{
			name:         "Update: Setting a rewrite with the same name replaces the old one",
			initialState: []*v1alpha2.InferenceModelRewrite{rewriteOld},
			op: func(store *modelRewriteStore) {
				store.set(rewriteUpdated)
			},
			modelToGet: "model1",
			wantRule:   &ruleModel1V2,
			wantName:   rewriteUpdated.Name,
			wantGetAll: []*v1alpha2.InferenceModelRewrite{rewriteUpdated},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			store := newModelRewriteStore()
			for _, r := range tc.initialState {
				store.set(r)
			}

			if tc.op != nil {
				tc.op(store)
			}

			gotRule, gotName := store.getRule(tc.modelToGet)
			if diff := cmp.Diff(tc.wantRule, gotRule); diff != "" {
				t.Errorf("GetRule() mismatch (-want +got):\n%s", diff)
			}
			if gotName != tc.wantName {
				t.Errorf("GetRule() returned incorrect name: got %s, want %s", gotName, tc.wantName)
			}

			if tc.wantGetAll != nil {
				gotAll := store.getAll()
				if diff := cmp.Diff(tc.wantGetAll, gotAll, cmpopts.SortSlices(func(a, b *v1alpha2.InferenceModelRewrite) bool {
					return a.Name < b.Name
				})); diff != "" {
					t.Errorf("GetAll() mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}
