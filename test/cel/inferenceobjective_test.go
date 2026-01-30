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

package main

import (
	"context"
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
)

func TestValidateInferenceObjective(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		desc       string
		objective  *v1alpha2.InferenceObjective
		wantErrors []string
	}{
		{
			desc: "passes validation with poolRef only",
			objective: &v1alpha2.InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("test-obj-poolref-%v", time.Now().UnixNano()),
					Namespace: metav1.NamespaceDefault,
				},
				Spec: v1alpha2.InferenceObjectiveSpec{
					PoolRef: &v1alpha2.PoolObjectReference{
						Name: "test-pool",
					},
				},
			},
			wantErrors: nil,
		},
		{
			desc: "passes validation with poolSelector only",
			objective: &v1alpha2.InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("test-obj-selector-%v", time.Now().UnixNano()),
					Namespace: metav1.NamespaceDefault,
				},
				Spec: v1alpha2.InferenceObjectiveSpec{
					PoolSelector: &v1alpha2.PoolSelector{
						MatchLabels: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
							"graduation": "beta",
						},
					},
				},
			},
			wantErrors: nil,
		},
		{
			desc: "passes validation with poolSelector using matchExpressions",
			objective: &v1alpha2.InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("test-obj-expr-%v", time.Now().UnixNano()),
					Namespace: metav1.NamespaceDefault,
				},
				Spec: v1alpha2.InferenceObjectiveSpec{
					PoolSelector: &v1alpha2.PoolSelector{
						MatchExpressions: []v1alpha2.LabelSelectorRequirement{
							{
								Key:      "plan",
								Operator: v1alpha2.LabelSelectorOpIn,
								Values:   []v1alpha2.LabelValue{"plus", "pro"},
							},
						},
					},
				},
			},
			wantErrors: nil,
		},
		{
			desc: "passes validation with poolRef and priority",
			objective: &v1alpha2.InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("test-obj-priority-%v", time.Now().UnixNano()),
					Namespace: metav1.NamespaceDefault,
				},
				Spec: v1alpha2.InferenceObjectiveSpec{
					Priority: ptr.To(10),
					PoolRef: &v1alpha2.PoolObjectReference{
						Name: "test-pool",
					},
				},
			},
			wantErrors: nil,
		},
		{
			desc: "fails validation with both poolRef and poolSelector",
			objective: &v1alpha2.InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("test-obj-both-%v", time.Now().UnixNano()),
					Namespace: metav1.NamespaceDefault,
				},
				Spec: v1alpha2.InferenceObjectiveSpec{
					PoolRef: &v1alpha2.PoolObjectReference{
						Name: "test-pool",
					},
					PoolSelector: &v1alpha2.PoolSelector{
						MatchLabels: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
							"graduation": "beta",
						},
					},
				},
			},
			wantErrors: []string{"exactly one of poolRef or poolSelector must be specified"},
		},
		{
			desc: "fails validation with neither poolRef nor poolSelector",
			objective: &v1alpha2.InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("test-obj-neither-%v", time.Now().UnixNano()),
					Namespace: metav1.NamespaceDefault,
				},
				Spec: v1alpha2.InferenceObjectiveSpec{
					Priority: ptr.To(5),
				},
			},
			wantErrors: []string{"exactly one of poolRef or poolSelector must be specified"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			err := k8sClient.Create(ctx, tc.objective)

			// This is a boolean XOR. It's true if one is true, but not both.
			// It ensures that an error is returned if and only if we expect one.
			if (len(tc.wantErrors) != 0) != (err != nil) {
				t.Fatalf("Unexpected response while creating InferenceObjective; got err=\n%v\n; want error=%v", err, tc.wantErrors != nil)
			}

			// If we got an error, check that it contains the expected substrings.
			if err != nil {
				var missingErrorStrings []string
				for _, wantError := range tc.wantErrors {
					if !celErrorStringMatches(err.Error(), wantError) {
						missingErrorStrings = append(missingErrorStrings, wantError)
					}
				}
				if len(missingErrorStrings) != 0 {
					t.Errorf("Unexpected response while creating InferenceObjective; got err=\n%v\n; missing strings within error=%q", err, missingErrorStrings)
				}
			}

			// Clean up if object was created successfully
			if err == nil {
				_ = k8sClient.Delete(ctx, tc.objective)
			}
		})
	}
}
