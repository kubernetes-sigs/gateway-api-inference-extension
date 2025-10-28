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
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

func TestValidateInferencePool(t *testing.T) {
	ctx := context.Background()

	// baseInferencePool is a valid, InferencePool resource.
	baseInferencePool := v1.InferencePool{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "base-pool",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: v1.InferencePoolSpec{
			TargetPorts: []v1.Port{
				{Number: 8000},
			},
			Selector: v1.LabelSelector{
				MatchLabels: map[v1.LabelKey]v1.LabelValue{
					"app": "model-server",
				},
			},
			EndpointPickerRef: v1.EndpointPickerRef{
				Name: "epp",
				Kind: "Service",
				Port: ptrTo(v1.Port{Number: 9002}),
			},
		},
	}

	testCases := []struct {
		desc       string
		mutate     func(ip *v1.InferencePool)
		wantErrors []string
	}{
		{
			desc: "passes validation with a valid configuration",
			mutate: func(ip *v1.InferencePool) {
			},
			wantErrors: nil,
		},
		{
			desc: "fails validation when kind is unset (defaults to Service) and port is missing",
			mutate: func(ip *v1.InferencePool) {
				// By setting Kind to an empty string, we rely on the API server's default value of "Service".
				ip.Spec.EndpointPickerRef.Kind = ""
				ip.Spec.EndpointPickerRef.Port = nil
			},
			wantErrors: []string{"port is required when kind is 'Service' or unspecified (defaults to 'Service')"},
		},
		{
			desc: "fails validation when kind is explicitly 'Service' and port is missing",
			mutate: func(ip *v1.InferencePool) {
				ip.Spec.EndpointPickerRef.Kind = "Service"
				ip.Spec.EndpointPickerRef.Port = nil
			},
			wantErrors: []string{"port is required when kind is 'Service' or unspecified (defaults to 'Service')"},
		},
		{
			desc: "passes validation with multiple unique port numbers",
			mutate: func(ip *v1.InferencePool) {
				ip.Spec.TargetPorts = []v1.Port{{Number: 8000}, {Number: 80}, {Number: 8081}, {Number: 443}}
			},
			wantErrors: nil,
		},
		{
			desc: "fails validation with port numbers containing duplicates",
			mutate: func(ip *v1.InferencePool) {
				ip.Spec.TargetPorts = []v1.Port{{Number: 8000}, {Number: 80}, {Number: 8000}, {Number: 443}}
			},
			wantErrors: []string{"port number must be unique"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ip := baseInferencePool.DeepCopy()
			// Use a unique name for each test case to avoid conflicts.
			ip.Name = fmt.Sprintf("test-pool-%v", time.Now().UnixNano())

			if tc.mutate != nil {
				tc.mutate(ip)
			}
			err := k8sClient.Create(ctx, ip)

			// This is a boolean XOR. It's true if one is true, but not both.
			// It ensures that an error is returned if and only if we expect one.
			if (len(tc.wantErrors) != 0) != (err != nil) {
				t.Fatalf("Unexpected response while creating InferencePool; got err=\n%v\n; want error=%v", err, tc.wantErrors != nil)
			}

			// If we got an error, check that it contains the expected substrings.
			var missingErrorStrings []string
			for _, wantError := range tc.wantErrors {
				if !celErrorStringMatches(err.Error(), wantError) {
					missingErrorStrings = append(missingErrorStrings, wantError)
				}
			}
			if len(missingErrorStrings) != 0 {
				t.Errorf("Unexpected response while creating InferencePool; got err=\n%v\n; missing strings within error=%q", err, missingErrorStrings)
			}
		})
	}
}
