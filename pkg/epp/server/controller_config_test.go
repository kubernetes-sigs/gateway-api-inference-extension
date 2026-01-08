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

package server

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/discovery/fake"
	k8stesting "k8s.io/client-go/testing"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
)

func TestNewControllerConfig(t *testing.T) {
	c := NewControllerConfig(true)
	if !c.startCrdReconcilers {
		t.Error("expected startCrdReconcilers to be true")
	}

	c = NewControllerConfig(false)
	if c.startCrdReconcilers {
		t.Error("expected startCrdReconcilers to be false")
	}
}

func TestPopulateWithDiscovery(t *testing.T) {
	tests := []struct {
		name                      string
		apiResources              []metav1.APIResource
		wantInferenceObjective    bool
		wantInferenceModelRewrite bool
	}{
		{
			name: "Both resources exist",
			apiResources: []metav1.APIResource{
				{Kind: "InferenceObjective"},
				{Kind: "InferenceModelRewrite"},
			},
			wantInferenceObjective:    true,
			wantInferenceModelRewrite: true,
		},
		{
			name:                      "Resources do not exist",
			apiResources:              []metav1.APIResource{},
			wantInferenceObjective:    false,
			wantInferenceModelRewrite: false,
		},
		{
			name: "Only InferenceObjective exists",
			apiResources: []metav1.APIResource{
				{Kind: "InferenceObjective"},
			},
			wantInferenceObjective:    true,
			wantInferenceModelRewrite: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup fake discovery for this specific test case
			fakeDiscovery := &fake.FakeDiscovery{
				Fake: &k8stesting.Fake{},
			}
			fakeDiscovery.Resources = []*metav1.APIResourceList{
				{
					GroupVersion: v1alpha2.GroupVersion.String(),
					APIResources: tt.apiResources,
				},
			}

			cc := &ControllerConfig{}
			cc.populateWithDiscovery(fakeDiscovery)

			if cc.hasInferenceObjective != tt.wantInferenceObjective {
				t.Errorf("populateWithDiscovery() hasInferenceObjective = %v, want %v", cc.hasInferenceObjective, tt.wantInferenceObjective)
			}
			if cc.hasInferenceModelRewrites != tt.wantInferenceModelRewrite {
				t.Errorf("populateWithDiscovery() hasInferenceModelRewrites = %v, want %v", cc.hasInferenceModelRewrites, tt.wantInferenceModelRewrite)
			}
		})
	}
}

func TestPopulateControllerConfig_Disable(t *testing.T) {
	c := NewControllerConfig(false)
	err := c.PopulateControllerConfig(nil)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
}
