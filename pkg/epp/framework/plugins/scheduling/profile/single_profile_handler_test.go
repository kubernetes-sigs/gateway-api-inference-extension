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

package profile

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

type fakeSchedulerProfile struct{}

func (f *fakeSchedulerProfile) Run(_ context.Context, _ *framework.LLMRequest, _ *framework.CycleState, _ []framework.Endpoint) (*framework.ProfileRunResult, error) {
	return &framework.ProfileRunResult{}, nil
}

func TestNewSingleProfileHandler(t *testing.T) {
	handler := NewSingleProfileHandler()

	wantTypedName := fwkplugin.TypedName{
		Type: SingleProfileHandlerType,
		Name: SingleProfileHandlerType,
	}
	if diff := cmp.Diff(wantTypedName, handler.TypedName()); diff != "" {
		t.Errorf("Unexpected TypedName (-want +got): %s", diff)
	}
}

func TestSingleProfileHandlerFactory(t *testing.T) {
	plugin, err := SingleProfileHandlerFactory("custom-name", nil, nil)
	if err != nil {
		t.Fatalf("SingleProfileHandlerFactory() returned unexpected error: %v", err)
	}

	handler, ok := plugin.(*SingleProfileHandler)
	if !ok {
		t.Fatalf("Expected *SingleProfileHandler, got %T", plugin)
	}

	wantTypedName := fwkplugin.TypedName{
		Type: SingleProfileHandlerType,
		Name: "custom-name",
	}
	if diff := cmp.Diff(wantTypedName, handler.TypedName()); diff != "" {
		t.Errorf("Unexpected TypedName (-want +got): %s", diff)
	}
}

func TestWithName(t *testing.T) {
	handler := NewSingleProfileHandler().WithName("renamed")

	if handler.TypedName().Name != "renamed" {
		t.Errorf("Expected Name to be %q, got %q", "renamed", handler.TypedName().Name)
	}
	if handler.TypedName().Type != SingleProfileHandlerType {
		t.Errorf("Expected Type to remain %q, got %q", SingleProfileHandlerType, handler.TypedName().Type)
	}
}

func TestPick(t *testing.T) {
	fakeProfile := &fakeSchedulerProfile{}

	tests := []struct {
		name           string
		profiles       map[string]framework.SchedulerProfile
		profileResults map[string]*framework.ProfileRunResult
		wantCount      int
	}{
		{
			name:           "no profiles executed yet, returns all",
			profiles:       map[string]framework.SchedulerProfile{"default": fakeProfile},
			profileResults: map[string]*framework.ProfileRunResult{},
			wantCount:      1,
		},
		{
			name:     "all profiles already executed, returns empty",
			profiles: map[string]framework.SchedulerProfile{"default": fakeProfile},
			profileResults: map[string]*framework.ProfileRunResult{
				"default": {TargetEndpoints: nil},
			},
			wantCount: 0,
		},
		{
			name:           "no profiles configured, returns empty",
			profiles:       map[string]framework.SchedulerProfile{},
			profileResults: map[string]*framework.ProfileRunResult{},
			wantCount:      0,
		},
	}

	handler := NewSingleProfileHandler()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := handler.Pick(context.Background(), framework.NewCycleState(), nil, tt.profiles, tt.profileResults)
			if len(got) != tt.wantCount {
				t.Errorf("Pick() returned %d profiles, want %d", len(got), tt.wantCount)
			}
		})
	}
}

func TestProcessResults(t *testing.T) {
	successResult := &framework.ProfileRunResult{
		TargetEndpoints: nil,
	}

	tests := []struct {
		name           string
		profileResults map[string]*framework.ProfileRunResult
		wantResult     *framework.SchedulingResult
		wantErr        bool
	}{
		{
			name: "single successful profile",
			profileResults: map[string]*framework.ProfileRunResult{
				"default": successResult,
			},
			wantResult: &framework.SchedulingResult{
				ProfileResults: map[string]*framework.ProfileRunResult{
					"default": successResult,
				},
				PrimaryProfileName: "default",
			},
		},
		{
			name:           "no profiles returns error",
			profileResults: map[string]*framework.ProfileRunResult{},
			wantErr:        true,
		},
		{
			name: "multiple profiles returns error",
			profileResults: map[string]*framework.ProfileRunResult{
				"profile-a": successResult,
				"profile-b": successResult,
			},
			wantErr: true,
		},
		{
			name: "nil result (profile execution failure) returns error",
			profileResults: map[string]*framework.ProfileRunResult{
				"default": nil,
			},
			wantErr: true,
		},
	}

	handler := NewSingleProfileHandler()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := handler.ProcessResults(context.Background(), framework.NewCycleState(), nil, tt.profileResults)

			if tt.wantErr {
				if err == nil {
					t.Fatalf("ProcessResults() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("ProcessResults() unexpected error: %v", err)
			}

			if diff := cmp.Diff(tt.wantResult, got); diff != "" {
				t.Errorf("Unexpected SchedulingResult (-want +got): %s", diff)
			}
		})
	}
}
