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

package runner

import (
	"context"
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
)

func TestParseConfigurationPhaseTwoAppliesPrepareDataTimeout(t *testing.T) {
	t.Parallel()

	r := NewRunner()
	r.registerInTreePlugins()

	rawConfig := &configapi.EndpointPickerConfig{
		RequestControl: &configapi.RequestControlConfig{
			PrepareDataTimeout: &metav1.Duration{Duration: 125 * time.Millisecond},
		},
	}

	ctx := context.Background()
	epFactory := datalayer.NewEndpointFactory(nil, time.Millisecond)
	ds := datastore.NewDatastore(ctx, epFactory, 0)

	if _, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds); err != nil {
		t.Fatalf("parseConfigurationPhaseTwo failed: %v", err)
	}

	got := readPrepareDataTimeout(t, r.requestControlConfig)
	want := 125 * time.Millisecond
	if got != want {
		t.Fatalf("prepareDataTimeout = %v, want %v", got, want)
	}
}

func readPrepareDataTimeout(t *testing.T, cfg any) time.Duration {
	t.Helper()

	v := reflect.ValueOf(cfg).Elem().FieldByName("prepareDataTimeout")
	if v.Kind() != reflect.Int64 {
		t.Fatalf("unexpected kind for prepareDataTimeout: %v", v.Kind())
	}
	return time.Duration(v.Int())
}
