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

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	testutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

func TestPool(t *testing.T) {
	pool1Selector := map[string]string{"app": "vllm_v1"}
	pool1 := testutil.MakeInferencePool("pool1").
		Namespace("default").
		Selector(pool1Selector).ObjRef()
	tests := []struct {
		name            string
		inferencePool   *v1alpha2.InferencePool
		labels          map[string]string
		wantSynced      bool
		wantPool        *v1alpha2.InferencePool
		wantErr         error
		wantLabelsMatch bool
	}{
		{
			name:            "Ready when InferencePool exists in data store",
			inferencePool:   pool1,
			labels:          pool1Selector,
			wantSynced:      true,
			wantPool:        pool1,
			wantLabelsMatch: true,
		},
		{
			name:            "Labels not matched",
			inferencePool:   pool1,
			labels:          map[string]string{"app": "vllm_v2"},
			wantSynced:      true,
			wantPool:        pool1,
			wantLabelsMatch: false,
		},
		{
			name:       "Not ready when InferencePool is nil in data store",
			wantErr:    errPoolNotSynced,
			wantSynced: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			datastore := NewDatastore()
			datastore.PoolSet(tt.inferencePool)
			gotPool, gotErr := datastore.PoolGet()
			if diff := cmp.Diff(tt.wantErr, gotErr, cmpopts.EquateErrors()); diff != "" {
				t.Errorf("Unexpected error diff (+got/-want): %s", diff)
			}
			if diff := cmp.Diff(tt.wantPool, gotPool); diff != "" {
				t.Errorf("Unexpected pool diff (+got/-want): %s", diff)
			}
			gotSynced := datastore.PoolHasSynced()
			if diff := cmp.Diff(tt.wantSynced, gotSynced); diff != "" {
				t.Errorf("Unexpected synced diff (+got/-want): %s", diff)
			}
			if tt.labels != nil {
				gotLabelsMatch := datastore.PoolLabelsMatch(tt.labels)
				if diff := cmp.Diff(tt.wantLabelsMatch, gotLabelsMatch); diff != "" {
					t.Errorf("Unexpected labels match diff (+got/-want): %s", diff)
				}
			}
		})
	}
}

func TestModel(t *testing.T) {
	chatModel := "chat"
	tsModel := "tweet-summary"
	model1ts := testutil.MakeInferenceModel("model1").
		CreationTimestamp(metav1.Unix(1000, 0)).
		ModelName(tsModel).ObjRef()
	// Same model name as model1ts, different object name.
	model2ts := testutil.MakeInferenceModel("model2").
		CreationTimestamp(metav1.Unix(1001, 0)).
		ModelName(tsModel).ObjRef()
	// Same model name as model1ts, newer timestamp
	model1tsNewer := testutil.MakeInferenceModel("model1").
		CreationTimestamp(metav1.Unix(1002, 0)).
		Criticality(v1alpha2.Critical).
		ModelName(tsModel).ObjRef()
	model2tsNewer := testutil.MakeInferenceModel("model2").
		CreationTimestamp(metav1.Unix(1003, 0)).
		ModelName(tsModel).ObjRef()
	// Same object name as model2ts, different model name.
	model2chat := testutil.MakeInferenceModel(model2ts.Name).
		CreationTimestamp(metav1.Unix(1005, 0)).
		ModelName(chatModel).ObjRef()

	tests := []struct {
		name           string
		existingModels []*v1alpha2.InferenceModel
		op             func(ds Datastore) bool
		wantOpResult   bool
		wantModels     []*v1alpha2.InferenceModel
	}{
		{
			name: "Add model1 with tweet-summary as modelName",
			op: func(ds Datastore) bool {
				return ds.ModelSetIfOlder(model1ts)
			},
			wantModels:   []*v1alpha2.InferenceModel{model1ts},
			wantOpResult: true,
		},
		{
			name:           "Set model1 with the same modelName, but with diff criticality and newer creation timestamp, should update.",
			existingModels: []*v1alpha2.InferenceModel{model1ts},
			op: func(ds Datastore) bool {
				return ds.ModelSetIfOlder(model1tsNewer)
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceModel{model1tsNewer},
		},
		{
			name:           "set model2 with the same modelName, but newer creation timestamp, should not update.",
			existingModels: []*v1alpha2.InferenceModel{model1tsNewer},
			op: func(ds Datastore) bool {
				return ds.ModelSetIfOlder(model2tsNewer)
			},
			wantOpResult: false,
			wantModels:   []*v1alpha2.InferenceModel{model1tsNewer},
		},
		{
			name:           "Set model2 with the same modelName, but older creation timestamp, should update",
			existingModels: []*v1alpha2.InferenceModel{model1tsNewer},
			op: func(ds Datastore) bool {
				return ds.ModelSetIfOlder(model2ts)
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceModel{model2ts},
		},
		{
			name:           "Set model1 with the tweet-summary modelName, both models should exist",
			existingModels: []*v1alpha2.InferenceModel{model2chat},
			op: func(ds Datastore) bool {
				return ds.ModelSetIfOlder(model1ts)
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceModel{model2chat, model1ts},
		},
		{
			name:           "Set model1 with the tweet-summary modelName, both models should exist",
			existingModels: []*v1alpha2.InferenceModel{model2chat, model1ts},
			op: func(ds Datastore) bool {
				return ds.ModelSetIfOlder(model1ts)
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceModel{model2chat, model1ts},
		},
		{
			name:           "Getting by model name, chat -> model2",
			existingModels: []*v1alpha2.InferenceModel{model2chat, model1ts},
			op: func(ds Datastore) bool {
				gotChat, exists := ds.ModelGet(chatModel)
				return exists && cmp.Diff(model2chat, gotChat) == ""
			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceModel{model2chat, model1ts},
		},
		{
			name:           "Delete the model",
			existingModels: []*v1alpha2.InferenceModel{model2chat, model1ts},
			op: func(ds Datastore) bool {
				_, existed := ds.ModelDelete(types.NamespacedName{Name: model1ts.Name, Namespace: model1ts.Namespace})
				_, exists := ds.ModelGet(tsModel)
				return existed && !exists

			},
			wantOpResult: true,
			wantModels:   []*v1alpha2.InferenceModel{model2chat},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ds := NewFakeDatastore(nil, test.existingModels, nil)
			gotOpResult := test.op(ds)
			if gotOpResult != test.wantOpResult {
				t.Errorf("Unexpected operation result, want: %v, got: %v", test.wantOpResult, gotOpResult)
			}

			if diff := testutil.DiffModelLists(test.wantModels, ds.ModelGetAll()); diff != "" {
				t.Errorf("Unexpected models diff: %s", diff)
			}

		})
	}
}

func TestRandomWeightedDraw(t *testing.T) {
	logger := logutil.NewTestLogger()
	tests := []struct {
		name  string
		model *v1alpha2.InferenceModel
		want  string
	}{
		{
			name: "'random' distribution",
			model: &v1alpha2.InferenceModel{
				Spec: v1alpha2.InferenceModelSpec{
					TargetModels: []v1alpha2.TargetModel{
						{
							Name:   "canary",
							Weight: pointer(50),
						},
						{
							Name:   "v1",
							Weight: pointer(50),
						},
					},
				},
			},
			want: "canary",
		},
		{
			name: "'random' distribution",
			model: &v1alpha2.InferenceModel{
				Spec: v1alpha2.InferenceModelSpec{
					TargetModels: []v1alpha2.TargetModel{
						{
							Name:   "canary",
							Weight: pointer(25),
						},
						{
							Name:   "v1.1",
							Weight: pointer(55),
						},
						{
							Name:   "v1",
							Weight: pointer(50),
						},
					},
				},
			},
			want: "v1",
		},
		{
			name: "'random' distribution",
			model: &v1alpha2.InferenceModel{
				Spec: v1alpha2.InferenceModelSpec{
					TargetModels: []v1alpha2.TargetModel{
						{
							Name:   "canary",
							Weight: pointer(20),
						},
						{
							Name:   "v1.1",
							Weight: pointer(20),
						},
						{
							Name:   "v1",
							Weight: pointer(10),
						},
					},
				},
			},
			want: "v1.1",
		},
		{
			name: "weighted distribution with weight unset",
			model: &v1alpha2.InferenceModel{
				Spec: v1alpha2.InferenceModelSpec{
					TargetModels: []v1alpha2.TargetModel{
						{
							Name: "canary",
						},
						{
							Name: "v1.1",
						},
						{
							Name: "v1",
						},
					},
				},
			},
			want: "canary",
		},
	}
	var seedVal int64 = 420
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for range 10000 {
				model := RandomWeightedDraw(logger, test.model, seedVal)
				if model != test.want {
					t.Errorf("Model returned: %v != %v", model, test.want)
					break
				}
			}
		})
	}
}

func pointer(v int32) *int32 {
	return &v
}
