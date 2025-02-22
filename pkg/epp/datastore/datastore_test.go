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

	ds := NewDatastore()
	dsImpl := ds.(*datastore)

	// Step 1: add model1 with tweet-summary as modelName.
	ds.ModelSetIfOlder(model1ts)
	if diff := diffModelMaps(dsImpl, []*v1alpha2.InferenceModel{model1ts}); diff != "" {
		t.Errorf("Unexpected models diff: %s", diff)
	}

	// Step 2: set model1 with the same modelName, but with criticality set and newer creation timestamp, should update.
	ds.ModelSetIfOlder(model1tsNewer)
	if diff := diffModelMaps(dsImpl, []*v1alpha2.InferenceModel{model1tsNewer}); diff != "" {
		t.Errorf("Unexpected models diff: %s", diff)
	}

	// Step 3: set model2 with the same modelName, but newer creation timestamp, should not update.
	ds.ModelSetIfOlder(model2tsNewer)
	if diff := diffModelMaps(dsImpl, []*v1alpha2.InferenceModel{model1tsNewer}); diff != "" {
		t.Errorf("Unexpected models diff: %s", diff)
	}

	// Step 4: set model2 with the same modelName, but older creation timestamp, should update.
	ds.ModelSetIfOlder(model2ts)
	if diff := diffModelMaps(dsImpl, []*v1alpha2.InferenceModel{model2ts}); diff != "" {
		t.Errorf("Unexpected models diff: %s", diff)
	}

	// Step 5: set model2 updated with a new modelName, should update modelName.
	ds.ModelSetIfOlder(model2chat)
	if diff := diffModelMaps(dsImpl, []*v1alpha2.InferenceModel{model2chat}); diff != "" {
		t.Errorf("Unexpected models diff: %s", diff)
	}

	// Step 6: set model1 with the tweet-summary modelName, both models should exist.
	ds.ModelSetIfOlder(model1ts)
	if diff := diffModelMaps(dsImpl, []*v1alpha2.InferenceModel{model2chat, model1ts}); diff != "" {
		t.Errorf("Unexpected models diff: %s", diff)
	}

	// Step 7: getting the models by model name, chat -> model2; tweet-summary -> model1
	gotChat, exists := ds.ModelGetByModelName(chatModel)
	if !exists {
		t.Error("Chat model should exist!")
	}
	if diff := cmp.Diff(model2chat, gotChat); diff != "" {
		t.Errorf("Unexpected chat model diff: %s", diff)
	}
	gotSummary, exists := ds.ModelGetByModelName(tsModel)
	if !exists {
		t.Error("Summary model should exist!")
	}
	if diff := cmp.Diff(model1ts, gotSummary); diff != "" {
		t.Errorf("Unexpected summary model diff: %s", diff)
	}

	// Step 6: delete model1, summary model should not exist.
	ds.ModelDelete(types.NamespacedName{Name: model1ts.Name, Namespace: model1ts.Namespace})
	_, exists = ds.ModelGetByModelName(tsModel)
	if exists {
		t.Error("Summary model should not exist!")
	}

}

func diffModelMaps(ds *datastore, want []*v1alpha2.InferenceModel) string {
	byObjName := ds.ModelGetAll()
	byModelName := []*v1alpha2.InferenceModel{}
	for _, v := range ds.modelsByModelName {
		byModelName = append(byModelName, v)
	}
	if diff := testutil.DiffModelLists(byObjName, byModelName); diff != "" {
		return "Inconsistent maps diff: " + diff
	}
	return testutil.DiffModelLists(want, byObjName)
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
	}
	var seedVal int64 = 420
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for range 10000 {
				model := RandomWeightedDraw(logger, test.model, seedVal)
				if model != test.want {
					t.Errorf("Model returned!: %v", model)
					break
				}
			}
		})
	}
}

func pointer(v int32) *int32 {
	return &v
}
