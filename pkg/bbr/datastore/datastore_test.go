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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const baseModel = "base-model-1"
const cmName = "cm-1"
const cmNS = "default"

func makeConfigMap(name, baseModel, adaptersYAML string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cmNS,
		},
		Data: map[string]string{
			"baseModel": baseModel,
			"adapters":  adaptersYAML,
		},
	}
}

func TestConfigMapUpdateOrAddIfNotExist(t *testing.T) {
	ds := NewDatastore()
	cm := makeConfigMap(cmName, "  "+baseModel+"  ", "-  a1  \n- \n- a2\n")

	err := ds.ConfigMapUpdateOrAddIfNotExist(cm)
	if err != nil {
		t.Fatalf("ConfigMapUpdateOrAddIfNotExist() error = %v", err)
	}

	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"adapter a1", "a1", baseModel},
		{"adapter a2", "a2", baseModel},
		{"baseModel", baseModel, baseModel},
		{"unknown", "unknown", ""},
	}

	for _, tt := range tests {
		if got := ds.GetBaseModel(tt.input); got != tt.want {
			t.Errorf("%s: GetBaseModel(%q) = %q, want %q", tt.name, tt.input, got, tt.want)
		}
	}
}

func TestConfigMapUpdateOrAddIfNotExist_UpdateAdapters(t *testing.T) {
	ds := NewDatastore()
	cm := makeConfigMap(cmName, baseModel, "- a1\n- a2\n")
	if err := ds.ConfigMapUpdateOrAddIfNotExist(cm); err != nil {
		t.Fatalf("ConfigMapUpdateOrAddIfNotExist() error = %v", err)
	}

	cmUpdated := makeConfigMap(cmName, baseModel, "- a2\n- a3\n")
	if err := ds.ConfigMapUpdateOrAddIfNotExist(cmUpdated); err != nil {
		t.Fatalf("ConfigMapUpdateOrAddIfNotExist() update error = %v", err)
	}

	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"adapter a1 removed", "a1", ""},
		{"adapter a2 retained", "a2", baseModel},
		{"adapter a3 added", "a3", baseModel},
	}

	for _, tt := range tests {
		if got := ds.GetBaseModel(tt.input); got != tt.want {
			t.Errorf("%s: GetBaseModel(%q) = %q, want %q", tt.name, tt.input, got, tt.want)
		}
	}
}

func TestConfigMapUpdateOrAddIfNotExist_InvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		cm   *corev1.ConfigMap
	}{
		{
			name: "missing baseModel",
			cm: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: cmName, Namespace: cmNS},
				Data:       map[string]string{"adapters": "- a1\n"},
			},
		},
		{
			name: "empty baseModel",
			cm: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: cmName, Namespace: cmNS},
				Data:       map[string]string{"baseModel": "   ", "adapters": "- a1\n"},
			},
		},
		{
			name: "bad adapters yaml",
			cm:   makeConfigMap(cmName, baseModel, "["),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds := NewDatastore()
			if err := ds.ConfigMapUpdateOrAddIfNotExist(tt.cm); err == nil {
				t.Fatalf("expected error, got nil")
			}

			if got := ds.GetBaseModel("a1"); got != "" {
				t.Errorf("GetBaseModel(a1) = %q, want \"\"", got)
			}
			if got := ds.GetBaseModel(baseModel); got != "" {
				t.Errorf("GetBaseModel(baseModel) = %q, want \"\"", got)
			}
		})
	}
}

func TestConfigMapDelete(t *testing.T) {
	ds := NewDatastore()
	cm := makeConfigMap(cmName, cmNS, baseModel, "- a1\n- a2\n")
	if err := ds.ConfigMapUpdateOrAddIfNotExist(cm); err != nil {
		t.Fatalf("ConfigMapUpdateOrAddIfNotExist() error = %v", err)
	}

	ds.ConfigMapDelete(cm)

	tests := []struct {
		name  string
		input string
	}{
		{"adapter a1 removed", "a1"},
		{"adapter a2 removed", "a2"},
		{"baseModel removed", baseModel},
	}
	for _, tt := range tests {
		if got := ds.GetBaseModel(tt.input); got != "" {
			t.Errorf("%s: GetBaseModel(%q) = %q, want \"\"", tt.name, tt.input, got)
		}
	}
}

func TestConfigMapDelete_NoExistingConfigMap(t *testing.T) {
	ds := NewDatastore()
	cm := makeConfigMap(cmName, cmNS, baseModel, "- a1\n")

	ds.ConfigMapDelete(cm)

	if got := ds.GetBaseModel("a1"); got != "" {
		t.Errorf("GetBaseModel(a1) = %q, want \"\"", got)
	}
}

func TestConfigMapDelete_MissingBaseModel(t *testing.T) {
	ds := NewDatastore()
	cm := makeConfigMap(cmName, cmNS, baseModel, "- a1\n")
	if err := ds.ConfigMapUpdateOrAddIfNotExist(cm); err != nil {
		t.Fatalf("ConfigMapUpdateOrAddIfNotExist() error = %v", err)
	}

	cmMissingBase := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: cmName, Namespace: cmNS},
		Data:       map[string]string{"adapters": "- a1\n"},
	}
	ds.ConfigMapDelete(cmMissingBase)

	if got := ds.GetBaseModel("a1"); got != "" {
		t.Errorf("GetBaseModel(a1) = %q, want \"\"", got)
	}
}

func TestConfigMapDelete_BaseModelCount(t *testing.T) {
	ds := NewDatastore()
	cm1 := makeConfigMap(cmName, cmNS, baseModel, "- a1\n")
	cm2 := makeConfigMap("cm-2", cmNS, baseModel, "- a2\n")

	if err := ds.ConfigMapUpdateOrAddIfNotExist(cm1); err != nil {
		t.Fatalf("ConfigMapUpdateOrAddIfNotExist() error = %v", err)
	}
	if err := ds.ConfigMapUpdateOrAddIfNotExist(cm2); err != nil {
		t.Fatalf("ConfigMapUpdateOrAddIfNotExist() error = %v", err)
	}

	ds.ConfigMapDelete(cm1)

	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"adapter a1 removed", "a1", ""},
		{"adapter a2 retained", "a2", baseModel},
		{"baseModel retained", baseModel, baseModel},
	}
	for _, tt := range tests {
		if got := ds.GetBaseModel(tt.input); got != tt.want {
			t.Errorf("%s: GetBaseModel(%q) = %q, want %q", tt.name, tt.input, got, tt.want)
		}
	}
}
