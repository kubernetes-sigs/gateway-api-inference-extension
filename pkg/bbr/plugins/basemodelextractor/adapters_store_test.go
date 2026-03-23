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

package basemodelextractor

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
	as := NewAdaptersStore()
	cm := makeConfigMap(cmName, "  "+baseModel+"  ", "-  a1  \n- \n- a2\n")

	err := as.configMapUpdateOrAddIfNotExist(cm)
	if err != nil {
		t.Fatalf("configMapUpdateOrAddIfNotExist() error = %v", err)
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
		if got := as.getBaseModel(tt.input); got != tt.want {
			t.Errorf("%s: getBaseModel(%q) = %q, want %q", tt.name, tt.input, got, tt.want)
		}
	}
}

func TestConfigMapUpdateOrAddIfNotExist_UpdateAdapters(t *testing.T) {
	as := NewAdaptersStore()
	cm := makeConfigMap(cmName, baseModel, "- a1\n- a2\n")
	if err := as.configMapUpdateOrAddIfNotExist(cm); err != nil {
		t.Fatalf("configMapUpdateOrAddIfNotExist() error = %v", err)
	}

	cmUpdated := makeConfigMap(cmName, baseModel, "- a2\n- a3\n")
	if err := as.configMapUpdateOrAddIfNotExist(cmUpdated); err != nil {
		t.Fatalf("configMapUpdateOrAddIfNotExist() update error = %v", err)
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
		if got := as.getBaseModel(tt.input); got != tt.want {
			t.Errorf("%s: getBaseModel(%q) = %q, want %q", tt.name, tt.input, got, tt.want)
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
			as := NewAdaptersStore()
			if err := as.configMapUpdateOrAddIfNotExist(tt.cm); err == nil {
				t.Fatalf("expected error, got nil")
			}

			if got := as.getBaseModel("a1"); got != "" {
				t.Errorf("getBaseModel(a1) = %q, want \"\"", got)
			}
			if got := as.getBaseModel(baseModel); got != "" {
				t.Errorf("getBaseModel(baseModel) = %q, want \"\"", got)
			}
		})
	}
}

func TestConfigMapDelete(t *testing.T) {
	as := NewAdaptersStore()
	cm := makeConfigMap(cmName, baseModel, "- a1\n- a2\n")
	if err := as.configMapUpdateOrAddIfNotExist(cm); err != nil {
		t.Fatalf("configMapUpdateOrAddIfNotExist() error = %v", err)
	}

	as.configMapDelete(cm)

	tests := []struct {
		name  string
		input string
	}{
		{"adapter a1 removed", "a1"},
		{"adapter a2 removed", "a2"},
		{"baseModel removed", baseModel},
	}
	for _, tt := range tests {
		if got := as.getBaseModel(tt.input); got != "" {
			t.Errorf("%s: getBaseModel(%q) = %q, want \"\"", tt.name, tt.input, got)
		}
	}
}

func TestConfigMapDelete_NoExistingConfigMap(t *testing.T) {
	as := NewAdaptersStore()
	cm := makeConfigMap(cmName, baseModel, "- a1\n")

	as.configMapDelete(cm)

	if got := as.getBaseModel("a1"); got != "" {
		t.Errorf("getBaseModel(a1) = %q, want \"\"", got)
	}
}

func TestConfigMapDelete_MissingBaseModel(t *testing.T) {
	as := NewAdaptersStore()
	cm := makeConfigMap(cmName, baseModel, "- a1\n")
	if err := as.configMapUpdateOrAddIfNotExist(cm); err != nil {
		t.Fatalf("configMapUpdateOrAddIfNotExist() error = %v", err)
	}

	cmMissingBase := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: cmName, Namespace: cmNS},
		Data:       map[string]string{"adapters": "- a1\n"},
	}
	as.configMapDelete(cmMissingBase)

	if got := as.getBaseModel("a1"); got != "" {
		t.Errorf("getBaseModel(a1) = %q, want \"\"", got)
	}
}

func TestGetBaseModel_ConcurrentRace(t *testing.T) {
	as := NewAdaptersStore()
	cm := makeConfigMap(cmName, baseModel, "- a1\n")

	go func() {
		for i := 0; i < 10000; i++ {
			_ = as.configMapUpdateOrAddIfNotExist(cm)
		}
	}()

	for i := 0; i < 10000; i++ {
		as.getBaseModel("a1")
	}
}

func TestConfigMapDelete_BaseModelCount(t *testing.T) {
	as := NewAdaptersStore()
	cm1 := makeConfigMap(cmName, baseModel, "- a1\n")
	cm2 := makeConfigMap("cm-2", baseModel, "- a2\n")

	if err := as.configMapUpdateOrAddIfNotExist(cm1); err != nil {
		t.Fatalf("configMapUpdateOrAddIfNotExist() error = %v", err)
	}
	if err := as.configMapUpdateOrAddIfNotExist(cm2); err != nil {
		t.Fatalf("configMapUpdateOrAddIfNotExist() error = %v", err)
	}

	as.configMapDelete(cm1)

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
		if got := as.getBaseModel(tt.input); got != tt.want {
			t.Errorf("%s: getBaseModel(%q) = %q, want %q", tt.name, tt.input, got, tt.want)
		}
	}
}
