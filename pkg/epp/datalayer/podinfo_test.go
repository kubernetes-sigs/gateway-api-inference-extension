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

package datalayer

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

const (
	name      = "test-pod"
	namespace = "default"
	podip     = "192.168.1.123"
)

var (
	labels = map[string]string{
		"app":  "inference-server",
		"env":  "prod",
		"team": "ml",
	}
	pod = &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    labels,
		},
		Status: corev1.PodStatus{
			PodIP: podip,
		},
	}
	expected = &PodInfo{
		NamespacedName: types.NamespacedName{Name: name, Namespace: namespace},
		Address:        podip,
		Labels:         labels,
	}
)

func TestToPodInfo(t *testing.T) {
	podinfo := ToPodInfo(pod)
	if podinfo.RunningRequests == nil {
		t.Fatal("Expected RunningRequests to be initialized")
	}
	podinfo.RunningRequests = nil // Reset to nil for comparison, this is necessary because the podinfo is created with a new map each time
	if diff := cmp.Diff(expected, podinfo); diff != "" {
		t.Errorf("Unexpected output (-want +got): %v", diff)
	}
}

func TestPodInfoClone(t *testing.T) {
	clone := expected.Clone()
	assert.NotSame(t, expected, clone)
	if diff := cmp.Diff(expected, clone); diff != "" {
		t.Errorf("Unexpected output (-want +got): %v", diff)
	}

	clone.Labels["env"] = "staging"
	assert.Equal(t, "prod", expected.Labels["env"], "mutating clone should not affect original")
}

func TestPodInfoString(t *testing.T) {
	podinfo := ToPodInfo(pod)

	s := podinfo.String()
	assert.Contains(t, s, name)
	assert.Contains(t, s, namespace)
	assert.Contains(t, s, podip)
}
