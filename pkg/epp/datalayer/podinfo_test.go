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

package datalayer_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
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
)

func TestToPodInfo(t *testing.T) {
	podinfo := datalayer.ToPodInfo(pod)

	assert.Equal(t, name, podinfo.NamespacedName.Name)
	assert.Equal(t, namespace, podinfo.NamespacedName.Namespace)
	assert.Equal(t, podip, podinfo.Address)
	assert.Equal(t, labels, podinfo.Labels)
}

func TestPodInfoClone(t *testing.T) {
	podinfo := &datalayer.PodInfo{
		NamespacedName: types.NamespacedName{Name: name, Namespace: namespace},
		Address:        podip,
		Labels:         labels,
	}

	clone := podinfo.Clone()

	assert.Equal(t, podinfo, clone)
	assert.NotSame(t, podinfo, clone)
	assert.Equal(t, podinfo.Labels, clone.Labels)
	clone.Labels["env"] = "staging"
	assert.Equal(t, "prod", podinfo.Labels["env"], "mutating clone should not affect original")
}

func TestPodInfoString(t *testing.T) {
	podinfo := datalayer.ToPodInfo(pod)

	s := podinfo.String()
	assert.Contains(t, s, name)
	assert.Contains(t, s, namespace)
	assert.Contains(t, s, podip)
}
