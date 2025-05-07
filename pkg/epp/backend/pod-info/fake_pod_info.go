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

package podinfo

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
)

var _ PodInfo = &FakePodInfo{}

func NewFakePodInfo(namespacedName types.NamespacedName) *FakePodInfo {
	return &FakePodInfo{
		pod:  &backend.Pod{NamespacedName: namespacedName},
		data: map[string]ScrapedData{},
	}
}

// FakePodInfo is an implementation of PodInfo that doesn't run the async scrape loop.
type FakePodInfo struct {
	pod  *backend.Pod
	data map[string]ScrapedData
}

func (fpi *FakePodInfo) WithData(data map[string]ScrapedData) *FakePodInfo {
	fpi.data = data
	return fpi
}

func (fpi *FakePodInfo) GetPod() *backend.Pod {
	return fpi.pod
}
func (fpi *FakePodInfo) GetData(key string) (ScrapedData, bool) {
	data, ok := fpi.data[key]
	return data, ok
}

func (fpi *FakePodInfo) GetDataKeys() []string {
	result := []string{}
	for key := range fpi.data {
		result = append(result, key)
	}
	return result
}

func (fpi *FakePodInfo) UpdatePod(pod *corev1.Pod) {
	fpi.pod = toInternalPod(pod)
}

func (fpi *FakePodInfo) Stop() {} // noop

func (fpi *FakePodInfo) String() string {
	return fmt.Sprintf("Pod: %v; Data: %v", fpi.GetPod(), fpi.data)
}
