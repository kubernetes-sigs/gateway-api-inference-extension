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

package pod

import (
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

const PortDiscoveryAnnotationName = "inference.networking.x-k8s.io/port-discovery"

func IsPodReady(pod *corev1.Pod) bool {
	if !pod.DeletionTimestamp.IsZero() {
		return false
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			if condition.Status == corev1.ConditionTrue {
				return true
			}
			break
		}
	}
	return false
}

func ExtractInferencePortsDeclaration(pod *corev1.Pod) (sets.Set[int], bool) {
	inferencePorts := sets.New[int]()
	annotations := pod.GetAnnotations()
	if portDiscoveryAnnotation, ok := annotations[PortDiscoveryAnnotationName]; ok {
		portStrs := strings.Split(portDiscoveryAnnotation, ",")
		for _, portStr := range portStrs {
			var portNum int
			_, err := fmt.Sscanf(strings.TrimSpace(portStr), "%d", &portNum)
			if err == nil {
				inferencePorts.Insert(portNum)
			}
		}
		return inferencePorts, true
	} else {
		return nil, false
	}
}
