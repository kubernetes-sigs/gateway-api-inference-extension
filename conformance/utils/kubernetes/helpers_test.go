/*
Copyright 2026 The Kubernetes Authors.

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

package kubernetes

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// TestMakeServiceUnavailableRestoresSelectorOnTimeout covers the case where the selector is
// changed but the EndpointSlices never drain. The returned cleanup must still restore the
// original selector, otherwise the Service stays broken for the rest of the suite.
func TestMakeServiceUnavailableRestoresSelectorOnTimeout(t *testing.T) {
	svcNN := types.NamespacedName{Namespace: "infra", Name: "epp"}
	originalSelector := map[string]string{"app": "epp"}
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: svcNN.Namespace, Name: svcNN.Name},
		Spec:       corev1.ServiceSpec{Selector: originalSelector},
	}
	// Nothing reconciles EndpointSlices in the fake client, so this endpoint never goes away.
	slice := &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: svcNN.Namespace,
			Name:      "epp-abcde",
			Labels:    map[string]string{discoveryv1.LabelServiceName: svcNN.Name},
		},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints:   []discoveryv1.Endpoint{{Addresses: []string{"10.0.0.1"}}},
	}
	c := fake.NewClientBuilder().WithScheme(clientgoscheme.Scheme).WithObjects(svc, slice).Build()

	restore, err := MakeServiceUnavailable(t, c, svcNN, 1500*time.Millisecond)
	require.Error(t, err)
	require.NotNil(t, restore, "cleanup must be returned once the selector has been changed")

	got := &corev1.Service{}
	require.NoError(t, c.Get(context.Background(), svcNN, got))
	require.NotEqual(t, originalSelector, got.Spec.Selector)

	restore()
	require.NoError(t, c.Get(context.Background(), svcNN, got))
	require.Equal(t, originalSelector, got.Spec.Selector)
}
