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

package prefix

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestPrefixPluginScore(t *testing.T) {
	p, _ := New(context.Background())
	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), nil)
	endpoint1.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(5, 10, 1))

	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, fwkdl.NewMetrics(), nil)
	endpoint2.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(2, 10, 1))

	endpoints := []fwksched.Endpoint{endpoint1, endpoint2}
	scores := p.Score(context.Background(), fwksched.NewCycleState(), nil, endpoints)

	assert.Equal(t, 0.5, scores[endpoint1])
	assert.Equal(t, 0.2, scores[endpoint2])
	}

