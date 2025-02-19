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

package backend

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/datastore"
)

var (
	pod1 = &datastore.PodMetrics{
		Pod: datastore.Pod{
			NamespacedName: types.NamespacedName{
				Name: "pod1",
			},
		},
		Metrics: datastore.Metrics{
			WaitingQueueSize:    0,
			KVCacheUsagePercent: 0.2,
			MaxActiveModels:     2,
			ActiveModels: map[string]int{
				"foo": 1,
				"bar": 1,
			},
		},
	}
	pod2 = &datastore.PodMetrics{
		Pod: datastore.Pod{
			NamespacedName: types.NamespacedName{
				Name: "pod2",
			},
		},
		Metrics: datastore.Metrics{
			WaitingQueueSize:    1,
			KVCacheUsagePercent: 0.2,
			MaxActiveModels:     2,
			ActiveModels: map[string]int{
				"foo1": 1,
				"bar1": 1,
			},
		},
	}
)

func TestProvider(t *testing.T) {
	tests := []struct {
		name      string
		pmc       PodMetricsClient
		datastore datastore.Datastore
		want      []*datastore.PodMetrics
	}{
		{
			name: "Probing metrics success",
			pmc: &FakePodMetricsClient{
				Res: map[types.NamespacedName]*datastore.PodMetrics{
					pod1.NamespacedName: pod1,
					pod2.NamespacedName: pod2,
				},
			},
			datastore: datastore.NewFakeDatastore(populateMap(pod1, pod2), nil, nil),
			want: []*datastore.PodMetrics{
				pod1,
				pod2,
			},
		},
		{
			name: "Only pods in the datastore are probed",
			pmc: &FakePodMetricsClient{
				Res: map[types.NamespacedName]*datastore.PodMetrics{
					pod1.NamespacedName: pod1,
					pod2.NamespacedName: pod2,
				},
			},
			datastore: datastore.NewFakeDatastore(populateMap(pod1), nil, nil),
			want: []*datastore.PodMetrics{
				pod1,
			},
		},
		{
			name: "Probing metrics error",
			pmc: &FakePodMetricsClient{
				Err: map[types.NamespacedName]error{
					pod2.NamespacedName: errors.New("injected error"),
				},
				Res: map[types.NamespacedName]*datastore.PodMetrics{
					pod1.NamespacedName: pod1,
				},
			},
			datastore: datastore.NewFakeDatastore(populateMap(pod1, pod2), nil, nil),

			want: []*datastore.PodMetrics{
				pod1,
				// Failed to fetch pod2 metrics so it remains the default values.
				{
					Pod: datastore.Pod{NamespacedName: pod2.NamespacedName},
					Metrics: datastore.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0,
						MaxActiveModels:     0,
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p := NewProvider(test.pmc, test.datastore)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			_ = p.Init(ctx, time.Millisecond, time.Millisecond)
			assert.EventuallyWithT(t, func(t *assert.CollectT) {
				metrics := test.datastore.PodGetAll()
				diff := cmp.Diff(test.want, metrics, cmpopts.SortSlices(func(a, b *datastore.PodMetrics) bool {
					return a.String() < b.String()
				}))
				assert.Equal(t, "", diff, "Unexpected diff (+got/-want)")
			}, 5*time.Second, time.Millisecond)
		})
	}
}

func populateMap(pods ...*datastore.PodMetrics) *sync.Map {
	newMap := &sync.Map{}
	for _, pod := range pods {
		newMap.Store(pod.NamespacedName, &datastore.PodMetrics{Pod: datastore.Pod{NamespacedName: pod.NamespacedName, Address: pod.Address}})
	}
	return newMap
}
