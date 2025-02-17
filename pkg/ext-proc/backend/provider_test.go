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
)

var (
	pod1 = &PodMetrics{
		NamespacedName: types.NamespacedName{
			Name: "pod1",
		},
		Metrics: Metrics{
			WaitingQueueSize:    0,
			KVCacheUsagePercent: 0.2,
			MaxActiveModels:     2,
			ActiveModels: map[string]int{
				"foo": 1,
				"bar": 1,
			},
		},
	}
	pod2 = &PodMetrics{
		NamespacedName: types.NamespacedName{
			Name: "pod2",
		},
		Metrics: Metrics{
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
		datastore Datastore
		want      []*PodMetrics
	}{
		{
			name: "Probing metrics success",
			pmc: &FakePodMetricsClient{
				Res: map[types.NamespacedName]*PodMetrics{
					pod1.NamespacedName: pod1,
					pod2.NamespacedName: pod2,
				},
			},
			datastore: &datastore{
				pods: populateMap(pod1, pod2),
			},
			want: []*PodMetrics{
				pod1,
				pod2,
			},
		},
		{
			name: "Only pods in the datastore are probed",
			pmc: &FakePodMetricsClient{
				Res: map[types.NamespacedName]*PodMetrics{
					pod1.NamespacedName: pod1,
					pod2.NamespacedName: pod2,
				},
			},
			datastore: &datastore{
				pods: populateMap(pod1),
			},
			want: []*PodMetrics{
				pod1,
			},
		},
		{
			name: "Probing metrics error",
			pmc: &FakePodMetricsClient{
				Err: map[types.NamespacedName]error{
					pod2.NamespacedName: errors.New("injected error"),
				},
				Res: map[types.NamespacedName]*PodMetrics{
					pod1.NamespacedName: pod1,
				},
			},
			datastore: &datastore{
				pods: populateMap(pod1, pod2),
			},
			want: []*PodMetrics{
				pod1,
				// Failed to fetch pod2 metrics so it remains the default values.
				{
					NamespacedName: pod2.NamespacedName,
					Metrics: Metrics{
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
				diff := cmp.Diff(test.want, metrics, cmpopts.SortSlices(func(a, b *PodMetrics) bool {
					return a.String() < b.String()
				}))
				assert.Equal(t, "", diff, "Unexpected diff (+got/-want)")
			}, 5*time.Second, time.Millisecond)
		})
	}
}

func populateMap(pods ...*PodMetrics) *sync.Map {
	newMap := &sync.Map{}
	for _, pod := range pods {
		newMap.Store(pod.NamespacedName, &PodMetrics{NamespacedName: pod.NamespacedName})
	}
	return newMap
}
