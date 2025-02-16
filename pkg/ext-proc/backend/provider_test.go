package backend

import (
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"k8s.io/apimachinery/pkg/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
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
	logger := logutil.NewTestLogger()

	tests := []struct {
		name      string
		pmc       PodMetricsClient
		datastore Datastore
		want      []*PodMetrics
	}{
		{
			name: "Fetch metrics error",
			pmc: &FakePodMetricsClient{
				// Err: map[string]error{
				// 	pod2.Name: errors.New("injected error"),
				// },
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
				// // Failed to fetch pod2 metrics so it remains the default values.
				// {
				// 	Name: "pod2",
				// 	Metrics: Metrics{
				// 		WaitingQueueSize:    0,
				// 		KVCacheUsagePercent: 0,
				// 		MaxActiveModels:     0,
				// 		ActiveModels:        map[string]int{},
				// 	},
				// },
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p := NewProvider(test.pmc, test.datastore)
			// if err := p.refreshMetricsOnce(logger); err != nil {
			// 	t.Fatalf("Unexpected error: %v", err)
			// }
			_ = p.refreshMetricsOnce(logger)
			metrics := test.datastore.PodGetAll()
			lessFunc := func(a, b *PodMetrics) bool {
				return a.String() < b.String()
			}
			if diff := cmp.Diff(test.want, metrics, cmpopts.SortSlices(lessFunc)); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

func populateMap(pods ...*PodMetrics) *sync.Map {
	newMap := &sync.Map{}
	for _, pod := range pods {
		newMap.Store(pod.NamespacedName, pod)
	}
	return newMap
}
