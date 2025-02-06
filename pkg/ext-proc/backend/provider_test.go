package backend

import (
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

var (
	pod1 = &PodMetrics{
		Pod: Pod{Name: "pod1"},
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
		Pod: Pod{Name: "pod2"},
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
		datastore *K8sDatastore
		initErr   bool
		want      []*PodMetrics
	}{
		{
			name: "Init success",
			datastore: &K8sDatastore{
				pods: populateMap(pod1.Pod, pod2.Pod),
			},
			pmc: &FakePodMetricsClient{
				Res: map[Pod]*PodMetrics{
					pod1.Pod: pod1,
					pod2.Pod: pod2,
				},
			},
			want: []*PodMetrics{pod1, pod2},
		},
		{
			name: "Fetch metrics error",
			pmc: &FakePodMetricsClient{
				Err: map[Pod]error{
					pod2.Pod: errors.New("injected error"),
				},
				Res: map[Pod]*PodMetrics{
					pod1.Pod: pod1,
				},
			},
			datastore: &K8sDatastore{
				pods: populateMap(pod1.Pod, pod2.Pod),
			},
			want: []*PodMetrics{
				pod1,
				// Failed to fetch pod2 metrics so it remains the default values.
				{
					Pod: Pod{Name: "pod2"},
					Metrics: Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0,
						MaxActiveModels:     0,
						ActiveModels:        map[string]int{},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p := NewProvider(test.pmc, test.datastore)
			err := p.Init(time.Millisecond, time.Millisecond)
			if test.initErr != (err != nil) {
				t.Fatalf("Unexpected error, got: %v, want: %v", err, test.initErr)
			}
			metrics := p.AllPodMetrics()
			lessFunc := func(a, b *PodMetrics) bool {
				return a.String() < b.String()
			}
			if diff := cmp.Diff(test.want, metrics, cmpopts.SortSlices(lessFunc)); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

func populateMap(pods ...Pod) *sync.Map {
	newMap := &sync.Map{}
	for _, pod := range pods {
		newMap.Store(pod, true)
	}
	return newMap
}
