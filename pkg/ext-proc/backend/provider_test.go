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
		Pod: Pod{Name: "pod1", Address: "127.0.0.1"},
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
		Pod: Pod{Name: "pod2", Address: "127.0.0.2"},
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
		name               string
		pmc                PodMetricsClient
		datastore          *K8sDatastore
		initErr            bool
		want               []*PodMetrics
		wantIncludingStale []*PodMetrics
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
			want:               []*PodMetrics{pod1, pod2},
			wantIncludingStale: []*PodMetrics{pod1, pod2},
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
				// Failed to fetch pod2 metrics so it remains the default values,
				// which is stale.
			},
			wantIncludingStale: []*PodMetrics{
				pod1,
				// Failed to fetch pod2 metrics so it remains the default values.
				{
					Pod: Pod{Name: "pod2", Address: "127.0.0.2"},
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
			err := p.Init(10*time.Millisecond, time.Millisecond)
			if test.initErr != (err != nil) {
				t.Fatalf("Unexpected error, got: %v, want: %v", err, test.initErr)
			}

			// do some turns of refreshing...
			time.Sleep(3 * time.Millisecond)

			metrics := p.AllPodMetrics()
			metricsCopy := make([]*PodMetrics, len(metrics))
			for i, pm := range metrics {
				if pm.UpdatedTime.IsZero() {
					t.Errorf("Expected non-zero UpdatedTime, got %v", pm.UpdatedTime)
				}
				cp := pm.Clone()
				// reset the UpdatedTime for comparison
				cp.UpdatedTime = time.Time{}
				metricsCopy[i] = cp
			}

			lessFunc := func(a, b *PodMetrics) bool {
				return a.String() < b.String()
			}
			if diff := cmp.Diff(test.want, metricsCopy, cmpopts.SortSlices(lessFunc)); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}

			// then check for AllPodMetricsIncludingStale
			if len(test.wantIncludingStale) > 0 {
				metricsIncludingStale := p.AllPodMetricsIncludingStale()
				metricsCopy := make([]*PodMetrics, len(metricsIncludingStale))
				for i, pm := range metricsIncludingStale {
					cp := pm.Clone()
					// reset the UpdatedTime for comparison
					cp.UpdatedTime = time.Time{}
					metricsCopy[i] = cp
				}

				if diff := cmp.Diff(test.wantIncludingStale, metricsCopy, cmpopts.SortSlices(lessFunc)); diff != "" {
					t.Errorf("Unexpected output (-want +got): %v", diff)
				}
			}

			// simulate pod deletion
			p.datastore.pods.Range(func(k, v any) bool {
				p.datastore.pods.Delete(k)
				return true
			})
			time.Sleep(20 * time.Millisecond)
			metrics = p.AllPodMetrics()
			// ensure no update is writing to the PodMetrics by background refreshing
			if len(metrics) != 0 {
				t.Errorf("Expected no metrics, got %v", metrics)
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
