package backend

import (
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	testingutil "inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/testing"
	corev1 "k8s.io/api/core/v1"
)

var (
	pod1 = &PodMetrics{
		Pod: Pod{Name: "pod1", Address: "address1:9009"},
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
		Pod: Pod{Name: "pod2", Address: "address2:9009"},
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
	allPodsLister := &testingutil.FakePodLister{
		PodsList: []*corev1.Pod{
			testingutil.MakePod(pod1.Pod.Name).SetReady().SetPodIP("address1").Obj(),
			testingutil.MakePod(pod2.Pod.Name).SetReady().SetPodIP("address2").Obj(),
		},
	}
	allPodsMetricsClient := &FakePodMetricsClient{
		Res: map[string]*PodMetrics{
			pod1.Pod.Name: pod1,
			pod2.Pod.Name: pod2,
		},
	}

	tests := []struct {
		name           string
		initPodMetrics []*PodMetrics
		lister         *testingutil.FakePodLister
		pmc            PodMetricsClient
		step           func(*Provider)
		want           []*PodMetrics
	}{
		{
			name:           "Init without refreshing pods",
			initPodMetrics: []*PodMetrics{pod1, pod2},
			lister:         allPodsLister,
			pmc:            allPodsMetricsClient,
			step: func(p *Provider) {
				_ = p.refreshMetricsOnce()
			},
			want: []*PodMetrics{pod1, pod2},
		},
		{
			name:   "Fetching all success",
			lister: allPodsLister,
			pmc:    allPodsMetricsClient,
			step: func(p *Provider) {
				p.refreshPodsOnce()
				_ = p.refreshMetricsOnce()
			},
			want: []*PodMetrics{pod1, pod2},
		},
		{
			name:   "Fetch metrics error",
			lister: allPodsLister,
			pmc: &FakePodMetricsClient{
				Err: map[string]error{
					pod2.Pod.Name: errors.New("injected error"),
				},
				Res: map[string]*PodMetrics{
					pod1.Pod.Name: pod1,
				},
			},
			step: func(p *Provider) {
				p.refreshPodsOnce()
				_ = p.refreshMetricsOnce()
			},
			want: []*PodMetrics{
				pod1,
				// Failed to fetch pod2 metrics so it remains the default values.
				{
					Pod: pod2.Pod,
					Metrics: Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0,
						MaxActiveModels:     0,
						ActiveModels:        map[string]int{},
					},
				},
			},
		},
		{
			name:           "A new pod added",
			initPodMetrics: []*PodMetrics{pod2},
			lister:         allPodsLister,
			pmc:            allPodsMetricsClient,
			step: func(p *Provider) {
				p.refreshPodsOnce()
				_ = p.refreshMetricsOnce()
			},
			want: []*PodMetrics{pod1, pod2},
		},
		{
			name:           "A pod removed",
			initPodMetrics: []*PodMetrics{pod1, pod2},
			lister: &testingutil.FakePodLister{
				PodsList: []*corev1.Pod{
					testingutil.MakePod(pod2.Pod.Name).SetReady().SetPodIP("address2").Obj(),
				},
			},
			pmc: allPodsMetricsClient,
			step: func(p *Provider) {
				p.refreshPodsOnce()
				_ = p.refreshMetricsOnce()
			},
			want: []*PodMetrics{pod2},
		},
		{
			name:           "A pod removed, another added",
			initPodMetrics: []*PodMetrics{pod1},
			lister: &testingutil.FakePodLister{
				PodsList: []*corev1.Pod{
					testingutil.MakePod(pod1.Pod.Name).SetReady().SetPodIP("address1").Obj(),
				},
			},
			pmc: allPodsMetricsClient,
			step: func(p *Provider) {
				p.refreshPodsOnce()
				_ = p.refreshMetricsOnce()
			},
			want: []*PodMetrics{pod1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			datastore := NewK8sDataStore(WithPodListerFactory(
				func(pool *v1alpha1.InferencePool) *PodLister {
					return &PodLister{
						Lister: test.lister,
					}
				}))
			datastore.setInferencePool(&v1alpha1.InferencePool{
				Spec: v1alpha1.InferencePoolSpec{TargetPortNumber: 9009},
			})
			p := NewProvider(test.pmc, datastore)
			for _, m := range test.initPodMetrics {
				p.UpdatePodMetrics(m.Pod, m)
			}
			test.step(p)
			metrics := p.AllPodMetrics()
			lessFunc := func(a, b *PodMetrics) bool {
				return a.String() < b.String()
			}
			if diff := cmp.Diff(test.want, metrics, cmpopts.SortSlices(lessFunc),
				cmpopts.IgnoreFields(PodMetrics{}, "revision")); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
