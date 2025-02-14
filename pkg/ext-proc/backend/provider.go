package backend

import (
	"context"
	"sync"
	"time"

	klog "k8s.io/klog/v2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

const (
	// TODO: https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/336
	metricsValidityPeriod = 5 * time.Second
)

func NewProvider(pmc PodMetricsClient, datastore *K8sDatastore) *Provider {
	p := &Provider{
		podMetrics:          sync.Map{},
		podMetricsRefresher: sync.Map{},
		pmc:                 pmc,
		datastore:           datastore,
	}
	return p
}

// Provider provides backend pods and information such as metrics.
type Provider struct {
	// key: Pod, value: *PodMetrics
	podMetrics sync.Map
	// key: Pod, value: *PodMetricsRefresher
	podMetricsRefresher sync.Map
	pmc                 PodMetricsClient
	datastore           *K8sDatastore
}

type PodMetricsClient interface {
	// FetchMetrics fetches metrics for the given pod.
	// The returned PodMetrics and the existing one should not be the same object.
	// Otherwise, there will be race.
	FetchMetrics(ctx context.Context, pod Pod, existing *PodMetrics) (*PodMetrics, error)
}

func isPodMetricsStale(pm *PodMetrics) bool {
	return time.Since(pm.Metrics.UpdatedTime) > metricsValidityPeriod
}

func (p *Provider) AllFreshPodMetrics() []*PodMetrics {
	return p.allPodMetrics(false)
}

func (p *Provider) AllStalePodMetrics() []*PodMetrics {
	return p.allPodMetrics(true)
}

func (p *Provider) allPodMetrics(stale bool) []*PodMetrics {
	res := []*PodMetrics{}
	fn := func(k, v any) bool {
		m := v.(*PodMetrics)

		if !stale {
			if isPodMetricsStale(m) {
				// exclude stale metrics for scheduler
				klog.V(logutil.DEBUG).InfoS("Pod metrics is stale, skipping", "pod", m.Pod)
			} else {
				res = append(res, m)
			}

		} else {
			if isPodMetricsStale(m) {
				res = append(res, m)
			}
		}

		return true
	}
	p.podMetrics.Range(fn)
	return res
}

func (p *Provider) UpdatePodMetrics(pod Pod, pm *PodMetrics) {
	pm.Metrics.UpdatedTime = time.Now()
	p.podMetrics.Store(pod, pm)
	klog.V(logutil.DEBUG).InfoS("Updated metrics", "pod", pod, "metrics", pm.Metrics)
}

func (p *Provider) GetPodMetrics(pod Pod) (*PodMetrics, bool) {
	val, ok := p.podMetrics.Load(pod)
	if ok {
		// For now, the only caller of GetPodMetrics is the refresher, so we
		// don't need to exclude the stale metrics.
		return val.(*PodMetrics), true
	}
	return nil, false
}

func (p *Provider) Init(refreshPodsInterval, refreshMetricsInterval, refreshMetricsTimeout, refreshPrometheusMetricsInterval time.Duration) error {
	// periodically refresh pods
	go func() {
		for {
			p.refreshPodsOnce(refreshMetricsInterval, refreshMetricsTimeout)
			time.Sleep(refreshPodsInterval)
		}
	}()

	// Periodically flush prometheus metrics for inference pool
	go func() {
		for {
			time.Sleep(refreshPrometheusMetricsInterval)
			p.flushPrometheusMetricsOnce()
		}
	}()

	// Periodically print out the pods and metrics for DEBUGGING.
	if klogV := klog.V(logutil.DEBUG); klogV.Enabled() {
		go func() {
			for {
				time.Sleep(5 * time.Second)
				klogV.InfoS("Current Pods and metrics gathered", "fresh metrics", p.AllFreshPodMetrics(),
					"stale metrics", p.AllStalePodMetrics())
			}
		}()
	}

	return nil
}

// refreshPodsOnce lists pods and updates keys in the podMetrics map.
// Note this function doesn't update the PodMetrics value, it's done separately.
func (p *Provider) refreshPodsOnce(refreshMetricsInterval, refreshMetricsTimeout time.Duration) {
	// merge new pods with cached ones.
	// add new pod to the map
	addNewPods := func(k, v any) bool {
		pod := k.(Pod)
		if _, ok := p.podMetrics.Load(pod); !ok {
			new := &PodMetrics{
				Pod: pod,
				Metrics: Metrics{
					ActiveModels: make(map[string]int),
				},
				// Metrics are considered stale until they are first refreshed.
			}
			p.podMetrics.Store(pod, new)

			refresher := NewPodMetricsRefresher(p, pod, refreshMetricsInterval, refreshMetricsTimeout)
			refresher.start()
			p.podMetricsRefresher.Store(pod, refresher)
		}
		return true
	}
	// remove pods that don't exist any more.
	mergeFn := func(k, v any) bool {
		pod := k.(Pod)
		if _, ok := p.datastore.pods.Load(pod); !ok {
			p.podMetrics.Delete(pod)
			if v, ok := p.podMetricsRefresher.LoadAndDelete(pod); ok {
				refresher := v.(*PodMetricsRefresher)
				refresher.stop()
			}
		}
		return true
	}
	p.podMetrics.Range(mergeFn)
	p.datastore.pods.Range(addNewPods)
}

func (p *Provider) flushPrometheusMetricsOnce() {
	klog.V(logutil.DEBUG).InfoS("Flushing Prometheus Metrics")

	pool, _ := p.datastore.getInferencePool()
	if pool == nil {
		// No inference pool or not initialize.
		return
	}

	var kvCacheTotal float64
	var queueTotal int

	podMetrics := append(p.AllFreshPodMetrics(), p.AllStalePodMetrics()...)
	if len(podMetrics) == 0 {
		return
	}

	for _, pod := range podMetrics {
		kvCacheTotal += pod.KVCacheUsagePercent
		queueTotal += pod.WaitingQueueSize
	}

	podTotalCount := len(podMetrics)
	metrics.RecordInferencePoolAvgKVCache(pool.Name, kvCacheTotal/float64(podTotalCount))
	metrics.RecordInferencePoolAvgQueueSize(pool.Name, float64(queueTotal/podTotalCount))
}
