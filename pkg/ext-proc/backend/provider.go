package backend

import (
	"context"
	"sync"
	"time"

	klog "k8s.io/klog/v2"
)

const (
	// TODO: make it configurable. One idea is to provide a configuration singleton
	// and put fields like refreshMetricsInterval in it. So far, we have to pass these
	// fields across several layers.
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
	return time.Since(pm.UpdatedTime) > metricsValidityPeriod
}

func (p *Provider) AllPodMetrics() []*PodMetrics {
	return p.allPodMetrics(false)
}

func (p *Provider) AllPodMetricsIncludingStale() []*PodMetrics {
	return p.allPodMetrics(true)
}

func (p *Provider) allPodMetrics(staleIncluded bool) []*PodMetrics {
	res := []*PodMetrics{}
	fn := func(k, v any) bool {
		m := v.(*PodMetrics)

		if !staleIncluded && isPodMetricsStale(m) {
			// exclude stale metrics for scheduler
			klog.V(4).Infof("Pod metrics for %s is stale, skipping", m.Pod)
			return true
		}

		res = append(res, m)
		return true
	}
	p.podMetrics.Range(fn)
	return res
}

func (p *Provider) UpdatePodMetrics(pod Pod, pm *PodMetrics) {
	pm.UpdatedTime = time.Now()
	p.podMetrics.Store(pod, pm)
	klog.V(4).Infof("Updated metrics for pod %s: %v", pod, pm.Metrics)
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

func (p *Provider) Init(refreshPodsInterval, refreshMetricsInterval time.Duration) error {
	p.refreshPodsOnce(refreshMetricsInterval)

	// periodically refresh pods
	go func() {
		for {
			time.Sleep(refreshPodsInterval)
			p.refreshPodsOnce(refreshMetricsInterval)
		}
	}()

	// Periodically print out the pods and metrics for DEBUGGING.
	if klog.V(4).Enabled() {
		go func() {
			for {
				time.Sleep(5 * time.Second)
				podMetrics := p.AllPodMetricsIncludingStale()
				stalePodMetrics := make([]*PodMetrics, 0)
				freshPodMetrics := make([]*PodMetrics, 0, len(podMetrics))
				for _, pm := range podMetrics {
					if isPodMetricsStale(pm) {
						stalePodMetrics = append(stalePodMetrics, pm)
					} else {
						freshPodMetrics = append(freshPodMetrics, pm)
					}
				}
				klog.Infof("===DEBUG: Current Pods and metrics: %+v", freshPodMetrics)
				klog.Infof("===DEBUG: Stale Pods and metrics: %+v", stalePodMetrics)
			}
		}()
	}

	return nil
}

// refreshPodsOnce lists pods and updates keys in the podMetrics map.
// Note this function doesn't update the PodMetrics value, it's done separately.
func (p *Provider) refreshPodsOnce(refreshMetricsInterval time.Duration) {
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

			refresher := NewPodMetricsRefresher(p, pod, refreshMetricsInterval)
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
			if v, ok := p.podMetrics.LoadAndDelete(pod); ok {
				refresher := v.(*PodMetricsRefresher)
				refresher.stop()
			}
		}
		return true
	}
	p.podMetrics.Range(mergeFn)
	p.datastore.pods.Range(addNewPods)
}
