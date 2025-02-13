package backend

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/multierr"
	klog "k8s.io/klog/v2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

const (
	fetchMetricsTimeout = 5 * time.Second
)

func NewProvider(pmc PodMetricsClient, datastore *K8sDatastore) *Provider {
	p := &Provider{
		podMetrics: sync.Map{},
		pmc:        pmc,
		datastore:  datastore,
	}
	return p
}

// Provider provides backend pods and information such as metrics.
type Provider struct {
	// key: Pod, value: *PodMetrics
	podMetrics sync.Map
	pmc        PodMetricsClient
	datastore  *K8sDatastore
}

type PodMetricsClient interface {
	FetchMetrics(ctx context.Context, pod Pod, existing *PodMetrics) (*PodMetrics, error)
}

func (p *Provider) AllPodMetrics() []*PodMetrics {
	res := []*PodMetrics{}
	fn := func(k, v any) bool {
		res = append(res, v.(*PodMetrics))
		return true
	}
	p.podMetrics.Range(fn)
	return res
}

func (p *Provider) UpdatePodMetrics(pod Pod, pm *PodMetrics) {
	p.podMetrics.Store(pod, pm)
}

func (p *Provider) GetPodMetrics(pod Pod) (*PodMetrics, bool) {
	val, ok := p.podMetrics.Load(pod)
	if ok {
		return val.(*PodMetrics), true
	}
	return nil, false
}

func (p *Provider) Init(refreshPodsInterval, refreshMetricsInterval, refreshPrometheusMetricsInterval time.Duration) error {
	p.refreshPodsOnce()

	if err := p.refreshMetricsOnce(); err != nil {
		klog.ErrorS(err, "Failed to init metrics")
	}

	klog.InfoS("Initialized pods and metrics", "metrics", p.AllPodMetrics())

	// periodically refresh pods
	go func() {
		for {
			time.Sleep(refreshPodsInterval)
			p.refreshPodsOnce()
		}
	}()

	// periodically refresh metrics
	go func() {
		for {
			time.Sleep(refreshMetricsInterval)
			if err := p.refreshMetricsOnce(); err != nil {
				klog.V(logutil.TRACE).ErrorS(err, "Failed to refresh metrics")
			}
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
				klogV.InfoS("Current Pods and metrics gathered", "metrics", p.AllPodMetrics())
			}
		}()
	}

	return nil
}

// refreshPodsOnce lists pods and updates keys in the podMetrics map.
// Note this function doesn't update the PodMetrics value, it's done separately.
func (p *Provider) refreshPodsOnce() {
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
			}
			p.podMetrics.Store(pod, new)
		}
		return true
	}
	// remove pods that don't exist any more.
	mergeFn := func(k, v any) bool {
		pod := k.(Pod)
		if _, ok := p.datastore.pods.Load(pod); !ok {
			p.podMetrics.Delete(pod)
		}
		return true
	}
	p.podMetrics.Range(mergeFn)
	p.datastore.pods.Range(addNewPods)
}

func (p *Provider) refreshMetricsOnce() error {
	klogV := klog.V(logutil.TRACE)
	ctx, cancel := context.WithTimeout(context.Background(), fetchMetricsTimeout)
	defer cancel()
	start := time.Now()
	defer func() {
		d := time.Since(start)
		// TODO: add a metric instead of logging
		klogV.InfoS("Metrics refreshed", "duration", d)
	}()
	var wg sync.WaitGroup
	errCh := make(chan error)
	processOnePod := func(key, value any) bool {
		klogV.InfoS("Pod and metric being processed", "pod", key, "metric", value)
		pod := key.(Pod)
		existing := value.(*PodMetrics)
		wg.Add(1)
		go func() {
			defer wg.Done()
			updated, err := p.pmc.FetchMetrics(ctx, pod, existing)
			if err != nil {
				errCh <- fmt.Errorf("failed to parse metrics from %s: %v", pod, err)
				return
			}
			p.UpdatePodMetrics(pod, updated)
			klogV.InfoS("Updated metrics for pod", "pod", pod, "metrics", updated.Metrics)
		}()
		return true
	}
	p.podMetrics.Range(processOnePod)

	// Wait for metric collection for all pods to complete and close the error channel in a
	// goroutine so this is unblocking, allowing the code to proceed to the error collection code
	// below.
	// Note we couldn't use a buffered error channel with a size because the size of the podMetrics
	// sync.Map is unknown beforehand.
	go func() {
		wg.Wait()
		close(errCh)
	}()

	var errs error
	for err := range errCh {
		errs = multierr.Append(errs, err)
	}
	return errs
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

	podMetrics := p.AllPodMetrics()
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
