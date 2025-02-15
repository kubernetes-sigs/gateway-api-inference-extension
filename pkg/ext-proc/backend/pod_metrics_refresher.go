package backend

import (
	"context"
	"time"

	klog "k8s.io/klog/v2"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

type PodMetricsRefresher struct {
	done     chan struct{}
	interval time.Duration
	timeout  time.Duration

	// refresher holds provider & pod so it can update the metrics concurrent-safely.
	pod      Pod
	provider *Provider
}

func NewPodMetricsRefresher(provider *Provider, pod Pod, interval, timeout time.Duration) *PodMetricsRefresher {
	return &PodMetricsRefresher{
		done:     make(chan struct{}),
		interval: interval,
		timeout:  timeout,
		pod:      pod,
		provider: provider,
	}
}

func (r *PodMetricsRefresher) start() {
	go func() {
		klog.V(logutil.DEFAULT).InfoS("Starting refresher", "pod", r.pod)
		for {
			select {
			case <-r.done:
				return
			default:
			}

			err := r.refreshMetrics()
			if err != nil {
				klog.ErrorS(err, "Failed to refresh metrics", "pod", r.pod)
			}

			time.Sleep(r.interval)
		}
	}()
}

func (r *PodMetricsRefresher) refreshMetrics() error {
	ctx, cancel := context.WithTimeout(context.Background(), r.timeout)
	defer cancel()

	pod := r.pod
	existing, found := r.provider.GetPodMetrics(pod)
	if !found {
		// As refresher is running in the background, it's possible that the pod is deleted but
		// the refresh goroutine doesn't read the done channel yet. In this case, we just return nil.
		// The refresher will be stopped after this interval.
		return nil
	}

	klog.V(logutil.DEBUG).InfoS("Refresh metrics", "pod", pod, "metrics", existing.Metrics)
	updated, err := r.provider.pmc.FetchMetrics(ctx, r.pod, existing)
	if err != nil {
		return err
	}

	r.provider.UpdatePodMetrics(pod, updated)
	return nil
}

func (r *PodMetricsRefresher) stop() {
	klog.V(logutil.DEFAULT).InfoS("Stopping refresher", "pod", r.pod)
	close(r.done)
}
