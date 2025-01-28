package backend

import (
	"context"
	"time"

	klog "k8s.io/klog/v2"
)

const (
	// TODO: make it configurable
	fetchMetricsTimeout = 1 * time.Second
)

type PodMetricsRefresher struct {
	done     chan struct{}
	interval time.Duration

	// refresher holds provider & pod so it can update the metrics concurrent-safely.
	pod      Pod
	provider *Provider
}

func NewPodMetricsRefresher(provider *Provider, pod Pod, interval time.Duration) *PodMetricsRefresher {
	return &PodMetricsRefresher{
		done:     make(chan struct{}),
		interval: interval,
		pod:      pod,
		provider: provider,
	}
}

func (r *PodMetricsRefresher) start() {
	go func() {
		klog.V(2).Infof("Starting refresher for pod %v", r.pod)
		for {
			select {
			case <-r.done:
				return
			default:
			}

			err := r.refreshMetrics()
			if err != nil {
				klog.Errorf("Failed to refresh metrics for pod %s: %v", r.pod, err)
			}

			time.Sleep(r.interval)
		}
	}()
}

func (r *PodMetricsRefresher) refreshMetrics() error {
	ctx, cancel := context.WithTimeout(context.Background(), fetchMetricsTimeout)
	defer cancel()

	pod := r.pod
	existing, found := r.provider.GetPodMetrics(pod)
	if !found {
		// As refresher is running in the background, it's possible that the pod is deleted but
		// the refresh goroutine doesn't read the done channel yet. In this case, we just return nil.
		// The refresher will be stopped after this interval.
		return nil
	}

	klog.V(4).Infof("Processing pod %v and metric %v", pod, existing.Metrics)
	updated, err := r.provider.pmc.FetchMetrics(ctx, r.pod, existing)
	if err != nil {
		return err
	}

	r.provider.UpdatePodMetrics(pod, updated)
	return nil
}

func (r *PodMetricsRefresher) stop() {
	klog.V(2).Infof("Stopping refresher for pod %v", r.pod)
	close(r.done)
}
