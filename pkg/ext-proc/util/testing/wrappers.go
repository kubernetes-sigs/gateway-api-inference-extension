package testing

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// PodWrapper wraps a Pod inside.
type PodWrapper struct{ corev1.Pod }

// MakePod creates a Pod wrapper.
func MakePod(name string) *PodWrapper {
	return &PodWrapper{
		corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
		},
	}
}

// Obj returns the inner Pod.
func (p *PodWrapper) Obj() *corev1.Pod {
	return &p.Pod
}

func (p *PodWrapper) SetReady() *PodWrapper {
	p.Status.Conditions = []corev1.PodCondition{{
		Type:   corev1.PodReady,
		Status: corev1.ConditionTrue,
	}}
	return p
}

func (p *PodWrapper) SetPodIP(podIP string) *PodWrapper {
	p.Status.PodIP = podIP
	return p
}
