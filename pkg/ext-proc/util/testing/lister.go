package testing

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	listersv1 "k8s.io/client-go/listers/core/v1"
)

type FakePodLister struct {
	PodsList []*v1.Pod
}

func (l *FakePodLister) List(selector labels.Selector) (ret []*v1.Pod, err error) {
	return l.PodsList, nil
}

func (l *FakePodLister) Pods(namespace string) listersv1.PodNamespaceLister {
	panic("not implemented")
}
