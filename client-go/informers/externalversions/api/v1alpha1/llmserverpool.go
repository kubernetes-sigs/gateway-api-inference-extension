/*
Copyright 2023.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
// Code generated by informer-gen. DO NOT EDIT.

package v1alpha1

import (
	context "context"
	time "time"

	llminstancegatewayapiv1alpha1 "inference.k8s.io/llm-instance-gateway/api/v1alpha1"
	versioned "inference.k8s.io/llm-instance-gateway/client-go/clientset/versioned"
	internalinterfaces "inference.k8s.io/llm-instance-gateway/client-go/informers/externalversions/internalinterfaces"
	apiv1alpha1 "inference.k8s.io/llm-instance-gateway/client-go/listers/api/v1alpha1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
	watch "k8s.io/apimachinery/pkg/watch"
	cache "k8s.io/client-go/tools/cache"
)

// LLMServerPoolInformer provides access to a shared informer and lister for
// LLMServerPools.
type LLMServerPoolInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() apiv1alpha1.LLMServerPoolLister
}

type lLMServerPoolInformer struct {
	factory          internalinterfaces.SharedInformerFactory
	tweakListOptions internalinterfaces.TweakListOptionsFunc
	namespace        string
}

// NewLLMServerPoolInformer constructs a new informer for LLMServerPool type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func NewLLMServerPoolInformer(client versioned.Interface, namespace string, resyncPeriod time.Duration, indexers cache.Indexers) cache.SharedIndexInformer {
	return NewFilteredLLMServerPoolInformer(client, namespace, resyncPeriod, indexers, nil)
}

// NewFilteredLLMServerPoolInformer constructs a new informer for LLMServerPool type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func NewFilteredLLMServerPoolInformer(client versioned.Interface, namespace string, resyncPeriod time.Duration, indexers cache.Indexers, tweakListOptions internalinterfaces.TweakListOptionsFunc) cache.SharedIndexInformer {
	return cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				if tweakListOptions != nil {
					tweakListOptions(&options)
				}
				return client.ApiV1alpha1().LLMServerPools(namespace).List(context.TODO(), options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				if tweakListOptions != nil {
					tweakListOptions(&options)
				}
				return client.ApiV1alpha1().LLMServerPools(namespace).Watch(context.TODO(), options)
			},
		},
		&llminstancegatewayapiv1alpha1.LLMServerPool{},
		resyncPeriod,
		indexers,
	)
}

func (f *lLMServerPoolInformer) defaultInformer(client versioned.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	return NewFilteredLLMServerPoolInformer(client, f.namespace, resyncPeriod, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}, f.tweakListOptions)
}

func (f *lLMServerPoolInformer) Informer() cache.SharedIndexInformer {
	return f.factory.InformerFor(&llminstancegatewayapiv1alpha1.LLMServerPool{}, f.defaultInformer)
}

func (f *lLMServerPoolInformer) Lister() apiv1alpha1.LLMServerPoolLister {
	return apiv1alpha1.NewLLMServerPoolLister(f.Informer().GetIndexer())
}
