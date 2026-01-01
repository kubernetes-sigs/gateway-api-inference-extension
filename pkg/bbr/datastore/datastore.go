/*
Copyright 2025 The Kubernetes Authors.

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

package datastore

import (
	"fmt"
	"strings"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/yaml"
)

const (
	baseModelKey = "baseModel"
	adaptersKey  = "adapters"
)

// The bbr datastore stores a mapping between a LoRA adapter and its corresponding base model.
// Each ConfigMap object stores a mapping between base model and a set of LoRA adapters and datastore translates this into an in-memory cache.
// In case the number of LoRA adapters is extremely large, it's possible to use multiple ConfigMap objects, each storing a slice of the mapping.
// Base models are not stored in the cache.
type Datastore interface {
	ConfigMapUpdateOrAddIfNotExist(configmap *corev1.ConfigMap) error
	ConfigMapDelete(configmap *corev1.ConfigMap)
}

// NewDatastore creates a new bbr data store.
func NewDatastore() Datastore {
	return &datastore{
		loraAdapterToBaseModel: map[string]string{},
		configmapAdapters:      map[types.NamespacedName]sets.Set[string]{}, // map from configmap namespaced name to its adapters
		lock:                   sync.RWMutex{},
	}
}

type datastore struct {
	loraAdapterToBaseModel map[string]string // a mapping between a lora adapter and its corresponding base model
	configmapAdapters      map[types.NamespacedName]sets.Set[string]
	lock                   sync.RWMutex
}

func (ds *datastore) ConfigMapUpdateOrAddIfNotExist(configmap *corev1.ConfigMap) error {
	baseModel, newAdapters, err := ds.parseConfigMap(configmap)
	if err != nil {
		return fmt.Errorf("failed to parse configmap - %w", err)
	}

	configmapNamespacedName := types.NamespacedName{Namespace: configmap.GetNamespace(), Name: configmap.GetName()}
	existingAdapters := sets.Set[string]{}
	ds.lock.Lock()
	defer ds.lock.Unlock()
	if previousAdapters, found := ds.configmapAdapters[configmapNamespacedName]; found {
		existingAdapters = previousAdapters
	}
	adaptersToRemove := existingAdapters.Difference(newAdapters) // adapters in existingAdapters but not in newAdapters.
	// update loraAdapterToBaseModel mapping
	for adapterToUpdate := range newAdapters {
		ds.loraAdapterToBaseModel[adapterToUpdate] = baseModel
	}
	for adapterToRemove := range adaptersToRemove {
		delete(ds.loraAdapterToBaseModel, adapterToRemove)
	}
	// update configmap NamespacedName to current set of adapters mapping
	ds.configmapAdapters[configmapNamespacedName] = newAdapters

	return nil
}

func (ds *datastore) ConfigMapDelete(configmap *corev1.ConfigMap) {
	configmapNamespacedName := types.NamespacedName{Namespace: configmap.GetNamespace(), Name: configmap.GetName()}
	ds.lock.Lock()
	defer ds.lock.Unlock()
	// delete adapters from loraAdapterToBaseModel mapping
	adapters := ds.configmapAdapters[configmapNamespacedName]
	for adapter := range adapters {
		delete(ds.loraAdapterToBaseModel, adapter)
	}
	// delete configmap NamespacedName to current set of adapters mapping
	delete(ds.configmapAdapters, configmapNamespacedName)
}

// parseConfigMap returns a tuple consisting (base model, set of adapters, error)
// error is set in case the configmap data section is not in the expected format.
func (ds *datastore) parseConfigMap(configmap *corev1.ConfigMap) (string, sets.Set[string], error) {
	// parse base model
	baseModel, ok := configmap.Data[baseModelKey]
	if !ok || strings.TrimSpace(baseModel) == "" {
		return "", nil, fmt.Errorf("missing or empty baseModel in ConfigMap %s/%s", configmap.Namespace, configmap.Name)
	}

	adapters := sets.Set[string]{}
	// parse adapters
	if raw, ok := configmap.Data[adaptersKey]; ok && strings.TrimSpace(raw) != "" {
		var list []string
		if err := yaml.Unmarshal([]byte(raw), &list); err != nil {
			return "", nil, fmt.Errorf("failed to parse adapters: %w", err)
		}

		for _, adapter := range list {
			if strings.TrimSpace(adapter) == "" {
				continue // skip empty entries
			}
			adapters.Insert(adapter)
		}
	}

	return baseModel, adapters, nil
}
