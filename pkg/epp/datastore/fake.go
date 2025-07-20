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
	"context"
	"fmt"
	"sync"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
)

// FakeDatastore is a fake implementation of the Datastore interface for testing
type FakeDatastore struct {
	mu                sync.RWMutex
	pool              *v1alpha2.InferencePool
	models            map[string]*v1alpha2.InferenceModel
	pods              map[types.NamespacedName]backendmetrics.PodMetrics
	
	// Control behavior
	poolSynced        bool
	poolGetError      error
	modelResyncError  error
	
	// Call tracking
	clearCalled       bool
	poolSetCalled     bool
	modelDeleteCalled bool
}

// NewFakeDatastore creates a new fake datastore
func NewFakeDatastore() *FakeDatastore {
	return &FakeDatastore{
		models:     make(map[string]*v1alpha2.InferenceModel),
		pods:       make(map[types.NamespacedName]backendmetrics.PodMetrics),
		poolSynced: true, // Default to synced
	}
}

// SetPoolGetError sets an error to be returned by PoolGet
func (f *FakeDatastore) SetPoolGetError(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.poolGetError = err
}

// SetModelResyncError sets an error to be returned by ModelResync
func (f *FakeDatastore) SetModelResyncError(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.modelResyncError = err
}

// SetPoolSynced controls whether the pool appears synced
func (f *FakeDatastore) SetPoolSynced(synced bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.poolSynced = synced
}

// WasClearCalled returns true if Clear was called
func (f *FakeDatastore) WasClearCalled() bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.clearCalled
}

// WasPoolSetCalled returns true if PoolSet was called
func (f *FakeDatastore) WasPoolSetCalled() bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.poolSetCalled
}

// WasModelDeleteCalled returns true if ModelDelete was called
func (f *FakeDatastore) WasModelDeleteCalled() bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.modelDeleteCalled
}

// InferencePool operations
func (f *FakeDatastore) PoolSet(ctx context.Context, reader client.Reader, pool *v1alpha2.InferencePool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.poolSetCalled = true
	
	if pool == nil {
		f.Clear()
		return nil
	}
	
	f.pool = pool
	return nil
}

func (f *FakeDatastore) PoolGet() (*v1alpha2.InferencePool, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	if f.poolGetError != nil {
		return nil, f.poolGetError
	}
	
	if !f.poolSynced {
		return nil, errPoolNotSynced
	}
	
	return f.pool, nil
}

func (f *FakeDatastore) PoolHasSynced() bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.poolSynced && f.pool != nil
}

func (f *FakeDatastore) PoolLabelsMatch(podLabels map[string]string) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	if f.pool == nil {
		return false
	}
	
	// Simple implementation - in real datastore this would use label selectors
	// For testing, we can just return true if pool exists
	return true
}

// InferenceModel operations
func (f *FakeDatastore) ModelSetIfOlder(infModel *v1alpha2.InferenceModel) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	
	existing, exists := f.models[infModel.Spec.ModelName]
	if exists {
		// Check if existing is older (simple comparison for testing)
		if existing.ObjectMeta.CreationTimestamp.Before(&infModel.ObjectMeta.CreationTimestamp) {
			f.models[infModel.Spec.ModelName] = infModel
			return true
		}
		return false
	}
	
	f.models[infModel.Spec.ModelName] = infModel
	return true
}

func (f *FakeDatastore) ModelGet(modelName string) *v1alpha2.InferenceModel {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.models[modelName]
}

func (f *FakeDatastore) ModelDelete(namespacedName types.NamespacedName) *v1alpha2.InferenceModel {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.modelDeleteCalled = true
	
	for modelName, model := range f.models {
		if model.Name == namespacedName.Name && model.Namespace == namespacedName.Namespace {
			delete(f.models, modelName)
			return model
		}
	}
	return nil
}

func (f *FakeDatastore) ModelResync(ctx context.Context, reader client.Reader, modelName string) (bool, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	if f.modelResyncError != nil {
		return false, f.modelResyncError
	}
	
	// Simple implementation for testing
	_, exists := f.models[modelName]
	return exists, nil
}

func (f *FakeDatastore) ModelGetAll() []*v1alpha2.InferenceModel {
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	result := make([]*v1alpha2.InferenceModel, 0, len(f.models))
	for _, model := range f.models {
		result = append(result, model)
	}
	return result
}

// PodMetrics operations
func (f *FakeDatastore) PodGetAll() []backendmetrics.PodMetrics {
	return f.PodList(func(backendmetrics.PodMetrics) bool { return true })
}

func (f *FakeDatastore) PodList(predicate func(backendmetrics.PodMetrics) bool) []backendmetrics.PodMetrics {
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	result := make([]backendmetrics.PodMetrics, 0, len(f.pods))
	for _, pod := range f.pods {
		if predicate(pod) {
			result = append(result, pod)
		}
	}
	return result
}

func (f *FakeDatastore) PodUpdateOrAddIfNotExist(pod *corev1.Pod) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	
	namespacedName := types.NamespacedName{
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}
	
	_, existed := f.pods[namespacedName]
	if !existed {
		// Create a fake pod metrics for testing
		f.pods[namespacedName] = NewFakePodMetrics(pod)
	} else {
		// Update existing pod
		f.pods[namespacedName].UpdatePod(pod)
	}
	
	return existed
}

func (f *FakeDatastore) PodDelete(namespacedName types.NamespacedName) {
	f.mu.Lock()
	defer f.mu.Unlock()
	
	if pod, exists := f.pods[namespacedName]; exists {
		pod.StopRefreshLoop()
		delete(f.pods, namespacedName)
	}
}

// Request management operations
func (f *FakeDatastore) PodAddRequest(podName types.NamespacedName, requestID string, tpot float64) error {
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	pod, exists := f.pods[podName]
	if !exists {
		return fmt.Errorf("pod %s not found in datastore", podName)
	}
	
	runningRequests := pod.GetRunningRequests()
	if runningRequests == nil {
		return fmt.Errorf("pod %s does not have running requests queue initialized", podName)
	}
	
	if !runningRequests.Add(requestID, tpot) {
		return fmt.Errorf("request %s already exists in pod %s", requestID, podName)
	}
	
	return nil
}

func (f *FakeDatastore) PodRemoveRequest(podName types.NamespacedName, requestID string) error {
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	pod, exists := f.pods[podName]
	if !exists {
		return fmt.Errorf("pod %s not found in datastore", podName)
	}
	
	runningRequests := pod.GetRunningRequests()
	if runningRequests == nil {
		return fmt.Errorf("pod %s does not have running requests queue initialized", podName)
	}
	
	_, removed := runningRequests.Remove(requestID)
	if !removed {
		return fmt.Errorf("request %s not found in pod %s", requestID, podName)
	}
	
	return nil
}

func (f *FakeDatastore) PodUpdateRequest(podName types.NamespacedName, requestID string, tpot float64) error {
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	pod, exists := f.pods[podName]
	if !exists {
		return fmt.Errorf("pod %s not found in datastore", podName)
	}
	
	runningRequests := pod.GetRunningRequests()
	if runningRequests == nil {
		return fmt.Errorf("pod %s does not have running requests queue initialized", podName)
	}
	
	if !runningRequests.Update(requestID, tpot) {
		return fmt.Errorf("request %s not found in pod %s", requestID, podName)
	}
	
	return nil
}

func (f *FakeDatastore) PodGetRunningRequests(podName types.NamespacedName) (*backend.RequestPriorityQueue, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	pod, exists := f.pods[podName]
	if !exists {
		return nil, fmt.Errorf("pod %s not found in datastore", podName)
	}
	
	runningRequests := pod.GetRunningRequests()
	if runningRequests == nil {
		return nil, fmt.Errorf("pod %s does not have running requests queue initialized", podName)
	}
	
	return runningRequests, nil
}

func (f *FakeDatastore) PodGetRequestCount(podName types.NamespacedName) (int, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	pod, exists := f.pods[podName]
	if !exists {
		return 0, fmt.Errorf("pod %s not found in datastore", podName)
	}
	
	runningRequests := pod.GetRunningRequests()
	if runningRequests == nil {
		return 0, fmt.Errorf("pod %s does not have running requests queue initialized", podName)
	}
	
	return runningRequests.GetSize(), nil
}

func (f *FakeDatastore) Clear() {
	f.clearCalled = true
	f.pool = nil
	f.models = make(map[string]*v1alpha2.InferenceModel)
	
	// Stop all pod refresh loops
	for _, pod := range f.pods {
		pod.StopRefreshLoop()
	}
	f.pods = make(map[types.NamespacedName]backendmetrics.PodMetrics)
}

// Helper methods for testing
func (f *FakeDatastore) AddPod(namespacedName types.NamespacedName, pod backendmetrics.PodMetrics) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.pods[namespacedName] = pod
}

func (f *FakeDatastore) AddModel(modelName string, model *v1alpha2.InferenceModel) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.models[modelName] = model
}

func (f *FakeDatastore) SetPool(pool *v1alpha2.InferencePool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.pool = pool
}

func (f *FakeDatastore) GetPodCount() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.pods)
}

func (f *FakeDatastore) GetModelCount() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.models)
}

// FakePodMetrics implements the PodMetrics interface for testing
type FakePodMetrics struct {
	pod             *backend.Pod
	metrics         *backendmetrics.MetricsState
	runningRequests *backend.RequestPriorityQueue
	stopped         bool
}

func NewFakePodMetrics(k8sPod *corev1.Pod) *FakePodMetrics {
	pod := &backend.Pod{
		NamespacedName: types.NamespacedName{
			Name:      k8sPod.Name,
			Namespace: k8sPod.Namespace,
		},
		Address:         k8sPod.Status.PodIP,
		Labels:          make(map[string]string),
		RunningRequests: backend.NewRequestPriorityQueue(),
	}
	
	// Copy labels
	for k, v := range k8sPod.Labels {
		pod.Labels[k] = v
	}
	
	return &FakePodMetrics{
		pod:             pod,
		metrics:         &backendmetrics.MetricsState{},
		runningRequests: pod.RunningRequests,
	}
}

func (f *FakePodMetrics) GetPod() *backend.Pod {
	return f.pod
}

func (f *FakePodMetrics) GetMetrics() *backendmetrics.MetricsState {
	return f.metrics
}

func (f *FakePodMetrics) UpdatePod(k8sPod *corev1.Pod) {
	f.pod.NamespacedName = types.NamespacedName{
		Name:      k8sPod.Name,
		Namespace: k8sPod.Namespace,
	}
	f.pod.Address = k8sPod.Status.PodIP
	
	// Update labels
	f.pod.Labels = make(map[string]string)
	for k, v := range k8sPod.Labels {
		f.pod.Labels[k] = v
	}
	// Note: RunningRequests queue is preserved
}

func (f *FakePodMetrics) StopRefreshLoop() {
	f.stopped = true
}

func (f *FakePodMetrics) String() string {
	return fmt.Sprintf("FakePodMetrics{%s}", f.pod.NamespacedName)
}

func (f *FakePodMetrics) GetRunningRequests() *backend.RequestPriorityQueue {
	return f.runningRequests
}

func (f *FakePodMetrics) AddRequest(requestID string, tpot float64) bool {
	if f.runningRequests == nil {
		return false
	}
	return f.runningRequests.Add(requestID, tpot)
}

func (f *FakePodMetrics) RemoveRequest(requestID string) bool {
	if f.runningRequests == nil {
		return false
	}
	_, success := f.runningRequests.Remove(requestID)
	return success
}

func (f *FakePodMetrics) UpdateRequest(requestID string, tpot float64) bool {
	if f.runningRequests == nil {
		return false
	}
	return f.runningRequests.Update(requestID, tpot)
}

func (f *FakePodMetrics) GetRequestCount() int {
	if f.runningRequests == nil {
		return 0
	}
	return f.runningRequests.GetSize()
}

func (f *FakePodMetrics) ContainsRequest(requestID string) bool {
	if f.runningRequests == nil {
		return false
	}
	return f.runningRequests.Contains(requestID)
}

func (f *FakePodMetrics) IsStopped() bool {
	return f.stopped
}

// Helper functions for creating test objects
func NewFakeInferencePool(name, namespace string) *v1alpha2.InferencePool {
	return &v1alpha2.InferencePool{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1alpha2.InferencePoolSpec{
			TargetPortNumber: 8080,
		},
	}
}

func NewFakeInferenceModel(name, namespace, modelName string) *v1alpha2.InferenceModel {
	return &v1alpha2.InferenceModel{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1alpha2.InferenceModelSpec{
			ModelName: modelName,
		},
	}
}

func NewFakePod(name, namespace, ip string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    map[string]string{"app": "test"},
		},
		Status: corev1.PodStatus{
			PodIP: ip,
		},
	}
}