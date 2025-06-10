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
package saturationdetector

import (
	"time"

	commonconfig "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/common/config"
)

// Default configuration values
const (
	DefaultQueueDepthThreshold  = commonconfig.DefaultQueueThresholdCritical
	DefaultKVCacheUtilThreshold = commonconfig.DefaultKVCacheThreshold
	// DefaultMetricsStalenessThreshold defines how old metrics can be before they
	// are considered stale.
	// Given the pod metrics refresh interval is 50ms, a threshold slightly above
	// that should be fine.
	DefaultMetricsStalenessThreshold = 200 * time.Millisecond
)
