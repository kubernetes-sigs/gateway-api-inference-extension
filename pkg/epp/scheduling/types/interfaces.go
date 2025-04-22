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

package types

import (
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
)

const (
	PreSchedulerPluginType = "PreSchedule"
	PostSchedulePluginType = "PostSchedule"
	FilterPluginType       = "Filter"
	ScorerPluginType       = "Scorer"
	PickerPluginType       = "Picker"
)

type Pod interface {
	GetPod() *backendmetrics.Pod
	GetMetrics() *backendmetrics.Metrics
	SetScore(float64)
	Score() float64
	String() string
}

// Plugin defines the interface for scheduler plugins, combining scoring, filtering,
// and event handling capabilities.
type Plugin interface {
	// Name returns the name of the plugin.
	Name() string
}

// PreSchedule is called when the scheduler receives a new request. It can be used for various
// initialization work.
type PreSchedule interface {
	Plugin
	PreSchedule(ctx *Context)
}

// PostSchedule is called by the scheduler after it selects a targetPod for the request.
type PostSchedule interface {
	Plugin
	PostSchedule(ctx *Context, res *Result)
}

// Filter defines the interface for filtering a list of pods based on context.
type Filter interface {
	Plugin
	Filter(ctx *Context, pods []Pod) ([]Pod, error)
}

// Scorer defines the interface for scoring pods based on context.
type Scorer interface {
	Plugin
	Score(ctx *Context, pod Pod) (float64, error)
}

// Picker picks the final pod(s) to send the request to.
type Picker interface {
	Plugin
	Pick(ctx *Context, pods []Pod) (*Result, error)
}
