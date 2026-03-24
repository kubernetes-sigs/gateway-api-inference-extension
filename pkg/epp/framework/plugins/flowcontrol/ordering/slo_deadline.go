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

package ordering

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/common/request"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	// SLODeadlineOrderingPolicyType orders requests by an SLO-based deadline
	//
	// It selects the request with the earliest SLO-based deadline, computed as ReceivedTimestamp plus the TTFT SLO header value (request.TTFTSLOMsHeaderKey, milliseconds).
	// Requests without a valid TTFT SLO header are treated as having no deadline and are scheduled after SLO-bound requests,
	// with FCFS as a tie-breaker.
	SLODeadlineOrderingPolicyType = "slo-deadline-ordering-policy"
)

func SLODeadlineOrderingPolicyFactory(name string, _ json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	return newSLODeadlinePolicy().withName(name), nil
}

type sloDeadlinePolicy struct {
	name string
}

var _ flowcontrol.OrderingPolicy = &sloDeadlinePolicy{}

func newSLODeadlinePolicy() *sloDeadlinePolicy {
	return &sloDeadlinePolicy{
		name: SLODeadlineOrderingPolicyType,
	}
}

func (p *sloDeadlinePolicy) withName(name string) *sloDeadlinePolicy {
	if name != "" {
		p.name = name
	}
	return p
}

func (p *sloDeadlinePolicy) Name() string {
	return p.name
}

// RequiredQueueCapabilities returns the queue capabilities required by this policy.
func (p *sloDeadlinePolicy) RequiredQueueCapabilities() []flowcontrol.QueueCapability {
	return []flowcontrol.QueueCapability{flowcontrol.CapabilityPriorityConfigurable}
}

func (p *sloDeadlinePolicy) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: SLODeadlineOrderingPolicyType,
		Name: p.name,
	}
}

var sloMaxDeadlineTime = time.Unix(0, 1<<63-1)

// calculateSLODeadline computes the SLO-based deadline for a request: ReceivedTimestamp + TTFT SLO header (ms).
// The header is read from the InferenceRequest()'s headers. If the header is missing, empty, or invalid,
// the request is assigned a far-future deadline so it sorts after SLO-bound requests.
func calculateSLODeadline(item flowcontrol.QueueItemAccessor) time.Time {
	req := item.OriginalRequest()
	if req == nil {
		return sloMaxDeadlineTime
	}
	infReq := req.InferenceRequest()
	if infReq == nil || infReq.Headers == nil {
		return sloMaxDeadlineTime
	}
	sloTTFT := request.GetHeader(infReq.Headers, request.TTFTSLOMsHeaderKey)
	if sloTTFT == "" {
		return sloMaxDeadlineTime
	}
	ms, err := strconv.ParseInt(strings.TrimSpace(sloTTFT), 10, 64)
	if err != nil || ms < 0 {
		return sloMaxDeadlineTime
	}
	return req.ReceivedTimestamp().Add(time.Duration(ms) * time.Millisecond)
}

// Less returns true if item 'a' should be dispatched before item 'b'.
// It orders by SLO deadline (earliest first), using FCFS as a tie-breaker.
func (p *sloDeadlinePolicy) Less(a, b flowcontrol.QueueItemAccessor) bool {
	if a == nil && b == nil {
		return false
	}
	if a == nil {
		return false
	}
	if b == nil {
		return true
	}
	deadlineA := calculateSLODeadline(a)
	deadlineB := calculateSLODeadline(b)
	if !deadlineA.Equal(deadlineB) {
		return deadlineA.Before(deadlineB)
	}
	reqA := a.OriginalRequest()
	reqB := b.OriginalRequest()
	if reqA == nil && reqB == nil {
		return false
	}
	if reqA == nil {
		return false
	}
	if reqB == nil {
		return true
	}
	return reqA.ReceivedTimestamp().Before(reqB.ReceivedTimestamp())
}
