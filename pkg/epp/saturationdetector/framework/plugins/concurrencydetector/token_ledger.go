/*
Copyright 2026 The Kubernetes Authors.

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

package concurrencydetector

import (
	"sync"
)

type tokenCountByEp struct {
	endpointID string
	tokenCount int64
	stage      string
}

// tokenLedger tracks the number of tokens added for each request.
// This is required in the Post Request hook to decrement the token count for the endpoint.
// In disaggregated flows, a single request may be associated with multiple endpoints/stages.
type tokenLedger struct {
	mu sync.Mutex
	// tokenCountByReq tracks the number of tokens added for each request and its associated endpoints.
	tokenCountByReq map[string][]tokenCountByEp // requestID -> []{endpointID, tokenCount, stage}
}

func newTokenLedger() *tokenLedger {
	return &tokenLedger{
		tokenCountByReq: make(map[string][]tokenCountByEp),
	}
}

// add records the number of tokens added for a request and endpoint to the ledger.
func (tl *tokenLedger) add(requestID string, endpointID string, tokenCount int64, stage string) {
	tl.mu.Lock()
	defer tl.mu.Unlock()
	tl.tokenCountByReq[requestID] = append(tl.tokenCountByReq[requestID], tokenCountByEp{
		endpointID: endpointID,
		tokenCount: tokenCount,
		stage:      stage,
	})
}

// removeAll removes and returns all token counts associated with a given request.
func (tl *tokenLedger) removeAll(requestID string) []tokenCountByEp {
	tl.mu.Lock()
	defer tl.mu.Unlock()
	counts, ok := tl.tokenCountByReq[requestID]
	if !ok {
		return nil
	}
	delete(tl.tokenCountByReq, requestID)
	return counts
}

// removeStage removes and returns the token count for a specific stage of a request.
// This is used to signal completion of a partial stage (e.g., prefill) before the full request finishes.
func (tl *tokenLedger) removeStage(requestID string, stage string) (endpointID string, tokenCount int64, ok bool) {
	tl.mu.Lock()
	defer tl.mu.Unlock()

	counts, ok := tl.tokenCountByReq[requestID]
	if !ok {
		return "", 0, false
	}

	for i, c := range counts {
		if c.stage == stage {
			// Found the stage. Remove it from the list.
			tl.tokenCountByReq[requestID] = append(counts[:i], counts[i+1:]...)
			if len(tl.tokenCountByReq[requestID]) == 0 {
				delete(tl.tokenCountByReq, requestID)
			}
			return c.endpointID, c.tokenCount, true
		}
	}
	return "", 0, false
}

// removeForEndpoint removes and returns the token counts added for all requests for an endpoint from the ledger.
func (tl *tokenLedger) removeForEndpoint(endpointID string) []tokenCountByEp {
	tl.mu.Lock()
	defer tl.mu.Unlock()
	var out []tokenCountByEp

	for rid, counts := range tl.tokenCountByReq {
		var active []tokenCountByEp
		for _, c := range counts {
			if c.endpointID == endpointID {
				out = append(out, c)
			} else {
				active = append(active, c)
			}
		}

		if len(active) == 0 {
			delete(tl.tokenCountByReq, rid)
		} else if len(active) < len(counts) {
			tl.tokenCountByReq[rid] = active
		}
	}
	return out
}
