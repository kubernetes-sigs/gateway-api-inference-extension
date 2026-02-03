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
}


// tokenLedger tracks the number of tokens added for each request. 
// This is required in the Post Request hook to decrement the token count for the endpoint.
type tokenLedger struct {
	mu sync.Mutex
	// tokenCountByReq tracks the number of tokens added for each request and endpoint.
	tokenCountByReq map[string]tokenCountByEp // requestID -> {endpointID -> tokenCount}
}

func newTokenLedger() *tokenLedger {
	return &tokenLedger{
		tokenCountByReq: make(map[string]tokenCountByEp),
	}
}

// adds the number of tokens added for a request and endpoint to the ledger.
func (tl *tokenLedger) add(requestID string, endpointID string, tokenCount int64) {
	tl.mu.Lock()
	defer tl.mu.Unlock()
	tl.tokenCountByReq[requestID] = tokenCountByEp{endpointID: endpointID, tokenCount: tokenCount}
}

// removes the number of tokens added for a request from the ledger.
func (tl *tokenLedger) remove(requestID string) (tokenCount int64, ok bool) {
	tl.mu.Lock()
	defer tl.mu.Unlock()
	e, ok := tl.tokenCountByReq[requestID]
	if !ok {
		return 0, false
	}
	delete(tl.tokenCountByReq, requestID)
	return e.tokenCount, true
}

// removes the number of tokens added for all requests for an endpoint from the ledger.
func (tl *tokenLedger) removeForEndpoint(endpointID string) []tokenCountByEp {
	tl.mu.Lock()
	defer tl.mu.Unlock()
	var out []tokenCountByEp
	for rid, e := range tl.tokenCountByReq {
		if e.endpointID == endpointID {
			out = append(out, e)
			delete(tl.tokenCountByReq, rid)
		}
	}
	return out
}
