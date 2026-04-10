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

package concurrency

import (
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

const (
	InFlightLoadKey = "InFlightLoadKey"
)

// InFlightLoad captures the current real-time load of an endpoint as tracked by the EPP.
type InFlightLoad struct {
	Tokens   int64
	Requests int64
}

func (l *InFlightLoad) Clone() fwkdl.Cloneable {
	if l == nil {
		return nil
	}
	return &InFlightLoad{
		Tokens:   l.Tokens,
		Requests: l.Requests,
	}
}
