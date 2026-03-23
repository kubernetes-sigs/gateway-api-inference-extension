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

package request

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetHeader(t *testing.T) {
	headers := map[string]string{"X-SLO-TTFT-MS": "42", "Other": "x"}
	assert.Equal(t, "42", GetHeader(headers, "X-SLO-TTFT-MS"))
	assert.Equal(t, "42", GetHeader(headers, "x-slo-ttft-ms"))
	assert.Equal(t, "", GetHeader(headers, "missing"))
	assert.Equal(t, "", GetHeader(nil, "k"))
}
