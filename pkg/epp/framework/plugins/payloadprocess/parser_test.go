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

package payloadprocess

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestNewParserFromConfig(t *testing.T) {
	tests := []struct {
		name                  string
		enablePluggableParser bool
		customParserName      string
		want                  *OpenAIParser
	}{
		{
			name:                  "Enabled with correct name returns parser",
			enablePluggableParser: true,
			customParserName:      OpenAIParserName,
			want:                  NewOpenAIParser(),
		},
		{
			name:                  "Enabled with wrong name returns nil",
			enablePluggableParser: true,
			customParserName:      "wrong-parser",
			want:                  nil,
		},
		{
			name:                  "Disabled with correct name returns nil",
			enablePluggableParser: false,
			customParserName:      OpenAIParserName,
			want:                  nil,
		},
		{
			name:                  "Disabled with empty name returns nil",
			enablePluggableParser: false,
			customParserName:      "",
			want:                  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewParserFromConfig(tt.enablePluggableParser, tt.customParserName)

			if diff := cmp.Diff(tt.want, got, cmp.AllowUnexported(OpenAIParser{})); diff != "" {
				t.Errorf("NewParserFromConfig() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
