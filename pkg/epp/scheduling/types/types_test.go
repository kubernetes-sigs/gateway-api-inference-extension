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
	"bytes"
	"testing"
)

func TestMarshalMessagesToJSON(t *testing.T) {
	testCases := []struct {
		name     string
		messages []Message
		want     []byte
		wantErr  bool
	}{
		{
			name:     "empty messages",
			messages: []Message{},
			want:     []byte{},
			wantErr:  false,
		},
		{
			name: "single message",
			messages: []Message{
				{Role: "user", Content: Content{Raw: "Hello"}},
			},
			want:    []byte(`{"role":"user","content":"Hello"},`),
			wantErr: false,
		},
		{
			name: "multiple messages",
			messages: []Message{
				{Role: "user", Content: Content{Raw: "Hello"}},
				{Role: "assistant", Content: Content{Raw: "Hi there!"}},
			},
			want:    []byte(`{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there!"},`),
			wantErr: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := MarshalMessagesToJSON(tc.messages...)
			if (err != nil) != tc.wantErr {
				t.Errorf("MarshalMessagesToJSON() error = %v, wantErr %v", err, tc.wantErr)
				return
			}

			if !bytes.Equal(got, tc.want) {
				t.Errorf("MarshalMessagesToJSON() got = %s, want %s", string(got), string(tc.want))
			}
		})
	}
}
