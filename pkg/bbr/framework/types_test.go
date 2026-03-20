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

package framework

import (
	"testing"
)

func TestSetBodyField(t *testing.T) {
	msg := newInferenceMessage()
	if msg.BodyMutated() {
		t.Error("new message should not be marked as body-mutated")
	}

	msg.SetBodyField("key", "value")

	if !msg.BodyMutated() {
		t.Error("expected BodyMutated() to return true after SetBodyField")
	}
	if got, ok := msg.Body["key"]; !ok || got != "value" {
		t.Errorf("Body[\"key\"] = %v, %v; want \"value\", true", got, ok)
	}
}

func TestSetBodyField_Overwrite(t *testing.T) {
	msg := newInferenceMessage()
	msg.Body["existing"] = "old"

	msg.SetBodyField("existing", "new")

	if !msg.BodyMutated() {
		t.Error("expected BodyMutated() to return true after overwriting a field")
	}
	if got := msg.Body["existing"]; got != "new" {
		t.Errorf("Body[\"existing\"] = %v; want \"new\"", got)
	}
}

func TestRemoveBodyField(t *testing.T) {
	msg := newInferenceMessage()
	msg.Body["key"] = "value"

	msg.RemoveBodyField("key")

	if !msg.BodyMutated() {
		t.Error("expected BodyMutated() to return true after RemoveBodyField")
	}
	if _, ok := msg.Body["key"]; ok {
		t.Error("expected key to be removed from Body")
	}
}

func TestRemoveBodyField_NonExistent(t *testing.T) {
	msg := newInferenceMessage()

	msg.RemoveBodyField("missing")

	if msg.BodyMutated() {
		t.Error("removing a non-existent field should not mark body as mutated")
	}
}

func TestSetBody(t *testing.T) {
	msg := newInferenceMessage()

	msg.SetBody(map[string]any{"model": "llama", "prompt": "hello"})

	if !msg.BodyMutated() {
		t.Error("expected BodyMutated() to return true after SetBody")
	}
	if got, ok := msg.Body["model"]; !ok || got != "llama" {
		t.Errorf("Body[\"model\"] = %v, %v; want \"llama\", true", got, ok)
	}
	if got, ok := msg.Body["prompt"]; !ok || got != "hello" {
		t.Errorf("Body[\"prompt\"] = %v, %v; want \"hello\", true", got, ok)
	}
}

func TestBodyMutated_FalseByDefault(t *testing.T) {
	req := NewInferenceRequest()
	if req.BodyMutated() {
		t.Error("new InferenceRequest should not be marked as body-mutated")
	}

	resp := NewInferenceResponse()
	if resp.BodyMutated() {
		t.Error("new InferenceResponse should not be marked as body-mutated")
	}
}
