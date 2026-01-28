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

package plugins

import (
	"encoding/json"
	"errors"
	"strings"
)

// BBRPluginSpec implements flag.Value interface and defines a repeatable configuration block specified in CLI: --plugin <type>:<name>:<json>
type BBRPluginSpec struct {
	Type string
	Name string
	JSON json.RawMessage // raw JSON representing parameters for the plugin factory
	Raw  string          // original parameters string (for error messages)
}

// BBRPluginSpecs Slice (because the plugin flag is repeatable)
type BBRPluginSpecs []BBRPluginSpec

func (p *BBRPluginSpecs) Set(s string) error {
	spec := BBRPluginSpec{Raw: s}

	// Accept either:
	//   <type>:<name>
	//   <type>:<name>:<json>
	segments := strings.SplitN(s, ":", 3)
	if len(segments) < 2 {
		return errors.New(`usage: --plugin <type>:<name>:<json>`)
	}
	spec.Type = strings.TrimSpace(segments[0])
	spec.Name = strings.TrimSpace(segments[1])

	if spec.Type == "" {
		return errors.New("bbr plugin type cannot be empty")
	}
	if spec.Name == "" {
		return errors.New("bbr plugin name cannot be empty")
	}

	// If the third segment is provided, validate JSON if non-empty.
	if len(segments) == 3 {
		jsonSegment := strings.TrimSpace(segments[2])
		// Allow empty JSON segment (means "no parameters").
		if jsonSegment != "" {
			if !json.Valid([]byte(jsonSegment)) {
				return errors.New("invalid json")
			}
		}
		spec.JSON = json.RawMessage(jsonSegment)
	}

	*p = append(*p, spec)
	return nil
}

func (p *BBRPluginSpecs) String() string {
	specs := *p
	out := make([]string, 0, len(specs)) // preallocate memory

	for _, s := range *p {
		out = append(out, s.Raw)
	}
	return strings.Join(out, " ")
}
