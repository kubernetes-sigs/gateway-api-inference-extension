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

package plugins

const (
	separator = "/"
)

// TypedName is a utility struct providing a type and a name to plugins.
type TypedName struct {
	// Type returns the type of a plugin.
	Type string
	// Name returns the name of a plugin instance.
	Name string
}

// String returns the type and name rendered as "<name>/<type>".
func (tn TypedName) String() string {
	return tn.Name + separator + tn.Type
}
