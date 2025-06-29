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

// TypedName is a utility struct providing a type and a name
// to plugins.
// It implements the Plugin interface and can be embedded in
// plugins across the code to reduce boilerplate.
type TypedName struct {
	objType string
	objName string
}

// NewTypedName returns a new TypedName struct configured with
// the given type and name.
func NewTypedName(objtype, objname string) TypedName {
	return TypedName{
		objType: objtype,
		objName: objname,
	}
}

// Type returns the type of the plugin.
func (tn *TypedName) Type() string {
	return tn.objType
}

// Name returns the name of this plugin instance.
func (tn *TypedName) Name() string {
	return tn.objName
}

// SetName sets the instance name.
func (tn *TypedName) SetName(name string) {
	tn.objName = name
}
