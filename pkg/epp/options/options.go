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

package options

import (
	"flag"
	"fmt"
	"io"
	"time"
)

// Flag defines parameters needed to manage command line flags.
type Flag struct {
	Name       string // CLI flag name.
	DefValue   any    // default value, required (to ensure Flag value type is defined).
	Usage      string // help text.
	Deprecated bool   // optional mark as deprecated.
	ReplacedBy string // optional replacement message.
}

// AddFlags registers a list of Flag definitions with a FlagSet (defaulting to
// flag.CommandLine if unspecified)), binding them to the pointer variables provided
// in the vars map.
func AddFlags(fs *flag.FlagSet, flags []Flag, vars map[string]any) error {
	if len(flags) != len(vars) {
		return fmt.Errorf("mismatch flags (%d) and vars (%d) count", len(flags), len(vars))
	}

	if fs == nil {
		fs = flag.CommandLine
	}

	for _, f := range flags {
		if f.DefValue == nil { // a default value is required to determine types
			return fmt.Errorf("flag %q must have a non-nil default value", f.Name)
		}

		ptr, ok := vars[f.Name]
		if !ok { // no destination variable
			return fmt.Errorf("variable pointer for flag %q not provided", f.Name)
		}

		switch def := f.DefValue.(type) {
		case string:
			p, ok := ptr.(*string)
			if !ok {
				return typeError(f.Name, ptr, "string")
			}
			*p = def
			fs.StringVar(p, f.Name, def, f.Usage)
		case int:
			p, ok := ptr.(*int)
			if !ok {
				return typeError(f.Name, ptr, "int")
			}
			*p = def
			fs.IntVar(p, f.Name, def, f.Usage)
		case bool:
			p, ok := ptr.(*bool)
			if !ok {
				return typeError(f.Name, ptr, "bool")
			}
			*p = def
			fs.BoolVar(p, f.Name, def, f.Usage)
		case time.Duration:
			p, ok := ptr.(*time.Duration)
			if !ok {
				return typeError(f.Name, ptr, "time.Duration")
			}
			*p = def
			fs.DurationVar(p, f.Name, def, f.Usage)
		default:
			return fmt.Errorf("unsupported flag type for %q: %T", f.Name, def)
		}

		if f.Deprecated { // wrap the value with deprecation warning
			fl := fs.Lookup(f.Name)
			if fl == nil {
				return fmt.Errorf("failed to lookup deprecated flag %q in set", f.Name)
			}
			fl.Value = &deprecatedValue{
				Value:      fl.Value,
				name:       f.Name,
				writer:     fs.Output(),
				replacedBy: f.ReplacedBy,
			}
		}
	}

	return nil
}

// deprecatedValue wraps a standard flag.Value to inject a warning message
// when the deprecated flag is used.
type deprecatedValue struct {
	flag.Value
	warned     bool
	name       string
	replacedBy string
	writer     io.Writer
}

// Set is called when the flag is parsed from the command line.
func (d *deprecatedValue) Set(s string) error {
	err := d.Value.Set(s) // delegate to the flag.Value

	if err == nil && !d.warned {
		d.warned = true
		if d.replacedBy != "" {
			fmt.Fprintf(d.writer, "Warning: --%s is deprecated; use %s instead.\n", d.name, d.replacedBy)
		} else {
			fmt.Fprintf(d.writer, "Warning: --%s is deprecated and will be removed in an upcoming release.\n", d.name)
		}
	}
	return err
}

// typeError creates a clear error message for flag type mismatches.
func typeError(name string, got any, expected string) error {
	return fmt.Errorf("flag %q: variable must be *%s, got %T", name, expected, got)
}

// GetStringFlagValue retrieves the current value (default or set) of a string flag
// by name from the specified FlagSet (or flag.CommandLine if nil).
func GetStringFlagValue(fs *flag.FlagSet, name string) (string, error) {
	if fs == nil {
		fs = flag.CommandLine
	}

	f := fs.Lookup(name)
	if f == nil {
		return "", fmt.Errorf("flag not found: %s", name)
	}
	val, ok := f.Value.(flag.Getter)
	if !ok {
		return "", fmt.Errorf("flag %s value does not support flag.Getter interface", name)
	}
	underlying := val.Get()
	strptr, ok := underlying.(*string)
	if !ok {
		return "", fmt.Errorf("flag %s is not a string type, got %T", name, underlying)
	}
	return *strptr, nil
}

// GetBoolFlagValue retrieves the current value (default or set) of a boolean flag
// by name from the specified FlagSet (or flag.CommandLine if nil).
func GetBoolFlagValue(fs *flag.FlagSet, name string) (bool, error) {
	if fs == nil {
		fs = flag.CommandLine
	}

	f := fs.Lookup(name)
	if f == nil {
		return false, fmt.Errorf("flag not found: %s", name)
	}
	val, ok := f.Value.(flag.Getter)
	if !ok {
		return false, fmt.Errorf("flag %s value does not support flag.Getter interface", name)
	}
	underlying := val.Get()
	boolptr, ok := underlying.(*bool)
	if !ok {
		return false, fmt.Errorf("flag %s is not a bool type, got %T", name, underlying)
	}
	return *boolptr, nil
}
