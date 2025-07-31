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

package mocks

import (
	"time"
)

// -- Ticker is a mock time source --
type Ticker struct {
	ch chan time.Time
}

func NewTicker() *Ticker {
	return &Ticker{
		ch: make(chan time.Time, 10),
	}
}

func (t *Ticker) Channel() <-chan time.Time {
	return t.ch
}

func (t *Ticker) Tick() {
	select {
	case t.ch <- time.Now():
	default: // if buffer is full, or channel closed
	}
}

func (t *Ticker) Stop() {}
