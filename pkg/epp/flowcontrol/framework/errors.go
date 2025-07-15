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

package framework

import (
	"errors"
)

// `SafeQueue` Errors
//
// These errors relate to operations directly on a `SafeQueue` implementation. They are returned by `SafeQueue` methods
// and might be handled or wrapped by the `ports.FlowRegistry`'s `ports.ManagedQueue` or the
// `controller.FlowController`.
var (
	// ErrNilQueueItem indicates that a nil `types.QueueItemAccessor` was passed to `SafeQueue.Add()`.
	ErrNilQueueItem = errors.New("queue item cannot be nil")

	// ErrQueueEmpty indicates an attempt to perform an operation on an empty `SafeQueue` that requires one or more items
	// (e.g., calling `SafeQueue.PeekHead()`).
	ErrQueueEmpty = errors.New("queue is empty")

	// ErrInvalidQueueItemHandle indicates that a `types.QueueItemHandle` provided to a `SafeQueue` operation (e.g.,
	// `SafeQueue.Remove()`) is not valid for that queue, has been invalidated, or does not correspond to an actual item
	// in the queue.
	ErrInvalidQueueItemHandle = errors.New("invalid queue item handle")

	// ErrQueueItemNotFound indicates that a `SafeQueue.Remove(handle)` operation did not find an item matching the
	// provided, valid `types.QueueItemHandle`. This can occur if the item was removed by a concurrent operation.
	ErrQueueItemNotFound = errors.New("queue item not found for the given handle")
)
