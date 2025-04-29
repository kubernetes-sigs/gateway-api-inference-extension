/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package prefix

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

type linkedList struct {
	dummyHead *node // The head of the linked list (dummy node).
	tail      *node // The tail of the linked list.
	size      int   // The size of the linked list (excluding dummy head).
}

// newLinkedList initializes a new linked list with a dummy head node.
// Using a dummy head simplifies the implementation by eliminating nil checks.
func newLinkedList() *linkedList {
	dummy := &node{} // Create dummy head node
	return &linkedList{
		dummyHead: dummy,
		tail:      dummy,
		size:      0,
	}
}

type node struct {
	prev   *node
	next   *node
	server types.ServerID
	hash   types.BlockHash
}

// add adds a node to the end of the linked list.
func (ll *linkedList) add(n *node) {
	ll.size++

	n.prev = ll.tail
	ll.tail.next = n
	ll.tail = n
}

// delete removes a node from the linked list.
// Note the method assumes the input node exists in the list.
func (ll *linkedList) delete(n *node) {
	ll.size--
	n.prev.next = n.next

	// If it's the tail node
	if n.next == nil {
		ll.tail = n.prev
	} else {
		n.next.prev = n.prev
	}
}

// moveToTail moves an existing node to the end of the linked list (most recent).
func (ll *linkedList) moveToTail(n *node) {
	if n.next == nil {
		// Already the tail, no need to move.
		return
	}

	n.prev.next = n.next
	n.next.prev = n.prev

	// Move it to the tail position
	n.prev = ll.tail
	n.next = nil
	ll.tail.next = n
	ll.tail = n
}
