package backend

import (
	"container/heap"
	"fmt"
	"strings"
	"sync"
)

// Request represents an element in the priority queue.
// The index is needed by heap.Remove and is maintained by the heap.Interface methods.
type Request struct {
	ID    string  // Unique identifier
	TPOT  float64 // The priority value (lower is higher priority)
	index int
}

// RequestPriorityQueue implements a priority queue with item removal by ID.
type RequestPriorityQueue struct {
	items  []*Request
	lookup map[string]*Request
	mutex  sync.RWMutex
}

// NewRequestPriorityQueue initializes and returns a new PriorityQueue.
func NewRequestPriorityQueue() *RequestPriorityQueue {
	return &RequestPriorityQueue{
		lookup: make(map[string]*Request),
		items:  []*Request{},
	}
}

// Clone creates a deep copy of the priority queue.
// The new queue is completely independent of the original.
func (pq *RequestPriorityQueue) Clone() *RequestPriorityQueue {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()
	
	// Initialize a new priority queue with pre-allocated capacity.
	clonedPq := &RequestPriorityQueue{
		items:  make([]*Request, len(pq.items)),
		lookup: make(map[string]*Request, len(pq.lookup)),
	}

	// Iterate through the original items to create deep copies.
	for i, oldItem := range pq.items {
		// Create a new Request struct, copying all values.
		newItem := &Request{
			ID:    oldItem.ID,
			TPOT:  oldItem.TPOT,
			index: oldItem.index,
		}

		// Assign the new item to the cloned queue's items slice.
		clonedPq.items[i] = newItem
		// Update the lookup map in the cloned queue to point to the new item.
		clonedPq.lookup[newItem.ID] = newItem
	}

	return clonedPq
}

// Len is the number of items in the queue.
func (pq *RequestPriorityQueue) Len() int { return len(pq.items) }

// Less reports whether the item with index i should sort before the item with index j.
func (pq *RequestPriorityQueue) Less(i, j int) bool {
	return pq.items[i].TPOT < pq.items[j].TPOT
}

// Swap swaps the items with indexes i and j.
func (pq *RequestPriorityQueue) Swap(i, j int) {
	pq.items[i], pq.items[j] = pq.items[j], pq.items[i]
	pq.items[i].index = i
	pq.items[j].index = j
}

// Push adds an item to the heap.
func (pq *RequestPriorityQueue) Push(x any) {
	item := x.(*Request)
	item.index = len(pq.items)
	pq.items = append(pq.items, item)
}

// Pop removes and returns the minimum item from the heap.
func (pq *RequestPriorityQueue) Pop() any {
	n := len(pq.items)
	item := pq.items[n-1]
	pq.items[n-1] = nil // avoid memory leak
	item.index = -1     // for safety
	pq.items = pq.items[0 : n-1]
	return item
}

// Add adds a new item to the queue.
// Returns true if the item was added, false if an item with the same ID already exists.
func (pq *RequestPriorityQueue) Add(id string, tpot float64) bool {
	pq.mutex.Lock()
	defer pq.mutex.Unlock()
	
	// Validate input
	if id == "" {
		return false
	}
	if tpot < 0 {
		return false
	}
	
	// If item already exists, do not add
	if _, exists := pq.lookup[id]; exists {
		return false
	}
	
	item := &Request{
		ID:   id,
		TPOT: tpot,
	}
	pq.lookup[id] = item
	heap.Push(pq, item)
	return true
}

// Update modifies the TPOT value of an existing item in the queue.
// If the item doesn't exist, this method does nothing.
func (pq *RequestPriorityQueue) Update(id string, tpot float64) bool {
	pq.mutex.Lock()
	defer pq.mutex.Unlock()
	
	// Validate input
	if tpot < 0 {
		return false
	}
	
	item, exists := pq.lookup[id]
	if !exists {
		return false
	}
	
	item.TPOT = tpot
	heap.Fix(pq, item.index)
	return true
}

// Remove removes an item from the queue by its ID.
func (pq *RequestPriorityQueue) Remove(id string) (*Request, bool) {
	pq.mutex.Lock()
	defer pq.mutex.Unlock()
	
	item, ok := pq.lookup[id]
	if !ok {
		return nil, false
	}
	removed := heap.Remove(pq, item.index).(*Request)
	delete(pq.lookup, id)
	return removed, true
}

// Peek returns the item with the lowest value without removing it.
func (pq *RequestPriorityQueue) Peek() *Request {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()
	
	if len(pq.items) == 0 {
		return nil
	}
	return pq.items[0]
}

// GetSize returns the current number of items in the queue.
func (pq *RequestPriorityQueue) GetSize() int {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()
	return len(pq.items)
}

// Contains checks if an item with the given ID exists in the queue.
func (pq *RequestPriorityQueue) Contains(id string) bool {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()
	_, exists := pq.lookup[id]
	return exists
}

// String returns a string representation of the queue for debugging.
func (pq *RequestPriorityQueue) String() string {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()
	
	if len(pq.items) == 0 {
		return "RequestPriorityQueue: []"
	}
	
	var builder strings.Builder
	builder.WriteString("RequestPriorityQueue: [")
	
	for i, item := range pq.items {
		if i > 0 {
			builder.WriteString(", ")
		}
		builder.WriteString(item.ID)
		builder.WriteString("(")
		builder.WriteString(fmt.Sprintf("%.2f", item.TPOT))
		builder.WriteString(")")
	}
	
	builder.WriteString("]")
	return builder.String()
}