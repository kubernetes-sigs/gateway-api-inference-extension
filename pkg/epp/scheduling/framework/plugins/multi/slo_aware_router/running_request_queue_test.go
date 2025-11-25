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

package slo_aware_router

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestNewRequestPriorityQueue(t *testing.T) {
	pq := newRequestPriorityQueue()

	if pq == nil {
		t.Fatal("NewRequestPriorityQueue returned nil")
	}

	if pq.GetSize() != 0 {
		t.Errorf("Expected empty queue, got size %d", pq.GetSize())
	}

	if pq.Peek() != nil {
		t.Error("Expected nil from Peek on empty queue")
	}
}

func TestAdd(t *testing.T) {
	pq := newRequestPriorityQueue()

	// Test successful add
	if !pq.Add("req1", 2.5) {
		t.Error("Expected Add to return true for new item")
	}

	if pq.GetSize() != 1 {
		t.Errorf("Expected size 1, got %d", pq.GetSize())
	}

	// Test duplicate add
	if pq.Add("req1", 3.0) {
		t.Error("Expected Add to return false for duplicate ID")
	}

	if pq.GetSize() != 1 {
		t.Errorf("Expected size 1 after duplicate add, got %d", pq.GetSize())
	}

	// Test validation
	if pq.Add("", 1.0) {
		t.Error("Expected Add to return false for empty ID")
	}

	if pq.Add("req2", -1.0) {
		t.Error("Expected Add to return false for negative TPOT")
	}
}

func TestPriorityOrdering(t *testing.T) {
	pq := newRequestPriorityQueue()

	// Add items with different priorities
	pq.Add("high", 1.0)   // highest priority (lowest TPOT)
	pq.Add("medium", 5.0) // medium priority
	pq.Add("low", 10.0)   // lowest priority (highest TPOT)

	// Check that highest priority item is at the top
	peek := pq.Peek()
	if peek == nil || peek.id != "high" || peek.tpot != 1.0 {
		t.Errorf("Expected high priority item at top, got %+v", peek)
	}

	// Test removal order
	expected := []struct {
		id   string
		tpot float64
	}{
		{"high", 1.0},
		{"medium", 5.0},
		{"low", 10.0},
	}

	for _, exp := range expected {
		item := pq.Peek()
		if item.id != exp.id || item.tpot != exp.tpot {
			t.Errorf("Expected %s(%.1f), got %s(%.1f)", exp.id, exp.tpot, item.id, item.tpot)
		}

		removed, ok := pq.Remove(item.id)
		if !ok || removed.id != exp.id {
			t.Errorf("Failed to remove %s", exp.id)
		}
	}
}

func TestRemove(t *testing.T) {
	pq := newRequestPriorityQueue()

	// Test remove from empty queue
	if _, ok := pq.Remove("nonexistent"); ok {
		t.Error("Expected Remove to return false for empty queue")
	}

	// Add some items
	pq.Add("req1", 1.0)
	pq.Add("req2", 2.0)
	pq.Add("req3", 3.0)

	// Test successful remove
	removed, ok := pq.Remove("req2")
	if !ok || removed.id != "req2" || removed.tpot != 2.0 {
		t.Errorf("Expected to remove req2(2.0), got %+v, ok=%v", removed, ok)
	}

	if pq.GetSize() != 2 {
		t.Errorf("Expected size 2 after removal, got %d", pq.GetSize())
	}

	// Test remove nonexistent
	if _, ok := pq.Remove("req2"); ok {
		t.Error("Expected Remove to return false for already removed item")
	}

	// Verify remaining items are still in correct order
	if peek := pq.Peek(); peek.id != "req1" {
		t.Errorf("Expected req1 at top, got %s", peek.id)
	}
}

func TestUpdate(t *testing.T) {
	pq := newRequestPriorityQueue()

	// Test update nonexistent item
	if pq.Update("nonexistent", 1.0) {
		t.Error("Expected Update to return false for nonexistent item")
	}

	// Add items
	pq.Add("req1", 1.0)
	pq.Add("req2", 2.0)
	pq.Add("req3", 3.0)

	// Update to make req3 highest priority
	if !pq.Update("req3", 0.5) {
		t.Error("Expected Update to return true for existing item")
	}

	// Check that req3 is now at the top
	if peek := pq.Peek(); peek.id != "req3" || peek.tpot != 0.5 {
		t.Errorf("Expected req3(0.5) at top, got %s(%.1f)", peek.id, peek.tpot)
	}

	// Test validation
	if pq.Update("req1", -1.0) {
		t.Error("Expected Update to return false for negative TPOT")
	}
}

func TestContains(t *testing.T) {
	pq := newRequestPriorityQueue()

	// Test empty queue
	if pq.Contains("req1") {
		t.Error("Expected Contains to return false for empty queue")
	}

	// Add item
	pq.Add("req1", 1.0)

	// Test existing item
	if !pq.Contains("req1") {
		t.Error("Expected Contains to return true for existing item")
	}

	// Test nonexistent item
	if pq.Contains("req2") {
		t.Error("Expected Contains to return false for nonexistent item")
	}

	// Test after removal
	pq.Remove("req1")
	if pq.Contains("req1") {
		t.Error("Expected Contains to return false after removal")
	}
}

func TestClone(t *testing.T) {
	pq := newRequestPriorityQueue()

	// Test clone of empty queue
	clone := pq.Clone()
	if clone.GetSize() != 0 {
		t.Error("Expected cloned empty queue to be empty")
	}

	// Add items to original
	pq.Add("req1", 1.0)
	pq.Add("req2", 2.0)
	pq.Add("req3", 3.0)

	// Clone with items
	clone = pq.Clone()

	// Verify clone has same items
	if clone.GetSize() != pq.GetSize() {
		t.Errorf("Expected clone size %d, got %d", pq.GetSize(), clone.GetSize())
	}

	// Verify independence - modify original
	pq.Add("req4", 4.0)
	if clone.GetSize() == pq.GetSize() {
		t.Error("Clone should be independent of original")
	}

	// Verify independence - modify clone
	clone.Remove("req1")
	if !pq.Contains("req1") {
		t.Error("Original should not be affected by clone modifications")
	}

	// Verify deep copy - items should be different instances
	origPeek := pq.Peek()
	clonePeek := clone.Peek()
	if origPeek == clonePeek {
		t.Error("Clone should create new Request instances, not share pointers")
	}
}

func TestString(t *testing.T) {
	pq := newRequestPriorityQueue()

	// Test empty queue
	str := pq.String()
	expected := "RequestPriorityQueue: []"
	if str != expected {
		t.Errorf("Expected %q, got %q", expected, str)
	}

	// Test with items
	pq.Add("req1", 1.5)
	pq.Add("req2", 2.25)

	str = pq.String()
	// Should contain both items in priority order
	if !contains(str, "req1(1.50)") || !contains(str, "req2(2.25)") {
		t.Errorf("String output missing expected items: %s", str)
	}
}

func TestConcurrency(t *testing.T) {
	pq := newRequestPriorityQueue()
	const numWorkers = 10
	const itemsPerWorker = 100

	var wg sync.WaitGroup

	// Launch workers that add items
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < itemsPerWorker; j++ {
				id := fmt.Sprintf("worker%d-item%d", workerID, j)
				tpot := float64(j) + float64(workerID)*0.1
				pq.Add(id, tpot)
			}
		}(i)
	}

	// Launch workers that read from the queue
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < itemsPerWorker/2; j++ {
				pq.Peek()
				pq.GetSize()
				time.Sleep(time.Microsecond)
			}
		}()
	}

	wg.Wait()

	// Verify final state
	expectedSize := numWorkers * itemsPerWorker
	if pq.GetSize() != expectedSize {
		t.Errorf("Expected final size %d, got %d", expectedSize, pq.GetSize())
	}
}

func TestLargeQueue(t *testing.T) {
	pq := newRequestPriorityQueue()
	const numItems = 10000

	// Add many items
	for i := 0; i < numItems; i++ {
		id := fmt.Sprintf("item%d", i)
		tpot := float64(numItems - i) // Reverse order so item0 has highest priority
		pq.Add(id, tpot)
	}

	if pq.GetSize() != numItems {
		t.Errorf("Expected size %d, got %d", numItems, pq.GetSize())
	}

	// Verify priority ordering by removing items
	lastTPOT := -1.0
	for i := 0; i < numItems; i++ {
		item := pq.Peek()
		if item.tpot < lastTPOT {
			t.Errorf("Priority order violated: %.1f < %.1f", item.tpot, lastTPOT)
		}
		lastTPOT = item.tpot
		pq.Remove(item.id)
	}

	if pq.GetSize() != 0 {
		t.Errorf("Expected empty queue after removing all items, got size %d", pq.GetSize())
	}
}

func BenchmarkAdd(b *testing.B) {
	pq := newRequestPriorityQueue()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := fmt.Sprintf("item%d", i)
		pq.Add(id, float64(i))
	}
}

func BenchmarkPeek(b *testing.B) {
	pq := newRequestPriorityQueue()

	// Pre-populate queue
	for i := 0; i < 1000; i++ {
		pq.Add(fmt.Sprintf("item%d", i), float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pq.Peek()
	}
}

func BenchmarkRemove(b *testing.B) {
	pq := newRequestPriorityQueue()

	// Pre-populate queue
	for i := 0; i < b.N; i++ {
		pq.Add(fmt.Sprintf("item%d", i), float64(i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pq.Remove(fmt.Sprintf("item%d", i))
	}
}

// Helper function to check if a string contains a substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) &&
		(s == substr ||
			s[:len(substr)] == substr ||
			s[len(s)-len(substr):] == substr ||
			containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
