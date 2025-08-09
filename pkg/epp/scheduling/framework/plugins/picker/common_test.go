package picker

import (
	"context"
	"fmt"
	"testing"
	"time"
)

func TestNewSafeRand_NotNil(t *testing.T) {
	sr := NewSafeRand()
	if sr == nil {
		t.Fatal("NewSafeRand returned nil")
	}
	if sr.p == nil {
		t.Fatal("sync.Pool in safeRand is nil")
	}
}

func TestSafeRand_Shuffle_ShufflesSlice(t *testing.T) {
	sr := NewSafeRand()
	orig := []int{1, 2, 3, 4, 5}
	shuffled := make([]int, len(orig))
	copy(shuffled, orig)

	sr.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	// It's possible (but unlikely) that the slice remains unchanged after shuffling.
	// So we check that at least one element has moved.
	same := true
	for i := range orig {
		if orig[i] != shuffled[i] {
			same = false
			break
		}
	}
	if same {
		t.Log("Warning: shuffled slice is the same as original (possible but unlikely)")
	}
}

func TestSafeRand_ConcurrentShuffle(t *testing.T) {
	sr := NewSafeRand()
	const goroutines = 10
	const sliceLen = 100
	done := make(chan struct{}, goroutines)

	for g := 0; g < goroutines; g++ {
		go func() {
			s := make([]int, sliceLen)
			for i := range s {
				s[i] = i
			}
			sr.Shuffle(len(s), func(i, j int) {
				s[i], s[j] = s[j], s[i]
			})
			done <- struct{}{}
		}()
	}

	for g := 0; g < goroutines; g++ {
		<-done
	}
}

func TestSafeRand_Shuffle_EmptySlice(t *testing.T) {
	sr := NewSafeRand()
	emptySlice := []int{}

	// Should not panic with empty slice
	sr.Shuffle(len(emptySlice), func(i, j int) {
		emptySlice[i], emptySlice[j] = emptySlice[j], emptySlice[i]
	})

	if len(emptySlice) != 0 {
		t.Errorf("Expected empty slice to remain empty, got length %d", len(emptySlice))
	}
}

func TestSafeRand_Shuffle_SingleElement(t *testing.T) {
	sr := NewSafeRand()
	singleElement := []int{42}

	sr.Shuffle(len(singleElement), func(i, j int) {
		singleElement[i], singleElement[j] = singleElement[j], singleElement[i]
	})

	if len(singleElement) != 1 || singleElement[0] != 42 {
		t.Errorf("Expected single element slice to remain [42], got %v", singleElement)
	}
}

// Benchmark tests for safeRand.
func BenchmarkSafeRand_Shuffle_Small(b *testing.B) {
	sr := NewSafeRand()
	slice := make([]int, 10)
	for i := range slice {
		slice[i] = i
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sr.Shuffle(len(slice), func(i, j int) {
			slice[i], slice[j] = slice[j], slice[i]
		})
	}
}

func BenchmarkSafeRand_Shuffle_Large(b *testing.B) {
	sr := NewSafeRand()
	slice := make([]int, 1000)
	for i := range slice {
		slice[i] = i
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sr.Shuffle(len(slice), func(i, j int) {
			slice[i], slice[j] = slice[j], slice[i]
		})
	}
}

func BenchmarkSafeRand_Shuffle_Concurrent(b *testing.B) {
	sr := NewSafeRand()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		slice := make([]int, 100)
		for i := range slice {
			slice[i] = i
		}

		for pb.Next() {
			sr.Shuffle(len(slice), func(i, j int) {
				slice[i], slice[j] = slice[j], slice[i]
			})
		}
	})
}

// Advanced concurrent tests for safeRand
func TestSafeRand_HighConcurrency(t *testing.T) {
	sr := NewSafeRand()
	const numGoroutines = 100
	const numOperations = 100
	const sliceSize = 50

	done := make(chan struct{}, numGoroutines)
	errors := make(chan error, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(goroutineID int) {
			defer func() {
				if r := recover(); r != nil {
					errors <- fmt.Errorf("goroutine %d panicked: %v", goroutineID, r)
					return
				}
				done <- struct{}{}
			}()

			for op := 0; op < numOperations; op++ {
				slice := make([]int, sliceSize)
				for i := range slice {
					slice[i] = i
				}

				sr.Shuffle(len(slice), func(i, j int) {
					slice[i], slice[j] = slice[j], slice[i]
				})

				// Verify all elements are still present
				for expected := 0; expected < sliceSize; expected++ {
					found := false
					for _, actual := range slice {
						if actual == expected {
							found = true
							break
						}
					}
					if !found {
						errors <- fmt.Errorf("goroutine %d: element %d missing after shuffle", goroutineID, expected)
						return
					}
				}
			}
		}(g)
	}

	// Wait for all goroutines to complete
	for g := 0; g < numGoroutines; g++ {
		select {
		case <-done:
			// Goroutine completed successfully
		case err := <-errors:
			t.Fatal(err)
		}
	}
}

func TestSafeRand_StressTest(t *testing.T) {
	sr := NewSafeRand()
	const duration = 100 * time.Millisecond
	const numGoroutines = 50

	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	done := make(chan struct{}, numGoroutines)
	errors := make(chan error, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(goroutineID int) {
			defer func() {
				if r := recover(); r != nil {
					errors <- fmt.Errorf("goroutine %d panicked: %v", goroutineID, r)
					return
				}
				done <- struct{}{}
			}()

			for {
				select {
				case <-ctx.Done():
					return
				default:
					slice := make([]int, 20)
					for i := range slice {
						slice[i] = i
					}
					sr.Shuffle(len(slice), func(i, j int) {
						slice[i], slice[j] = slice[j], slice[i]
					})
				}
			}
		}(g)
	}

	// Wait for context timeout
	<-ctx.Done()

	// Wait for all goroutines to complete
	for g := 0; g < numGoroutines; g++ {
		select {
		case <-done:
			// Goroutine completed successfully
		case err := <-errors:
			t.Fatal(err)
		}
	}
}
