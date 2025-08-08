package picker

import (
	"math/rand/v2"
	"testing"
	"time"
)

// TestNewSafeRand_NilRand checks that NewSafeRand initializes a non-nil rand.Rand when passed nil.
func TestNewSafeRand_NilRand(t *testing.T) {
	sr := NewSafeRand(nil)
	if sr == nil || sr.r == nil {
		t.Errorf("Expected non-nil safeRand and rand.Rand, got nil")
	}
}

// TestNewSafeRand_CustomRand checks that NewSafeRand uses the provided rand.Rand.
func TestNewSafeRand_CustomRand(t *testing.T) {
	src := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 42))
	sr := NewSafeRand(src)
	if sr.r != src {
		t.Errorf("Expected safeRand to use provided rand.Rand")
	}
}

// TestSafeRand_Uint64_ThreadSafe checks that Uint64 returns a value and is thread-safe.
func TestSafeRand_Uint64_ThreadSafe(t *testing.T) {
	sr := NewSafeRand(nil)
	const goroutines = 10
	results := make(chan uint64, goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			results <- sr.Uint64()
		}()
	}
	seen := make(map[uint64]bool)
	for i := 0; i < goroutines; i++ {
		val := <-results
		if seen[val] {
			t.Errorf("Duplicate value %v from Uint64, possible lack of randomness", val)
		}
		seen[val] = true
	}
}

// TestSafeRand_Shuffle checks that Shuffle shuffles a slice.
func TestSafeRand_Shuffle(t *testing.T) {
	sr := NewSafeRand(nil)
	orig := []int{1, 2, 3, 4, 5}
	shuffled := make([]int, len(orig))
	copy(shuffled, orig)
	sr.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	// It's possible (but unlikely) that the slice remains unchanged after shuffling.
	same := true
	for i := range orig {
		if orig[i] != shuffled[i] {
			same = false
			break
		}
	}
	if same {
		t.Log("Shuffle did not change the slice; this is rare but possible")
	}
}

// TestSafeRand_Shuffle_ThreadSafe checks Shuffle is thread-safe.
func TestSafeRand_Shuffle_ThreadSafe(t *testing.T) {
	sr := NewSafeRand(nil)
	const goroutines = 5
	done := make(chan bool, goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			arr := []int{1, 2, 3, 4, 5}
			sr.Shuffle(len(arr), func(i, j int) {
				arr[i], arr[j] = arr[j], arr[i]
			})
			done <- true
		}()
	}
	for i := 0; i < goroutines; i++ {
		<-done
	}
}
