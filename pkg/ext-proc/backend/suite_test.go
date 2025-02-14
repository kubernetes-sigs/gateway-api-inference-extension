package backend

import (
	"flag"
	"os"
	"testing"

	klog "k8s.io/klog/v2"
)

// Turn on klog in test so we can observe the behavior, which is good
// for debugging and catching error.
func TestMain(m *testing.M) {
	klog.InitFlags(nil)
	_ = flag.Set("logtostderr", "true") // Log to stderr instead of files
	_ = flag.Set("v", "4")              // Log level
	flag.Parse()

	os.Exit(m.Run())
}
