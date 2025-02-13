package logging

import "k8s.io/klog/v2"

// Fatal calls klog.ErrorS followed by klog.FlushAndExit(1).
//
// This is a utility function and should not be used in production code!
func Fatal(err error, msg string, keysAndValues ...interface{}) {
	klog.ErrorS(err, msg, keysAndValues...)
	klog.FlushAndExit(klog.ExitFlushTimeout, 1)
}
