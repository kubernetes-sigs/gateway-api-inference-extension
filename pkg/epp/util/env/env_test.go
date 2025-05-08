package env

import (
	"os"
	"testing"

	"github.com/go-logr/logr/testr"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func TestGetEnvFloat(t *testing.T) {
	logger := testr.New(t)

	tests := []struct {
		name       string
		key        string
		value      string
		defaultVal float64
		expected   float64
		setup      func()
		teardown   func()
	}{
		{
			name:       "env variable exists and is valid",
			key:        "TEST_FLOAT",
			value:      "123.456",
			defaultVal: 0.0,
			expected:   123.456,
			setup: func() {
				os.Setenv("TEST_FLOAT", "123.456")
			},
			teardown: func() {
				os.Unsetenv("TEST_FLOAT")
			},
		},
		{
			name:       "env variable exists but is invalid",
			key:        "TEST_FLOAT",
			value:      "invalid",
			defaultVal: 99.9,
			expected:   99.9,
			setup: func() {
				os.Setenv("TEST_FLOAT", "invalid")
			},
			teardown: func() {
				os.Unsetenv("TEST_FLOAT")
			},
		},
		{
			name:       "env variable does not exist",
			key:        "TEST_FLOAT_MISSING",
			defaultVal: 42.42,
			expected:   42.42,
			setup:      func() {},
			teardown:   func() {},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.setup()
			defer tc.teardown()

			result := GetEnvFloat(tc.key, tc.defaultVal, logger.V(logutil.VERBOSE))
			if result != tc.expected {
				t.Errorf("GetEnvFloat(%s, %f) = %f, expected %f", tc.key, tc.defaultVal, result, tc.expected)
			}
		})
	}
}

func TestGetEnvInt(t *testing.T) {
	logger := testr.New(t)

	tests := []struct {
		name       string
		key        string
		value      string
		defaultVal int
		expected   int
		setup      func()
		teardown   func()
	}{
		{
			name:       "env variable exists and is valid",
			key:        "TEST_INT",
			value:      "123",
			defaultVal: 0,
			expected:   123,
			setup: func() {
				os.Setenv("TEST_INT", "123")
			},
			teardown: func() {
				os.Unsetenv("TEST_INT")
			},
		},
		{
			name:       "env variable exists but is invalid",
			key:        "TEST_INT",
			value:      "invalid",
			defaultVal: 99,
			expected:   99,
			setup: func() {
				os.Setenv("TEST_INT", "invalid")
			},
			teardown: func() {
				os.Unsetenv("TEST_INT")
			},
		},
		{
			name:       "env variable does not exist",
			key:        "TEST_INT_MISSING",
			defaultVal: 42,
			expected:   42,
			setup:      func() {},
			teardown:   func() {},
		},
		{
			name:       "env variable is empty string",
			key:        "TEST_INT_EMPTY",
			value:      "",
			defaultVal: 77,
			expected:   77,
			setup: func() {
				os.Setenv("TEST_INT_EMPTY", "")
			},
			teardown: func() {
				os.Unsetenv("TEST_INT_EMPTY")
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.setup()
			defer tc.teardown()

			result := GetEnvInt(tc.key, tc.defaultVal, logger.V(logutil.VERBOSE))
			if result != tc.expected {
				t.Errorf("GetEnvInt(%s, %d) = %d, expected %d", tc.key, tc.defaultVal, result, tc.expected)
			}
		})
	}
}

func TestGetEnvString(t *testing.T) {
	logger := testr.New(t)

	tests := []struct {
		name       string
		key        string
		value      string
		defaultVal string
		expected   string
		setup      func()
		teardown   func()
	}{
		{
			name:       "env variable exists and is valid",
			key:        "TEST_STR",
			value:      "123",
			defaultVal: "default",
			expected:   "123",
			setup: func() {
				os.Setenv("TEST_STR", "123")
			},
			teardown: func() {
				os.Unsetenv("TEST_STR")
			},
		},
		{
			name:       "env variable does not exist",
			key:        "TEST_STR_MISSING",
			defaultVal: "default",
			expected:   "default",
			setup:      func() {},
			teardown:   func() {},
		},
		{
			name:       "env variable is empty string",
			key:        "TEST_STR_EMPTY",
			value:      "",
			defaultVal: "default",
			expected:   "",
			setup: func() {
				os.Setenv("TEST_STR_EMPTY", "")
			},
			teardown: func() {
				os.Unsetenv("TEST_STR_EMPTY")
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.setup()
			defer tc.teardown()

			result := GetEnvString(tc.key, tc.defaultVal, logger.V(logutil.VERBOSE))
			if result != tc.expected {
				t.Errorf("GetEnvString(%s, %s) = %s, expected %s", tc.key, tc.defaultVal, result, tc.expected)
			}
		})
	}
}
