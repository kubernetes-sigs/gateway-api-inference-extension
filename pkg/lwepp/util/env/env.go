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

package env

import (
	"fmt"
	"os"
	"reflect"
	"strconv"
	"time"

	"github.com/go-logr/logr"
)

// getEnvWithParser retrieves an environment variable. If set, it uses the provided parser to parse it.
// It logs success or failure and returns the parsed value or the default value in case of a failure.
func getEnvWithParser[T any](key string, defaultVal T, parser func(string) (T, error), logger logr.Logger) T {
	valueStr, exists := os.LookupEnv(key)
	if !exists {
		logger.Info("Environment variable not set, using default value", "key", key, "defaultValue", defaultVal)
		return defaultVal
	}

	parsedValue, err := parser(valueStr)
	if err != nil {
		logger.Info(fmt.Sprintf("Failed to parse environment variable as %s, using default value", reflect.TypeOf(defaultVal)),
			"key", key, "rawValue", valueStr, "error", err, "defaultValue", defaultVal)
		return defaultVal
	}

	logger.Info("Successfully loaded environment variable", "key", key, "value", parsedValue)
	return parsedValue
}

// GetEnvFloat gets a float64 from an environment variable with a default value.
func GetEnvFloat(key string, defaultVal float64, logger logr.Logger) float64 {
	parser := func(s string) (float64, error) { return strconv.ParseFloat(s, 64) }
	return getEnvWithParser(key, defaultVal, parser, logger)
}

// GetEnvInt gets an int from an environment variable with a default value.
func GetEnvInt(key string, defaultVal int, logger logr.Logger) int {
	return getEnvWithParser(key, defaultVal, strconv.Atoi, logger)
}

// GetEnvDuration gets a time.Duration from an environment variable with a default value.
func GetEnvDuration(key string, defaultVal time.Duration, logger logr.Logger) time.Duration {
	return getEnvWithParser(key, defaultVal, time.ParseDuration, logger)
}

// GetEnvBool gets a boolean from an environment variable with a default value.
func GetEnvBool(key string, defaultVal bool, logger logr.Logger) bool {
	return getEnvWithParser(key, defaultVal, strconv.ParseBool, logger)
}

// GetEnvString gets a string from an environment variable with a default value.
func GetEnvString(key string, defaultVal string, logger logr.Logger) string {
	parser := func(s string) (string, error) { return s, nil }
	return getEnvWithParser(key, defaultVal, parser, logger)
}
