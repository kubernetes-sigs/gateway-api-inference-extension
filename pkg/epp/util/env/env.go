package env

import (
	"os"
	"strconv"

	"github.com/go-logr/logr"
)

// getEnvFloat gets a float64 from an environment variable with a default value
func GetEnvFloat(key string, defaultVal float64, logger logr.Logger) float64 {
	val, exists := os.LookupEnv(key)
	if !exists {
		logger.Info("Environment variable not set, using default value",
			"key", key, "defaultValue", defaultVal)
		return defaultVal
	}

	floatVal, err := strconv.ParseFloat(val, 64)
	if err != nil {
		logger.Info("Failed to parse environment variable as float, using default value",
			"key", key, "value", val, "error", err, "defaultValue", defaultVal)
		return defaultVal
	}

	logger.Info("Successfully loaded environment variable",
		"key", key, "value", floatVal)
	return floatVal
}

// getEnvInt gets an int from an environment variable with a default value
func GetEnvInt(key string, defaultVal int, logger logr.Logger) int {
	val, exists := os.LookupEnv(key)
	if !exists {
		logger.Info("Environment variable not set, using default value",
			"key", key, "defaultValue", defaultVal)
		return defaultVal
	}

	intVal, err := strconv.Atoi(val)
	if err != nil {
		logger.Info("Failed to parse environment variable as int, using default value",
			"key", key, "value", val, "error", err, "defaultValue", defaultVal)
		return defaultVal
	}

	logger.Info("Successfully loaded environment variable",
		"key", key, "value", intVal)
	return intVal
}

// GetEnvString gets a string from an environment variable with a default value
func GetEnvString(key string, defaultVal string, logger logr.Logger) string {
	val, exists := os.LookupEnv(key)
	if !exists {
		logger.Info("Environment variable not set, using default value",
			"key", key, "defaultValue", defaultVal)
		return defaultVal
	}
	return val
}
