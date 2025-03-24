package utils

import (
	"fmt"
	"os"
	"path/filepath"

	klog "k8s.io/klog/v2"
)

func SaveFile(path string, data []byte, override bool) error {
	// Check if the file already exists
	if _, err := os.Stat(path); err == nil {
		if !override {
			klog.V(1).Infof("File %v already exists, skipping saving", path)
			return nil // File already exists, skip saving
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("error checking if file exists: %v", err)
	}

	folder := filepath.Dir(path)
	// Create the directory if it doesn't exist
	if err := os.MkdirAll(folder, os.ModePerm); err != nil {
		return fmt.Errorf("error creating directory: %v", err)
	}
	klog.V(2).Infof("Writing output to %v", path)
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write path %v: %v", path, err)
	}
	return nil
}
