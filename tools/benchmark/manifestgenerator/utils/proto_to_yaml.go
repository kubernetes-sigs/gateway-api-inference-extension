package utils

import (
	"fmt"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"gopkg.in/yaml.v3"
	// Import your generated protobuf package
	// Replace with the actual import path
)

// protobufToYAML converts a proto.Message to a YAML file.
func protobufToYAML(message proto.Message) ([]byte, error) {
	// 1. Marshal the Protobuf message to JSON bytes using protojson.
	jsonBytes, err := protojson.MarshalOptions{
		Multiline:       true, // Output multi-line JSON (easier to read before YAML)
		Indent:          "  ", // Indent JSON for readability
		UseProtoNames:   true, // Use the names from the .proto file (snake_case)
		EmitUnpopulated: true, // include fields with default values
	}.Marshal(message)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal protobuf to JSON: %w", err)
	}

	// 2. Unmarshal the JSON bytes into a generic YAML structure (map[string]interface{}).
	var yamlData interface{}
	if err := yaml.Unmarshal(jsonBytes, &yamlData); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON to YAML data structure: %w", err)
	}

	// 3. Marshal the YAML data structure to YAML bytes.
	return yaml.Marshal(yamlData)
}
