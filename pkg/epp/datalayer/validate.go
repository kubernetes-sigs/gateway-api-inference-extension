/*
Copyright 2026 The Kubernetes Authors.

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

package datalayer

import (
	"errors"
	"fmt"
	"reflect"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

var (
	ExtractorType             = reflect.TypeOf((*datalayer.Extractor)(nil)).Elem()
	NotificationExtractorType = reflect.TypeOf((*datalayer.NotificationExtractor)(nil)).Elem()
	NotificationEventType     = reflect.TypeOf(datalayer.NotificationEvent{})
)

// ValidateInputTypeCompatible checks if the extractor's expected input type is
// compatible with the DataSource's output type.
func ValidateInputTypeCompatible(dataSourceOutput, extractorInput reflect.Type) error {
	if dataSourceOutput == nil || extractorInput == nil {
		return errors.New("data source output type or extractor input type can't be nil")
	}
	if dataSourceOutput == extractorInput ||
		(extractorInput.Kind() == reflect.Interface && extractorInput.NumMethod() == 0) ||
		(extractorInput.Kind() == reflect.Interface && dataSourceOutput.Implements(extractorInput)) {
		return nil
	}
	return fmt.Errorf("extractor input type %v is not compatible with data source output type %v",
		extractorInput, dataSourceOutput)
}

// ValidateExtractorCompatible checks if the extractor type is compatible with the
// expected extractor interface type. For notification DataSources, this validates
// that the extractor implements NotificationExtractor.
func ValidateExtractorCompatible(extractorType reflect.Type, expectedInterfaceType reflect.Type) error {
	if extractorType == nil || expectedInterfaceType == nil {
		return errors.New("extractor type or expected interface type can't be nil")
	}
	if expectedInterfaceType.Kind() != reflect.Interface {
		return fmt.Errorf("expected type must be an interface, got %v", expectedInterfaceType.Kind())
	}
	if !extractorType.Implements(expectedInterfaceType) {
		return fmt.Errorf("extractor type %v does not implement interface %v",
			extractorType, expectedInterfaceType)
	}
	return nil
}
