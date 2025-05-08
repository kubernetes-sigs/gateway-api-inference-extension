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

package metrics

import (
	"sync"

	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const component = "bbr"

var (
	successCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      component,
			Name:           "success_total",
			Help:           "Count of successes pulling model name from body and injecting it in the request headers.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{},
	)
	modelNotInBodyCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      component,
			Name:           "model_not_in_body_total",
			Help:           "Count of times the model was not present in the request body.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{},
	)
	modelNotParsedCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      component,
			Name:           "model_not_parsed_total",
			Help:           "Count of times the model was in the request body but we could not parse it.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{},
	)

	// TODO: Uncomment and use this metrics once the core server implementation has handling to skip body parsing if header exists.
	/*
		modelAlreadyPresentInHeaderCounter = compbasemetrics.NewCounterVec(
			&compbasemetrics.CounterOpts{
				Subsystem:      component,
				Name:           "model_already_present_in_header_total",
				Help:           "Count of times the model was already present in request headers.",
				StabilityLevel: compbasemetrics.ALPHA,
			},
			[]string{},
		)
	*/
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(successCounter)
		legacyregistry.MustRegister(modelNotInBodyCounter)
		legacyregistry.MustRegister(modelNotParsedCounter)
		// legacyregistry.MustRegister(modelAlreadyPresentInHeaderCounter)
	})
}

// RecordSuccessCounter records the number of successful requests to inject the model name into request headers.
func RecordSuccessCounter() {
	successCounter.WithLabelValues().Inc()
}

// RecordModelNotInBodyCounter records the number of times the model was not found in the request body.
func RecordModelNotInBodyCounter() {
	modelNotInBodyCounter.WithLabelValues().Inc()
}

// RecordModelNotParsedCounter records the number of times the model was found in the body but it could not be parsed.
func RecordModelNotParsedCounter() {
	modelNotParsedCounter.WithLabelValues().Inc()
}

/*
// RecordModelAlreadyInHeaderCounter records the number of times the model was already found in the request headers.
func RecordModelAlreadyInHeaderCounter() {
	modelAlreadyPresentInHeaderCounter.WithLabelValues().Inc()
}
*/
