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

package datalayer

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	ctrl "sigs.k8s.io/controller-runtime"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

var (
	ExtractorType             = reflect.TypeOf((*fwkdl.Extractor)(nil)).Elem()
	NotificationExtractorType = reflect.TypeOf((*fwkdl.NotificationExtractor)(nil)).Elem()
	NotificationEventType     = reflect.TypeOf(fwkdl.NotificationEvent{})
)

func validateInputTypeCompatible(dataSourceOutput, extractorInput reflect.Type) error {
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

func validateExtractorCompatible(extractorType reflect.Type, expectedInterfaceType reflect.Type) error {
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

// Runtime manages data sources, extractors, and endpoint lifecycle.
type Runtime struct {
	pollingInterval time.Duration
	state           atomic.Int64 // stores RuntimeState

	// sync.Map[string]PollingDataSource
	pollers sync.Map

	// sync.Map[string]NotificationSource
	notifiers sync.Map

	// sync.Map[string][]fwkdl.Extractor - source name to extractors
	sourceExtractors sync.Map

	// Per-endpoint collectors: key=namespaced name, value=*Collector
	collectors sync.Map

	ctx    context.Context
	cancel context.CancelFunc
}

// NewRuntime creates a new Runtime with the given polling interval.
func NewRuntime(pollingInterval time.Duration) *Runtime {
	r := &Runtime{
		pollingInterval: pollingInterval,
		state:           atomic.Int64{},
	}
	r.state.Store(int64(StateInitial))
	return r
}

// State returns the current state of the Runtime.
func (r *Runtime) State() RuntimeState {
	return RuntimeState(r.state.Load())
}

func (r *Runtime) Configure(cfg *Config, enableNewMetrics bool, disallowedExtractorType string, logger logr.Logger) (err error) {
	// Check state: must be initial
	if err = r.transition(StateInitial, StateConfiguring); err != nil {
		return err
	}

	// Ensure we transition to error state if there's any error
	defer func() {
		if err != nil {
			_ = r.transition(StateConfiguring, StateError)
		}
	}()

	if cfg == nil {
		return errors.New("config cannot be nil")
	}

	logger.Info("Configuring datalayer runtime", "numSources", len(cfg.Sources))

	pollersCount := 0
	notifiersCount := 0
	gvkToSource := make(map[string]string, len(cfg.Sources)) // track GVK uniqueness

	for _, srcCfg := range cfg.Sources {
		src := srcCfg.Plugin
		srcName := src.TypedName().Name

		logger.V(1).Info("Processing source", "source", srcName, "numExtractors", len(srcCfg.Extractors))

		if err := r.validateSourceExtractors(src, srcCfg.Extractors, disallowedExtractorType); err != nil {
			return err
		}

		if poller, ok := src.(fwkdl.PollingDataSource); ok { // Register to appropriate map based on type
			r.pollers.Store(srcName, poller)
			pollersCount++
		} else if notifier, ok := src.(fwkdl.NotificationSource); ok {
			gvk := notifier.GVK().String()
			r.notifiers.Store(srcName, notifier)
			gvkToSource[gvk] = srcName
			notifiersCount++
		} else {
			return fmt.Errorf("invalid source %s", src.TypedName().String())
		}

		if len(srcCfg.Extractors) > 0 { // Store extractors mapped to source
			r.sourceExtractors.Store(srcName, srcCfg.Extractors)
		}

		extractorNames := make([]string, len(srcCfg.Extractors))
		for i, ext := range srcCfg.Extractors {
			extractorNames[i] = ext.TypedName().String()
		}
		logger.V(1).Info("Source configured", "source", srcName, "extractors", extractorNames)
	}

	logger.Info("Datalayer runtime configured", "pollers", pollersCount, "notifiers", notifiersCount)

	return r.transition(StateConfiguring, StateConfigured)
}

func (r *Runtime) validateSourceExtractors(src fwkdl.DataSource, extractors []fwkdl.Extractor, disallowedExtractorType string) error {
	// Get GVK if source is a NotificationSource
	var srcGVK string
	if ns, ok := src.(fwkdl.NotificationSource); ok {
		srcGVK = ns.GVK().String()
	}

	for _, ext := range extractors {
		// Check disallowed extractor type
		if disallowedExtractorType != "" && ext.TypedName().Type == disallowedExtractorType {
			return fmt.Errorf("disallowed Extractor %s is configured for source %s",
				ext.TypedName().String(), src.TypedName().String())
		}
		// Validate GVK match for notification source/extractor
		if srcGVK != "" {
			if ne, ok := ext.(fwkdl.NotificationExtractor); ok {
				if ne.GVK().String() != srcGVK {
					return fmt.Errorf("extractor %s GVK %s does not match source %s GVK %s",
						ext.TypedName(), ne.GVK().String(), src.TypedName(), srcGVK)
				}
			}
		}
		// Validate extractor type compatibility first
		extractorType := reflect.TypeOf(ext)
		if err := validateExtractorCompatible(extractorType, src.ExtractorType()); err != nil {
			return fmt.Errorf("extractor %s type incompatible with datasource %s: %w",
				ext.TypedName(), src.TypedName(), err)
		}
		// Validate input type compatibility
		if err := validateInputTypeCompatible(src.OutputType(), ext.ExpectedInputType()); err != nil {
			return fmt.Errorf("extractor %s input type incompatible with datasource %s: %w",
				ext.TypedName(), src.TypedName(), err)
		}
		// Allow datasource custom validation
		if validator, ok := src.(fwkdl.ValidatingDataSource); ok {
			if err := validator.ValidateExtractor(ext); err != nil {
				return fmt.Errorf("extractor %s failed custom validation for datasource %s: %w",
					ext.TypedName(), src.TypedName(), err)
			}
		}
	}
	return nil
}

func (r *Runtime) Start(ctx context.Context, mgr ctrl.Manager) (err error) {
	// Check state: must be configured
	if err = r.transition(StateConfigured, StateStarting); err != nil {
		return err
	}

	// Ensure we transition to error state if there's any error
	defer func() {
		if err != nil {
			_ = r.transition(StateStarting, StateError)
		}
	}()

	r.ctx, r.cancel = context.WithCancel(ctx)

	return r.transition(StateStarting, StateStarted)
}

func (r *Runtime) Stop() (err error) {
	// Check state: must be started
	if err = r.transition(StateStarted, StateStopping); err != nil {
		return err
	}

	// Ensure we transition to error state if there's any error
	defer func() {
		if err != nil {
			_ = r.transition(StateStopping, StateError)
		}
	}()

	if r.cancel != nil {
		r.cancel()
	}

	r.collectors.Range(func(_, val any) bool {
		if c, ok := val.(*Collector); ok {
			_ = c.Stop()
		}
		return true
	})

	return r.transition(StateStopping, StateStopped)
}

func (r *Runtime) NewEndpoint(ctx context.Context, endpointMetadata *fwkdl.EndpointMetadata, poolInfo PoolInfo) fwkdl.Endpoint {
	logger, _ := logr.FromContext(ctx)
	logger = logger.WithValues("endpoint", endpointMetadata.GetNamespacedName())

	// Collect all polling sources
	var pollers []fwkdl.PollingDataSource
	r.pollers.Range(func(_, val any) bool {
		if poller, ok := val.(fwkdl.PollingDataSource); ok {
			pollers = append(pollers, poller)
		}
		return true
	})

	// Collect all extractors
	var extractors []fwkdl.Extractor
	r.sourceExtractors.Range(func(_, val any) bool {
		extractors = append(extractors, val.([]fwkdl.Extractor)...)
		return true
	})

	if len(pollers) == 0 {
		logger.Info("No polling sources configured, creating endpoint without collector")
		return fwkdl.NewEndpoint(endpointMetadata, nil)
	}

	endpoint := fwkdl.NewEndpoint(endpointMetadata, nil)
	collector := NewCollectorWithExtractors(pollers, extractors)

	key := endpointMetadata.GetNamespacedName()
	if _, loaded := r.collectors.LoadOrStore(key, collector); loaded {
		logger.Info("collector already running for endpoint", "endpoint", key)
		return nil
	}

	ticker := NewTimeTicker(r.pollingInterval)
	if err := collector.Start(ctx, ticker, endpoint); err != nil {
		logger.Error(err, "failed to start collector for endpoint", "endpoint", key)
		r.collectors.Delete(key)
		return nil
	}

	return endpoint
}

func (r *Runtime) ReleaseEndpoint(ep fwkdl.Endpoint) {
	key := ep.GetMetadata().GetNamespacedName()

	if value, ok := r.collectors.LoadAndDelete(key); ok {
		collector := value.(*Collector)
		_ = collector.Stop()
	}
}
