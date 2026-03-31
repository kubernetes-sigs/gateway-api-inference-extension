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
	"time"

	"github.com/go-logr/logr"
	ctrl "sigs.k8s.io/controller-runtime"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

// Runtime manages data sources, extractors, their mapping, and endpoint lifecycle.
type Runtime struct {
	pollingInterval time.Duration // used for polling sources

	pollers          sync.Map // Map of polling sources (key=source name, value=PollingDataSource)
	notifiers        sync.Map // Map of k8s notification sources (key=source name, value=NotificationSource)
	endpointSources  sync.Map // Map of endpoint sources (key=source name, value=EndpointSource)
	sourceExtractors sync.Map // Map sources to extractors (key=source name. value=[]Extractor)

	collectors sync.Map    // Per-endpoint poller (key=namespaced name, value=*Collector)
	logger     logr.Logger // Set in Configure; used where no context is available (e.g. ReleaseEndpoint).
}

const (
	defaultRefreshInterval = 50 * time.Millisecond
)

// NewRuntime creates a new Runtime with the given polling interval.
// If duration is <= 0, uses the defaultRefreshInterval.
func NewRuntime(pollingInterval time.Duration) *Runtime {
	interval := defaultRefreshInterval
	if pollingInterval > 0 {
		interval = pollingInterval
	}
	return &Runtime{
		pollingInterval: interval,
		logger:          logr.Discard(),
	}
}

// Configure is called to transform the configuration information into the Runtime's
// internal fields.
func (r *Runtime) Configure(cfg *Config, enableNewMetrics bool, disallowedExtractorType string, logger logr.Logger) error {
	if cfg == nil || len(cfg.Sources) == 0 {
		if enableNewMetrics {
			return errors.New("data layer enabled but no data sources configured")
		}
		return nil
	}

	r.logger = logger
	logger.Info("Configuring datalayer runtime", "numSources", len(cfg.Sources))

	pollersCount := 0
	notifiersCount := 0
	endpointSourcesCount := 0
	gvkToSource := make(map[string]string, len(cfg.Sources)) // track GVK uniqueness

	for _, srcCfg := range cfg.Sources {
		src := srcCfg.Plugin
		srcName := src.TypedName().Name

		logger.V(logging.DEFAULT).Info("Processing source", "source", srcName, "numExtractors", len(srcCfg.Extractors))
		if err := r.validateSourceExtractors(src, srcCfg.Extractors, disallowedExtractorType); err != nil {
			return err
		}

		if poller, ok := src.(fwkdl.PollingDataSource); ok { // Register to appropriate map based on type
			r.pollers.Store(srcName, poller)
			pollersCount++
		} else if notifier, ok := src.(fwkdl.NotificationSource); ok {
			gvk := notifier.GVK().String()
			if existingSource, exists := gvkToSource[gvk]; exists {
				return fmt.Errorf("duplicate notification source GVK %s: already used by source %s, cannot add %s",
					gvk, existingSource, src.TypedName().String())
			}
			r.notifiers.Store(srcName, notifier)
			gvkToSource[gvk] = srcName
			notifiersCount++
		} else if epSrc, ok := src.(fwkdl.EndpointSource); ok {
			r.endpointSources.Store(srcName, epSrc)
			endpointSourcesCount++
		} else {
			return fmt.Errorf("skipping unknown datasource plugin type %s", src.TypedName().String())
		}

		if len(srcCfg.Extractors) > 0 { // Store extractors mapped to source
			r.sourceExtractors.Store(srcName, srcCfg.Extractors)
		}

		extractorNames := make([]string, len(srcCfg.Extractors))
		for i, ext := range srcCfg.Extractors {
			extractorNames[i] = ext.TypedName().String()
		}
		logger.V(logging.DEFAULT).Info("Source configured", "source", srcName, "extractors", extractorNames)
	}

	logger.Info("Datalayer runtime configured", "pollers", pollersCount, "notifiers", notifiersCount, "endpointSources", endpointSourcesCount)
	return nil
}

// Start is called to enable the Runtime to start processing data collection. It wires
// Kubernetes notifications into the manager.
func (r *Runtime) Start(ctx context.Context, mgr ctrl.Manager) error {
	var err error

	r.notifiers.Range(func(key, val any) bool { // bind notification sources to the manager
		ns := val.(fwkdl.NotificationSource)
		srcName := ns.TypedName().Name

		var extractors []fwkdl.NotificationExtractor
		if rawExts, ok := r.sourceExtractors.Load(srcName); ok {
			raw := rawExts.([]fwkdl.Extractor)
			extractors = make([]fwkdl.NotificationExtractor, len(raw))
			for i, e := range raw {
				extractors[i] = e.(fwkdl.NotificationExtractor)
			}
		}

		if bindErr := BindNotificationSource(ns, extractors, mgr); bindErr != nil {
			err = fmt.Errorf("failed to bind notification source %s: %w", ns.TypedName(), bindErr)
			return false
		}
		return true
	})
	return err
}

// Stop is called to terminate the Runtime's data collection. It terminates all
// go routines used for polling data sources.
func (r *Runtime) Stop() error {
	r.collectors.Range(func(_, val any) bool {
		if c, ok := val.(*Collector); ok {
			_ = c.Stop()
		}
		return true
	})
	return nil
}

// NewEndpoint sets up data polling on the provided endpoint.
func (r *Runtime) NewEndpoint(ctx context.Context, endpointMetadata *fwkdl.EndpointMetadata, _ PoolInfo) fwkdl.Endpoint {
	// TODO: should we cache the sources and map after Configure? Or just replace with maps and Mutex?
	// The code could be simpler and also would benefit from using RLock mutex for concurrent access
	// (no change expected) instead of using sync.Map (avoid use of Range just to count, more idiomatic code, etc.).
	logger, _ := logr.FromContext(ctx)
	logger = logger.WithValues("endpoint", endpointMetadata.GetNamespacedName())

	var pollers []fwkdl.PollingDataSource
	r.pollers.Range(func(_, val any) bool {
		if poller, ok := val.(fwkdl.PollingDataSource); ok {
			pollers = append(pollers, poller)
		}
		return true
	})

	if len(pollers) == 0 {
		logger.Info("No polling sources configured, creating endpoint without collector")
		return fwkdl.NewEndpoint(endpointMetadata, nil)
	}

	extractors := make(map[string][]fwkdl.Extractor, len(pollers))
	r.sourceExtractors.Range(func(key, val any) bool {
		srcName := key.(string)
		exts := val.([]fwkdl.Extractor)
		extractors[srcName] = exts
		return true
	})

	endpoint := fwkdl.NewEndpoint(endpointMetadata, nil)
	collector := NewCollector()

	key := endpointMetadata.GetNamespacedName()
	if _, loaded := r.collectors.LoadOrStore(key, collector); loaded {
		logger.V(logging.DEFAULT).Info("collector already running for endpoint", "endpoint", key)
		return nil
	}

	ticker := NewTimeTicker(r.pollingInterval)
	if err := collector.Start(ctx, ticker, endpoint, pollers, extractors); err != nil {
		logger.Error(err, "failed to start collector for endpoint", "endpoint", key)
		r.collectors.Delete(key)
		return nil
	}

	r.dispatchEndpointEvent(ctx, logger, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint})
	return endpoint
}

// ReleaseEndpoint terminates polling for data on the given endpoint.
func (r *Runtime) ReleaseEndpoint(ep fwkdl.Endpoint) {
	r.dispatchEndpointEvent(context.Background(), r.logger, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: ep})

	key := ep.GetMetadata().GetNamespacedName()
	if value, ok := r.collectors.LoadAndDelete(key); ok {
		collector := value.(*Collector)
		_ = collector.Stop()
	}
}

// dispatchEndpointEvent routes an endpoint lifecycle event to all registered
// EndpointSources and their extractors.
func (r *Runtime) dispatchEndpointEvent(ctx context.Context, logger logr.Logger, event fwkdl.EndpointEvent) {
	if isEmpty(&r.endpointSources) {
		return
	}
	r.endpointSources.Range(func(key, val any) bool {
		srcName := key.(string)
		epSrc := val.(fwkdl.EndpointSource)

		processed, err := epSrc.NotifyEndpoint(ctx, event)
		if err != nil {
			logger.Error(err, "endpoint source failed to process event", "source", srcName)
			return true
		}
		if processed == nil {
			return true
		}

		rawExts, ok := r.sourceExtractors.Load(srcName)
		if !ok {
			return true
		}
		for _, ext := range rawExts.([]fwkdl.Extractor) {
			if epExt, ok := ext.(fwkdl.EndpointExtractor); ok {
				if err := epExt.ExtractEndpoint(ctx, *processed); err != nil {
					logger.Error(err, "endpoint extractor failed", "extractor", ext.TypedName())
				}
			}
		}
		return true
	})
}

// validates the compatibility of data source and configured extractors. This includes
// expected Extractor type, source output and extractor input type compatibility and
// optionally source specific validation.
func (r *Runtime) validateSourceExtractors(src fwkdl.DataSource, extractors []fwkdl.Extractor, disallowedExtractorType string) error {
	for _, ext := range extractors {
		// check if disallowed extractor type
		if disallowedExtractorType != "" && ext.TypedName().Type == disallowedExtractorType {
			return fmt.Errorf("disallowed Extractor %s is configured for source %s",
				ext.TypedName().String(), src.TypedName().String())
		}

		// validate extractor type
		extractorType := reflect.TypeOf(ext)
		if err := validateExtractorCompatible(extractorType, src.ExtractorType()); err != nil {
			return fmt.Errorf("extractor %s type incompatible with datasource %s: %w",
				ext.TypedName(), src.TypedName(), err)
		}

		// validate input/output types match
		if err := validateInputTypeCompatible(src.OutputType(), ext.ExpectedInputType()); err != nil {
			return fmt.Errorf("extractor %s input type incompatible with datasource %s: %w",
				ext.TypedName(), src.TypedName(), err)
		}
		if notifySrc, ok := src.(fwkdl.NotificationSource); ok {
			if notifyExt, ok := ext.(fwkdl.NotificationExtractor); ok {
				if notifySrc.GVK().String() != notifyExt.GVK().String() {
					return fmt.Errorf("extractor %s GVK %s does not match source %s GVK %s",
						ext.TypedName(), notifyExt.GVK().String(), src.TypedName(), notifySrc.GVK().String())
				}
			}
		}

		// allow datasource custom validation
		if validator, ok := src.(fwkdl.ValidatingDataSource); ok {
			if err := validator.ValidateExtractor(ext); err != nil {
				return fmt.Errorf("extractor %s failed custom validation for datasource %s: %w",
					ext.TypedName(), src.TypedName(), err)
			}
		}
	}
	return nil
}

// validate input/output type compatibility.
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

// validate extractor compatibility.
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

// isEmpty reports whether the sync.Map has no entries.
func isEmpty(m *sync.Map) bool {
	empty := true
	m.Range(func(_, _ any) bool {
		empty = false
		return false // stop immediately
	})
	return empty
}

var _ EndpointFactory = (*Runtime)(nil)
