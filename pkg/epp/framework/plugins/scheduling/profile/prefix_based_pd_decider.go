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

package profile

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	// PrefixBasedPDDeciderPluginType is the type-name of the prefixBasedPDDecider plugin.
	PrefixBasedPDDeciderPluginType = "prefix-based-pd-decider"
)

// PrefixBasedPDDeciderConfig holds the configuration for the prefixBasedPDDecider plugin.
type PrefixBasedPDDeciderConfig struct {
	// NonCachedTokens non cached minimum tokens that triggers disaggregated PD
	NonCachedTokens int `json:"nonCachedTokens"`
}

func (p PrefixBasedPDDeciderConfig) validate() error {
	if p.NonCachedTokens < 0 {
		return errors.New("nonCachedTokens parameter of prefix disaggregation decider cannot be negative")
	}

	return nil
}

// compile-time type assertion
var _ pdDeciderPlugin = &PrefixBasedPDDecider{}

// PrefixBasedPDDecider is a PD decider plugin which decision is based prefix aware
type PrefixBasedPDDecider struct {
	typedName plugin.TypedName
	config    PrefixBasedPDDeciderConfig
}

// PrefixBasedPDDeciderPluginFactory defines the factory function for creating
// a new instance of the prefixBasedPDDecider.
func PrefixBasedPDDeciderPluginFactory(name string, rawParameters json.RawMessage,
	handle plugin.Handle) (plugin.Plugin, error) {
	config := PrefixBasedPDDeciderConfig{
		NonCachedTokens: 0,
	}

	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", PrefixBasedPDDeciderPluginType, err)
		}
	}

	decider, err := NewPrefixBasedPDDecider(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s plugin: %w", PrefixBasedPDDeciderPluginType, err)
	}

	return decider.WithName(name), nil
}

// NewPrefixBasedPDDecider initializes a NewPrefixBasedPDDecider prefix based PD decider Plugin and returns its pointer.
// If the configuration is invalid an error is returned.
func NewPrefixBasedPDDecider(config PrefixBasedPDDeciderConfig) (*PrefixBasedPDDecider, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}

	return &PrefixBasedPDDecider{
		config: config,
	}, nil
}

// TypedName returns the typed name of the plugin.
func (d *PrefixBasedPDDecider) TypedName() plugin.TypedName {
	return d.typedName
}

// WithName sets the name of the plugin.
func (d *PrefixBasedPDDecider) WithName(name string) *PrefixBasedPDDecider {
	d.typedName.Name = name
	return d
}

func (d *PrefixBasedPDDecider) disaggregate(ctx context.Context, inputTokens int, endpoint scheduling.Endpoint) bool {
	logger := log.FromContext(ctx)
	debugLogger := log.FromContext(ctx).V(logutil.DEBUG)

	if d.config.NonCachedTokens <= 0 { // always use disaggregation in case of non cached tokens number is 0
		return true
	}
	if endpoint == nil {
		logger.Error(nil, "prefix decider: endpoint is nil")
		return false
	}
	if inputTokens < d.config.NonCachedTokens {
		debugLogger.Info("Input is shorter than the nonCachedToken, no disaggregated PD")
		return false
	}
	// inspect the decode endpoint to decide if prefill should run or not.
	// if the non-cached part is short enough - no disaggregation.
	prefixInfoRaw, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoKey)
	if !ok || prefixInfoRaw == nil {
		logger.Error(nil, "unable to read prefix cache state")
		return false
	}
	prefixCacheMatchInfo, ok := prefixInfoRaw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok {
		logger.Error(nil, "wrong type of prefix cache match info")
		return false
	}

	// number of cached tokens
	hitPrefixTokens := prefixCacheMatchInfo.MatchBlocks() * prefixCacheMatchInfo.BlockSizeTokens()
	// length of non-cached suffix in tokens
	nonCachedTokens := inputTokens - hitPrefixTokens

	debugLogger.Info("Computed hit percentage for prefix cache",
		"absolute hit prefix len (tokens)", hitPrefixTokens,
		"prompt length (token)", inputTokens)

	if nonCachedTokens < d.config.NonCachedTokens {
		debugLogger.Info("Non-cached suffix is smaller than threshold, using decode profile only")
		return false // do not run prefill
	}

	return true
}

// Consumes defines data types consumed by this plugin
func (*PrefixBasedPDDecider) Consumes() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}
