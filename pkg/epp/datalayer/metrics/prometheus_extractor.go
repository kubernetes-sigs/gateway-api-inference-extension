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
	"context"
	"fmt"
	"reflect"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const PrometheusMetricPluginType = "prometheus-metric"

// PrometheusMetricPlugin extracts a specific metric from Prometheus format data.
type PrometheusMetricPlugin struct {
	typedName fwkplugin.TypedName
	spec      *Spec
}

var (
	_ fwkplugin.ProducerPlugin = &PrometheusMetricPlugin{}
	_ fwkdl.Extractor          = &PrometheusMetricPlugin{}
)

// NewPrometheusMetricPlugin returns a new PrometheusMetricPlugin.
func NewPrometheusMetricPlugin(metricName string, labels map[string]string) *PrometheusMetricPlugin {
	return &PrometheusMetricPlugin{
		typedName: fwkplugin.TypedName{
			Type: PrometheusMetricPluginType,
			Name: PrometheusMetricPluginType,
		},
		spec: &Spec{
			Name:   metricName,
			Labels: labels,
		},
	}
}

// TypedName returns the type and name of the plugin.
func (p *PrometheusMetricPlugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// ExpectedInputType defines the type expected by the extractor.
func (p *PrometheusMetricPlugin) ExpectedInputType() reflect.Type {
	return fwkdl.PrometheusMetricType
}

// Produces returns the dynamic metric key this plugin produces.
func (p *PrometheusMetricPlugin) Produces() map[string]any {
	return map[string]any{
		p.spec.Name: float64(0),
	}
}

// Extract transforms the data source output into a concrete attribute.
func (p *PrometheusMetricPlugin) Extract(ctx context.Context, data any, ep fwkdl.Endpoint) error {
	families, ok := data.(fwkdl.PrometheusMetricMap)
	if !ok {
		return fmt.Errorf("unexpected input in Extract: %T", data)
	}

	metric, err := p.spec.getLatestMetric(families)
	if err != nil {
		return err
	}

	val := extractValue(metric)
	logger := log.FromContext(ctx).WithValues("endpoint", ep.GetMetadata().NamespacedName, "metric", p.spec.Name)
	logger.V(logutil.TRACE).Info("Extracted custom metric", "value", val)

	current := ep.GetMetrics()
	clone := current.Clone()
	clone.Custom[p.spec.Name] = val
	clone.UpdateTime = time.Now()
	ep.UpdateMetrics(clone)

	return nil
}
