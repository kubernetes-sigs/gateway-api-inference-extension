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

package costreporting

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/go-logr/logr"
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log"

	extproc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

const (
	// CostReportingPluginType is the type of the cost reporting plugin.
	CostReportingPluginType = "cost-reporter"
	// DefaultNamespace is the default namespace for the dynamic metadata.
	DefaultNamespace = "envoy.lb"
)

// Plugin is a plugin that reports the cost of a request based on the response body.
type Plugin struct {
	config         Config
	logger         logr.Logger
	env            *cel.Env
	expressionProg cel.Program
	conditionProg  cel.Program // Can be nil if no condition
}

// Config is the configuration for the cost reporting plugin.
type Config struct {
	Metric     Metric `json:"metric"`
	DataSource string `json:"dataSource"`
	Expression string `json:"expression"`
	Condition  string `json:"condition,omitempty"`
}

// Metric defines a single cost metric to be reported.
type Metric struct {
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
}

// New creates a new CostReporting plugin.
func New(config Config, logger logr.Logger) (*Plugin, error) {
	metric := &config.Metric
	if metric.Name == "" {
		return nil, fmt.Errorf("metric.Name cannot be empty")
	}
	if config.Expression == "" {
		return nil, fmt.Errorf("config.Expression cannot be empty")
	}
	if config.DataSource == "" {
		config.DataSource = "responseBody" // Default data source
	}
	if config.DataSource != "responseBody" {
		return nil, fmt.Errorf("unsupported dataSource: %s", config.DataSource)
	}
	if metric.Namespace == "" {
		metric.Namespace = DefaultNamespace
	}

	env, err := cel.NewEnv(
		cel.Declarations(
			decls.NewVar("responseBody", decls.Dyn),
		),
	)
	if err != nil {
		return nil, err
	}

	// Compile Expression
	exprAst, issues := env.Compile(config.Expression)
	if issues != nil && issues.Err() != nil {
		return nil, fmt.Errorf("failed to compile expression: %w", issues.Err())
	}
	expressionProg, err := env.Program(exprAst)
	if err != nil {
		return nil, fmt.Errorf("failed to create program for expression: %w", err)
	}

	// Compile Condition (if provided)
	var conditionProg cel.Program
	if config.Condition != "" {
		condAst, issues := env.Compile(config.Condition)
		if issues != nil && issues.Err() != nil {
			return nil, fmt.Errorf("failed to compile condition: %w", issues.Err())
		}
		conditionProg, err = env.Program(condAst)
		if err != nil {
			return nil, fmt.Errorf("failed to create program for condition: %w", err)
		}
	}

	return &Plugin{
		config:         config,
		logger:         logger,
		env:            env,
		expressionProg: expressionProg,
		conditionProg:  conditionProg,
	}, nil
}

// Type returns the type of the plugin.
func (c *Plugin) Type() string {
	return CostReportingPluginType
}

// TypedName returns the typed name of the plugin.
func (c *Plugin) TypedName() plugins.TypedName {
	return plugins.TypedName{
		Type: c.Type(),
	}
}

// ResponseStreaming implements the requestcontrol.ResponseStreaming interface.
func (c *Plugin) ResponseStreaming(
	ctx context.Context, req *extproc.ProcessingRequest_ResponseHeaders, resp *extproc.ProcessingResponse_ResponseHeaders, bodyChunk []byte) (*extproc.ProcessingResponse, error) {
	// For streaming, we'll assume the relevant information is in a single chunk.
	// More complex scenarios might require buffering and re-parsing.
	return c.processBody(ctx, bodyChunk)
}

// ResponseComplete implements the requestcontrol.ResponseComplete interface.
func (c *Plugin) ResponseComplete(
	ctx context.Context, req *extproc.ProcessingRequest_ResponseBody, resp *extproc.ProcessingResponse_ResponseBody) (*extproc.ProcessingResponse, error) {
	return c.processBody(ctx, req.ResponseBody.Body)
}

func (c *Plugin) processBody(_ context.Context, body []byte) (*extproc.ProcessingResponse, error) {
	logger := c.logger.WithValues("plugin", CostReportingPluginType)
	var data map[string]interface{}
	if err := json.Unmarshal(body, &data); err != nil {
		logger.V(1).Info("Failed to unmarshal response body as JSON", "error", err)
		return &extproc.ProcessingResponse{}, nil // Don't fail the request
	}

	vars := map[string]interface{}{
		"responseBody": data,
	}

	response := &extproc.ProcessingResponse{}

	shouldCalculateCost, err := c.shouldCalculateCost(vars)
	if err != nil {
		return response, err
	} else if !shouldCalculateCost {
		return response, nil
	}

	intVal, err := c.calculateCost(vars)
	if err != nil {
		return response, err
		// TODO we should not fail the request here
	}

	if response.Response == nil {
		response.Response = &extproc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &extproc.ImmediateResponse{
				Details: "cost reporting plugin",
			},
		}
	}

	if response.DynamicMetadata == nil {
		response.DynamicMetadata = &structpb.Struct{
			Fields: map[string]*structpb.Value{},
		}
	}

	metric := c.config.Metric
	ns := metric.Namespace
	nsData, ok := response.DynamicMetadata.Fields[ns]
	if !ok {
		nsData = &structpb.Value{
			Kind: &structpb.Value_StructValue{
				StructValue: &structpb.Struct{
					Fields: map[string]*structpb.Value{},
				},
			},
		}
		response.DynamicMetadata.Fields[ns] = nsData
	}

	nsData.GetStructValue().Fields[metric.Name] = &structpb.Value{
		Kind: &structpb.Value_StringValue{
			StringValue: fmt.Sprintf("%d", intVal),
		},
	}
	logger.V(1).Info("Successfully calculated metric", "namespace", ns, "name", metric.Name, "value", intVal)

	if response.DynamicMetadata != nil && len(response.DynamicMetadata.GetFields()) > 0 {
		logger.Info("Dynamic metadata to be set", "metadata", response.DynamicMetadata)
	}

	return response, nil
}

func (c *Plugin) shouldCalculateCost(vars map[string]interface{}) (bool, error) {
	if c.conditionProg != nil {
		val, err := c.maybeExecuteProg(c.conditionProg, vars, "condition", c.config.Condition)
		if err != nil {
			return false, nil // Error already logged
		}
		if bVal, ok := val.(bool); !ok || !bVal {
			c.logger.V(1).Info("Condition not met", "condition", c.config.Condition)
			return false, nil
		}
	}
	return true, nil
}

func (c *Plugin) calculateCost(vars map[string]interface{}) (int64, error) {
	val, err := c.maybeExecuteProg(c.expressionProg, vars, "expression", c.config.Expression)
	if err != nil {
		return -1, nil // Error already logged
	}

	doubleVal, ok := val.(float64)
	if !ok {
		// Try int64 as well
		int64Val, ok := val.(int64)
		if !ok {
			c.logger.Error(fmt.Errorf("type conversion error"), "Expression result could not be converted to float64 or int64", "expression", c.config.Expression, "result", val)
			return -1, nil
		}
		doubleVal = float64(int64Val)
	}
	return int64(doubleVal), nil
}

func (c *Plugin) maybeExecuteProg(prog cel.Program, vars map[string]interface{}, exprType string, expression string) (any, error) {
	val, _, err := prog.Eval(vars)
	if err != nil {
		c.logger.Error(err, fmt.Sprintf("Failed to evaluate %s", exprType), exprType, expression)
		return nil, err
	}
	return val.Value(), nil
}

// CostReportingPluginFactory is the factory function for the cost reporting plugin.
func CostReportingPluginFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	logger := log.FromContext(handle.Context()).WithName(name)
	parameters := Config{}
	if err := json.Unmarshal(rawParameters, &parameters); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	plugin, err := New(parameters, logger)
	if err != nil {
		return nil, err
	}
	return plugin, nil
}
