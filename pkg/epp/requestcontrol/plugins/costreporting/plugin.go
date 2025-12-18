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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
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
	Metric Metric `json:"metric"`
	// The CEL expression to calculate the cost. Must return an integer.
	Expression string `json:"expression"`
	// Optional: CEL expression to determine if this metric should be calculated/reported.
	// Must return a boolean.
	Condition string `json:"condition,omitempty"`
}

// Defines where in dynamic metadata to return the data
type Metric struct {
	// Which top-level key to use in dynamic metadata. Optional. Defaults to envoy.lb if omitted
	Namespace string `json:"namespace"`
	// What key to use in the provided namespace for the value from the expression
	Name string `json:"name"`
}

func CostReportingPluginFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
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

func New(config Config, logger logr.Logger) (*Plugin, error) {
	metric := &config.Metric
	if metric.Name == "" {
		return nil, fmt.Errorf("metric.name cannot be empty")
	}
	if config.Expression == "" {
		return nil, fmt.Errorf("config.expression cannot be empty")
	}
	if metric.Namespace == "" {
		metric.Namespace = DefaultNamespace
	}

	env, err := cel.NewEnv(
		cel.Declarations(
			decls.NewVar("usage", decls.NewObjectType("google.protobuf.Struct")),
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
func (c *Plugin) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: c.Type(),
	}
}

// ResponseComplete implements the requestcontrol.ResponseComplete interface.
func (c *Plugin) ResponseComplete(
	ctx context.Context, request *scheduling.LLMRequest, response *requestcontrol.Response, _ *datalayer.EndpointMetadata) {
	logger := c.logger.WithValues("plugin", CostReportingPluginType)

	// Convert the request usage Go struct into a protobuf struct so that it can be used as a CEL variable.
	celData, err := c.getCelData(response)
	if err != nil {
		logger.V(1).Error(err, "Failed to convert usage into CEL data")
		return
	}

	shouldCalculateCost, err := c.shouldCalculateCost(celData)
	if err != nil {
		logger.V(1).Error(err, "Error in shouldCalculateCost")
		return
	}
	if !shouldCalculateCost {
		logger.V(1).Info("shouldCalculateCost is false, returning")
		return
	}

	intVal, err := c.calculateCost(celData)
	if err != nil {
		return // Error already logged
	}
	if intVal == -1 {
		return // Type error in calculateCost
	}

	// Write the calculated cost to dynamic metadata so it can be returned via the ext_proc response.

	if response.DynamicMetadata == nil {
		response.DynamicMetadata = &structpb.Struct{Fields: make(map[string]*structpb.Value)}
	}
	if response.DynamicMetadata.Fields == nil {
		response.DynamicMetadata.Fields = make(map[string]*structpb.Value)
	}

	metric := c.config.Metric
	metricValue := &structpb.Value{Kind: &structpb.Value_NumberValue{NumberValue: float64(intVal)}}

	namespaceMap, ok := response.DynamicMetadata.Fields[metric.Namespace]
	if !ok {
		namespaceMap = &structpb.Value{Kind: &structpb.Value_StructValue{StructValue: &structpb.Struct{Fields: make(map[string]*structpb.Value)}}}
		response.DynamicMetadata.Fields[metric.Namespace] = namespaceMap
	}

	namespaceMap.GetStructValue().Fields[metric.Name] = metricValue

	logger.V(1).Info("Wrote dynamic metadata value to dynamic metadata", "value", intVal)
}

func (c *Plugin) getCelData(response *requestcontrol.Response) (any, error) {
	usageBytes, err := json.Marshal(response.Usage)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request.Usage to JSON: %w", err)
	}
	usageStruct := &structpb.Struct{}
	if err := usageStruct.UnmarshalJSON(usageBytes); err != nil {
		return nil, fmt.Errorf("failed to unmarshal request.Usage JSON to structpb: %w", err)
	}
	return usageStruct, nil
}

func (c *Plugin) shouldCalculateCost(celData any) (bool, error) {
	if c.conditionProg != nil {
		val, err := c.maybeExecuteProg(c.conditionProg, celData, "condition", c.config.Condition)
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

func (c *Plugin) calculateCost(celData any) (int64, error) {
	val, err := c.maybeExecuteProg(c.expressionProg, celData, "expression", c.config.Expression)
	if err != nil {
		return -1, err // Error already logged
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

func (c *Plugin) maybeExecuteProg(prog cel.Program, celData any, exprType string, expression string) (any, error) {
	val, _, err := prog.Eval(map[string]any{"usage": celData})
	if err != nil {
		c.logger.Error(err, fmt.Sprintf("Failed to evaluate %s", exprType), exprType, expression)
		return nil, err
	}
	return val.Value(), nil
}
