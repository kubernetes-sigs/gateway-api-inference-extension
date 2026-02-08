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

package requestattributereporter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/cel-go/cel"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	// RequestAttributeReporterType is the type of this plugin.
	RequestAttributeReporterType = "request-attribute-reporter"
	// defaultNamespace is the default namespace for the dynamic metadata.
	defaultNamespace = "envoy.lb"
)

// Plugin state
type Plugin struct {
	typedName      plugin.TypedName
	config         Config
	env            *cel.Env
	expressionProg cel.Program
	conditionProg  cel.Program // Can be nil if no condition
}

// Plugin config
type Config struct {
	Attributes []Attribute `json:"attributes"`
}

type Attribute struct {
	Key AttributeKey `json:"key"`
	// The CEL expression to calculate the value. Must return an integer.
	Expression string `json:"expression"`
	// Optional: CEL expression to determine if this metric should be calculated/reported.
	// Must return a boolean.
	Condition string `json:"condition,omitempty"`
}

// Defines where in dynamic metadata to return the data
type AttributeKey struct {
	// Which top-level key to use in dynamic metadata. Optional. Defaults to envoy.lb if omitted
	Namespace string `json:"namespace"`
	// What key to use in the provided namespace for the value from the expression
	Name string `json:"name"`
}

func RequestAttributeReporterPluginFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	pluginConfig := Config{}
	if err := json.Unmarshal(rawParameters, &pluginConfig); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	plugin, err := New(handle.Context(), pluginConfig)
	if err != nil {
		return nil, err
	}
	return plugin.WithName(name), nil
}

func New(ctx context.Context, config Config) (*Plugin, error) {
	if len(config.Attributes) != 1 {
		return nil, errors.New("attributes must contain exactly one entry")
	}
	attributeKey := &config.Attributes[0].Key
	if attributeKey.Name == "" {
		return nil, errors.New("attributeKey.name cannot be empty")
	}
	if config.Attributes[0].Expression == "" {
		return nil, errors.New("attributes[0].expression cannot be empty")
	}
	if attributeKey.Namespace == "" {
		attributeKey.Namespace = defaultNamespace
	}

	env, err := cel.NewEnv(cel.Variable("usage", cel.ObjectType("google.protobuf.Struct")))
	if err != nil {
		return nil, err
	}

	// Compile Expression
	exprAst, issues := env.Compile(config.Attributes[0].Expression)
	if issues != nil && issues.Err() != nil {
		return nil, fmt.Errorf("failed to compile expression: %w", issues.Err())
	}
	expressionProg, err := env.Program(exprAst)
	if err != nil {
		return nil, fmt.Errorf("failed to create program for expression: %w", err)
	}

	// Compile Condition (if provided)
	var conditionProg cel.Program
	if config.Attributes[0].Condition != "" {
		condAst, issues := env.Compile(config.Attributes[0].Condition)
		if issues != nil && issues.Err() != nil {
			return nil, fmt.Errorf("failed to compile condition: %w", issues.Err())
		}
		conditionProg, err = env.Program(condAst)
		if err != nil {
			return nil, fmt.Errorf("failed to create program for condition: %w", err)
		}
	}

	return &Plugin{
		typedName:      plugin.TypedName{Type: RequestAttributeReporterType, Name: RequestAttributeReporterType},
		config:         config,
		env:            env,
		expressionProg: expressionProg,
		conditionProg:  conditionProg,
	}, nil
}

// WithName sets the name of the plugin.
func (p *Plugin) WithName(name string) *Plugin {
	p.typedName.Name = name
	return p
}

// TypedName returns the typed name of the plugin.
func (c *Plugin) TypedName() plugin.TypedName {
	return c.typedName
}

// ResponseComplete implements the requestcontrol.ResponseComplete interface.
func (c *Plugin) ResponseComplete(ctx context.Context, request *scheduling.LLMRequest, response *requestcontrol.Response,
	_ *datalayer.EndpointMetadata) {
	// Convert the request usage Go struct into a protobuf struct so that it can be used as a CEL variable.
	celData, err := c.getCelData(response)
	if err != nil {
		log.FromContext(ctx).Error(err, "Failed to convert usage into CEL data")
		return
	}

	shouldCalculateValue, err := c.shouldCalculateValue(ctx, celData)
	if err != nil {
		log.FromContext(ctx).Error(err, "Error in shouldCalculateValue")
		return
	}
	if !shouldCalculateValue {
		log.FromContext(ctx).Info("shouldCalculateValue is false, returning")
		return
	}

	intVal, err := c.calculateValue(ctx, celData)
	if err != nil {
		return // Error already logged
	}
	if intVal == -1 {
		return // Type error in calculateValue
	}

	// Write the calculated value to dynamic metadata so it can be returned via the ext_proc response.

	if response.DynamicMetadata == nil {
		response.DynamicMetadata = &structpb.Struct{Fields: make(map[string]*structpb.Value)}
	}
	if response.DynamicMetadata.Fields == nil {
		response.DynamicMetadata.Fields = make(map[string]*structpb.Value)
	}

	attributeKey := c.config.Attributes[0].Key
	attributeValue := &structpb.Value{Kind: &structpb.Value_NumberValue{NumberValue: float64(intVal)}}

	namespaceMap, ok := response.DynamicMetadata.Fields[attributeKey.Namespace]
	if !ok {
		namespaceMap = &structpb.Value{Kind: &structpb.Value_StructValue{StructValue: &structpb.Struct{Fields: make(map[string]*structpb.Value)}}}
		response.DynamicMetadata.Fields[attributeKey.Namespace] = namespaceMap
	}

	namespaceMap.GetStructValue().Fields[attributeKey.Name] = attributeValue

	log.FromContext(ctx).V(logutil.VERBOSE).Info("Wrote dynamic metadata value to dynamic metadata", "value", intVal)
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

func (c *Plugin) shouldCalculateValue(ctx context.Context, celData any) (bool, error) {
	if c.conditionProg != nil {
		val, err := c.maybeExecuteProg(ctx, c.conditionProg, celData, "condition", c.config.Attributes[0].Condition)
		if err != nil {
			return false, err // Error already logged in maybeExecuteProg
		}
		if bVal, ok := val.(bool); !ok || !bVal {
			log.FromContext(ctx).V(logutil.VERBOSE).Info("Condition not met", "condition", c.config.Attributes[0].Condition)
			return false, nil
		}
	}
	return true, nil
}

func (c *Plugin) calculateValue(ctx context.Context, celData any) (int64, error) {
	val, err := c.maybeExecuteProg(ctx, c.expressionProg, celData, "expression", c.config.Attributes[0].Expression)
	if err != nil {
		return -1, err // Error already logged
	}

	doubleVal, ok := val.(float64)
	if !ok {
		// Try int64 as well
		int64Val, ok := val.(int64)
		if !ok {
			log.FromContext(ctx).Error(errors.New("type conversion error"), "Expression result could not be converted to float64 or int64", "expression", c.config.Attributes[0].Expression, "result", val)
			return -1, nil
		}
		doubleVal = float64(int64Val)
	}
	return int64(doubleVal), nil
}

func (c *Plugin) maybeExecuteProg(ctx context.Context, prog cel.Program, celData any, exprType string, expression string) (any, error) {
	val, _, err := prog.Eval(map[string]any{"usage": celData})
	if err != nil {
		log.FromContext(ctx).Error(err, "Failed to evaluate "+exprType, exprType, expression)
		return nil, err
	}
	return val.Value(), nil
}
