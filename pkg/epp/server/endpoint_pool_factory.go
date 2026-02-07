package server

import (
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/labels"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

// NewEndpointPoolFromOptions constructs an EndpointPool from standalone options.
// This is shared between the production runner and standalone integration tests.
func NewEndpointPoolFromOptions(
	namespace string,
	name string,
	endpointSelector string,
	endpointTargetPorts []int,
) (*datalayer.EndpointPool, error) {

	if namespace == "" {
		return nil, errors.New("namespace must not be empty")
	}
	if name == "" {
		return nil, errors.New("name must not be empty")
	}
	if endpointSelector == "" {
		return nil, errors.New("endpoint selector must not be empty")
	}
	if len(endpointTargetPorts) == 0 {
		return nil, errors.New("endpoint target ports must not be empty")
	}

	selectorMap, err := labels.ConvertSelectorToLabelsMap(endpointSelector)
	if err != nil {
		return nil, fmt.Errorf("failed to parse endpoint selector %q: %w", endpointSelector, err)
	}

	pool := datalayer.NewEndpointPool(namespace, name)
	pool.Selector = selectorMap
	pool.TargetPorts = append(pool.TargetPorts, endpointTargetPorts...)

	return pool, nil
}
