package requestdeadlineenricher

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

const (
	// RequestDeadlineEnricherType is the type of this plugin.
	RequestDeadlineEnricherType = "request-deadline-enricher"
	// defaultDeadlineKey is the default key for the deadline in the metadata.
	defaultDeadlineKey = "deadline"
)

type Plugin struct {
	typedName plugin.TypedName
	cfg       config
}

type config struct {
	DeadlineKey string `json:"deadline_key"`
}

func RequestDeadlineEnricherPluginFactory(name string, params json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	cfg := config{}
	if err := json.Unmarshal(params, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return newRequestDeadlineEnricher(cfg).withName(name), nil
}

func newRequestDeadlineEnricher(cfg config) *Plugin {
	return &Plugin{
		typedName: plugin.TypedName{
			Type: RequestDeadlineEnricherType,
			Name: RequestDeadlineEnricherType,
		},
		cfg: cfg,
	}
}

func (p *Plugin) withName(name string) *Plugin {
	if name != "" {
		p.typedName.Name = name
	}
	return p
}

func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

func (p *Plugin) EnrichRequest(ctx context.Context, request *scheduling.LLMRequest, reqMetadata map[string]any) error {
	val, ok := request.Headers["x-slo-ttft"]
	if !ok || val == "" {
		return nil
	}
	slo, err := strconv.Atoi(val)
	if err != nil {
		return fmt.Errorf("failed to create deadline from x-slo-ttft header: %w", err)
	}
	deadline := request.ReceivedAt.Add(time.Duration(slo) * time.Millisecond)
	key := p.cfg.DeadlineKey
	if key == "" {
		key = defaultDeadlineKey
	}
	custMetadata, ok := reqMetadata[metadata.CustomOrderingNamespace].(map[string]float64)
	if !ok {
		custMetadata = make(map[string]float64)
		reqMetadata[metadata.CustomOrderingNamespace] = custMetadata
	}
	custMetadata[key] = float64(deadline.UnixNano())
	return nil
}
