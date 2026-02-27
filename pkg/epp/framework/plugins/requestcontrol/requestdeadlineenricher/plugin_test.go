package requestdeadlineenricher

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestPluginCreation(t *testing.T) {
	cfg := config{}
	plugin := newRequestDeadlineEnricher(cfg)
	assert.Equal(t, plugin.TypedName().Type, RequestDeadlineEnricherType)
	assert.Equal(t, plugin.TypedName().Name, RequestDeadlineEnricherType)
}

func TestRequestDeadlineEnriching(t *testing.T) {
	cfg := config{
		DeadlineKey: "test-deadline",
	}
	plugin := newRequestDeadlineEnricher(cfg)
	receivedAt := time.Now()
	request := &scheduling.LLMRequest{
		ReceivedAt: receivedAt,
		Headers: map[string]string{
			"x-slo-ttft": "200",
		},
	}
	reqMetadata := make(map[string]any)
	err := plugin.EnrichRequest(context.Background(), request, reqMetadata)
	assert.NoError(t, err)
	assert.Equal(t, reqMetadata["ordering.custom"].(map[string]float64)["test-deadline"], float64(receivedAt.Add(200*time.Millisecond).UnixNano()))
}

func TestRequestDeadlineEnrichingWithDefaultKey(t *testing.T) {
	cfg := config{}
	plugin := newRequestDeadlineEnricher(cfg)
	receivedAt := time.Now()
	request := &scheduling.LLMRequest{
		ReceivedAt: receivedAt,
		Headers: map[string]string{
			"x-slo-ttft": "200",
		},
	}
	reqMetadata := make(map[string]any)
	err := plugin.EnrichRequest(context.Background(), request, reqMetadata)
	assert.NoError(t, err)
	assert.Equal(t, reqMetadata["ordering.custom"].(map[string]float64)["deadline"], float64(receivedAt.Add(200*time.Millisecond).UnixNano()))
}

func TestRequestDeadlineEnrichingWithInvalidTTFT(t *testing.T) {
	cfg := config{}
	plugin := newRequestDeadlineEnricher(cfg)
	receivedAt := time.Now()
	request := &scheduling.LLMRequest{
		ReceivedAt: receivedAt,
		Headers: map[string]string{
			"x-slo-ttft": "invalid",
		},
	}
	reqMetadata := make(map[string]any)
	err := plugin.EnrichRequest(context.Background(), request, reqMetadata)
	assert.Error(t, err)
	assert.Equal(t, reqMetadata["ordering.custom"], nil)
}
