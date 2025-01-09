package scheduling

import "testing"

func TestFilterGenValidation(t *testing.T) {
	testCases := []struct {
		name      string
		fo        *FilterOption
		filterGen FilterGen
		wantErr   bool
	}{
		{
			name: "valid sheddable_request filter option",
			fo: &FilterOption{
				KvCacheThreshold:       toPtr(0.8),
				QueueThresholdCritical: toPtr(5),
			},
			filterGen: FilterSheddableRequest,
			wantErr:   false,
		},
		{
			name:      "valid sheddable_request filter option, nil option",
			fo:        &FilterOption{},
			filterGen: FilterSheddableRequest,
			wantErr:   false,
		},
		{
			name: "valid sheddable_request filter option, nil QueueThresholdCritical",
			fo: &FilterOption{
				KvCacheThreshold: toPtr(0.8),
			},
			filterGen: FilterSheddableRequest,
			wantErr:   false,
		},
		{
			name: "invalid sheddable_request filter option",
			fo: &FilterOption{
				KvCacheThreshold:       toPtr(-1.0),
				QueueThresholdCritical: toPtr(5),
			},
			filterGen: FilterSheddableRequest,
			wantErr:   true,
		},
		{
			name: "valid low_latency filter option",
			fo: &FilterOption{
				QueueingThresholdLoRA: toPtr(50),
			},
			filterGen: FilterLowLatency,
			wantErr:   false,
		},
		{
			name:      "valid low_latency filter option, nil option",
			fo:        &FilterOption{},
			filterGen: FilterLowLatency,
			wantErr:   false,
		},
		{
			name: "invalid low_latency filter option",
			fo: &FilterOption{
				QueueingThresholdLoRA: toPtr(-1),
			},
			filterGen: FilterLowLatency,
			wantErr:   true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.filterGen.Validate(tc.fo)
			if (err != nil) != tc.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func toPtr[T any](v T) *T {
	return &v
}
