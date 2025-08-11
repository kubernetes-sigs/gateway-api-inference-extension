package test

const (
	// HeaderTestEppEndPointSelectionKey is the header used for testing purposes to make EPP behavior controllable.
	// The header value should be a comma-separated list of endpoint IP addresses.
	// E.g., "test-epp-endpoint-selection": "10.0.0.7,10.0.0.8"
	// The returned order is the same as the order provided in the header.
	HeaderTestEppEndPointSelectionKey = "test-epp-endpoint-selection"
)
