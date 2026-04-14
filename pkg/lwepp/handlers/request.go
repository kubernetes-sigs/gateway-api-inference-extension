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

package handlers

import (
	"context"
	"net"
	"strings"
	"sync/atomic"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"sigs.k8s.io/controller-runtime/pkg/log"

	envoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/datastore"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

  func (s *StreamingServer) handleRequestHeaders(ctx context.Context, reqCtx *RequestContext, fullReq *extProcPb.ProcessingRequest,
  req *extProcPb.ProcessingRequest_RequestHeaders) error {	logger := log.FromContext(ctx)

  var filterEndpoints []string

  // Production path: read endpoint subset from filter metadata per the EPP protocol.
  // The data plane sets envoy.lb.subset_hint / x-gateway-destination-endpoint-subset
  // to constrain which endpoints the EPP may pick from.
  if !s.disableEndpointSubsetFilter {
      requestMetadata := envoy.ExtractMetadataValues(fullReq)
      if subsetMap, ok := requestMetadata[metadata.SubsetFilterNamespace].(map[string]any); ok {
          if endpointList, ok := subsetMap[metadata.SubsetFilterKey].([]any); ok {
              for _, ep := range endpointList {
                  if epStr, ok := ep.(string); ok {
                      filterEndpoints = append(filterEndpoints, epStr)
                  }
              }
          }
      }
  }

  // Test path: read endpoint selection from the test-only request header.
  // This is used by conformance tests to steer routing to a specific pod, analogous
  // to the HeaderBasedTestingFilter plugin in the main EPP.
  if len(filterEndpoints) == 0 {
      for _, header := range req.RequestHeaders.Headers.Headers {
          if header.Key == "test-epp-endpoint-selection" {
              val := envoy.GetHeaderValue(header)
              if val != "" {
                  filterEndpoints = strings.Split(val, ",")
              }
              break
          }
      }
  }

	allPods := s.datastore.PodList(datastore.AllPodsPredicate)
	if len(allPods) == 0 {
		return status.Errorf(codes.Unavailable, "no pods available")
	}

  var candidates []*datastore.Endpoint
  if len(filterEndpoints) > 0 {
      // Build a set of IP addresses from the filter list. Filter entries may be
      // "ip" or "ip:port"; we match only on the IP portion.
      allowedIPs := make(map[string]struct{}, len(filterEndpoints))
      for _, ep := range filterEndpoints {
          ep = strings.TrimSpace(ep)
          if host, _, err := net.SplitHostPort(ep); err == nil {
              allowedIPs[host] = struct{}{}
          } else {
              allowedIPs[ep] = struct{}{}
          }
      }
      for _, pod := range allPods {
          if _, ok := allowedIPs[pod.Address]; ok {
              candidates = append(candidates, pod)
          }
      }
  }

	// If no matches or header not present, use all pods
	if len(candidates) == 0 {
		candidates = allPods
	}

	// Round-robin selection
	index := atomic.AddUint64(&s.rrIndex, 1)
	selectedPod := candidates[index%uint64(len(candidates))]

	reqCtx.SelectedPodIP = selectedPod.Address
	reqCtx.TargetEndpoint = net.JoinHostPort(selectedPod.Address, selectedPod.Port)

	logger.Info("Selected endpoint", "podIP", selectedPod.Address, "endpoint", reqCtx.TargetEndpoint)

	return nil
}
