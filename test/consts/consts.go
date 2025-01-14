/*
Copyright 2024 The Kubernetes Authors.

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

package consts

const (
	// TestModelServerName is the name used for test model server resources.
	TestModelServerName = "vllm-llama2-7b-pool"
	// TestModelName is the test model name.
	TestModelName = "tweet-summary"
	// TestEnvoyName is the name used for test envoy proxy resources.
	TestEnvoyName = "envoy"
	// TestEnvoyPort is the stringified port number used for the test envoy proxy.
	TestEnvoyPort = "8081"
	// TestInferExtName is the name used for test inference extension resources.
	TestInferExtName = "inference-gateway-ext-proc"
)
