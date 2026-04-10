/*
Copyright 2026 The Kubernetes Authors.

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

package approximateprefix

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"

	"github.com/cespare/xxhash/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// hashPrompt divides the prompt into blocks and calculates a prefix cache hash for each block.
// The first block hash includes the model name and cache salt (if provided).
// For subsequent blocks, the hash is calculated as: hash(block i content, hash(i-1)).
func hashPrompt(ctx context.Context, request *scheduling.InferenceRequest, blockSizeTokens int, maxPrefixBlocks int) []blockHash {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	if request == nil || request.Body == nil {
		loggerDebug.Info("Request or request data is nil, skipping hashing")
		return nil
	}

	userInput, err := getUserInputBytes(request)
	if err != nil {
		loggerDebug.Error(err, "Failed to get user input bytes")
		return nil
	}

	// convert block size from tokens to characters
	cacheBlockSizeChars := blockSizeTokens * averageCharactersPerToken

	if cacheBlockSizeChars <= 0 {
		loggerDebug.Info("Skipping prefix hashing: block size in characters must be positive",
			"blockSizeTokens", blockSizeTokens,
			"cacheBlockSizeChars", cacheBlockSizeChars)
		return nil
	}

	if len(userInput) < cacheBlockSizeChars {
		loggerDebug.Info("Request body too small for prefix cache", "size", len(userInput), "block size in chars", cacheBlockSizeChars)
		return nil
	}

	if len(userInput) > cacheBlockSizeChars*maxPrefixBlocks {
		loggerDebug.Info("Truncating input", "size", len(userInput), "max prefix blocks", maxPrefixBlocks, "block size in chars", cacheBlockSizeChars)
		userInput = userInput[:maxPrefixBlocks*cacheBlockSizeChars]
	}

	// Split the body into blocks of size cacheBlockSizeChars.
	res := make([]blockHash, 0, len(userInput)/cacheBlockSizeChars)

	h := xxhash.New()
	// Different models should have different hashes even with the same body.
	_, _ = h.Write([]byte(request.TargetModel))
	if cacheSalt := request.Body.CacheSalt(); cacheSalt != "" {
		_, _ = h.Write([]byte(cacheSalt))
	}

	prevBlockHash := blockHash(h.Sum64())
	for i := 0; i+cacheBlockSizeChars <= len(userInput); i += cacheBlockSizeChars {
		h.Reset()
		_, _ = h.Write(userInput[i : i+cacheBlockSizeChars])
		_, _ = h.Write(toBytes(prevBlockHash))
		res = append(res, blockHash(h.Sum64()))

		prevBlockHash = res[len(res)-1]
	}

	return res
}

func toBytes(i blockHash) []byte {
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, uint64(i))
	return bytes
}

func getUserInputBytes(request *scheduling.InferenceRequest) ([]byte, error) {
	switch {
	case request.Body.Conversations != nil:
		return json.Marshal(request.Body.Conversations.Items)

	case request.Body.Responses != nil:
		var combined []map[string]interface{}
		if request.Body.Responses.Instructions != nil {
			combined = append(combined, map[string]interface{}{"instructions": request.Body.Responses.Instructions})
		}
		if request.Body.Responses.Tools != nil {
			combined = append(combined, map[string]interface{}{"tools": request.Body.Responses.Tools})
		}
		combined = append(combined, map[string]interface{}{"input": request.Body.Responses.Input})
		return json.Marshal(combined)

	case request.Body.ChatCompletions != nil:
		return json.Marshal(request.Body.ChatCompletions.Messages)

	case request.Body.Completions != nil:
		return []byte(request.Body.Completions.Prompt.PlainText()), nil

	case request.Body.Embeddings != nil:
		// Handle embeddings API - marshal input for cache key generation
		return json.Marshal(request.Body.Embeddings.Input)

	default:
		return nil, errors.New("invalid request body: no recognized API format found")
	}
}
