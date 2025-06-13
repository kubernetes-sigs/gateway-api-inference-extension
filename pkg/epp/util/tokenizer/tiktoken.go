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

package tokenizer

import (
	"bytes"
	"encoding/binary"

	"github.com/pkoukk/tiktoken-go"
	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"
)

// reference: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
const encoding = tiktoken.MODEL_CL100K_BASE

type tiktokenizer struct{}

func NewTiktoken() Tokenizer {
	return &tiktokenizer{}
}

func (s tiktokenizer) Tokenize(text string) ([]byte, error) {
	tiktoken.SetBpeLoader(tiktoken_loader.NewOfflineLoader())

	encoder, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		return nil, err
	}
	token := encoder.Encode(text, nil, nil)

	var buf bytes.Buffer
	for _, num := range token {
		err := binary.Write(&buf, binary.BigEndian, int32(num))
		if err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}
