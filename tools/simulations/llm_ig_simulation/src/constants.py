# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PREFILL_LATENCY_CONST_2 = 0
PREFILL_LATENCY_CONST_1 = 0.00006769375513
PREFILL_LATENCY_CONST_0 = 0.01969
PREFILL_LATENCY_CONST_MIN = 0.04

DECODE_LATENCY_CONST_BATCH = 0.0001026494433
DECODE_LATENCY_CONST_1 = 0.0000005353485087
DECODE_LATENCY_CONST_0 = 0.014
TOKENIZE_LATENCY_CONST = 0

MAX_NUM_BATCH_TOKENS = 512 # in prefill

TOTAL_NUM_GPU_BLOCKS = 2810
NUMBER_OF_TOKENS_PER_BLOCK = 16
MAX_NUM_TOKENS_ALLOWED = TOTAL_NUM_GPU_BLOCKS * NUMBER_OF_TOKENS_PER_BLOCK - MAX_NUM_BATCH_TOKENS # in kv cache
MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE = 0.9
MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE_NON_CRITICAL = 0.8
MAX_NUM_SEQ = 256

# size of each lora in units of KV Cache
LORA_DICT = {"tweet": 1600, "sql": 1600, "dummy-1": 0, "dummy-2": 0}
