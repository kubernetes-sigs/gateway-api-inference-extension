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

import numpy as np

class Request:
    def __init__(self, id, arrival_time, input_size, output_size, lora):
        self.id = id
        self.arrival_time = arrival_time
        self.input_size = input_size
        self.start_prefill_time = None
        self.prefill_time = None
        self.tokens_in_kv_cache_at_start_of_decode = None
        self.start_decode_time = None
        self.end_first_token_decode_time = None
        self.end_decode_time = None
        self.output_size = output_size
        self.output_size_remaining = output_size
        self.recompute_count = 0
        self.target_pod = None
        self.target_latency = np.inf
        self.queue_size_before_prefill = None
        self.estimated_latency = 0
        self.lora = lora
        self.pending_tokens_at_arrival_perc = 0
        self.actual_tokens_at_arrival_perc = 0

def create_request(id, time, input_size, output_size, lora=None):
    """Creates a new request with given parameters."""
    return Request(id, time, input_size, output_size, lora)

def determine_size(distribution_mean, distribution_std, sizes_dict=None, id=None):
    """Determines the size of a request component based on a dictionary, normal distribution, or fixed size."""
    if sizes_dict and id in sizes_dict:
        return max(1, sizes_dict[id])
    return max(1.0, np.round(np.abs(np.random.normal(distribution_mean, distribution_std, 1))).astype(int)[0])
