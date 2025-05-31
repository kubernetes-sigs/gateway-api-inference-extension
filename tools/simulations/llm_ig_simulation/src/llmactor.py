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

from constants import MAX_NUM_TOKENS_ALLOWED
import simpy 
import numpy as np

class LLMActor(object):
    """This class represents the propagation through an LLM Inference Actor managing multiple stores with Request objects."""

    def __init__(self, env, number_of_actors=1, actorId=0):
        """Initialize the simulation environment and stores."""
        self.env = env
        self.prefill_store = simpy.Store(env)
        self.decode_store = simpy.FilterStore(env)
        self.decoded_store = simpy.Store(env)
        self.recompute_store = simpy.PriorityStore(env)
        self.actor = simpy.Resource(env, capacity=number_of_actors)  # Now dynamically set capacity
        self.user = simpy.Resource(env, capacity=1)
        self.id = actorId
        self.lora_loaded = set()
        self.max_num_tokens_allowed = MAX_NUM_TOKENS_ALLOWED

    def get_num_tokens(self, store, include_remaining=True):
        """Calculate the total number of tokens in a given store, optionally including remaining output tokens."""
        if include_remaining:
            return np.sum([x.input_size + x.output_size - x.output_size_remaining for x in store.items])
        return np.sum([x.input_size for x in store.items])

    def get_num_tokens_in_decode(self):
        """Return the number of total tokens currently in the decode store."""
        return self.get_num_tokens(self.decode_store)

    def get_num_prompt_tokens_in_decode(self):
        """Return the number of input tokens currently in the decode store."""
        return np.sum([x.input_size for x in self.decode_store.items])

    def get_num_gen_tokens_in_decode(self):
        """Return the number of output tokens remaining to be generated in the decode store."""
        return np.sum([x.output_size - x.output_size_remaining for x in self.decode_store.items])

    def get_num_gen_tokens_in_decoded(self):
        """Return the number of output tokens remaining to be generated in the decode store."""
        return np.sum([x.output_size - x.output_size_remaining for x in self.decoded_store.items])

    def get_num_prompt_tokens_in_decoded(self):
        """Return the number of output tokens remaining to be generated in the decode store."""
        return np.sum([x.input_size for x in self.decoded_store.items])

    def get_queue_size(self, store):
        """Return the current queue size of a given store."""
        return len(store.items)

    def get_decode_queue_size(self):
        return self.get_queue_size(self.decode_store)

    def get_prefill_queue_size(self):
        return self.get_queue_size(self.prefill_store)

    def get_recompute_queue_size(self):
        return self.get_queue_size(self.recompute_store)

    def get_decoded_queue_size(self):
        return self.get_queue_size(self.decoded_store)

    def get_min_expected_num_tokens_in_kvcache_after_prefill(self):
        """Calculate the minimum expected number of tokens in the key-value cache after prefill."""
        num_tokens_decode = self.get_num_tokens_in_decode()
        if self.get_queue_size(self.recompute_store) > 0:
            item = self.recompute_store.items[0].item
            return num_tokens_decode + item.input_size + item.output_size - item.output_size_remaining
        elif self.get_queue_size(self.prefill_store) > 0:
            item = self.prefill_store.items[0]
            return num_tokens_decode + item.input_size + item.output_size - item.output_size_remaining

        return num_tokens_decode

    def get_expected_num_tokens_in_kvcache_after_decode(self):
        return self.get_decode_queue_size() + self.get_num_tokens_in_decode()
