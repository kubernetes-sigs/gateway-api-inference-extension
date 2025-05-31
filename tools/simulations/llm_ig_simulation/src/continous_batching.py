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

from constants import MAX_NUM_SEQ, MAX_NUM_BATCH_TOKENS, MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE, TOKENIZE_LATENCY_CONST, PREFILL_LATENCY_CONST_2, PREFILL_LATENCY_CONST_1, PREFILL_LATENCY_CONST_0, PREFILL_LATENCY_CONST_MIN, DECODE_LATENCY_CONST_1, DECODE_LATENCY_CONST_0, DECODE_LATENCY_CONST_BATCH, LORA_DICT

import simpy
import numpy as np

def should_process_prefill_or_recompute(llmactor, env):
    """Check if the system should process prefill or recompute based on queue sizes and memory constraints."""
    return can_prefill_items(llmactor, env)
           
def can_prefill_items(llmactor, env):
    """Are there items I can prefill?"""
    prefill_batch_size = 0
    num_new_seq = 0

    while llmactor.get_recompute_queue_size() > 0:
        oldest_item = llmactor.recompute_store.items[0].item
        oldest_item_len = oldest_item.input_size + oldest_item.output_size - oldest_item.output_size_remaining
        oldest_item_input_len = oldest_item.input_size 

        if any([
            llmactor.get_decode_queue_size() + num_new_seq + 1 > MAX_NUM_SEQ,
            prefill_batch_size + oldest_item_input_len > MAX_NUM_BATCH_TOKENS,
            (prefill_batch_size + num_new_seq + llmactor.get_num_tokens_in_decode()) / (llmactor.max_num_tokens_allowed + 0.0) >= MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE
        ]):
            break

        return True

    while llmactor.get_prefill_queue_size() > 0:
        oldest_item = llmactor.prefill_store.items[0]
        oldest_item_len = oldest_item.input_size + oldest_item.output_size - oldest_item.output_size_remaining
        oldest_item_input_len = oldest_item.input_size 

        if any([
            llmactor.get_decode_queue_size() + num_new_seq + 1 > MAX_NUM_SEQ,
            prefill_batch_size + oldest_item_input_len > MAX_NUM_BATCH_TOKENS,
            (prefill_batch_size + num_new_seq + llmactor.get_num_tokens_in_decode()) / (llmactor.max_num_tokens_allowed + 0.0) >= MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE
        ]):
            break

        return True

    return False

def fetch_prefill_items(llmactor, env):
    """Fetch items to prefill if there is memory either from recompute (p0) or from prefill (p1)"""
    items_to_prefill = []
    prefill_batch_size = 0
    num_new_seq = 0

    while llmactor.get_recompute_queue_size() > 0:
        oldest_item = llmactor.recompute_store.items[0].item
        oldest_item_len = oldest_item.input_size + oldest_item.output_size - oldest_item.output_size_remaining
        oldest_item_input_len = oldest_item.input_size 

        if any([
            llmactor.get_decode_queue_size() + num_new_seq + 1 > MAX_NUM_SEQ,
            prefill_batch_size + oldest_item_input_len > MAX_NUM_BATCH_TOKENS,
            (prefill_batch_size + num_new_seq + llmactor.get_num_tokens_in_decode()) / (llmactor.max_num_tokens_allowed + 0.0) >= MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE
        ]):
            break

        prefill_batch_size += oldest_item_len
        num_new_seq += 1
        msg = yield llmactor.recompute_store.get()
        items_to_prefill.append(msg.item)

    while llmactor.get_prefill_queue_size() > 0:
        oldest_item = llmactor.prefill_store.items[0]
        oldest_item_len = oldest_item.input_size + oldest_item.output_size - oldest_item.output_size_remaining
        oldest_item_input_len = oldest_item.input_size 

        if any([
            llmactor.get_decode_queue_size() + num_new_seq + 1 > MAX_NUM_SEQ,
            prefill_batch_size + oldest_item_input_len > MAX_NUM_BATCH_TOKENS,
            (prefill_batch_size + num_new_seq + llmactor.get_num_tokens_in_decode()) / (llmactor.max_num_tokens_allowed + 0.0) >= MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE
        ]):
            break

        prefill_batch_size += oldest_item_len
        num_new_seq += 1
        msg = yield llmactor.prefill_store.get()
        items_to_prefill.append(msg)

    return items_to_prefill

def process_prefill_items(llmactor, env, items_to_prefill, req_dict_prefill, req_dict, logging=False):
    """Process prefill items, updating times and managing item states."""
    prefill_len = np.sum([x.input_size + x.output_size - x.output_size_remaining for x in items_to_prefill])
    prefill_delay = calculate_prefill_delay(prefill_len, len(items_to_prefill), TOKENIZE_LATENCY_CONST, PREFILL_LATENCY_CONST_2, PREFILL_LATENCY_CONST_1, PREFILL_LATENCY_CONST_0, PREFILL_LATENCY_CONST_MIN)

    for item in items_to_prefill:
        # lora stuff
        if item.lora is not None:
            if item.lora not in llmactor.lora_loaded:
                llmactor.lora_loaded.add(item.lora)
                llmactor.max_num_tokens_allowed -= LORA_DICT[item.lora]

        if item.start_prefill_time is None:
            item.start_prefill_time = env.now
            item.end_prefill_time = item.start_prefill_time + prefill_delay
        item.end_decode_time = llmactor.env.now + prefill_delay
        item.output_size_remaining -= 1

        if item.output_size_remaining == 0:
            llmactor.decoded_store.put(item)
        else:
            llmactor.decode_store.put(item)
            if item.output_size_remaining <= 0:
                if logging:
                    print(f'llmactor {llmactor.id} {item.id} item.output_size_remaining {item.output_size_remaining}')
                assert item.output_size_remaining > 0
        req_dict_prefill[item.id] = item
        req_dict[item.id] = item
    return prefill_delay

def should_recompute(llmactor, env):
    """Determine if items should be moved to recompute based on memory usage."""
    return llmactor.get_expected_num_tokens_in_kvcache_after_decode() / (llmactor.max_num_tokens_allowed + 0.0) > MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE

def remove_from_decode_store(llmactor, env, req_dict_prefill, req_dict, logging=False):
    """Manage the recomputation of items based on priority and conditions."""
    while should_recompute(llmactor, env):
        if llmactor.get_decode_queue_size() > 0:
            newest_decode_item_id = llmactor.decode_store.items[-1].id  # newest item goes to recompute
            if logging:
                print(f'llmactor {llmactor.id} removing from decode store sequence {newest_decode_item_id}')
            req_dict[newest_decode_item_id].recompute_count += 1

            newest_decode_item = yield llmactor.decode_store.get(lambda req: req.id == newest_decode_item_id)
            llmactor.recompute_store.put(simpy.PriorityItem(item=newest_decode_item, priority=newest_decode_item_id))

def decode_items(llmactor, env, req_dict_prefill, req_dict, logging=False):
    """Process decoding of items, handling them appropriately based on their remaining output size."""
    num_items_to_decode = llmactor.get_decode_queue_size()
    before_decoding_token_count = llmactor.get_num_tokens_in_decode()
    temp_items = []
    decode_delay = calculate_decode_delay(before_decoding_token_count, num_items_to_decode, TOKENIZE_LATENCY_CONST, DECODE_LATENCY_CONST_1, DECODE_LATENCY_CONST_0, DECODE_LATENCY_CONST_BATCH)
    if logging:
        print(f'llmactor {llmactor.id} Decoding sequences {[x.id for x in llmactor.decode_store.items]} items with delay {decode_delay}')

    for _ in range(num_items_to_decode):
        msg = yield llmactor.decode_store.get()
        if msg.output_size_remaining == msg.output_size - 1:
            msg.start_decode_time = env.now
            msg.tokens_in_kv_cache_at_start_of_decode = before_decoding_token_count
        msg.output_size_remaining -= 1
        if msg.output_size_remaining < 0:
            raise ValueError(f'Output size remaining negative for {msg.id}')

        temp_items.append(msg)
        req_dict_prefill[msg.id] = msg
        req_dict[msg.id] = msg

    for item in temp_items:
        if item.output_size_remaining == 0:
            item.end_decode_time = env.now + decode_delay
            llmactor.decoded_store.put(item)
        else:
            item.end_decode_time = env.now + decode_delay
            llmactor.decode_store.put(item)

    return decode_delay

def calculate_decode_delay(token_count, num_items_to_decode, tokenize_latency_const, decode_latency_const_1, decode_latency_const_0, decode_latency_const_batch):
    """Calculate delay based on the token count and latency constants."""
    return token_count * decode_latency_const_1 + decode_latency_const_0 + (tokenize_latency_const + decode_latency_const_batch) * num_items_to_decode

def calculate_prefill_delay(token_count, num_items_to_prefill, tokenize_latency_const, prefill_latency_const_2, prefill_latency_const_1, prefill_latency_const_0, prefill_latency_const_min):
    """Calculate delay based on the token count and latency constants."""
    return max(prefill_latency_const_min, (token_count * token_count * prefill_latency_const_2 + token_count * prefill_latency_const_1 + prefill_latency_const_0 + num_items_to_prefill * tokenize_latency_const))

def prefill_or_decode(env, llmactor, req_dict_prefill, req_dict, logging=False):
    """Main process for managing prefill, decode, or recompute operations."""
    while True:
        with llmactor.actor.request() as req:
            yield req
            if (llmactor.get_decode_queue_size() == 0) and (llmactor.get_prefill_queue_size() == 0) and (llmactor.get_recompute_queue_size() == 0):
                yield env.timeout(1 / 1000.0)
            elif should_process_prefill_or_recompute(llmactor, env):
                items_to_prefill = yield from fetch_prefill_items(llmactor, env)
                prefill_delay = process_prefill_items(llmactor, env, items_to_prefill, req_dict_prefill, req_dict)
                if logging:
                    print(f'llmactor {llmactor.id} Processed prefill for sequences {[x.id for x in items_to_prefill]} with delay {prefill_delay}')
                yield env.timeout(prefill_delay)  # Assume prefill_delay is calculated somewhere
            else:
                if should_recompute(llmactor, env):
                    yield from remove_from_decode_store(llmactor, env, req_dict_prefill, req_dict)
                if llmactor.get_decode_queue_size() > 0:
                    decode_delay = yield from decode_items(llmactor, env, req_dict_prefill, req_dict)
                    yield env.timeout(decode_delay)

def metrics(env, llmactor):
    while True:
        yield env.timeout(10)
        cur_time = env.now
        num_of_prompt_tokens = llmactor.get_num_prompt_tokens_in_decode() + llmactor.get_num_prompt_tokens_in_decoded()
        num_of_gen_tokens = llmactor.get_num_gen_tokens_in_decode() + llmactor.get_num_gen_tokens_in_decoded()
        running_req = llmactor.get_decode_queue_size()
        pending_req = llmactor.get_prefill_queue_size()
        gpu_kv_cache_usage = llmactor.get_num_tokens_in_decode() / llmactor.max_num_tokens_allowed * 100
        print(f'llmactor {llmactor.id} Avg prompt throughput: {num_of_prompt_tokens / cur_time} tokens/s, Avg generation throughput: {num_of_gen_tokens / cur_time}, Running: {running_req} reqs, Pending: {pending_req} reqs, GPU KV cache usage: {gpu_kv_cache_usage}%')
