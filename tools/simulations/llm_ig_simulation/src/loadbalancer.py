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

from queue import Queue
import random
from re import I
from threading import active_count
from typing import Dict, List, Optional
import simpy
import numpy as np
from constants import MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE, MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE_NON_CRITICAL, MAX_NUM_TOKENS_ALLOWED, MAX_NUM_BATCH_TOKENS
from request import Request, create_request, determine_size
from continous_batching import prefill_or_decode, metrics
from llmactor import LLMActor
from typing import Set

class LoadBalancer:
    def __init__(self, 
                 env: simpy.Environment, 
                 number_of_servers: int = 1, 
                 list_of_llmactors: List[LLMActor] = None, 
                 messages_remaining_cnt: Optional[int] = None, 
                 req_dict_prefill = {}, req_dict = {},
                 queueing_perc: float = np.inf):
        self.number_of_servers = number_of_servers
        self.list_of_llmactors = list_of_llmactors or []
        assert len(self.list_of_llmactors) == number_of_servers, "Number of actors must match number of servers"
        self.env = env
        self.req_dict_prefill: Dict[str, Request] = req_dict_prefill
        self.req_dict: Dict[str, Request] = req_dict
        self.queues: Dict[float, Queue] = {}
        self.messages_remaining_cnt = messages_remaining_cnt
        self.queueing_perc = queueing_perc # KV Cache saturation threshold for queueing
        self.max_prefill_queue_size = 5 # max prefill queue size before queing


    def estimate_avg_latency(self, llmactor, input_size, output_size, include_running_requests=False, percentile=95, TTL = 300):
        """
        Estimates total latency for the request processing by calculating prefill, decode, and queue times.
        If include_running_requests is True, it considers both finished and running requests; otherwise, only finished requests are considered.

     """
        estimate_total_latency_list = []
        estimated_prefill_latency_list = []
        estimated_decode_latency_list = []
        estimated_queue_time_list = []
    
        current_tokens_in_kv_cache = llmactor.get_num_tokens_in_decode()
        total_queue_length = 0
    
        # Choose which request set to process
        items = llmactor.decode_store.items if include_running_requests else llmactor.decoded_store.items
    
        for item in items:
            if self.env.now - item.arrival_time > TTL:
                continue  # Skip long-running requests
        
            tokens_in_kv_cache_at_start_of_decode = item.tokens_in_kv_cache_at_start_of_decode or 0
        
            if tokens_in_kv_cache_at_start_of_decode > 0:
                decode_delays_per_output_token_normalized_by_batch_size = (
                ((item.end_decode_time - item.end_prefill_time) / tokens_in_kv_cache_at_start_of_decode)
                / (item.output_size - item.output_size_remaining)
            )
                estimated_decode_time = decode_delays_per_output_token_normalized_by_batch_size * current_tokens_in_kv_cache * output_size
            else:
                estimated_decode_time = 0

            estimated_prefill_time = (item.end_prefill_time - item.arrival_time) / item.input_size * input_size

            estimated_prefill_latency_list.append(estimated_prefill_time)
            estimated_decode_latency_list.append(estimated_decode_time)


        total_queue_length = llmactor.get_prefill_queue_size()

        # Calculate latencies
        estimated_prefill_latency = 0 if len(estimated_prefill_latency_list) == 0 else (
        np.percentile(estimated_prefill_latency_list, percentile) if include_running_requests else np.mean(estimated_prefill_latency_list)
        )
        estimated_decode_latency = 0 if len(estimated_decode_latency_list) == 0 else (
         np.percentile(estimated_decode_latency_list, percentile) if include_running_requests else np.mean(estimated_decode_latency_list)
        )
    
        estimated_queue_time = estimated_prefill_latency * total_queue_length
        estimated_total_latency = estimated_prefill_latency + estimated_decode_latency + estimated_queue_time

        return estimated_total_latency, estimated_prefill_latency, estimated_decode_latency



    def check_saturations(self, use_pseudo_kv_cache = False, max_saturation: float = MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE_NON_CRITICAL) -> bool:
        if use_pseudo_kv_cache:
          return all(
              self.get_pending_tokens_perc(llmactor)  >= max_saturation
              for llmactor in self.list_of_llmactors
          )
        return all(
            llmactor.get_min_expected_num_tokens_in_kvcache_after_prefill() / llmactor.max_num_tokens_allowed >= max_saturation
            for llmactor in self.list_of_llmactors
        )
    def get_actual_tokens_perc(self, llmactor: LLMActor) -> float:
        actual_tokens = sum(
            item.output_size + item.input_size - item.output_size_remaining
            for item in llmactor.decode_store.items
        )
        return actual_tokens / llmactor.max_num_tokens_allowed

    def get_pending_tokens_perc(self, llmactor: LLMActor) -> float:
        pending_tokens = sum(
            item.output_size + item.input_size
            for store in (llmactor.decode_store, llmactor.prefill_store)
            for item in store.items
        )
        return pending_tokens / llmactor.max_num_tokens_allowed

    def get_overall_pending_tokens_perc(self) -> float:
        total_pending_tokens = sum(
            self.get_pending_tokens_perc(llmactor) * llmactor.max_num_tokens_allowed
            for llmactor in self.list_of_llmactors
        )
        total_max_tokens = sum(llmactor.max_num_tokens_allowed for llmactor in self.list_of_llmactors)
        return total_pending_tokens / total_max_tokens if total_max_tokens else 0
      
    def get_overall_actual_tokens_perc(self) -> float:
        total_actual_tokens = sum(
            self.get_actual_tokens_perc(llmactor) * llmactor.max_num_tokens_allowed
            for llmactor in self.list_of_llmactors
        )
        total_max_tokens = sum(llmactor.max_num_tokens_allowed for llmactor in self.list_of_llmactors)
        return total_actual_tokens / total_max_tokens if total_max_tokens else 0

    def get_lora_affinity(self, lora_requested: str) -> List[LLMActor]:
        if not lora_requested:
            return self.list_of_llmactors
        
        pods_with_lora = [llmactor for llmactor in self.list_of_llmactors if lora_requested in llmactor.lora_loaded]
        if pods_with_lora:
            return pods_with_lora
        
        min_lora_count = min(len(llmactor.lora_loaded) for llmactor in self.list_of_llmactors)
        return [llmactor for llmactor in self.list_of_llmactors if len(llmactor.lora_loaded) == min_lora_count]


    def all_servers_queued(self) -> bool:
        
        return min(llmactor.get_prefill_queue_size() for llmactor in self.list_of_llmactors) > self.max_prefill_queue_size

    
    def find_target_pod_based_on_min_latency(self, pods, input_size, output_size,  target_latency = np.inf, output_error = 0, max_tokens = 1024, buffer = 1):

        """
        Finds the target pod with the minimum latency for processing a request.
        """
        all_candiated_pods_based_on_min_expected_latency = []
        min_expected_latency = np.inf




        for i, llmactor in enumerate(pods):

            #estimate latencies
            output_size_for_estimation = min(max(1.0, np.round(np.abs(np.random.normal(output_size, output_size * output_error, 1))).astype(int)[0]), max_tokens)
            estimated_latency, estimated_prefill_latency, estimated_decode_latency = self.estimate_avg_latency(llmactor, input_size, output_size_for_estimation)
            estimated_latency_per_output_token = (estimated_latency) / output_size_for_estimation


            if estimated_latency_per_output_token < min_expected_latency:
              min_expected_latency = estimated_latency_per_output_token
              all_candiated_pods_based_on_min_expected_latency = [i]
            elif estimated_latency_per_output_token == min_expected_latency:
              all_candiated_pods_based_on_min_expected_latency.append(i)


        # return a random one
        if len(all_candiated_pods_based_on_min_expected_latency) > 0:
          random_pod_index = random.choice(all_candiated_pods_based_on_min_expected_latency)
          return self.list_of_llmactors[random_pod_index], estimated_latency_per_output_token
        else:
          return None, estimated_latency_per_output_token


    def find_target_pod_based_on_max_pending(self, pods, input_size, output_size,  target_latency = np.inf, output_error = 0, max_tokens = 1024, buffer = 0.5):

        """
        Finds the target pod based on maximum pending requests under a certain latency.
        """
        all_candiated_pods_based_on_maximum_pending_req = []
        min_pending_token_perc = np.inf
        max_pending_token_below_target_perc = 0
        min_kv_cache_usage = np.inf
        drop_request = False
        if target_latency == np.inf:
          drop_request = True
        max_tokens_in_kv_before_eviction = MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE
        latency_estimations = {}



        for i, llmactor in enumerate(pods):

            #estimate latencies
            expected_kvcache_usage_after_prefill = llmactor.get_min_expected_num_tokens_in_kvcache_after_prefill() / (llmactor.max_num_tokens_allowed + 0.0)
            output_size_for_estimation = min(max(1.0, np.round(np.abs(np.random.normal(output_size, output_size * output_error, 1))).astype(int)[0]), max_tokens)
            estimated_latency, estimated_prefill_latency, estimated_decode_latency = self.estimate_avg_latency(llmactor, input_size, output_size_for_estimation, True, 95, 300)
            estimated_latency_per_output_token = (estimated_latency) / output_size_for_estimation
            latency_estimations[i] = estimated_latency_per_output_token

            #get pending tokens
            pending_token_perc = self.get_pending_tokens_perc(llmactor)
            prefill_queue_size = llmactor.get_prefill_queue_size()

            # Get matching pods
            if (estimated_latency_per_output_token < buffer * target_latency and 
              pending_token_perc > max_pending_token_below_target_perc and 
              expected_kvcache_usage_after_prefill < max_tokens_in_kv_before_eviction and 
              prefill_queue_size < self.max_prefill_queue_size):
              
              max_pending_token_below_target_perc = pending_token_perc
              all_candiated_pods_based_on_maximum_pending_req = [i]
              
            elif (estimated_latency_per_output_token < buffer * target_latency and 
                pending_token_perc == max_pending_token_below_target_perc and 
                expected_kvcache_usage_after_prefill < max_tokens_in_kv_before_eviction and 
                prefill_queue_size < self.max_prefill_queue_size):
              
              all_candiated_pods_based_on_maximum_pending_req.append(i)


        # return a random one
        if len(all_candiated_pods_based_on_maximum_pending_req) > 0:
          random_pod_index = random.choice(all_candiated_pods_based_on_maximum_pending_req)
          return self.list_of_llmactors[random_pod_index],   latency_estimations[random_pod_index]
        else:
          return None, 0


    def find_target_pod_based_on_min_pending(self, pods, eviction_safe = False, max_kv_perc = MAX_GPU_MEMORY_PERC_BEFORE_RECOMPUTE ):


        """
        Finds the target pod based on the minimum pending tokens and safe eviction condition.
        """
        all_candiated_pods_based_on_minimum_pending_req = []
        min_pending_token_perc = np.inf

        for i, llmactor in enumerate(pods):
            pending_token_perc = self.get_pending_tokens_perc(llmactor)
            expected_kvcache_usage_after_prefill = llmactor.get_min_expected_num_tokens_in_kvcache_after_prefill() / (llmactor.max_num_tokens_allowed + 0.0)

            if eviction_safe:
              if pending_token_perc < min_pending_token_perc and expected_kvcache_usage_after_prefill < max_kv_perc:
                all_candiated_pods_based_on_minimum_pending_req = [i]
                min_pending_token_perc = pending_token_perc
              elif pending_token_perc == min_pending_token_perc:
                all_candiated_pods_based_on_minimum_pending_req.append(i)
            else:
               if pending_token_perc < min_pending_token_perc:
                all_candiated_pods_based_on_minimum_pending_req = [i]
                min_pending_token_perc = pending_token_perc
               elif pending_token_perc == min_pending_token_perc:
                all_candiated_pods_based_on_minimum_pending_req.append(i)


        # return a random one
        if len(all_candiated_pods_based_on_minimum_pending_req) > 0:
          random_pod_index = random.choice(all_candiated_pods_based_on_minimum_pending_req)
          return self.list_of_llmactors[random_pod_index]
        else:
          return None


    def find_target_pod_based_on_min_kv_cache(self, pods, ):

        """
        Finds the target pod with the minimum usage of KV cache.
        """

        all_candiated_pods_based_on_min_kv_cache_usage = []
        min_kv_cache_usage = np.inf

        for i, llmactor in enumerate(pods):
            expected_kvcache_usage_after_prefill = llmactor.get_min_expected_num_tokens_in_kvcache_after_prefill() / (llmactor.max_num_tokens_allowed + 0.0)

            if expected_kvcache_usage_after_prefill < min_kv_cache_usage:
              min_kv_cache_usage = expected_kvcache_usage_after_prefill
              all_candiated_pods_based_on_min_kv_cache_usage = [i]
            elif expected_kvcache_usage_after_prefill == min_kv_cache_usage:
              all_candiated_pods_based_on_min_kv_cache_usage.append(i)


        # return a random one
        if len(all_candiated_pods_based_on_min_kv_cache_usage) > 0:
          random_pod_index = random.choice(all_candiated_pods_based_on_min_kv_cache_usage)
          return self.list_of_llmactors[random_pod_index]
        else:
          return None




    def find_target_pod(self, routing_type, input_size, output_size,  target_latency = np.inf, lora_requested = "", output_error = 0, max_tokens = 1024):
        """
        Finds the target pod based on routing strategy and various factors like latency, pending tokens, or LoRA.
        """

        target_pod = None
        latency_esimated = 0

        active_req_target_latency_in_window = self.getActiveReqTargetLatencyInWindow()
        violations_present , _= self.getViolationsTargetLatencyInWindow()



        if target_latency == np.inf:
          all_pods_saturated = self.check_saturations()
          if all_pods_saturated and len(active_req_target_latency_in_window) > 0:
            return target_pod, latency_esimated
          if violations_present:
            return target_pod, latency_esimated

        if routing_type == "random":
          target_pod = self.list_of_llmactors[random.randint(0, self.number_of_servers-1)]
          return target_pod, latency_esimated
        elif routing_type == "least":
          target_pod = self.find_target_pod_based_on_min_kv_cache(self.list_of_llmactors, )
          return target_pod, latency_esimated
        elif routing_type == "leastPseudo":
          target_pod = self.find_target_pod_based_on_min_pending(self.list_of_llmactors, eviction_safe=False)
          return target_pod, latency_esimated
        elif routing_type == "leastlatency":
          target_pod, latency_esimated = self.find_target_pod_based_on_min_latency(self.list_of_llmactors, input_size, output_size, target_latency)
          return target_pod, latency_esimated
        elif lora_requested != "":
          lora_pods = self.get_lora_affinity(lora_requested)
          target_pod, latency_esimated = self.find_target_pod_based_on_max_pending(lora_pods, input_size, output_size, target_latency)
          if target_pod is  None:
            target_pod, latency_esimated = self.find_target_pod_based_on_max_pending(self.list_of_llmactors, input_size, output_size, target_latency)
          if target_pod is None:
            target_pod = self.find_target_pod_based_on_min_pending(lora_pods, eviction_safe=True)
          if target_pod is None:
            target_pod = self.find_target_pod_based_on_min_pending(self.list_of_llmactors,  eviction_safe=False)
          return target_pod, latency_esimated

        else:
          pods = self.list_of_llmactors
          target_pod, latency_esimated = self.find_target_pod_based_on_max_pending(pods, input_size, output_size, target_latency)
          if target_pod is None:
            target_pod = self.find_target_pod_based_on_min_pending(self.list_of_llmactors,  eviction_safe=False)
          return target_pod, latency_esimated


    def queueing_signal(self, routing_type = "smart") -> bool:
      if routing_type == "smart":
        return self.check_saturations(use_pseudo_kv_cache=False, max_saturation= self.queueing_perc) or self.all_servers_queued()
      else :
        return self.get_overall_pending_tokens_perc() > self.queueing_perc or self.all_servers_queued()

    def dequeueing_signal(self, routing_type = "smart") -> bool:
      if routing_type == "smart":
        return self.check_saturations(use_pseudo_kv_cache=False, max_saturation= self.queueing_perc)  == False and  self.all_servers_queued() == False
      else :
        return self.get_overall_pending_tokens_perc() < self.queueing_perc and self.all_servers_queued() == False




    def check_if_queues_empty(self) -> bool:
        for k, v in self.queues.items():
          if not v.empty():
            return False
        return True

    import random
    
    def slo_based_dequeue(self) -> Optional[Request]:
      # Get active targets and their latencies
      _, violation_dict = self.getViolationsTargetLatencyInWindow()
      # get list of active targets in order of violation dict
  
      active_targets = sorted(violation_dict.keys(), key=lambda x: violation_dict[x], reverse=True)

      for k in self.queues:
        if k not in active_targets and not self.queues[k].empty():
          req = self.queues[k].get()
          return req
        
      for k in active_targets:
          if k in self.queues and not self.queues[k].empty():
            req = self.queues[k].get()
            return req
      
      return None
    


    def weighted_dequeue(self) -> Optional[Request]:
      # Get active targets and their latencies
      active_targets = list(self.getActiveReqTargetLatencyInWindow(np.inf))
    
      # Calculate inverse weights based on latencies
      inverse_weights = {k: 1.0 / k for k in active_targets}
    
      # Calculate total weight to normalize
      total_weight = sum(inverse_weights.values())
    
      # Calculate the relative probabilities for each target
      target_probs = {k: inverse_weights[k] / total_weight for k in active_targets}
    
      # Use random.choices to select a target based on probabilities
      # Attempt to dequeue from the selected target's queue
      for _ in range(1000):  # Try up to the 100 times
        selected_target = random.choices(list(target_probs.keys()), weights=target_probs.values(), k=1)[0]
        
        # Check if the selected target's queue is non-empty
        if selected_target in self.queues and not self.queues[selected_target].empty():
            req = self.queues[selected_target].get()
            return req
    
      return None
    
    def dequeue(self) -> Optional[Request]:
        active_targets = sorted(self.getActiveReqTargetLatencyInWindow(np.inf))
        for k in active_targets:
          if k in self.queues and not self.queues[k].empty():
            req = self.queues[k].get()
            return req
        return None




    def dequeue_process(self, routing_type, drop_late_requests = False):
        while True:
            if not self.check_if_queues_empty() and self.dequeueing_signal(routing_type):
                # Get the request with the highest SLO violation
                req = self.weighted_dequeue()
                if   req:
                  if (drop_late_requests == False) or (self.env.now - req.arrival_time < 100*req.target_latency): #ad-hoc
                    target_pod, estimated_latency = self.find_target_pod(routing_type, req.input_size, req.output_size, req.target_latency, req.lora)
                    req.target_pod = target_pod.id
                    req.estimated_latency = estimated_latency
                    req.queue_size_before_prefill = target_pod.get_prefill_queue_size()
                    req.pending_tokens_at_arrival_perc = self.get_pending_tokens_perc(target_pod)
                    req.actual_tokens_at_arrival_perc = self.get_actual_tokens_perc(target_pod)


                    # Send it to the appropriate pod for processing
                    target_pod = self.list_of_llmactors[req.target_pod]
                    target_pod.prefill_store.put(req)
                    self.req_dict_prefill[req.id] = req
                else:
                    yield self.env.timeout(0.001)  # Adjust the delay as per your requirements
            # Check again after a small delay
            else:
              yield self.env.timeout(0.001)  # Adjust the delay as per your requirements






    def getActiveReqTargetLatencyInWindow(self, time_windows = 300) -> Set[float]:
      """
        Retrieves active target latencies within a specified time window.

        :param time_windows: Time window in which to check for active requests.
        :return: Set of active target latencies.
      """
      active_targets = set()
      for llmactor in self.list_of_llmactors:
        if llmactor.get_prefill_queue_size() > 0:
          for req in llmactor.prefill_store.items:
            if req.target_latency != np.inf:
              active_targets.add(req.target_latency)
        if llmactor.get_decode_queue_size() > 0:
          for req in llmactor.decode_store.items:
            if req.target_latency != np.inf:
              active_targets.add(req.target_latency)
        if llmactor.get_recompute_queue_size() > 0:
          for req in llmactor.recompute_store.items:
            if req.item.target_latency != np.inf:
              active_targets.add(req.item.target_latency)
        if llmactor.get_decoded_queue_size() > 0:
          for req in llmactor.decoded_store.items:
            if (req.target_latency != np.inf) and (self.env.now - req.arrival_time < time_windows) :
              active_targets.add(req.target_latency)
      return active_targets

    def getViolationsTargetLatencyInWindow(self, time_windows = 300, percentile = 0.04) -> bool:
      """
        Checks for latency violations within a specified time window.

        :param time_windows: Time window in which to check for latency violations.
        :param percentile: The violation threshold percentile.
        :return: Boolean indicating if violations occurred. And %  of violations per target latency.
      """
      didViolate = False
      violation_dict = {}
      req_dict = {}
      for llmactor in self.list_of_llmactors:
        if llmactor.get_decoded_queue_size() > 0:
          for req in llmactor.decoded_store.items :
              if  (req.target_latency == np.inf)  or  (self.env.now - req.arrival_time > time_windows):
                continue
              if req.target_latency not in violation_dict:
                violation_dict[req.target_latency] = 0
              if req.target_latency not in req_dict:
                req_dict[req.target_latency] = 0

              req_dict[req.target_latency] += 1
              if ((req.end_decode_time - req.arrival_time)/req.output_size > req.target_latency ):
                  violation_dict[req.target_latency] += 1


      for target_latency in violation_dict:
        if violation_dict[target_latency]/req_dict[target_latency] > percentile:
          didViolate = True
        violation_dict[target_latency] = violation_dict[target_latency]/req_dict[target_latency]
      return didViolate, violation_dict


    def allPodsRunningCritical(self):
      """
        Checks if all pods are running critical requests.
      """
      pods_running_critical = set()
      for llmactor in self.list_of_llmactors:
        if llmactor.get_prefill_queue_size() > 0:
          for req in llmactor.prefill_store.items:
            if req.target_latency != np.inf:
              pods_running_critical.add(llmactor.id)
        if llmactor.get_decode_queue_size() > 0:
          for req in llmactor.decode_store.items:
            if req.target_latency != np.inf:
              pods_running_critical.add(llmactor.id)
        if llmactor.get_recompute_queue_size() > 0:
          for req in llmactor.recompute_store.items:
            if req.item.target_latency != np.inf:
              pods_running_critical.add(llmactor.id)
      return len(pods_running_critical) == len(self.list_of_llmactors)



    def generate_request_inference_gateway(
        self, rate, lora_requested, target_latency_list, prefix_latency_list, 
        routing_type, prompt_output_tuple=None, mean_request_size=None, 
        std_request_size=None, mean_output_size=None, std_output_size=None, 
        estimated_output_size=None):
      """
      Generates and routes requests through the inference gateway based on the provided parameters.
      """
      cnt = 0
      timeout_interval = 1 / rate

      while True:
        if self.messages_remaining_cnt is not None and self.messages_remaining_cnt <= 0:
            break

        target_latency_index = random.choice(range(len(target_latency_list)))
        target_latency = target_latency_list[target_latency_index]
        prefix = prefix_latency_list[target_latency_index]

        input_size, output_size = self.determine_request_sizes(
            prompt_output_tuple, mean_request_size, std_request_size, 
            mean_output_size, std_output_size
        )
        input_size = min(input_size, MAX_NUM_BATCH_TOKENS)

        request_id = f"{prefix}: {cnt}"
        new_req = self.create_request(request_id, input_size, output_size, target_latency)
        cnt += 1
        self.messages_remaining_cnt -= 1

        if self.should_enqueue_request(routing_type):
            self.enqueue_request(new_req, lora_requested, target_latency)
        else:
            self.route_request(new_req, routing_type, input_size, output_size, target_latency, lora_requested, estimated_output_size)

        yield self.env.timeout(timeout_interval)

    def determine_request_sizes(self, prompt_output_tuple, mean_request_size, std_request_size, mean_output_size, std_output_size):
      if prompt_output_tuple is None:
        input_size = determine_size(mean_request_size, std_request_size, None, None)
        output_size = determine_size(mean_output_size, std_output_size, None, None)
      else:
        input_output_size = random.choice(prompt_output_tuple)
        input_size = input_output_size[0]
        output_size = input_output_size[1]
      return input_size, output_size

    def create_request(self, request_id, input_size, output_size, target_latency):
      new_req = create_request(request_id, self.env.now, input_size, output_size)
      new_req.target_latency = target_latency
      return new_req

    def should_enqueue_request(self, routing_type):
      if self.queueing_perc == np.inf:
        return False
      return self.queueing_signal(routing_type) or not self.check_if_queues_empty()

    def enqueue_request(self, new_req, lora_requested, target_latency):
      if lora_requested:
        new_req.lora = lora_requested
      target_queue = self.queues.get(target_latency, Queue())
      target_queue.put(new_req)
      self.queues[target_latency] = target_queue


    def route_request(self, new_req, routing_type, input_size, output_size, target_latency, lora_requested, estimated_output_size):
      estimated_output_size = output_size if estimated_output_size is None else estimated_output_size
      target_pod, estimated_latency = self.find_target_pod(
        routing_type, input_size, estimated_output_size, target_latency, lora_requested
      )

      if target_pod:
        new_req.target_pod = target_pod.id
        new_req.estimated_latency = estimated_latency
        new_req.queue_size_before_prefill = target_pod.get_prefill_queue_size()
        new_req.pending_tokens_at_arrival_perc = self.get_pending_tokens_perc(target_pod)
        new_req.actual_tokens_at_arrival_perc = self.get_actual_tokens_perc(target_pod)

        if lora_requested:
            new_req.lora = lora_requested

        target_pod.prefill_store.put(new_req)
        self.req_dict_prefill[new_req.id] = new_req





    def process(self,rate, lora_requested, target_latency_list, prefix_latency_list, routing_type, prompt_output_tuple, mean_request_size, std_request_size, mean_output_size, std_output_size, estimated_output_size):
           self.env.process(self.generate_request_inference_gateway( rate, lora_requested, target_latency_list, prefix_latency_list, routing_type, prompt_output_tuple,
                                       mean_request_size = mean_request_size,
                                       std_request_size=std_request_size,
                                       mean_output_size = mean_output_size,
                                       std_output_size=std_output_size, estimated_output_size = estimated_output_size))
           self.env.process(self.dequeue_process(routing_type), )
           for llmactor in self.list_of_llmactors:
              self.env.process(prefill_or_decode(self.env, llmactor, self.req_dict_prefill, self.req_dict))




    def metrics(self):
        for llmactor in self.list_of_llmactors:
              self.env.process(metrics(self.env, llmactor))

