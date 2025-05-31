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

import argparse
from collections import Counter
import csv
from datetime import datetime
import numpy as np
import simpy
from llmactor import LLMActor
from loadbalancer import LoadBalancer

def main():
    parser = argparse.ArgumentParser(description="Simulate LLM load balancing with configurable parameters.")
    parser.add_argument("--rates_lo", nargs='+', type=int, default=[35, 30, 25, 20, 15, 10, 5, 1], help="List of low rates.")
    parser.add_argument("--rates_hi", nargs='+', type=int, default=[35, 30, 25, 20, 15, 10, 5, 1], help="List of high rates.")
    parser.add_argument("--no_of_messages", type=int, default=2500, help="Number of messages to simulate.")
    parser.add_argument("--mean_request_size_1", type=int, default=202, help="Mean request size for set 1.")
    parser.add_argument("--std_request_size_1", type=int, default=20, help="Standard deviation of request size for set 1.")
    parser.add_argument("--mean_output_size_1", type=int, default=179, help="Mean output size for set 1.")
    parser.add_argument("--std_output_size_1", type=int, default=17, help="Standard deviation of output size for set 1.")
    parser.add_argument("--mean_request_size_2", type=int, default=202, help="Mean request size for set 2.")
    parser.add_argument("--std_request_size_2", type=int, default=20, help="Standard deviation of request size for set 2.")
    parser.add_argument("--mean_output_size_2", type=int, default=179, help="Mean output size for set 2.")
    parser.add_argument("--std_output_size_2", type=int, default=17, help="Standard deviation of output size for set 2.")
    parser.add_argument("--queueing_perc", type=float, default=np.inf, help="Queueing percentage.")
    parser.add_argument('--target-latency-lo', nargs='+', type=float, help='List of target latencies for low priority requests.')
    parser.add_argument('--target-latency-hi', nargs='+', type=float, help='List of target latencies for high priority requests.')
    parser.add_argument('--prefix-latency-lo', nargs='+', type=float, help='List of prefix of target latencies for low priority requests.')
    parser.add_argument('--prefix-latency-hi', nargs='+', type=float, help='List of prefix of target latencies for high priority requests.')
    parser.add_argument('--number-of-servers', type=int, default=1, help='List of target latencies for high priority requests.')
    
    args = parser.parse_args()

    # Use provided arguments or defaults
    rates_lo = args.rates_lo
    rates_hi = args.rates_hi
    no_of_messages = args.no_of_messages
    SIM_DURATIONS = [no_of_messages / r + 100 for r in rates_lo]
    mean_request_size_1 = args.mean_request_size_1
    std_request_size_1 = args.std_request_size_1
    mean_output_size_1 = args.mean_output_size_1
    std_output_size_1 = args.std_output_size_1
    mean_request_size_2 = args.mean_request_size_2
    std_request_size_2 = args.std_request_size_2
    mean_output_size_2 = args.mean_output_size_2
    std_output_size_2 = args.std_output_size_2
    queueing_perc = args.queueing_perc
    lora_requested_lo = ""
    lora_requested_hi = ""
    target_latency_list_lo = args.target_latency_lo if args.target_latency_lo else [0.025]
    target_latency_list_hi = args.target_latency_hi if args.target_latency_hi else [0.5]
    prefix_latency_list_lo = args.prefix_latency_lo if args.prefix_latency_lo else ['lo']
    prefix_latency_list_hi = args.prefix_latency_hi if args.prefix_latency_hi else ['hi']
    number_of_servers = args.number_of_servers

    # Define a structure to store results for all routing types
    results = {
        'leastPseudo': {'latency': [], 'latency_lo': [], 'latency_hi': [],
               'throughput_prefill': [], 'throughput_decode': [],
               'throughput_prefill_lo': [], 'throughput_decode_lo': [],
               'throughput_prefill_hi': [], 'throughput_decode_hi': [],
               'ttft': [], 'ttft_lo': [], 'ttft_hi': [],
               'tpot': [], 'tpot_lo': [], 'tpot_hi': [],
               'target_pods_lo': [], 'target_pods_hi': [],
               'recompute_cnt' : [], 'recompute_cnt_hi' : [], 'recompute_cnt_lo' : [],
               'pct_below_latency_target_lo': [], 'pct_below_latency_target_hi': [],  'queue_time_lo': [], 'queue_time_hi': [],
              'tol_lat_time_lo': [], 'tol_lat_time_hi': [], 
              'avg_prefill_queue_size' : [],
'avg_pending_tokens_perc' : [],
'avg_actual_tokens_perc' : []},

    'smart': {'latency': [], 'latency_lo': [], 'latency_hi': [],
              'estimated_latency': [], 'estimated_latency_lo': [], 'estimated_latency_hi': [],
              'throughput_prefill': [], 'throughput_decode': [],
              'throughput_prefill_lo': [], 'throughput_decode_lo': [],
              'throughput_prefill_hi': [], 'throughput_decode_hi': [],
              'ttft': [], 'ttft_lo': [], 'ttft_hi': [],
              'tpot': [], 'tpot_lo': [], 'tpot_hi': [],
              'target_pods_lo': [], 'target_pods_hi': [],
              'recompute_cnt' : [], 'recompute_cnt_hi' : [], 'recompute_cnt_lo' : [],
              'pct_below_latency_target_lo': [], 'pct_below_latency_target_hi': [],
               'pct_below_latency_target_lo': [], 'pct_below_latency_target_hi': [], 'queue_time_lo': [], 'queue_time_hi': [],
              'tol_lat_time_lo': [], 'tol_lat_time_hi': [], 
              'avg_prefill_queue_size' : [],
'avg_pending_tokens_perc' : [],
'avg_actual_tokens_perc' : []},


    'leastlatency': {'latency': [], 'latency_lo': [], 'latency_hi': [],
                'throughput_prefill': [], 'throughput_decode': [],
                'throughput_prefill_lo': [], 'throughput_decode_lo': [],
                'throughput_prefill_hi': [], 'throughput_decode_hi': [],
                'ttft': [], 'ttft_lo': [], 'ttft_hi': [],
                'tpot': [], 'tpot_lo': [], 'tpot_hi': [],
                'target_pods_lo': [], 'target_pods_hi': [],
                'recompute_cnt' : [], 'recompute_cnt_hi' : [], 'recompute_cnt_lo' : [],
                'pct_below_latency_target_lo': [], 'pct_below_latency_target_hi': [], 'queue_time_lo': [], 'queue_time_hi': [],
              'tol_lat_time_lo': [], 'tol_lat_time_hi': [], 
              'avg_prefill_queue_size' : [],
'avg_pending_tokens_perc' : [],
'avg_actual_tokens_perc' : []},

    'least': {'latency': [], 'latency_lo': [], 'latency_hi': [],
                'throughput_prefill': [], 'throughput_decode': [],
                'throughput_prefill_lo': [], 'throughput_decode_lo': [],
                'throughput_prefill_hi': [], 'throughput_decode_hi': [],
                'ttft': [], 'ttft_lo': [], 'ttft_hi': [],
                'tpot': [], 'tpot_lo': [], 'tpot_hi': [],
                'target_pods_lo': [], 'target_pods_hi': [],
                'recompute_cnt' : [], 'recompute_cnt_hi' : [], 'recompute_cnt_lo' : [],
               'pct_below_latency_target_lo': [], 'pct_below_latency_target_hi': [], 'queue_time_lo': [], 'queue_time_hi': [],
              'tol_lat_time_lo': [], 'tol_lat_time_hi': [], 
              'avg_prefill_queue_size' : [],
'avg_pending_tokens_perc' : [],
'avg_actual_tokens_perc' : []},

    'random': {'latency': [], 'latency_lo': [], 'latency_hi': [],
                'throughput_prefill': [], 'throughput_decode': [],
                'throughput_prefill_lo': [], 'throughput_decode_lo': [],
                'throughput_prefill_hi': [], 'throughput_decode_hi': [],
                'ttft': [], 'ttft_lo': [], 'ttft_hi': [],
                'tpot': [], 'tpot_lo': [], 'tpot_hi': [],
                'target_pods_lo': [], 'target_pods_hi': [],
                'recompute_cnt' : [], 'recompute_cnt_hi' : [], 'recompute_cnt_lo' : [],
               'pct_below_latency_target_lo': [], 'pct_below_latency_target_hi': [], 'queue_time_lo': [], 'queue_time_hi': [],
              'tol_lat_time_lo': [], 'tol_lat_time_hi': [], 
              'avg_prefill_queue_size' : [],
'avg_pending_tokens_perc' : [],
'avg_actual_tokens_perc' : []},

}

    all_routing_types = [ "random",  ]
    prompt_output_tuple = None

# Iterate over routing types
    for routing_type in all_routing_types:
        print(f'Routing Type: {routing_type}')

        for i, _ in enumerate(rates_lo):
            req_dict = {}
            req_dict_prefill = {}
            SIM_DURATION = SIM_DURATIONS[i]
            print(f'Simulate with rate: for lo {rates_lo[i]} and for hi {rates_hi[i]} and routing type: {routing_type}')

            # Simpy environment and LLM actors setup
            env = simpy.Environment()
            list_of_llmactors = [LLMActor(env, 1, id) for id in range(number_of_servers)]
            lb = LoadBalancer(env, number_of_servers=number_of_servers, list_of_llmactors=list_of_llmactors, req_dict_prefill=req_dict_prefill, req_dict=req_dict, messages_remaining_cnt=no_of_messages)
            lb.queueing_perc = queueing_perc

            estimated_output_size = mean_output_size_1
            lb.process(rates_lo[i], lora_requested_lo, target_latency_list_lo, prefix_latency_list_lo, routing_type, prompt_output_tuple, mean_request_size_1, std_request_size_1, mean_output_size_1, std_output_size_1, estimated_output_size)
            env.run(until=SIM_DURATION)

            # Completed requests
            completed_req = list(filter(lambda x: x.output_size_remaining == 0, req_dict.values()))
            completed_req_sorted = sorted(completed_req, key=lambda x: x.arrival_time)
            # Exclude the first 10% of requests based on end_decode_time
            exclude_count = int(0 * len(completed_req_sorted))
            # Filter out the first 10%
            filtered_req = completed_req_sorted[exclude_count:]

            # Calculate ttft, tpot, latency, and throughput
            ttft_cur = np.mean([x.end_prefill_time - x.arrival_time for x in req_dict.values()])
            tpot_cur = np.mean([(x.end_decode_time - x.start_prefill_time) / (x.output_size - x.output_size_remaining) for x in req_dict.values()])
            latency_cur = np.mean([(x.end_decode_time - x.arrival_time) / (x.output_size - x.output_size_remaining) for x in filtered_req])
            estimated_latency_cur = np.mean([x.estimated_latency for x in filtered_req])
            recompute_cur = np.sum([x.recompute_count for x in filtered_req]) / len(filtered_req)
            tt = SIM_DURATION
            throughput_prefill_cur = np.sum([x.input_size for x in filtered_req]) / tt
            throughput_decode_cur = np.sum([max(0, x.output_size - x.output_size_remaining - 1) for x in filtered_req]) / tt

            pending_tokens_at_arrival_perc = [x.pending_tokens_at_arrival_perc for x in completed_req]
            actual_tokens_at_arrival_perc = [x.actual_tokens_at_arrival_perc for x in completed_req]
            prefill_queue_size = [x.queue_size_before_prefill for x in completed_req]

            # Store results for the current routing type
            results[routing_type]['latency'].append(latency_cur)
            results[routing_type]['throughput_prefill'].append(throughput_prefill_cur)
            results[routing_type]['throughput_decode'].append(throughput_decode_cur)
            results[routing_type]['ttft'].append(ttft_cur)
            results[routing_type]['tpot'].append(tpot_cur)
            results[routing_type]['recompute_cnt'].append(recompute_cur)
            results[routing_type]['avg_prefill_queue_size'].append(np.mean(prefill_queue_size))
            results[routing_type]['avg_pending_tokens_perc'].append(np.mean(pending_tokens_at_arrival_perc))
            results[routing_type]['avg_actual_tokens_perc'].append(np.mean(actual_tokens_at_arrival_perc))

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create the output file name with the timestamp
    output_file = f"results_{timestamp}.csv"

    # Write results to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['RoutingType', 'RateIndex', 'Latency', 'avg_prefill_queue_size', 'avg_pending_tokens_perc', 'avg_actual_tokens_perc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Iterate over routing types and write each entry
        for routing_type in all_routing_types:
            for i in range(len(rates_lo)):
                writer.writerow({
                    'RoutingType': routing_type,
                    'RateIndex': rates_lo[i],
                    'Latency': results[routing_type]['latency'][i],
                    'avg_prefill_queue_size': results[routing_type]['avg_prefill_queue_size'][i],
                    'avg_pending_tokens_perc': results[routing_type]['avg_pending_tokens_perc'][i],
                    'avg_actual_tokens_perc': results[routing_type]['avg_actual_tokens_perc'][i],
                })

    print(f"Results have been saved to {output_file}")

if __name__ == "__main__":
    main()
