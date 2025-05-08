#!/bin/bash

# Downloads the benchmark result files from the benchmark tool pod.
download_benchmark_results() {
  until echo $(kubectl logs deployment/benchmark-tool -n ${namespace}) | grep -q -m 1 "LPG_FINISHED"; do sleep 30 ; done;
      benchmark_pod=$(kubectl get pods -l app=benchmark-tool -n ${namespace} -o jsonpath="{.items[0].metadata.name}")
      echo "Downloading JSON results from pod ${benchmark_pod}"
      kubectl exec ${benchmark_pod} -n ${namespace} -- rm -f ShareGPT_V3_unfiltered_cleaned_split.json
      for f in $(kubectl exec ${benchmark_pod} -n ${namespace} -- /bin/sh -c ls -l | grep json); do
        echo "Downloading json file ${f}"
        kubectl cp -n ${namespace} ${benchmark_pod}:$f ${benchmark_output_dir}/results/json/$f; 
      done
}

# Env vars to be passed when calling this script.
# The id of the benchmark. This is needed to identify what the benchmark is for.
# It decides the filepath to save the results, which later is used by the jupyter notebook to assign
# the benchmark_id as data labels for plotting. 
benchmark_id=${benchmark_id:-"inference-extension"}
# run_id can be used to group different runs of the same benchmarks for comparison.
run_id=${run_id:-"default-run"}
namespace=${namespace:-"default"}
output_dir=${output_dir:-'output'}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
benchmark_output_dir=${SCRIPT_DIR}/${output_dir}/${run_id}/${benchmark_id}

echo "Saving benchmark results to ${benchmark_output_dir}/results/json/"
download_benchmark_results
kubectl delete -f ${SCRIPT_DIR}/../../config/manifests/benchmark/benchmark.yaml