#!/bin/bash

# Env vars to be passed when calling this script.
# Example usage: benchmarks="example" benchmark_name_regex="c1.*" gcs_bucket="benchmark-inference-gateway" ./run_benchmark.bash
benchmarks=${benchmarks:-"example"}
dry_run=${dry_run:-"false"}
gcs_bucket=${gcs_bucket:-""}
# set skip_tear_down to true to preserve the env after benchmark.
skip_tear_down=${skip_tear_down:-"false"}
benchmark_name_regex=${benchmark_name_regex:-".*"}
output_dir=${output_dir:-'output'}


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/env.sh

if [[ -z ${run_id} ]]; then
    run_id=$(uuidgen | tr '[:upper:]' '[:lower:]'| cut -d'-' -f1)  # Get the first part (8 hex characters)
else
  echo "Using existing run id ${run_id}"
fi

# Generate benchmark manifests
# First we generate the ModelServer and LoadBalancer manifests. We will generate BenchmarkTool
# manifest after we deploy the ModelServer and LoadBalancer, because we need runtime information.
echo "Generating ModelServer and LoadBalancer manifests for benchmarks ${benchmarks}"
run_id=${run_id} benchmarks=${benchmarks} ${SCRIPT_DIR}/generate_manifests.bash
benchmarks_output_dir=${SCRIPT_DIR}/../${output_dir}/${run_id}
if [[ "${dry_run}" == "true" ]]; then
  echo "Dry-run=${dry_run}. Skipping deploying the benchmark. You can check the generated manifest at ${benchmarks_output_dir}"
  return 
fi

echo "Run generated benchmark one by one"
while read -r folder; do
  benchmark_output_dir=$(basename "$folder")
  echo "Running benchmark for $benchmark_output_dir"
  run_id=${run_id} benchmark_id=${benchmark_output_dir} ${SCRIPT_DIR}/run_one_benchmark.bash
done < <(find "${benchmarks_output_dir}/" -maxdepth 1 -mindepth 1 -type d -regex "$benchmark_name_regex" -print | sort )
