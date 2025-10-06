#!/bin/bash

# Downloads a file from a GCS bucket.

# Check if GCS_BUCKET is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <GCS_BUCKET> [GCS_FOLDER_PATH:DEFAULT=benchmark_results]"
  exit 1
fi

GCS_BUCKET="$1"
GCS_FOLDER_PATH="${2:-benchmark_results/}" # Default to benchmark_results/ if not provided

# Env vars to be passed when calling this script.
# The id of the benchmark. This is needed to identify what the benchmark is for.
# It decides the filepath to save the results, which later is used by the jupyter notebook to assign
# the benchmark_id as data labels for plotting.
benchmark_id=${benchmark_id:-"inference-extension"}
# run_id can be used to group different runs of the same benchmarks for comparison.
run_id=${run_id:-"default-run"}
output_dir=${output_dir:-'output'}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
benchmark_output_dir=${SCRIPT_DIR}/${output_dir}/${run_id}/${benchmark_id}

echo "Creating output directory: ${benchmark_output_dir}/results/json/"
mkdir -p "${benchmark_output_dir}/results/json/"

echo "Downloading gs://${GCS_BUCKET}/${GCS_FOLDER_PATH} to ${benchmark_output_dir}/results/json/"
gsutil cp -r "gs://${GCS_BUCKET}/${GCS_FOLDER_PATH}" "${benchmark_output_dir}/results/json/"

echo "Download complete."
