#!/bin/bash

# Downloads files from a GCS or S3 bucket.

# Check if provider and bucket are provided as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <gcs|s3> <BUCKET> [FOLDER_PATH:DEFAULT=benchmark_results]"
  exit 1
fi

PROVIDER="$1"
BUCKET="$2"
FOLDER_PATH="${3:-benchmark_results/}" # Default to benchmark_results/ if not provided

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

case "$PROVIDER" in
  gcs)
    echo "Downloading gs://${BUCKET}/${FOLDER_PATH} to ${benchmark_output_dir}/results/json/"
    gsutil cp -r "gs://${BUCKET}/${FOLDER_PATH}" "${benchmark_output_dir}/results/json/"
    ;;
  s3)
    echo "Downloading s3://${BUCKET}/${FOLDER_PATH} to ${benchmark_output_dir}/results/json/"
    aws s3 cp -r "s3://${BUCKET}/${FOLDER_PATH}" "${benchmark_output_dir}/results/json/"
    ;;
  *)
    echo "Invalid provider: $PROVIDER. Please use 'gcs' or 's3'."
    exit 1
    ;;
esac

echo "Download complete."
