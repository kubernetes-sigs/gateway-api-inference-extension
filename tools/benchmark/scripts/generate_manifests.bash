#!/bin/bash

main(){
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
  source ${SCRIPT_DIR}/env.sh

  if [[ -z ${run_id} ]]; then
      run_id=$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -d'-' -f1)  # Get the first part (8 hex characters)
  else
    echo "Generating manifests using run id ${run_id}"
  fi

  # Generate benchmark manifests
  # First we generate the ModelServer and LoadBalancer manifests. We will generate BenchmarkTool
  # manifest after we deploy the ModelServer and LoadBalancer, because we need runtime information.
  echo "Generating ModelServer and LoadBalancer manifests for benchmarks ${benchmarks}"
  go run ${SCRIPT_DIR}/../manifestgenerator/main.go \
  --catalogDir="${SCRIPT_DIR}/../catalog/" \
  --outputRootDir="${SCRIPT_DIR}/../${output_dir}" \
  --benchmarks="${benchmarks}.pbtxt" \
  --manifestTypes="ModelServer,LoadBalancer" \
  --runID="${run_id}" \
  --override=${override} \
  --v=1
}

# Env vars to be passed when calling this script.
# Example usage: benchmarks="example" ./generate_manifests.bash
# benchmarks is the file name of a benchmark pbtxt file under catalog/benchmark
benchmarks=${benchmarks:-"example"}
run_id=${run_id:-""}
override=${override:-"false"}
output_dir=${output_dir:-'output'}
main