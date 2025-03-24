#!/bin/bash

main(){
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
  source ${SCRIPT_DIR}/env.sh

  if [[ -z ${run_id} ]]; then
      run_id=$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -d'-' -f1)  # Get the first part (8 hex characters)
  fi
  echo "Using run id ${run_id}"

  namespace=${benchmark_id}
  benchmark_output_dir=${SCRIPT_DIR}/../${output_dir}/${run_id}/${benchmark_id}
  benchmark_file_path="${benchmark_output_dir}/benchmark.pbtxt"

  if [[ -d "${benchmark_output_dir}/results/json" ]]; then
    echo "The JSON results ${benchmark_output_dir}/results/json already exists, skipping. If you want to re-run the benchmark, delete the results directory and try again."
    echo "Attempting to tear down ${run_id}/${benchmark_id} anyway in case there were dangling resources"
    if [[ "${skip_tear_down}" == "true" ]]; then
      echo "Skipping tearing down benchmark"
    else
      run_id=${run_id} benchmark_id=${benchmark_id} ${SCRIPT_DIR}/teardown.bash
    fi
    return
  fi

  run_id=${run_id} benchmark_id=${benchmark_id} ${SCRIPT_DIR}/setup.bash

  start_time=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
  
  echo "Deploying benchmark environment"
  kubectl apply -f ${benchmark_output_dir}/manifests/ModelServer.yaml
  kubectl apply -f ${benchmark_output_dir}/manifests/LoadBalancer.yaml
  echo "Waiting for deployments to be ready before starting the benchmark tool"
  wait_deployments_ready

  echo "Generating BenchmarkTool manifest after the LoadBalancer is deployed, because we need to derive the IP of the load balancer to send traffic to"
  go run ${SCRIPT_DIR}/../manifestgenerator/main.go \
  --catalogDir="${SCRIPT_DIR}/../catalog/" \
  --outputRootDir="${SCRIPT_DIR}/../${output_dir}" \
  --benchmarkFilePath="${benchmark_file_path}" \
  --manifestTypes="BenchmarkTool" \
  --runID="${run_id}" \
  --v=1
  echo "Deploying benchmark tool"
  kubectl apply -f ${benchmark_output_dir}/manifests/BenchmarkTool.yaml
  wait_deployments_ready

  echo "Collecting benchmark results"
  download_benchmark_results

  if [[ -z ${gcs_bucket} ]]; then
    echo "Skipping uploading to GCS as gcs bucket is not provided"
  else
    echo "Uploading output ${benchmark_output_dir} to GCS bucket"
    gcloud storage rsync -r ${benchmark_output_dir} gs://${gcs_bucket}/${output_dir}/staging/${benchmark_id}
  fi

  end_time=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
  dashboard="https://pantheon.corp.google.com/monitoring/dashboards/builder/32b5c92a-a12d-40ba-8bd6-b8c0a62a8d8e;startTime=${start_time};endTime=${end_time};filters=type:rlabel,key:namespace,val:${namespace}?&mods=logs_tg_prod&project=${BENCHMARK_PROJECT}"
  metadata="start_time: $start_time\nend_time: $end_time\ncloud_monitoring_dashboard: ${dashboard}"
  echo "metadata:\n $metadata"
  echo -e "$metadata" > ${benchmark_output_dir}/metadata.txt

  # Tear down benchmark
  if [[ "${skip_tear_down}" == "true" ]]; then
    echo "Skipping tearing down benchmark"
  else
    run_id=${run_id} benchmark_id=${benchmark_id} ${SCRIPT_DIR}/teardown.bash
  fi
}

wait_deployments_ready() {
  kubectl wait --for=condition=available --timeout=6000s $(kubectl get deployments -o name -n ${namespace}) -n ${namespace}
}

# Downloads the benchmark result files from the benchmark tool pod.
download_benchmark_results() {
  local benchmark_pod
  local pod_finished=false

  while true; do
    # Check if the pod has finished
    if echo $(kubectl logs deployment/benchmark-tool -n "${namespace}") | grep -q -m 1 "LPG_FINISHED"; then
      pod_finished=true
      echo "Benchmark tool pod has finished."
    fi

    # Get the benchmark pod name
    benchmark_pod=$(kubectl get pods -l app=benchmark-tool -n "${namespace}" -o jsonpath="{.items[0].metadata.name}")
    if [[ -z "${benchmark_pod}" ]]; then
      echo "Benchmark pod not found yet. Retrying in 30 seconds..."
      sleep 30
      continue
    fi

    echo "Checking for new results from pod ${benchmark_pod}"

    # Download JSON files
    local json_files=$(kubectl exec "${benchmark_pod}" -n "${namespace}" -- /bin/sh -c "ls -l | grep benchmark-catalog.*json | awk '{print \$9}'")
    for f in $json_files; do
      local local_json_path="${benchmark_output_dir}/results/json/${f}"
      if [[ ! -f "${local_json_path}" ]]; then
        echo "Downloading json file ${f}"
        mkdir -p "$(dirname "${local_json_path}")"
        kubectl cp -n "${namespace}" "${benchmark_pod}:${f}" "${local_json_path}"
      else
        echo "json file ${f} already exists locally, skipping download."
      fi
    done

    # Download TXT files
    local txt_files=$(kubectl exec "${benchmark_pod}" -n "${namespace}" -- /bin/sh -c "ls -l | grep txt | awk '{print \$9}'")
    for f in $txt_files; do
      local local_txt_path="${benchmark_output_dir}/results/txt/${f}"
      if [[ ! -f "${local_txt_path}" ]]; then
        echo "Downloading txt file ${f}"
        mkdir -p "$(dirname "${local_txt_path}")"
        kubectl cp -n "${namespace}" "${benchmark_pod}:${f}" "${local_txt_path}"
      else
        echo "txt file ${f} already exists locally, skipping download."
      fi
    done

    if [[ "${pod_finished}" == "true" ]]; then
      # Download logs
      local local_log_path="${benchmark_output_dir}/results/benchmark-tool.log"
      echo "Downloading logs from pod ${benchmark_pod}"
      mkdir -p "$(dirname "${local_log_path}")"
      kubectl logs deployment/benchmark-tool -n "${namespace}" > "${local_log_path}"
      echo "All files downloaded and pod finished. Exiting."
      break
    else
      echo "Waiting for new files or pod to finish. Retrying in 30 seconds..."
      sleep 30
    fi
  done
}

# Env vars to be passed when calling this script.
# Example usage: benchmark="example-benchmark-config" skip_tear_down="true" gcs_bucket="benchmark-inference-gateway" ./run_benchmark.bash
gcs_bucket=${gcs_bucket:-""}
# The id of the benchmark under output/${run_id} folder. Make sure the manifests are already generated before calling this script.
benchmark_id=${benchmark_id:-"please-provide-benchmark-id"}
# set skip_tear_down to true to preserve the env after benchmark.
skip_tear_down=${skip_tear_down:-"false"}
output_dir=${output_dir:-'output'}
main