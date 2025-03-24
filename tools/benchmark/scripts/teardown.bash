#!/bin/bash

main(){
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
  source ${SCRIPT_DIR}/env.sh

  namespace=${benchmark_id}
  benchmark_output_dir=${SCRIPT_DIR}/../${output_dir}/${run_id}/${benchmark_id}

  echo "Tearing down benchmark ${run_id}/${benchmark_id}"

  if [[ ${env} == "gke" ]]; then
    echo "Tearing down GKE cluster"
    tear_down_gke
  fi


  kubectl delete -f ${benchmark_output_dir}/manifests/BenchmarkTool.yaml --grace-period=0 --force
  kubectl delete -f ${benchmark_output_dir}/manifests/ModelServer.yaml --grace-period=0 --force
  kubectl delete -f ${benchmark_output_dir}/manifests/LoadBalancer.yaml --grace-period=0 --force
  kubectl delete namespace ${namespace} --grace-period=0 --force
}

tear_down_gke() {
  gcloud iam service-accounts remove-iam-policy-binding \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:${BENCHMARK_PROJECT}.svc.id.goog[${namespace}/default]" \
  gmp-test-sa@${BENCHMARK_PROJECT}.iam.gserviceaccount.com --project ${BENCHMARK_PROJECT}
}

env=${env:-"gke"}
# The id of the benchmark under output/${run_id} folder. Make sure the manifests are already generated before calling this script.
benchmark_id=${benchmark_id:-"please-provide-benchmark-id"}
output_dir=${output_dir:-'output'}
main