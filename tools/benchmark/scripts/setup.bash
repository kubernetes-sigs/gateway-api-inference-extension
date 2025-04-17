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

  if [[ -d "${benchmark_output_dir}/results/json" ]]; then
    echo "The JSON results ${benchmark_output_dir}/results/json already exists, skipping. If you want to re-run the benchmark, delete the results directory and try again."
    return
  fi

  if [[ ${provider} == "gke" ]]; then
    echo "Configuring GKE cluster"
    configure_gke
  fi

  # Create namespace if it doesn't exist.
  kubectl create namespace "${namespace}" --dry-run=client -o yaml | kubectl apply -f -
  # Copy existing HF secret to the new namespace. This is assuming you created a hf secret in the default namespace following guide
  # https://gateway-api-inference-extension.sigs.k8s.io/guides/#deploy-sample-model-server
  kubectl get secret hf-token --namespace=default -oyaml | grep -v '^\s*namespace:\s' | kubectl apply --namespace=${namespace} -f -
  
}


configure_gke() {
  gcloud config configurations create benchmark-catalog
  gcloud config configurations activate benchmark-catalog
  gcloud config set project ${BENCHMARK_PROJECT}
  gcloud config set billing/quota_project ${BENCHMARK_PROJECT}
  gcloud config set container/cluster ${CLUSTER_NAME}
  gcloud config set compute/zone ${LOCATION}
  # Configure kubectl
  gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${LOCATION} --project ${BENCHMARK_PROJECT}

  echo "Binding KSA to GSA for metrics scraping"
  gcloud iam service-accounts add-iam-policy-binding \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:${BENCHMARK_PROJECT}.svc.id.goog[${namespace}/default]" \
  gmp-test-sa@${BENCHMARK_PROJECT}.iam.gserviceaccount.com  --project ${BENCHMARK_PROJECT}
  
  kubectl annotate serviceaccount \
  --namespace ${namespace} \
  default \
  iam.gke.io/gcp-service-account=gmp-test-sa@${BENCHMARK_PROJECT}.iam.gserviceaccount.com
}

# Cloud provider where the cluster runs. Optional.
# If provided, the tool can automate additional features applicable to this provider. For example,
# on GKE, it can configure permissions to query cloud monitoring to get model server metrics.
provider=${provider:-""}
benchmark_id=${benchmark_id:-"please-provide-benchmark-id"}
output_dir=${output_dir:-'output'}
main