## 🚀 Deploying `inference-perf` via Helm Chart

This guide explains how to deploy `inference-perf` to a Kubernetes cluster with Helm.

Note: This is a temporary chart added until remote chart is available.

---

### 1. Prerequisites

Make sure you have the following tools installed and configured:

* **Kubernetes Cluster:** Access to a functional cluster (e.g., GKE).
* **Helm:** The Helm CLI installed locally.

---

### 2. Configuration (`values.yaml`)

Before deployment, navigate to the **`deploy/inference-perf`** directory and edit the **`values.yaml`** file to customize your deployment and the benchmark parameters.

#### Optional Parameters

| Key | Description | Default |
| :--- | :--- | :--- |
| `hfToken` | Hugging Face API token. If provided, a Kubernetes `Secret` named `hf-token-secret` will be created for authentication. | `""` |
| `serviceAccountName` | Standard Kubernetes `serviceAccountName`. If not provided, default service account is used. | `""` |
| `nodeSelector` |  Standard Kubernetes `nodeSelector` map to constrain pod placement to nodes with matching labels. | `{}` |
| `resources` | Standard Kubernetes resource requests and limits for the main `inference-perf` container. | `{}` |
---

> **Example Resource Block:**
> ```yaml
> # resources:
> #   requests:
> #     cpu: "1"
> #     memory: "4Gi"
> #   limits:
> #     cpu: "2"
> #     memory: "8Gi"
> ```

#### GKE Specific Parameters

This section details the necessary configuration and permissions for using a Google Cloud Storage (GCS) path to manage your dataset, typical for deployments on GKE.

##### Required IAM Permissions

The identity executing the workload (e.g., the associated Kubernetes Service Account, often configured via **Workload Identity**) must possess the following IAM roles on the target GCS bucket for data transfer:

* **`roles/storage.objectViewer`** (Required to read/download the input dataset from GCS).
* **`roles/storage.objectCreator`** (Required to write/push benchmark results back to GCS).


| Key | Description | Default |
| :--- | :--- | :--- |
| `gcsPath` | A GCS URI pointing to the dataset file (e.g., `gs://my-bucket/dataset.json`). The file will be automatically copied to the running pod during initialization. | `""` |

---

### 3. Run Deployment

Use the **`helm install`** command from the **`deploy/inference-perf`** directory to deploy the chart.

* **Standard Install:** Deploy using the default `values.yaml`.
    ```bash
    helm install test .
    ```

* **Set `hfToken` Override:** Pass the Hugging Face token directly.
    ```bash
    helm install test . --set hfToken="<TOKEN>"
    ```

* **Custom Config Override:** Make changes to the values file for custom settings.
    ```bash
    helm install test . -f values.yaml
    ```

### 4. Cleanup

To remove the benchmark deployment.
    ```bash
    helm uninstall test
    ```