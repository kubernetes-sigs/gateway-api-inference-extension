Before starting, verify that your environment meets the prerequisites:

1. Check your Kubernetes version:

   ```bash
   kubectl version --short
   ```

   Ensure you're running one of the three most recent Kubernetes minor releases.

2. Verify cluster connectivity:

   ```bash
   kubectl get nodes
   ```

   All nodes should be in `Ready` status.

3. Check if required tools are installed:

   ```bash
   helm version
   jq --version
   ```

If any of these checks fail, please refer to the [Prerequisites](#prerequisites) section above for installation instructions.