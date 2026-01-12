#!/bin/bash
# Entrypoint for test container
# Handles optional training warmup before running tests

set -e

# Check if training warmup is enabled
if [[ "${RUN_TRAINING_WARMUP}" == "true" ]]; then
  # Only the first pod (index 0) sends training data
  # Other pods wait for models to be ready
  if [[ "$JOB_COMPLETION_INDEX" == "0" ]]; then
    echo "Pod 0: Running training warmup..."
    python3 setup_training_data.py || exit 1
  else
    echo "Pod $JOB_COMPLETION_INDEX: Waiting for models to be ready..."
    # Wait for pod 0 to finish training
    sleep 45
    # Reload models
    python3 -c "import requests; requests.post('${PREDICTION_SERVER_URL}/reload', timeout=20)" || echo "Warning: Model reload failed"
  fi
else
  echo "Skipping training warmup (RUN_TRAINING_WARMUP=${RUN_TRAINING_WARMUP})"
fi

# Run the actual test command
exec "$@"
