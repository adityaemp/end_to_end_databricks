#!/usr/bin/env bash
# Run every bundle-defined job sequentially (non-interactive).
# Usage: ./scripts/run_all_bundle_jobs.sh [target]
# Example: ./scripts/run_all_bundle_jobs.sh dev
#
# Order: features → training → batch inference → sensor ETL → monitoring retraining.
# Adjust JOBS_ORDER below if your pipeline needs a different sequence.

set -euo pipefail

TARGET="${1:-dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BUNDLE_ROOT}"

JOBS_ORDER=(
  write_feature_table_job
  model_training_job
  batch_inference_job
  sensor_streaming_job
  retraining_job
)

for key in "${JOBS_ORDER[@]}"; do
  echo "========== bundle run ${key} (-t ${TARGET}) =========="
  databricks bundle run "${key}" -t "${TARGET}"
done

echo "All jobs finished."
