#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../backend"
export SPECTRASCOPE_DATA_DIR="$(mktemp -d)"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
trap 'rmdir "$SPECTRASCOPE_DATA_DIR" 2>/dev/null || true' EXIT
python -m unittest benchmarks.test_catalog_egress benchmarks.test_data_policy benchmarks.test_db2_contract benchmarks.test_distributed_queue benchmarks.test_insight_llm_boundary benchmarks.test_internal_http benchmarks.test_native_bridge_policy benchmarks.test_public_raster_transport benchmarks.test_resource_admission benchmarks.test_worker_transport
