#!/usr/bin/env bash
#
# Builds this folder's Python environment and registers its Jupyter kernel.
#
#     ./setup.sh
#
# Then open encode_decode_test.ipynb and select "NeuralGCM release".
# See README.md if you would rather do it by hand.
#
# CPU-only, as the notebook's default path needs no GPU. For GPU acceleration
# (only relevant with REGENERATE_FROM_SCRATCH = True) see requirements.txt.

set -euo pipefail
cd "$(dirname "$0")"
exec ../common/make_env.sh \
    "neuralgcm-release" "NeuralGCM release" 3
