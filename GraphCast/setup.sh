#!/usr/bin/env bash
#
# Builds this folder's Python environment and registers its Jupyter kernel.
#
#     ./setup.sh
#
# Then open any notebook in this folder and select "GraphCast release".
# See README.md if you would rather do it by hand.
#
# This is the big one: jax[cuda12] + torch cu124 is about 6.4GB of wheels, so
# the script checks for free space first and passes --no-cache-dir to keep pip
# from holding a second copy. For a CPU-only install instead, edit
# requirements.txt as described at the bottom of that file and install by hand.

set -euo pipefail
cd "$(dirname "$0")"
exec ../common/make_env.sh \
    "graphcast-release" "GraphCast release" 8 --no-cache-dir
