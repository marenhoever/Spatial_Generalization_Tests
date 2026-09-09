#!/usr/bin/env bash
#
# Builds this folder's Python environment and registers its Jupyter kernel.
#
#     ./setup.sh
#
# Then open plot_published_figure.ipynb and select "SpeedyWeather repro (Python)".
# See README.md (one level up) if you would rather do it by hand.
#
# This is the cheapest thing in the repository to run: 5 direct dependencies,
# ~9MB of data, no GPU. A good first test that a new machine works.

set -euo pipefail
cd "$(dirname "$0")"
exec ../../common/make_env.sh \
    "speedyweather-repro-env" "SpeedyWeather repro (Python)" 1
