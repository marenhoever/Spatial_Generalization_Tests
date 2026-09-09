# GraphCast

Six notebooks, covering ERA5 data prep, the main spatial-generalization
experiment, and the TOA/datetime/rotated-geopotential perturbation
experiments.

| Notebook | Role | Produces |
|---|---|---|
| `compute_era5_climatology.ipynb` | Fetches (or optionally re-derives) 2 climatology files used for normalization by `plot_forecast_errors.ipynb`. Adapted from Emily Morris's `era5_data.ipynb` — see credit in its header cell. | — (intermediate data) |
| `prepare_era5_input.ipynb` | Builds the combined ERA5 file from the public ARCO-ERA5 archive. Runnable end to end, but slow and disk-hungry (~32GB of chunks + 33GB output); its final cell fetches the finished file instead if you don't want to re-derive it. | `regridded_era5_data_for_GraphCast.nc` |
| `run_forecasts.ipynb` | Generation notebook — produces the 2m-temperature forecast-error arrays `plot_forecast_errors.ipynb` reads. Only 1 of 4 experiment variants has fully-preserved runnable code (see its own top markdown cell for a real, unresolved provenance gap on the other 3); gated behind `REGENERATE_FROM_SCRATCH` (default off) since all 7 output arrays already exist. | prediction `.npy` arrays |
| `perturbation_experiments.ipynb` | Self-contained runs+plots notebook (cheap enough — ~8 forecasts — to not split generation from plotting). | Figures 4, 10, 7, 11 — `toa_datetime_perturbation_linearity.png`, `toa_datetime_rmse_vs_leadtime.png`, `input_transformations_overview.jpeg`, `rotated_geopotential_effect.png` |
| `plot_forecast_errors.ipynb` | Plotting-only — reads pre-computed prediction/error arrays (GraphCast's own, the climatology files, and SpeedyWeather's), no model-running needed. Not every figure needs every input — see the data-dependency note below. | Figures 5, 3, 9, 8 — `rmse_and_skill_score_24h.jpeg`, `graphcast_rmse_vs_leadtime.jpeg`, `speedyweather_rmse_vs_leadtime.jpeg`, `rmse_by_transform_and_season.jpeg` |
| `quickstart_demo.ipynb` | Standalone quickstart — runs the real GraphCast model on its own lightweight built-in demo data (no ERA5/large downloads needed). Kept separate deliberately, since its whole value is being cheap to run. | — (demo only, no target figure) |

Data-dependency note for `plot_forecast_errors.ipynb`: the "Produces" column lists
the notebook's four figures, and the "Role" column lists the notebook's inputs
— but the two lists do not pair up one-to-one. Only two of the four figures use
the SpeedyWeather error files:

| Figure | Cell | Needs SpeedyWeather `.nc`? |
|---|---|---|
| `rmse_and_skill_score_24h.jpeg` (Fig 5) | 38 | no — GraphCast error arrays + climatology only (and it is written *before* the SpeedyWeather files are opened, in cell 40) |
| `graphcast_rmse_vs_leadtime.jpeg` (Fig 3) | 43 | **yes** |
| `speedyweather_rmse_vs_leadtime.jpeg` (Fig 9) | 46 | **yes** |
| `rmse_by_transform_and_season.jpeg` (Fig 8) | 50 | no — GraphCast error arrays only |

Note that the Config cell defines all three `ZENODO_RECORD_*` variables up
front, so a straight top-to-bottom run still needs `ZENODO_RECORD_SPEEDYWEATHER`
filled in even if you only want the two figures that don't depend on it.

### Known difference from the published figure

`rmse_by_transform_and_season.jpeg` (Figure 8) produced here does **not**
match the version in arXiv v1, and the version here is the correct one.

The published render predates the final revision of the forecast-error arrays: it
shows a baseline 2m-temperature RMSE of ≈2.1 K at day 10, where the archived arrays
give **2.649 K**. The plotting code is unchanged and unaffected — only the input
data was newer when this repository's copy was made. The discrepancy is visible
inside the paper itself: the same quantity appears as the black dashed "Forecast
error (ERA5)" line of `graphcast_rmse_vs_leadtime.jpeg` (Figure 3), where it ends
at 2.6 K.
All four panels shift upward, so the shared y-axis runs to 7 here rather than 5;
the qualitative result — transformed inputs give larger errors than the baseline —
is unchanged. The figure will be corrected at the next manuscript revision.

## Setup

GPU by default — the demo notebook and any `REGENERATE_FROM_SCRATCH` path run
real GraphCast model forward passes, which are heavy on CPU. The other
notebooks' default (non-regenerate) paths are comparatively cheap (fetch
precomputed data + plot, or at most load checkpoint metadata) and would work
fine CPU-only, but GPU is set as the default to cover the demo notebook
without a separate environment.

**Python 3.11 or 3.12.** The pinned versions were resolved and validated on 3.12,
and re-checked to resolve identically on 3.11. Anything older will fail: several
of the pins require 3.11 or newer. If the machine has neither, `uv` can install a
Python without root — `curl -LsSf https://astral.sh/uv/install.sh | sh`, then
`uv python install 3.12` and `uv venv --python 3.12 env`.

The quickest way is to let the script do it — it finds a suitable interpreter
(or installs one via `uv` if you have neither), builds the environment from
`requirements.lock`, and registers the kernel:

```bash
cd GraphCast   # this folder
./setup.sh
```

On a cluster, note that `setup.sh` deliberately keeps `uv`'s interpreters and
all wheel caches in `.uv/` at the repository root rather than the usual
`~/.local` and `~/.cache`. A quota'd home directory is the normal case there,
and a per-user quota is invisible to `df`, so the install would otherwise pass
the free-space check and then fail partway with "Disk quota exceeded". Set
`UV_PYTHON_INSTALL_DIR`, `UV_CACHE_DIR` or `PIP_CACHE_DIR` to override.

By hand instead: check the interpreter first, and chain the steps, so a missing
one stops the sequence rather than quietly installing into whatever environment
is already active:

```bash
python3 --version   # must be 3.11 or 3.12; otherwise name one, e.g. python3.11
```

```bash
cd GraphCast && \
python3 -m venv env && \
source env/bin/activate && \
pip install --no-cache-dir -r requirements.lock && \
python -m ipykernel install --user --name graphcast-release --display-name "GraphCast release"
```

**`requirements.lock` vs `requirements.txt`.** The lock file is the fully
resolved environment — all ~150 packages, including the transitive ones
`requirements.txt` leaves to pip's resolver, which drift as PyPI moves. Install
the lock; read `requirements.txt`, which stays the hand-maintained list of
direct pins and the reasoning behind each. This environment is the one that has
already been broken once by an unpinned transitive dependency (dm-haiku 0.0.17
against jax 0.4.29). Regenerating the lock, if you ever change a pin, needs
`uv` — the command is in the lock file's own header.

The install is ~6.4GB of wheels (jax[cuda12] + torch cu124), so make sure that
much space is free first; `--no-cache-dir` keeps pip from keeping a second copy.
Then open any notebook and select that kernel.

For a CPU-only install instead (works for everything except actually running
the model), see the commented-out block at the bottom of `requirements.txt`.

**GraphCast package note:** DeepMind renamed/merged the GraphCast repository
into `google-deepmind/weathernext`
(`github.com/deepmind/graphcast` now redirects there). `requirements.txt`
installs from a pinned commit of the new repo rather than the notebooks'
original floating `.../archive/master.zip` reference, for reproducibility —
see the comment above that line in `requirements.txt` for what was verified
about compatibility with what these notebooks were actually validated
against.

## Data

All external data (regridded ERA5, climatology files, prediction arrays,
simulation outputs) is downloaded from Zenodo automatically via each
notebook's Config cell and checksum-verified, skipping anything already
present locally. It is spread over 8 records — a Zenodo record is capped both at
50GB and at 100 files, two independent limits — of which the notebooks in this
folder read seven (all but the NeuralGCM one). See the Data section of
`../README.md` for which record each notebook needs.

## License

This folder incorporates code adapted from Google DeepMind's GraphCast demo
(Apache License 2.0) — see `LICENSE-APACHE-2.0`. Four notebooks contain such
code and each carries the copyright notice and a statement of modification in
its header cell: `quickstart_demo.ipynb`, `prepare_era5_input.ipynb`,
`run_forecasts.ipynb` and `perturbation_experiments.ipynb`. The rest of this
repository is MIT-licensed — see `../LICENSE`.

The model weights and normalization statistics the notebooks download from
Google's `dm_graphcast` bucket are not covered by that Apache license — DeepMind
releases them under CC BY 4.0. They are fetched at runtime and are not
redistributed here.
