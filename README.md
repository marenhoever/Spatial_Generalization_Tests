# Spatial Generalization Tests for Machine Learning-based Weather Models

Code accompanying the paper *"Spatial Generalization Tests for Machine
Learning-based Weather Models to Assess Physical Consistency"* (Höver,
Klöwer, Schroeder de Witt, Christensen) —
[arXiv:2607.20716](https://arxiv.org/abs/2607.20716). See `CITATION.cff` if
you use this code.

The paper tests whether three weather models — a physics-based model
(SpeedyWeather) and two machine-learning models (NeuralGCM, GraphCast) —
behave consistently under spatial transformations of their inputs (rotating
longitude, reversing latitude/longitude) that should leave the underlying
physics unchanged. This repo reproduces the paper's figures from each model.

## Disclaimer

The code for this project was initially primarily human-written. However, Claude
Code has been tasked to clean up the code and ensure scientific reproducibility.
Some supporting code, such as the Zenodo data-fetching helper and the
configuration and data-preparation cells in the notebooks, was written by Claude
Code during that cleanup. For transparency, you can find the
`claude_code_clean.md` file in this repo.

## Layout

| Folder | Model | Language/framework |
|---|---|---|
| `SpeedyWeather/` | Physics-based baseline | Julia |
| `NeuralGCM/` | ML weather model | Python/JAX |
| `GraphCast/` | ML weather model | Python/JAX |

Each model folder is a self-contained environment with its own README —
start there for setup and run instructions. The three models are otherwise
independent of each other (no shared code between them besides
`common/zenodo_fetch.py`, a small shared helper for downloading external data).

## Figures

Each model writes its figures to `<model>/figures/`, and each one corresponds to
a figure in the paper:

| Paper | File | Produced by |
|---|---|---|
| Figure 3 | `GraphCast/figures/graphcast_rmse_vs_leadtime.jpeg` | `GraphCast/plot_forecast_errors.ipynb` |
| Figure 4 | `GraphCast/figures/toa_datetime_perturbation_linearity.png` | `GraphCast/perturbation_experiments.ipynb` |
| Figure 5 | `GraphCast/figures/rmse_and_skill_score_24h.jpeg` | `GraphCast/plot_forecast_errors.ipynb` |
| Figure 6 | `NeuralGCM/figures/neuralgcm_encode_decode_error_grid.png` | `NeuralGCM/encode_decode_test.ipynb` |
| Figure 7 | `GraphCast/figures/input_transformations_overview.jpeg` | `GraphCast/perturbation_experiments.ipynb` |
| Figure 8 | `GraphCast/figures/rmse_by_transform_and_season.jpeg` | `GraphCast/plot_forecast_errors.ipynb` |
| Figure 9 | `GraphCast/figures/speedyweather_rmse_vs_leadtime.jpeg` | `GraphCast/plot_forecast_errors.ipynb` |
| Figure 10 | `GraphCast/figures/toa_datetime_rmse_vs_leadtime.png` | `GraphCast/perturbation_experiments.ipynb` |
| Figure 11 | `GraphCast/figures/rotated_geopotential_effect.png` | `GraphCast/perturbation_experiments.ipynb` |

Figures 1 and 2 are not produced by this code.

Figure 9 is also produced by `SpeedyWeather/reproduce_published_figure/plot_published_figure.ipynb`,
into that folder — the same figure, but plotted from the original SpeedyWeather
data rather than from the copy staged for the GraphCast notebook. See
`SpeedyWeather/README.md` for why both exist.

One known difference from the published paper: Figure 8 in arXiv v1 was rendered
from an earlier revision of the forecast-error arrays, so it does not match what
this repository produces. The version here is the correct one — see the note in
`GraphCast/README.md`.

## Data

None of the data lives in this repository. Every notebook downloads what it
needs from Zenodo the first time you run it, checks each file against a recorded
checksum, and skips anything already on disk. Files land under `data/` at the
repository root, in the layout the notebooks expect.

> **The data currently lives on Zenodo's sandbox, not on Zenodo itself.** The
> paper is still under review, and the real records will be published — with real
> DOIs — once it is accepted. The record ids and the sandbox address are already
> filled into each notebook's config cell, so everything runs as-is; the sandbox
> block is a clearly marked handful of lines to delete when the real records exist.
> Sandbox records are wiped periodically, so if a download starts failing, that is
> why.

The record ids are set in a config cell near the top of each notebook:

```python
ZENODO_RECORD_GRAPHCAST_ERA5 = "586387"
```

The data is split across eight records, because a single Zenodo record is
limited to 50GB *and* to 100 files — two independent caps, and the file count is
not raised by a storage-quota increase.

| Variable | Contents | Files | Size |
|---|---|---|---|
| `ZENODO_RECORD_SPEEDYWEATHER` | SpeedyWeather transformation-test error fields | 3 | 8.8 MiB |
| `ZENODO_RECORD_NEURALGCM` | ERA5 regridded onto the NeuralGCM grid | 1 | 12.0 GiB |
| `ZENODO_RECORD_GRAPHCAST_ERA5` | ERA5 regridded onto the GraphCast grid, the 30-year climatology, the precipitation accumulation, and the perturbation-experiment forecasts | 14 | 34.9 GiB |
| `ZENODO_RECORD_GRAPHCAST_PREDICTIONS` | 2m-temperature forecast-error arrays, one per spatial transform | 7 | 24.8 GiB |
| `ZENODO_RECORD_GRAPHCAST_FC_BASELINE` | daily forecasts, untransformed baseline | 13 | 24.9 GiB |
| `ZENODO_RECORD_GRAPHCAST_FC_REVLAT` | daily forecasts, reversed latitude | 13 | 24.9 GiB |
| `ZENODO_RECORD_GRAPHCAST_FC_REVLON` | daily forecasts, reversed longitude | 13 | 24.9 GiB |
| `ZENODO_RECORD_GRAPHCAST_FC_ROTLON` | daily forecasts, longitude rotated 180° | 13 | 24.9 GiB |

Which notebook needs which record:

| Notebook | Records it reads |
|---|---|
| `SpeedyWeather/reproduce_published_figure/plot_published_figure.ipynb` | SPEEDYWEATHER |
| `NeuralGCM/encode_decode_test.ipynb` | NEURALGCM |
| `GraphCast/compute_era5_climatology.ipynb`, `prepare_era5_input.ipynb`, `perturbation_experiments.ipynb` | GRAPHCAST_ERA5 |
| `GraphCast/run_forecasts.ipynb` | GRAPHCAST_ERA5, and the FC record for each test case it differences |
| `GraphCast/plot_forecast_errors.ipynb` | GRAPHCAST_ERA5, GRAPHCAST_PREDICTIONS, SPEEDYWEATHER |

### File names on Zenodo

A Zenodo record is a flat list of files — it has no directories. Most files are
therefore named the same on Zenodo as they are on disk here.

The seven forecast-error arrays are the exception. Six of them share just two
filenames, so on Zenodo each carries the name of its test case as a prefix:

| Prefix | Test case |
|---|---|
| `baseline` | untransformed |
| `revlon` | longitude reversed |
| `revlat` | latitude reversed |
| `rotlon` | longitude rotated by 180° |

These four names are the canonical ones, used throughout the paper and the
notebooks. For each of the three transforms there are two arrays:
`<case>__error_2mT_era5_corrected.npy` is that transform's forecast error, and
`<case>__error_2mT_era5_corrected_baseline.npy` is the untransformed error over
the same days, to difference it against. `baseline__error_2mT_era5.npy` is the
standalone untransformed array. Each is 365 days x 40 lead times x 181 latitudes
x 360 longitudes, float32.

The notebooks file these into subdirectories of `data/GraphCast/predictions/`
for you. You only need the mapping if you are downloading by hand:

| On Zenodo | Goes to, under `data/GraphCast/predictions/` |
|---|---|
| `baseline__error_2mT_era5.npy` | `not_flipped/error_2mT_era5.npy` |
| `revlon__error_2mT_era5_corrected.npy` | `axis_flip/incl_flipped_wind/error_2mT_era5_corrected.npy` |
| `revlon__error_2mT_era5_corrected_baseline.npy` | `axis_flip/incl_flipped_wind/error_2mT_era5_corrected_baseline.npy` |
| `revlat__error_2mT_era5_corrected.npy` | `equatorial_flip/incl_flipped_wind/error_2mT_era5_corrected.npy` |
| `revlat__error_2mT_era5_corrected_baseline.npy` | `equatorial_flip/incl_flipped_wind/error_2mT_era5_corrected_baseline.npy` |
| `rotlon__error_2mT_era5_corrected.npy` | `shift_180/minus_12h/error_2mT_era5_corrected.npy` |
| `rotlon__error_2mT_era5_corrected_baseline.npy` | `shift_180/minus_12h/error_2mT_era5_corrected_baseline.npy` |

### The daily forecasts

The four `..._FC_...` records hold the forecasts those error arrays were derived
from — 366 daily files per test case, each a 40-step (10-day) six-hourly rollout
carrying all seven variables GraphCast writes. The error arrays are 2m
temperature only, so six of the seven exist nowhere else.

366 files per record would exceed Zenodo's 100-file cap, so each record holds 13
uncompressed zip archives instead, one per calendar month:

| On Zenodo | Holds |
|---|---|
| `<case>__2021-12.zip` | day 000 |
| `<case>__2022-01.zip` … `<case>__2022-12.zip` | days 001–365, by month |

Forecast day *d* is initialised at 2021-12-31T18:00 + *d* days, so day 000 is the
only day in 2021-12 and days 001–365 are exactly the calendar year 2022. Each
archive stores its files under their on-disk paths, so unzipping all thirteen
into `data/GraphCast/` rebuilds the tree directly. `run_forecasts.ipynb` fetches
only the months it needs, verifies each archive and then each file inside it, and
deletes the archive once unpacked.

### Downloading by hand

If you would rather not let the notebooks download for you — or one fails
partway — fetch the files from the record pages and put them at the paths above.
The notebooks skip anything already present, so they will pick up from there. The
expected md5 of every file is written into the `ensure_zenodo_files(...)` call
that asks for it, so you can check a manual download against the same value.

`common/zenodo_fetch.py` does the fetching. It downloads to a `.part` file and
renames only once the checksum matches, so an interrupted download is never
mistaken for a complete one.

### Data not distributed here

One thing the notebooks produce is not archived, because it is an intermediate
rather than an input: the 151 per-chunk regridded ERA5 files (~32GB), which
`prepare_era5_input.ipynb` Part 1 writes and Part 2 consumes — only the combined
file they build is archived.

The daily forecasts *are* archived, in the four `..._FC_...` records above. If
you set `REGENERATE_FROM_SCRATCH = True` in `run_forecasts.ipynb` it writes them
itself; otherwise it downloads the months it needs. Either way the baseline case
is required alongside any other, since every test case is differenced against
it.

Both source archives the notebooks read from are public and need no credentials:
ERA5 from `gs://gcp-public-data-arco-era5`, and the model checkpoints from
Google's `dm_graphcast` and `neuralgcm` buckets.

## License

This repository's own code is MIT-licensed — see `LICENSE`. `GraphCast/`
additionally incorporates code adapted from Google DeepMind's GraphCast demo,
which remains under its original Apache License 2.0 — see
`GraphCast/LICENSE-APACHE-2.0`.
