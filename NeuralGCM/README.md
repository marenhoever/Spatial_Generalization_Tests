# NeuralGCM

`encode_decode_test.ipynb` produces `neuralgcm_encode_decode_error_grid.png`
(**Figure 6** of the paper):
a 4x4 grid comparing a pretrained NeuralGCM model's encode/decode
reconstruction error under 4 spatial transformations (baseline, rotate
longitude, reverse latitude, reverse longitude), plus each transform's
difference from the untransformed baseline.

## Setup

CPU-only by default (the notebook's own default path loads a cached 12.9GB
regridded-ERA5 file and a small pretrained checkpoint — no heavy on-the-fly
computation, so GPU isn't needed unless you set `REGENERATE_FROM_SCRATCH =
True` in the Config cell, which re-derives that file from the raw public
ARCO-ERA5 archive).

**Python 3.11 or 3.12.** The pinned versions were resolved and validated on 3.12,
and re-checked to resolve identically on 3.11. Anything older will fail: several
of the pins require 3.11 or newer. If the machine has neither, `uv` can install a
Python without root — `curl -LsSf https://astral.sh/uv/install.sh | sh`, then
`uv python install 3.12` and `uv venv --python 3.12 env`.

The quickest way is to let the script do it — it finds a suitable interpreter
(or installs one via `uv` if you have neither), builds the environment from
`requirements.lock`, and registers the kernel:

```bash
cd NeuralGCM   # this folder
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
cd NeuralGCM && \
python3 -m venv env && \
source env/bin/activate && \
pip install -r requirements.lock && \
python -m ipykernel install --user --name neuralgcm-release --display-name "NeuralGCM release"
```

**`requirements.lock` vs `requirements.txt`.** The lock file is the fully
resolved environment — all 118 packages, including the transitive ones
`requirements.txt` leaves to pip's resolver, which drift as PyPI moves. Install
the lock; read `requirements.txt`, which stays the hand-maintained list of
direct pins. Regenerating the lock, if you ever change a pin, needs `uv` — the
command is in the lock file's own header.
Then open `encode_decode_test.ipynb` and select that kernel.

For GPU acceleration (only relevant if regenerating ERA5 from scratch), see
the commented-out install line at the bottom of `requirements.txt`.

## Data

`combined_regridded_data_era5.nc` (12.9GB) is downloaded from Zenodo
automatically via the notebook's Config cell (`ZENODO_RECORD_NEURALGCM`) if it
is not already present in `../data/NeuralGCM/regridded_era5/`, and verified
against its checksum — see the Data section of `../README.md`.

The pretrained model checkpoint (`v1_precip/stochastic_precip_2_8_deg.pkl`)
and the raw ARCO-ERA5 archive used by the optional regeneration path are both
public GCS buckets, fetched anonymously — no credentials or extra setup
needed.
