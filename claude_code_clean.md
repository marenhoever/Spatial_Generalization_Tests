# Task: Clean spatial-generalization code for GitHub release

## Context

This folder contains the code used to produce the data and figures for the
paper at https://arxiv.org/abs/2607.20716. It uses three models:

- **SpeedyWeather** (Julia)
- **NeuralGCM** (Python)
- **GraphCast** (Python)

The goal is a single new folder that can be pushed to GitHub as-is and gives
another researcher everything needed to reproduce the paper's figures: clean
notebooks, working environments, and clear documentation of external data
dependencies.

## Non-negotiable rules

1. **Never modify the original notebooks.** Copy them into the new working
   folder first; all cleaning happens on copies.
2. **Never fix bugs or mistakes found in plot-producing code.** The code that
   generates a published figure must remain functionally identical to what
   produced the paper, even if you spot an error. Instead, log every
   suspected mistake in a separate `mistakes_log.md` (notebook, cell, and a
   one-line description) and leave the code untouched.
3. **Credit collaborators.** If `era5_data.ipynb` (written by Emily Morris,
   not me) ends up being the regridding approach kept in the final repo, add
   a credit line to her in that notebook's header and in the top-level
   README.
4. **Track every external file dependency as you go.** Any time a notebook
   reads from a path outside this repo (GWS, JASMIN, scratch space, etc.),
   add it to a running `external_data_inventory.md` so I know what needs to
   go to Zenodo before this is public.

## Scope, model by model

### SpeedyWeather (Julia)
- Only notebook needed: `reverse_rotate_full_model_with_saving.ipynb`, in the
  `SpeedyWeather/` folder. This one is already clean and produces all needed
  outputs — do not restructure its content, only handle environment/paths.
- It was built on a different server, so assume the Julia environment does
  not transfer as-is. Rebuild it fresh (see Environments section).

### NeuralGCM
- Only notebook needed: `rotate_180.ipynb`. All other NeuralGCM notebooks in
  the folder are not used and can be excluded from the new repo.
- This notebook is responsible for exactly one paper figure. Confirm which
  output file it produces and check it against
  `spatial_generalization_test_grid_with_manual_baseline_limits_thermal_map.png`
  in the target figure list below — the filename I have on hand
  (`..._thermal_map_150.png`) does not exactly match the published figure
  name, so verify this is the same plot before assuming it is done.

### GraphCast
- `TOA_clean.ipynb` and `the_final_notebook.ipynb` both produce figures used
  in the paper. Consolidate what you reasonably can, but for each remaining
  cell that produces a plot, note in a markdown comment which published
  figure(s) it corresponds to.
- `GraphCast_Testing_Spatial_Generalizability_Demo.ipynb` is a lightweight
  demo using GraphCast's built-in `example_batch` — much cheaper to run than
  the full pipeline. Keep this as a separate "quickstart" notebook rather
  than merging it into the main GraphCast notebook, since its value is being
  runnable without heavy compute or the full ERA5 dataset.
- `Coarsen_ERA5.ipynb` and `era5_data.ipynb` both regrid ERA5 data (plus some
  precipitation-related processing). Determine which one is actually in the
  dependency chain for the plot-producing notebooks above. If only one is
  used, drop the other. If both are genuinely needed for different things,
  keep both and document why. Remember the credit requirement for
  `era5_data.ipynb` above.

## Target figures

These are the exact filenames of every figure that appears in the
publication. For each one, trace it back to the specific notebook and cell
that generates it. Flag any you cannot locate rather than guessing.

```
Combined_perturbation_linearity_letter_labeled.png
combined_RMSE_SS_2rows_no_formula_letter_labels.jpeg
datetime_and_TOA_changes_error_over_time.png
Inputs_overview_selection2_corrected.jpeg
Only_speedy_errors_three_black.jpeg
RMSE_2mT_by_season_updated_incl_flipped_wind.jpeg
RMSE_combined_plot_updated_with_new_SpeedyRuns_colorblind2_dashed_line_different_title_simple.jpeg
Sim1_sim2_diff_labeled.png
spatial_generalization_test_grid_with_manual_baseline_limits_thermal_map.png
```

Note: `RMSE_2mT_by_season_updated_incl_flipped_wind.jpeg` is not obviously
covered by the notebooks named above — the code for it may be in an older,
not-yet-listed notebook. Search the full folder contents (not just the files
named in this prompt) for candidates before concluding it is missing, and
tell me what you find.

## Process

Work through these phases in order. Do not start Phase 2 for a given
notebook until Phase 1 is complete for it — consolidating before you know
what actually matters just means redoing work.

**Phase 1 — Inventory (read-only, no edits)**
For every notebook in scope, record: every file path it reads or writes
(local and external), every package it imports, and which cells produce
which output files. Write this to `inventory.md` before touching any code.

**Phase 2 — Consolidate and clean**
Remove unrelated/exploratory cells. Keep the logic of any cell that
influences a published figure byte-for-byte equivalent — reorganizing cell
order or removing genuinely dead code around it is fine, changing its
behavior is not. Add markdown cells documenting what each remaining section
does and which figure(s) it produces.

**Phase 3 — Hardcode paths at the top**
For every remaining hardcoded path (including my username, any
`atmlxgpu2`/JASMIN/GWS paths, or absolute paths from the other server the
SpeedyWeather notebook was built on), move it into a clearly labeled config
cell at the very start of the notebook, so a new user only has to edit one
place.

**Phase 4 — Environments**
Do not freeze the existing `GraphCast_env` / `NeuralGCM_env` directly, since
they likely contain packages that are no longer used. Instead, determine the
actual imports per notebook and build a minimal environment from that. Pin
exact versions for anything numerically sensitive (JAX, CUDA/cuDNN pairing,
numpy) and document the tested CUDA/driver combination explicitly, given
past environment issues on this setup. For SpeedyWeather, produce a
`Project.toml`/`Manifest.toml` pinning the Julia package versions, separate
from the Python environments. Rebuild each environment from scratch (not by
editing the existing one) and run each notebook top-to-bottom in it before
considering that model done.

**Phase 5 — External data inventory**
Consolidate `external_data_inventory.md`: every external file path found
during Phases 1–3, with a note on what it is and, if you can determine it,
roughly how large it is. This is what I will use to decide what needs to go
to Zenodo.

**Phase 6 — Repo hygiene**
- `.gitignore` covering model checkpoints, large data files, notebook
  checkpoints, and any SLURM log/output files.
- Check for and remove any hardcoded credentials or API keys.
- LICENSE file.
- Top-level README: what the repo is, links to the paper, how the three
  model folders relate, and how to set up and run each.
- One short README per model folder with setup + run instructions specific
  to that environment.
- `CITATION.cff` for the paper, since this accompanies a journal submission.

**Phase 7 — Final validation**
Clone the new folder into a clean location, build each environment from the
committed environment files only, and run every notebook top-to-bottom with
no reliance on anything outside the repo (no pre-set shell variables, no
cached data, nothing from the original working directories). Only report
this as done once that clean run actually succeeds.

## Deliverables

By the end, the new folder should contain:
- Cleaned, copied notebooks (SpeedyWeather ×1, NeuralGCM ×1, GraphCast ×3–4
  depending on the regridding-notebook decision)
- `inventory.md`
- `mistakes_log.md`
- `external_data_inventory.md`
- Environment files (Python envs + Julia `Project.toml`/`Manifest.toml`)
- Top-level README + per-model README
- `.gitignore`, `LICENSE`, `CITATION.cff`

Work incrementally and show me the inventory and mistake log as you find
things, rather than only at the very end.
