# FFT versus nested-AD Laplacians in windowed IMEX-PINNs for Schnakenberg Turing patterns

Code and results accompanying the manuscript *"A Seed-Resolved Reliability Study of FFT versus Nested
Automatic-Differentiation Laplacians in Windowed IMEX-PINNs for Schnakenberg Turing Patterns"*
(Ngueabou and Oloniiju, submitted to the Journal of Computational Physics).

Archived code and metrics: https://doi.org/10.5281/zenodo.23602120. Field arrays (separate dataset): https://doi.org/10.5281/zenodo.23597440.

The study compares two realisations of the Laplacian inside an otherwise identical windowed IMEX-PINN:
**Method A**, a Fourier pseudospectral Laplacian of the sampled network output, and **Method B**, a nested
reverse-mode automatic-differentiation Laplacian. Nine parameter regimes, up to four seeds each.

## Contents

| Path | Description |
|---|---|
| `code/schnakenberg_v16_spectral_v10.py` | Method A (FFT Laplacian), stabiliser on |
| `code/schnakenberg_v16_spectral_v10_staboff.py` | Method A, stabiliser off |
| `code/schnakenberg_v16_autograd_squarev2.py` | Method B (nested AD Laplacian) |
| `analysis/` | Scripts that regenerate the figures and the cost table of the paper |
| `results/metrics/` | Per-run metrics (`*_all_metrics_s<seed>.json`): 21 folders, 4 seeds each (84 files) |
| `results/crosschecks/` | Text output of the per-seed cross-check of the final runs |
| `results/seed_outcome_table.csv` | One row per (regime, method, seed) |

The folders in `results/metrics/` are named by regime and method: `_A` is Method A (FFT), `_B` is Method B
(nested AD). `nearonset_A_stab_on/off` and `kstripes_A_stabon/staboff` hold the stabiliser-on and
stabiliser-off runs of Method A. `khpi_hexrect` and `khpi_B` are the `K_Hpi` main runs on the rectangular
domain; `khpi_square` is the square-domain control (Method A).

The large field arrays (`*_arrays.npz`, 84 files, 8.4 GB) are not in this repository. They are archived as a Zenodo dataset:
[https://doi.org/10.5281/zenodo.23597440](https://doi.org/10.5281/zenodo.23597440) (one zip per folder of `results/metrics/`,
plus `MANIFEST.csv` with SHA-256 checksums). Scripts in `analysis/` that read them need those files: unzip each
archive next to the matching folder of `results/metrics/`.

## Environment

- Python 3.12.13, TensorFlow 2.20.0 (Keras 3.13.2), NumPy 2.0.2, Matplotlib 3.10.0 (exact versions in `requirements.txt`)
- Results were produced on Kaggle sessions with NVIDIA Tesla T4 GPUs (16 GB; CUDA 12.5.1, cuDNN 9, float32). Runs are seed-controlled but not
  guaranteed bit-identical across hardware or library versions.

```bash
pip install -r requirements.txt
```

## Running an experiment

```bash
# list regimes
python code/schnakenberg_v16_spectral_v10.py --list-regimes

# Method A (FFT), regime S_spots, seed 42
python code/schnakenberg_v16_spectral_v10.py --regime S_spots --domain 2 --seed 42 --save-dir results/S_spots_A

# Method B (nested AD), same regime and seed
python code/schnakenberg_v16_autograd_squarev2.py --regime S_spots --domain 2 --seed 42 --save-dir results/S_spots_B
```

All reported runs used `--domain 2` (domain tier 2: $64\times64$ grid for the Regime I labels,
$96\times96$ for the `K_*` labels) and the seeds 42, 45, 49 and 52.
Regimes: `S_spots, M_spots, T_spots, near_onset, low_rho, min_rho` (Regime I) and
`K_stripes, K_mixed, K_Hpi` (Regime II).
Other options: `--quick` (short test run), `--no-early-stop`, `--ic-mode {eigenmode,broadband,triad}`,
`--force-domain-shape {square,hex_rect}`.
The `K_Hpi` main runs use the rectangular `hex_rect` domain with the triad initial condition; the square-domain control uses the eigenmode initial condition (see the paper and its domain table).

The scripts default to Kaggle/Drive output paths; always pass `--save-dir`.

## Outputs

Each run writes `<method>_d2_all_metrics_s<seed>.json` (`spectral_FFT_` for Method A, `autograd_` for Method B), a list of records with fields such as `regime`, `pattern`,
`anisotropy`, and the other per-run diagnostics reported in the paper. `results/seed_outcome_table.csv` lists the scalar fields of every record, one row per run.

## Reproducing the figures and the cost table

Each figure of the paper that is produced from the runs, with the script in `analysis/` that makes it.
The architecture figure and the figure of the two Laplacian realisations are TikZ drawings inside the
manuscript source and have no script.

| Figure in the paper | File | Script |
|---|---|---|
| Evolution strips, `T_spots`, seed 45, Methods A and B | `T_spots_A_s45_evolution.pdf`, `T_spots_B_s45_evolution.pdf` | `export_evolution_pdf.py` (commands below) |
| Evolution strips, `M_spots`, seed 45, Methods A and B | `M_spots_A_s45_evolution.pdf`, `M_spots_B_s45_evolution.pdf` | `make_evolution_figs_spots.py` (also writes the `S_spots` strips, which are not used in the paper) |
| Training curves, `T_spots` and `low_rho` | `fig_training_curves_Tspots.pdf`, `fig_training_curves_lowrho.pdf` | `fig3_training_curves.py` (`--regime`, `--out`) |
| `K_Hpi` seed 49 evolution, rectangular and square domain | `khpi_A_s49_evolution.pdf`, `khpi_square_A_s49_evolution.pdf` | `export_evolution_pdf.py` |
| Fourier montage | `figure_fourier_montage.pdf` | `fig_fourier_montage.py` |
| `K_Hpi` hexagon timelines, Methods A and B | `fig_khpi_A_timeline.pdf`, `fig_khpi_B_timeline.pdf` | `fig_khpi_hex_timeline.py` |
| Seed-outcome grid | `fig_seed_outcome_grid.pdf` | `make_fig_seed_outcome_grid.py` |
| Late-breakdown trajectories | `fig_late_breakdown.pdf` | `make_fig_late_breakdown.py` |
| Cost table (Table 5) | printed text and LaTeX | `make_cost_table.py` |

`fig_fourier_montage.py` and `fig_khpi_hex_timeline.py` import `verify_khpi_hexagonal_symmetry.py`, which is
in `analysis/` too; run them from that folder.

**Paths.** `make_fig_seed_outcome_grid.py`, `make_fig_late_breakdown.py` and `make_cost_table.py` were run on
Kaggle and set `ROOT` near the top to `/kaggle/working/...`. To run them on this repository, set `ROOT` to
`results/metrics` (they search it recursively). The scripts that read `*_arrays.npz` need the Zenodo dataset:
unpack its zips next to the corresponding metrics folders.

**`make_evolution_figs_spots.py`** calls `export_evolution_pdf.py` (3 time points, 300 dpi) for seed 45. It reads
`results/<regime>_<A|B>/*s45*arrays.npz`, so run it from a folder where `results/` contains those regime folders
(in this repository they are inside `results/metrics/`, so adjust the glob line or copy the folders) and where
`export_evolution_pdf.py` is present.

**Commands for the `T_spots` evolution strips** (run from `analysis/`; seed 45 as in the paper, any of 42, 45, 49, 52 works):

```bash
seed=45
python export_evolution_pdf.py \
    --npz ../results/metrics/T_spots_A/spectral_FFT_d2_T_spots_s${seed}_arrays.npz \
    --k-true 21.20 --n-mode 5 --domain-shape square \
    --n-timepoints 3 --dpi 600 --out T_spots_A_s${seed}_evolution.pdf

python export_evolution_pdf.py \
    --npz ../results/metrics/T_spots_B/autograd_d2_T_spots_s${seed}_arrays.npz \
    --k-true 21.20 --n-mode 5 --domain-shape square \
    --n-timepoints 3 --dpi 600 --out T_spots_B_s${seed}_evolution.pdf
```

The `*_arrays.npz` files are in the Zenodo dataset (DOI above); put them in the matching `results/metrics/` folders first.

**`K_Hpi` hexagon timelines** (run from `analysis/`; Method A on `khpi_hexrect`, Method B on `khpi_B`):

```bash
python fig_khpi_hex_timeline.py --npz-dir ../results/metrics/khpi_hexrect \
    --pattern "spectral_FFT_d2_K_Hpi_s{seed}_arrays.npz" --label "Method A (hex_rect)" \
    --seeds 42 45 49 52 --out fig_khpi_A_timeline

python fig_khpi_hex_timeline.py --npz-dir ../results/metrics/khpi_B \
    --pattern "autograd_d2_K_Hpi_s{seed}_arrays.npz" --label "Method B (hex_rect)" \
    --seeds 42 45 49 52 --out fig_khpi_B_timeline
```

**Fourier montage** (held seed 45 and transitioned seed 42 on the rectangular domain, square-domain control seed 45):

```bash
python fig_fourier_montage.py \
    --hold-npz ../results/metrics/khpi_hexrect/spectral_FFT_d2_K_Hpi_s45_arrays.npz \
    --hold-seed 45 --hold-time 15.0 --hold-label "Held hexagon (hex_rect)" \
    --transition-npz ../results/metrics/khpi_hexrect/spectral_FFT_d2_K_Hpi_s42_arrays.npz \
    --transition-seed 42 --transition-time 15.0 --transition-label "Transitioned (hex_rect)" \
    --square-npz ../results/metrics/khpi_square/spectral_FFT_d2_K_Hpi_s45_arrays.npz \
    --square-seed 45 --square-time 10.0 --square-label "Square domain (control)" \
    --out figure_fourier_montage
```

**Training curves** (`T_spots`: 800-window budget; `low_rho`: 2000-window budget):

```bash
python fig3_training_curves.py --npz-dir-a ../results/metrics/T_spots_A --npz-dir-b ../results/metrics/T_spots_B \
    --regime T_spots --budget-windows 800 --out fig_training_curves_Tspots

python fig3_training_curves.py --npz-dir-a ../results/metrics/low_rho_A --npz-dir-b ../results/metrics/low_rho_B \
    --regime low_rho --budget-windows 2000 --out fig_training_curves_lowrho
```

**`K_Hpi` seed 49 evolution strips** (rectangular `hex_rect` domain and square-domain control, Method A):

```bash
seed=49
python export_evolution_pdf.py \
    --npz ../results/metrics/khpi_hexrect/spectral_FFT_d2_K_Hpi_s${seed}_arrays.npz \
    --k-true 19.221 --n-mode 6 --domain-shape hex_rect \
    --times 2,4,6 --dpi 600 --out khpi_A_s${seed}_evolution.pdf

python export_evolution_pdf.py \
    --npz ../results/metrics/khpi_square/spectral_FFT_d2_K_Hpi_s${seed}_arrays.npz \
    --k-true 19.221 --n-mode 6 --domain-shape square \
    --times 2,4,6 --dpi 600 --out khpi_square_A_s${seed}_evolution.pdf
```

## Known limitations

One reaction system, one pair of Laplacian realisations, at most four seeds per regime. The results are
configuration-specific evidence, not failure-rate estimates. See the paper for details.

## License and citation

Code and metrics are released under the MIT licence (see `LICENSE`). The field-array dataset on Zenodo is released under CC BY 4.0.
Please cite via `CITATION.cff` (GitHub's "Cite this repository" button) or the Zenodo DOI.
