# FFT versus nested-AD Laplacians in windowed IMEX-PINNs for Schnakenberg Turing patterns

Code and results accompanying the manuscript *"A Seed-Resolved Reliability Study of FFT versus Nested
Automatic-Differentiation Laplacians in Windowed IMEX-PINNs for Schnakenberg Turing Patterns"*
(Ngueabou and Oloniiju, submitted to the Journal of Computational Physics).

The study compares two realisations of the Laplacian inside an otherwise identical windowed IMEX-PINN:
**Method A**, a Fourier pseudospectral Laplacian of the sampled network output, and **Method B**, a nested
reverse-mode automatic-differentiation Laplacian. Nine parameter regimes, up to four seeds each.

DOI of this archive: <!-- add Zenodo DOI -->

## Contents

| Path | Description |
|---|---|
| `code/schnakenberg_v16_spectral_v10.py` | Method A (FFT Laplacian), stabiliser on |
| `code/schnakenberg_v16_spectral_v10_staboff.py` | Method A, stabiliser off |
| `code/schnakenberg_v16_autograd_squarev2.py` | Method B (nested AD Laplacian) |
| `analysis/` | Scripts that regenerate the paper's figures and tables |
| `results/metrics/` | Per-run metrics (JSON), one folder per regime and method |
| `results/seed_outcome_table.csv` | One row per (regime, method, seed) |

Large field arrays (`.npz`) are in the Zenodo record, not in this repository. <!-- edit if not applicable -->

## Environment

- Python X.Y.Z, TensorFlow X.Y.Z, NumPy X.Y.Z (exact versions in `requirements.txt`)
- Results were produced on a Kaggle session with <!-- GPU model -->. Runs are seed-controlled but not
  guaranteed bit-identical across hardware or library versions.

```bash
pip install -r requirements.txt
```

## Running an experiment

```bash
# list regimes
python code/schnakenberg_v16_spectral_v10.py --list-regimes

# Method A, regime S_spots, seed 42
python code/schnakenberg_v16_spectral_v10.py --regime S_spots --seed 42 --save-dir results/S_spots_A

# Method B, same regime and seed
python code/schnakenberg_v16_autograd_squarev2.py --regime S_spots --seed 42 --save-dir results/S_spots_B

# Regime II (K_stripes, K_mixed, K_Hpi) uses the larger domain
python code/schnakenberg_v16_spectral_v10.py --regime K_stripes --domain 2 --seed 42 --save-dir results/K_stripes_A
```

Regimes: `S_spots, M_spots, T_spots, near_onset, low_rho, min_rho` (Regime I) and
`K_stripes, K_mixed, K_Hpi` (Regime II). Seeds used: 42, 45, 49, 52.
Other options: `--quick` (short test run), `--no-early-stop`, `--ic-mode {eigenmode,broadband,triad}`,
`--force-domain-shape {square,hex_rect}`.
<!-- Check every command here against what you actually ran, especially --domain and --ic-mode per regime. -->

The scripts default to Kaggle/Drive output paths; always pass `--save-dir`.

## Outputs

Each run writes `*_all_metrics_s<seed>.json`, a list of records with fields such as `regime`, `pattern`,
`anisotropy`, final amplitude and spectral error. <!-- add the fields you use in the paper -->

## Reproducing figures and tables

<!-- one line per script, e.g.: -->
- Cost table (Table 5): `python analysis/make_cost_table.py`
- Seed-outcome grid figure: `python analysis/make_fig_seed_outcome_grid.py`

## Known limitations

One reaction system, one pair of Laplacian realisations, at most four seeds per regime. The results are
configuration-specific evidence, not failure-rate estimates. See the paper for details.

## License and citation

Code: MIT (see `LICENSE`). Data in `results/`: CC BY 4.0. <!-- keep only if you choose this -->
Please cite via `CITATION.cff` (GitHub's "Cite this repository" button) or the Zenodo DOI.
