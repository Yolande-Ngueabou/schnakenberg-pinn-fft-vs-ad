"""
fig_khpi_hex_timeline.py

Produces a single-panel Gantt-style timeline of a seed's hexagonal
hold/transition status over the full simulated window: one horizontal
row per seed, coloured by whether the Fourier-peak classifier found the
hexagonal six-peak signature at each saved snapshot.

Designed to be run once per method, producing one PDF per method, which
the manuscript then composes into a two-panel stacked figure (Method A
above, Method B below, same four seeds in the same vertical order).
Composition is done in LaTeX via two \\includegraphics calls inside
subfigure environments -- not by this script -- so that the same axis
scale and row ordering appear in both panels and the reader can compare
seed i's row across methods directly.

Reuses analyze_seed() from verify_khpi_hexagonal_symmetry.py (same
directory), so the underlying computation is identical to the verifier's
own output, not a second implementation that could drift out of sync.

USAGE
  # Method A (hex_rect, spectral FFT):
  python fig_khpi_hex_timeline.py \
      --npz-dir results/khpi_hexrect \
      --pattern "spectral_FFT_d2_K_Hpi_s{seed}_arrays.npz" \
      --label "Method A (hex_rect)" \
      --out fig_khpi_A_timeline

  # Method B (hex_rect, autograd):
  python fig_khpi_hex_timeline.py \
      --npz-dir results/khpi_B \
      --pattern "autograd_d2_K_Hpi_s{seed}_arrays.npz" \
      --label "Method B (hex_rect)" \
      --out fig_khpi_B_timeline

Output: one PDF and one 600-dpi PNG per invocation. Rows are labelled
by seed only ("seed 42", "seed 45", ...); the method label lives in the
panel title, so the same four y-tick labels appear in both panels and
seed i's row is directly comparable across methods.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.rcParams.update({
    "pdf.fonttype": 42,   # embed TrueType; required by most journals
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_khpi_hexagonal_symmetry import analyze_seed

SEEDS = [42, 45, 49, 52]
DEFAULT_PATTERN = "spectral_FFT_d2_K_Hpi_s{seed}_arrays.npz"

COLOR_HEX = "#3498db"
COLOR_TRANSITION = "#95a5a6"
COLOR_MISSING = "#f1c40f"


def collect_rows(npz_dir, seeds, k_true, Lx, Ly, pattern):
    """Returns a list of (seed, ranges, t_max) tuples in the order
    requested. Rows are deliberately labelled by seed only -- the method
    label lives in the panel title, not the y-ticks, so that when the
    two panels are stacked in the manuscript the same four y-tick labels
    appear in both and seed i's row is directly comparable across
    methods."""
    rows = []
    for seed in seeds:
        path = os.path.join(npz_dir, pattern.format(seed=seed))
        if not os.path.exists(path):
            rows.append((seed, None, None))
            continue
        status, transition_time, t, status_arr, ranges = analyze_seed(
            path, k_true, Lx, Ly)
        rows.append((seed, ranges, float(t[-1])))
    return rows


def plot_timeline(rows, panel_title, out_path):
    n = len(rows)
    # Journal two-column width (7.0 in). Height scales with row count so
    # that bar height and label size stay constant regardless of n.
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * n + 1.0))

    yticks = []
    yticklabels = []
    t_max_overall = 0.0

    for i, (seed, ranges, t_max) in enumerate(rows):
        y = n - i - 1
        yticks.append(y)
        yticklabels.append(f"seed {seed}")

        if ranges is None:
            ax.broken_barh([(0, 1)], (y - 0.35, 0.7),
                            facecolors=COLOR_MISSING)
            ax.text(0.5, y, "FILE MISSING",
                     ha="center", va="center", fontsize=7)
            continue

        t_max_overall = max(t_max_overall, t_max)
        for is_hex, t_start, t_end in ranges:
            color = COLOR_HEX if is_hex else COLOR_TRANSITION
            # Minimum width so a zero-duration range (a single-snapshot
            # flicker between states) is still visible.
            width = max(t_end - t_start, t_max * 0.003)
            ax.broken_barh([(t_start, width)], (y - 0.35, 0.7),
                            facecolors=color,
                            edgecolors="white", linewidth=0.5)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlabel(r"$t$", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(0, t_max_overall * 1.02 if t_max_overall > 0 else 1)
    ax.set_ylim(-0.6, n - 0.4)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_HEX,
                       label="hexagonal"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_TRANSITION,
                       label="not hexagonal"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MISSING,
                       label="file missing"),
    ]
    # Legend sits directly above the axes (one row, flush against the
    # anchor); the panel title sits above the legend via the axes-title
    # pad. Both are captured correctly by savefig(bbox_inches="tight").
    ax.legend(handles=legend_handles,
              loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=7,
              borderaxespad=0.0)
    ax.set_title(panel_title, fontsize=9, pad=20)

    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_path}.png", bbox_inches="tight", dpi=600)
    print(f"Saved {out_path}.pdf / {out_path}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-dir", required=True,
                    help="Directory containing the method's .npz trajectories.")
    p.add_argument("--label", required=True,
                    help="Panel title, e.g. 'Method A (hex_rect)' or "
                         "'Method B (hex_rect)'. Keep short -- this is the "
                         "only method-identifying text on the panel, since "
                         "the y-ticks are labelled by seed only.")
    p.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help="Filename pattern for --npz-dir. Method A files use "
                         "'spectral_FFT_d2_...', Method B files use "
                         "'autograd_d2_...' -- set explicitly.")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                    help="Seeds to plot, in the order they should appear "
                         "top-to-bottom. Use the same order for both method "
                         "panels so that seed i's row in Method A lines up "
                         "with seed i's row in Method B.")
    p.add_argument("--k-true", type=float, default=19.220976475563766)
    p.add_argument("--Lx", type=float, default=1.9613525822169122)
    p.add_argument("--Ly", type=float, default=1.1323874413187018)
    p.add_argument("--out", default="fig_khpi_hex_timeline")
    args = p.parse_args()

    rows = collect_rows(args.npz_dir, args.seeds,
                         args.k_true, args.Lx, args.Ly, args.pattern)
    plot_timeline(rows, args.label, args.out)


if __name__ == "__main__":
    main()
