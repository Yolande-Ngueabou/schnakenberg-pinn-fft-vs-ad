"""
fig3_training_curves.py

Amplitude vs. simulated time, Method A and Method B overlaid, for ALL
FOUR seeds as a 2x2 small-multiples panel by default -- not a single
seed. This paper's methodology is built around seed-to-seed variability
(n=4 everywhere else, including Figure 2's per-seed grid); a training-
curve figure showing only one seed would understate that and could
read as cherry-picked to a reviewer. Directly visualizes the
collapse/divergence-flag discussion in Section 4.5 rather than
describing it only in prose. Works for any regime with real trajectory
data for both methods; T_spots and low_rho are the two immediately
available now.

USAGE
  python fig3_training_curves.py --npz-dir-a results/T_spots_A --npz-dir-b results/T_spots_B --regime T_spots
  python fig3_training_curves.py --npz-dir-a results/T_spots_A --npz-dir-b results/T_spots_B --regime T_spots --seed-only 42
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

SEEDS = [42, 45, 49, 52]


def find_npz(npz_dir, seed):
    matches = glob.glob(os.path.join(npz_dir, f"*s{seed}*arrays.npz"))
    return matches[0] if matches else None


def compute_amplitude_trajectory(npz_path):
    d = np.load(npz_path)
    t = d["t"]
    U = d["U"]
    u_amp = np.array([float(frame.max() - frame.min()) for frame in U])
    return t, u_amp


def plot_one_seed(ax, npz_dir_a, npz_dir_b, seed, divergence_ceiling,
                   log_scale=True, budget_windows=None):
    path_a = find_npz(npz_dir_a, seed)
    path_b = find_npz(npz_dir_b, seed)

    if path_a is None and path_b is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"seed {seed}", fontsize=10)
        return None, None

    final_a = final_b = None
    if path_a is not None:
        t_a, amp_a = compute_amplitude_trajectory(path_a)
        ax.plot(t_a, amp_a, color="#1f4e8c", lw=1.3, label="Method A (spectral)")
        final_a = amp_a[-1]
    if path_b is not None:
        t_b, amp_b = compute_amplitude_trajectory(path_b)
        ax.plot(t_b, amp_b, color="#c0392b", lw=1.3, label="Method B (autograd)")
        final_b = amp_b[-1]

        # mark early termination: this seed's trajectory stopped short of
        # the full window budget, meaning the divergence flag cut it off
        # rather than it running to the nominal end time
        if budget_windows is not None and len(t_b) < budget_windows:
            ax.axvline(t_b[-1], color="#c0392b", ls="--", lw=0.8, alpha=0.6)
            ax.annotate("early stop\n(divergence flag)", xy=(t_b[-1], amp_b[-1]),
                       xytext=(5, -10), textcoords="offset points", fontsize=7,
                       color="#c0392b", ha="left")

    if divergence_ceiling:
        ax.axhline(divergence_ceiling, color="grey", ls=":", lw=0.8)

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(f"seed {seed}", fontsize=10)
    return final_a, final_b


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-dir-a", required=True, help="Method A directory")
    p.add_argument("--npz-dir-b", required=True, help="Method B directory")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--seed-only", type=int, default=None,
                    help="Show just one seed instead of the default 2x2 panel "
                         "of all seeds -- use sparingly; the multi-seed panel "
                         "is the one that shows this isn't cherry-picked.")
    p.add_argument("--regime", default="")
    p.add_argument("--divergence-ceiling", type=float, default=None,
                    help="If known, draw a horizontal reference line at this "
                         "u_amp on every panel. Omit if not applicable.")
    p.add_argument("--linear-scale", action="store_true",
                    help="Use a linear y-axis instead of the default log scale. "
                         "Log scale is recommended when Method A stays small "
                         "and Method B diverges by orders of magnitude, since "
                         "linear scale crushes Method A's dynamics near zero.")
    p.add_argument("--budget-windows", type=int, default=None,
                    help="The nominal full window budget for this regime (e.g. "
                         "800). If a seed's saved trajectory is shorter than "
                         "this, it's marked as an early stop from the "
                         "divergence flag rather than a run that completed "
                         "the full window. Omit if not known/applicable.")
    p.add_argument("--out", default="fig3_training_curves")
    args = p.parse_args()

    log_scale = not args.linear_scale
    seeds_to_plot = [args.seed_only] if args.seed_only is not None else args.seeds

    if len(seeds_to_plot) == 1:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        final_a, final_b = plot_one_seed(ax, args.npz_dir_a, args.npz_dir_b,
                                          seeds_to_plot[0], args.divergence_ceiling,
                                          log_scale=log_scale,
                                          budget_windows=args.budget_windows)
        ax.set_xlabel("t (simulated time)")
        ax.set_ylabel(r"$u_{\mathrm{amp}}$" + (" (log scale)" if log_scale else ""))
        ax.legend(frameon=False, fontsize=9)
        fig.suptitle(f"{args.regime}, seed {seeds_to_plot[0]}: amplitude trajectory, "
                     f"both methods", fontsize=11)
    else:
        n = len(seeds_to_plot)
        ncols = 2
        nrows = (n + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * nrows), sharey=True)
        axes = np.atleast_1d(axes).flatten()

        finals = {}
        for ax, seed in zip(axes, seeds_to_plot):
            final_a, final_b = plot_one_seed(ax, args.npz_dir_a, args.npz_dir_b,
                                              seed, args.divergence_ceiling,
                                              log_scale=log_scale,
                                              budget_windows=args.budget_windows)
            finals[seed] = (final_a, final_b)

        for ax in axes[len(seeds_to_plot):]:
            ax.axis("off")

        for ax in axes[::ncols]:
            ax.set_ylabel(r"$u_{\mathrm{amp}}$" + (" (log scale)" if log_scale else ""))
        for ax in axes[-ncols:]:
            ax.set_xlabel("t (simulated time)")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"{args.regime}: amplitude trajectory, all seeds, both methods",
                    fontsize=12, y=1.06)

        for seed, (fa, fb) in finals.items():
            fa_str = f"{fa:.2f}" if fa is not None else "-"
            fb_str = f"{fb:.2f}" if fb is not None else "-"
            print(f"seed {seed}: Method A final={fa_str}  Method B final={fb_str}")

    fig.tight_layout()
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", bbox_inches="tight", dpi=300)
    print(f"\nSaved {args.out}.pdf / {args.out}.png")


if __name__ == "__main__":
    main()
