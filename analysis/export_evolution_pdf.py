"""
export_evolution_pdf.py

Compact evolution figure: two rows (U on top, V on bottom), N columns for
N selected timepoints. No colorbars, no caption.

v7: explicit outer margins.
    - Drop bbox_inches="tight" so subplots_adjust's top/bottom/left/right
      actually take effect.
    - Individual flags for padding above U, below V, and at the L/R corners.
    - hspace controls the gap between U and V rows.
    - Explicit time selection via --times, or --n-timepoints (3 or 4).

USAGE:

  python export_evolution_pdf.py --npz results/khpi_B/autograd_d2_K_Hpi_s42_arrays.npz \
      --k-true 19.221 --n-mode 6 --domain-shape hex_rect \
      --times 0,2,8,15 --dpi 600 --out khpi_B_s42_evolution.pdf
"""

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_domain(k_true, n_mode_target, domain_shape):
    if domain_shape == "hex_rect":
        n_mode = n_mode_target if n_mode_target % 2 == 0 else n_mode_target + 1
        Lx = n_mode * (2 * np.pi / k_true)
        Ly = Lx / np.sqrt(3)
    else:
        Lx = n_mode_target * (2 * np.pi / k_true)
        Ly = Lx
    return Lx, Ly


def guess_seed(npz_path):
    m = re.search(r"[sS](\d+)", npz_path.split("/")[-1])
    return m.group(1) if m else "?"


def pick_indices(t, times=None, n_timepoints=4):
    t_min, t_max = float(t.min()), float(t.max())
    dt_nominal = np.median(np.diff(t)) if len(t) > 1 else 0.0

    if times is not None:
        indices = []
        for tt in times:
            i = int(np.argmin(np.abs(t - tt)))
            err = abs(float(t[i]) - tt)
            if dt_nominal > 0 and err > 2 * dt_nominal:
                print(f"  WARNING: requested t={tt} -> nearest snapshot "
                      f"t={float(t[i]):.3f} (off by {err:.3f}, "
                      f"snapshot spacing ~{dt_nominal:.3f})")
            indices.append(i)
        seen, uniq = set(), []
        for i in indices:
            if i not in seen:
                seen.add(i); uniq.append(i)
        if len(uniq) < len(indices):
            print(f"  NOTE: collapsed {len(indices)} requested times to "
                  f"{len(uniq)} unique snapshots.")
        return uniq, [t[i] for i in uniq]

    targets = np.linspace(t_min, t_max, n_timepoints)
    indices = [int(np.argmin(np.abs(t - tt))) for tt in targets]
    return indices, [t[i] for i in indices]


def export_evolution(npz_path, k_true, n_mode, domain_shape, n_timepoints,
                      dpi, out, cmap="RdYlBu_r", shared_scale=False,
                      times=None,
                      panel_width=2.2,   # inches per column
                      hspace=0.02,       # gap between U and V rows
                      wspace=0.04,       # gap between columns
                      pad_top=0.35,      # inches above the U row
                      pad_bottom=0.15,   # inches below the V row
                      pad_left=0.15,     # inches at left corner
                      pad_right=0.15):   # inches at right corner
    Lx, Ly = compute_domain(k_true, n_mode, domain_shape)
    panel_aspect = Ly / Lx   # height/width of each panel

    print(f"Loading {npz_path} (once)...")
    d = np.load(npz_path)
    t, U, V = d["t"], d["U"], d["V"]
    t_min, t_max = float(t.min()), float(t.max())
    print(f"Experiment duration: t={t_min:.3f} to {t_max:.3f} "
          f"({len(t)} saved snapshots)")

    indices, actual_times = pick_indices(t, times=times,
                                          n_timepoints=n_timepoints)
    n_cols = len(indices)
    print(f"Using {n_cols} snapshots at t = "
          f"{[f'{x:.3f}' for x in actual_times]}")

    u_vmin, u_vmax = (U[indices].min(), U[indices].max()) if shared_scale else (None, None)
    v_vmin, v_vmax = (V[indices].min(), V[indices].max()) if shared_scale else (None, None)

    # ---- Figure size: panels + margins, in inches ----
    panel_w_in = panel_width
    panel_h_in = panel_w_in * panel_aspect
    fig_w = pad_left + n_cols * panel_w_in + pad_right
    fig_h = pad_top + 2.0 * panel_h_in + pad_bottom

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    # ---- Row 1: U ----
    for col, idx in enumerate(indices):
        ax = axes[0, col]
        ax.imshow(U[idx].T, origin="lower", extent=[0, Lx, 0, Ly],
                   cmap=cmap, vmin=u_vmin, vmax=u_vmax, aspect="equal")
        ax.set_title(f"u  t={actual_times[col]:.2f}", fontsize=8, pad=3)
        ax.set_xticks([]); ax.set_yticks([])

    # ---- Row 2: V ----
    for col, idx in enumerate(indices):
        ax = axes[1, col]
        ax.imshow(V[idx].T, origin="lower", extent=[0, Lx, 0, Ly],
                   cmap=cmap, vmin=v_vmin, vmax=v_vmax, aspect="equal")
        ax.set_title(f"v  t={actual_times[col]:.2f}", fontsize=8, pad=3)
        ax.set_xticks([]); ax.set_yticks([])

    # ---- Margins: convert inches -> fractions of figure size ----
    # subplots_adjust takes fractions in [0, 1], measured from the figure edge.
    fig.subplots_adjust(
        left   = pad_left   / fig_w,
        right  = 1.0 - pad_right  / fig_w,
        top    = 1.0 - pad_top    / fig_h,
        bottom = pad_bottom / fig_h,
        hspace = hspace,   # gap between the two rows
        wspace = wspace,   # gap between columns
    )

    # NO bbox_inches="tight" -- we want the figure canvas exactly as laid out.
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out} ({n_cols} timepoints x 2 fields, dpi={dpi})")
    print(f"Figure size: {fig_w:.2f} in x {fig_h:.2f} in "
          f"(margins: top={pad_top}in, bottom={pad_bottom}in, "
          f"left={pad_left}in, right={pad_right}in)")

    if shared_scale:
        print(f"U shared scale: [{u_vmin:.4f}, {u_vmax:.4f}]")
        print(f"V shared scale: [{v_vmin:.4f}, {v_vmax:.4f}]")
    else:
        print("Independent per-panel scaling (house convention default).")


def _parse_times(s):
    if s is None:
        return None
    return [float(x) for x in s.split(",") if x.strip() != ""]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--k-true", type=float, required=True)
    p.add_argument("--n-mode", type=int, required=True)
    p.add_argument("--domain-shape", default="hex_rect", choices=["square", "hex_rect"])
    p.add_argument("--times", type=str, default=None,
                    help="Comma-separated list of exact times to extract, e.g. "
                         "'0,2,8,15'. Snapped to nearest saved snapshot. "
                         "Overrides --n-timepoints.")
    p.add_argument("--n-timepoints", type=int, default=4,
                    help="Number of evenly-spaced timepoints (used only when "
                         "--times is not given). Default 4; use 3 for tighter "
                         "journal layout.")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--shared-scale", action="store_true",
                    help="One shared color scale across timepoints per field.")
    p.add_argument("--panel-width", type=float, default=2.3,
                    help="Width of each panel in inches (default 2.2).")
    p.add_argument("--hspace", type=float, default=0.20,
                    help="Vertical gap between U and V rows (fraction of axis "
                         "height). 0.0 = touching, 0.10 = clear separation.")
    p.add_argument("--wspace", type=float, default=0.08,
                    help="Horizontal gap between columns (fraction of axis width).")
    p.add_argument("--pad-top", type=float, default=0.17,
                    help="Inches of margin above the U row (default 0.35).")
    p.add_argument("--pad-bottom", type=float, default=0.13,
                    help="Inches of margin below the V row (default 0.15).")
    p.add_argument("--pad-left", type=float, default=0.12,
                    help="Inches of margin at the left corner (default 0.15).")
    p.add_argument("--pad-right", type=float, default=0.12,
                    help="Inches of margin at the right corner (default 0.15).")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    export_evolution(args.npz, args.k_true, args.n_mode, args.domain_shape,
                      args.n_timepoints, args.dpi, args.out,
                      shared_scale=args.shared_scale,
                      times=_parse_times(args.times),
                      panel_width=args.panel_width,
                      hspace=args.hspace, wspace=args.wspace,
                      pad_top=args.pad_top, pad_bottom=args.pad_bottom,
                      pad_left=args.pad_left, pad_right=args.pad_right)
