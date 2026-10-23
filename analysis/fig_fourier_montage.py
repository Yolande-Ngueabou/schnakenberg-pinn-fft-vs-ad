"""
fig_fourier_montage.py

Side-by-side 2D Fourier power spectra for three real, contrasting states
of the K_Hpi system: a held hexagonal state, a state that has
transitioned away from hexagonal symmetry, and a square-domain field
that is geometrically forbidden from exhibiting hexagonal symmetry.
Supports the paper's methodological claim that quantitative Fourier
verification is required -- visual/classifier-based inspection labels
all three of these as "spots" (see metrics JSONs), yet the spectra are
qualitatively different.

This version is formatted for direct journal submission:
  - Double-column width (7.0 in), three panels + one colorbar
  - No suptitle -- explanatory prose belongs in the caption
  - Panel labels (a), (b), (c) with verdict as a compact second title line
  - Fonts matched to reproduction at column width
  - Vector PDF with rasterised heatmap layer (small file, sharp text)
  - 600 dpi PNG fallback for review systems that require raster

SAFETY: domain size (Lx, Ly, k_true) is auto-detected per field via
autodetect_domain_size() from each seed's own metrics JSON -- nothing is
hardcoded or guessed.

PANEL LABELS: panel titles are supplied via --hold-label,
--transition-label, --square-label. The script cannot know which
experiment a given .npz came from, so labels must be set by the caller.
Keep labels under ~30 characters for a clean two-line title at column
width; longer labels are auto-wrapped.
"""

import argparse
import os
import sys
import textwrap

import matplotlib
matplotlib.rcParams.update({
    "pdf.fonttype": 42,   # embed TrueType; required by most journals
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_khpi_hexagonal_symmetry import autodetect_domain_size, find_hexagonal_peaks


def load_snapshot(npz_path, time):
    d = np.load(npz_path)
    t = d["t"]
    U = d["U"]
    idx = int(np.argmin(np.abs(t - time)))
    return U[idx], t[idx]


def plot_spectrum(ax, field, k_true, Lx, Ly):
    """Draw the 2D power spectrum. No titles or verdicts here -- those
    are set by the caller so all display decisions live in one place."""
    N = field.shape[0]
    fft = np.fft.fftshift(np.fft.fft2(field - field.mean()))
    power = np.abs(fft) ** 2
    power_db = 10 * np.log10(power / power.max() + 1e-12)

    kx_freqs = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(N, d=Lx / N))
    ky_freqs = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(N, d=Ly / N))

    im = ax.pcolormesh(kx_freqs, ky_freqs, power_db.T,
                        cmap="viridis", vmin=-40, vmax=0,
                        shading="auto", rasterized=True)

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(k_true * np.cos(theta), k_true * np.sin(theta),
            color="red", ls="--", lw=0.9, alpha=0.85, zorder=3)
    for angle_deg in range(0, 360, 60):
        r = k_true * 1.35
        ax.plot([0, r * np.cos(np.radians(angle_deg))],
                 [0, r * np.sin(np.radians(angle_deg))],
                 color="white", ls=":", lw=0.5, alpha=0.5, zorder=2)

    ax.set_xlim(-k_true * 1.5, k_true * 1.5)
    ax.set_ylim(-k_true * 1.5, k_true * 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$", fontsize=7)
    ax.set_ylabel(r"$k_y$", fontsize=7)
    ax.tick_params(axis="both", labelsize=6.5)
    return im


def process_panel(ax, npz_path, seed, time, panel_letter, short_label):
    detected_k, detected_Lx, detected_Ly, source = autodetect_domain_size(npz_path, seed)
    if detected_k is None:
        ax.text(0.5, 0.5, "no metrics JSON found",
                ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_title(f"({panel_letter}) {short_label}", loc="left", fontsize=8)
        return None
    field, actual_time = load_snapshot(npz_path, time)
    is_hex, _ = find_hexagonal_peaks(field, detected_k, detected_Lx, detected_Ly)
    im = plot_spectrum(ax, field, detected_k, detected_Lx, detected_Ly)

    verdict = "hexagonal" if is_hex else "not hexagonal"
    title_text = textwrap.fill(f"({panel_letter}) {short_label}", width=30)
    ax.set_title(f"{title_text}\n{verdict}",
                  loc="left", fontsize=8, pad=4)

    print(f"({panel_letter}) {short_label}: seed={seed} t={actual_time:.3f} "
          f"domain=({detected_Lx:.4f},{detected_Ly:.4f}) "
          f"[auto-detected from {os.path.basename(source)}] -> is_hex={is_hex}")
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hold-npz", required=True)
    p.add_argument("--hold-seed", type=int, required=True)
    p.add_argument("--hold-time", type=float, required=True)
    p.add_argument("--hold-label", required=True,
                    help="Panel (a) title, e.g. 'Held hexagon (hex_rect)'. "
                         "Keep under ~30 characters.")
    p.add_argument("--transition-npz", required=True)
    p.add_argument("--transition-seed", type=int, required=True)
    p.add_argument("--transition-time", type=float, required=True)
    p.add_argument("--transition-label", required=True)
    p.add_argument("--square-npz", required=True)
    p.add_argument("--square-seed", type=int, required=True)
    p.add_argument("--square-time", type=float, required=True)
    p.add_argument("--square-label", required=True)
    p.add_argument("--out", default="fig_fourier_montage")
    p.add_argument("--dpi", type=int, default=600,
                    help="DPI for PNG fallback (default 600).")
    args = p.parse_args()

    # Double-column journal width (7.0 in).
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7),
                              constrained_layout=True)

    im = process_panel(axes[0], args.hold_npz, args.hold_seed, args.hold_time,
                        "a", args.hold_label)
    process_panel(axes[1], args.transition_npz, args.transition_seed,
                   args.transition_time, "b", args.transition_label)
    process_panel(axes[2], args.square_npz, args.square_seed, args.square_time,
                   "c", args.square_label)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015, aspect=25)
        cbar.set_label("power (dB, normalized)", fontsize=7)
        cbar.ax.tick_params(labelsize=6.5)

    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=args.dpi)
    print(f"\nSaved {args.out}.pdf and {args.out}.png ({args.dpi} dpi)")


if __name__ == "__main__":
    main()
