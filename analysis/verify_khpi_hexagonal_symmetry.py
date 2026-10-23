"""
verify_khpi_hexagonal_symmetry.py

Checks each seed's FULL trajectory (not just the final snapshot) for the
hexagonal 6-peak, 60-degree-spaced Fourier signature, to determine
whether it held hexagonal symmetry throughout or transitioned away at
some point -- the same methodology already used to resolve Method B's
status in registry Section 13, applied here to the freshly-regenerated
Method A hex_rect trajectories.

METHOD: at each saved snapshot, take the 2D FFT of the u-field, find the
dominant peaks in the power spectrum (excluding the DC/k=0 component),
and check whether six peaks exist at the expected radius (k_true) spaced
at 60-degree intervals. A seed "holds" if this signature is present at
every snapshot; it "transitions" at the first snapshot where the
signature is lost and never recovers for the remainder of the window
(matching how seeds 49/52 were characterised for Method B: a single,
permanent transition, not oscillation back and forth).

USAGE
  python verify_khpi_hexagonal_symmetry.py --npz-dir results/khpi_hexrect --seeds 42 45 49 52

FIX APPLIED (this version):
  find_hexagonal_peaks() now includes a peak-to-background test using
  the ratio (max power in annulus) / (median power in annulus). The
  previously-relied-on "power ratio across top-6 peaks < 3.0" check is
  inert on this data (observed 2.0-2.4 at every snapshot including
  known non-hexagonal cases), so the verdict was being decided solely
  by the angle-gap check. That let square-domain fields at accidental
  angle alignments pass as "hexagonal" (observed on khpi_square s42 at
  t~1.9-2.5). top/median cleanly separates the populations on sampled
  data: square domain 32-75, genuine hexagons 515-88094. Threshold set
  at 200 for now; calibrate via the trace_top_median diagnostic before
  finalising any manuscript claims that depend on borderline verdicts.
"""

import argparse
import json
import os

import numpy as np


# Peak-to-background threshold. A snapshot with
#   max(power in k_true annulus) / median(power in k_true annulus)
# below this value is rejected as non-hexagonal regardless of angle
# alignment. Calibrated from: square-domain false positives 32-75;
# genuine hexagonal snapshots 515-88094. 200 sits in the gap.
PEAK_TO_BACKGROUND_THRESHOLD = 200.0


def find_hexagonal_peaks(field_2d, k_true, Lx, Ly, k_tolerance=0.25,
                          n_peaks_expected=6,
                          peak_to_background_threshold=PEAK_TO_BACKGROUND_THRESHOLD):
    """Peak-finding and classification logic matched directly against the
    original check_hexagonal_symmetry.py (registry Section 13's trusted
    tool), not independently re-derived. Two things corrected from an
    earlier version of this function after comparing against the actual
    original source: k_tolerance widened from 0.15 to the original's 0.25,
    and the power-ratio check (all 6 peaks must have comparable power,
    max/min < 3.0) added -- this check was missing entirely before, and
    is the most likely explanation for an earlier false "hold" result on
    one contested seed (elevated anisotropy is consistent with an uneven
    power distribution across the 6 candidate peaks, which an angle-only
    check cannot detect).

    ADDITIONAL FIX (this version):
      The power-ratio-across-top-6 check turned out to be inert on
      real data -- the ratio was 2.0-2.4 at every sampled snapshot,
      including non-hexagonal ones, so it never rejected anything.
      Verdicts were therefore decided entirely by the angle-gap check,
      which is fooled by accidental angle alignment on broad-spectrum
      fields (square domain, transitional snapshots). Added a
      peak-to-background test: max(annulus power) / median(annulus
      power). Genuine hexagonal fields have this ratio in the hundreds
      to tens of thousands; broad-support non-hexagonal fields sit at
      30-75. Threshold rejects below PEAK_TO_BACKGROUND_THRESHOLD."""

    N = field_2d.shape[0]
    F = np.fft.fft2(field_2d - field_2d.mean())
    P = np.abs(F) ** 2
    kx = np.fft.fftfreq(N, d=1.0 / N) * 2 * np.pi / Lx
    ky = np.fft.fftfreq(N, d=1.0 / N) * 2 * np.pi / Ly
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    KR = np.sqrt(KX ** 2 + KY ** 2)

    mask = np.abs(KR - k_true) < k_tolerance * k_true
    idx = np.argwhere(mask)
    powers = P[mask]
    if len(powers) == 0:
        return False, []

    # ---- NEW: peak-to-background check ----
    # Must come before any angle-based logic: a broad-support field can
    # accidentally align its top-6 angle positions within 60-degree
    # comb regardless of whether there are any real peaks. The
    # top/median ratio catches that case.
    annulus_median = float(np.median(powers))
    annulus_max = float(powers.max())
    if annulus_median > 0:
        peak_to_background = annulus_max / annulus_median
        if peak_to_background < peak_to_background_threshold:
            return False, []
    # ---- END NEW ----

    order = np.argsort(powers)[::-1]
    peaks = []
    used_angles = []
    for i in order:
        kx_i, ky_i = KX[tuple(idx[i])], KY[tuple(idx[i])]
        angle = np.degrees(np.arctan2(ky_i, kx_i)) % 360
        if any(abs((angle - a + 180) % 360 - 180) < 15 for a in used_angles):
            continue
        peaks.append((kx_i, ky_i, float(powers[i]), angle))
        used_angles.append(angle)
        if len(peaks) >= 8:
            break

    if len(peaks) < n_peaks_expected:
        return False, peaks

    top6 = sorted(peaks[:n_peaks_expected], key=lambda p: p[3])
    angles = [p[3] for p in top6]
    gaps = [(angles[(k + 1) % 6] - angles[k]) % 360 for k in range(6)]
    gaps_ok = all(abs(g - 60) < 15.0 for g in gaps)

    powers6 = [p[2] for p in top6]
    power_ratio = max(powers6) / min(powers6) if min(powers6) > 0 else float("inf")
    power_ok = power_ratio < 3.0

    return (gaps_ok and power_ok), peaks


def peak_to_background_ratio(field_2d, k_true, Lx, Ly, k_tolerance=0.25):
    """Diagnostic helper: return the top/median power ratio in the k_true
    annulus without making a classification. Useful for tracing where a
    trajectory sits relative to PEAK_TO_BACKGROUND_THRESHOLD."""
    N = field_2d.shape[0]
    F = np.fft.fft2(field_2d - field_2d.mean())
    P = np.abs(F) ** 2
    kx = np.fft.fftfreq(N, d=1.0 / N) * 2 * np.pi / Lx
    ky = np.fft.fftfreq(N, d=1.0 / N) * 2 * np.pi / Ly
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    KR = np.sqrt(KX ** 2 + KY ** 2)
    mask = np.abs(KR - k_true) < k_tolerance * k_true
    powers = P[mask]
    if len(powers) == 0:
        return float("nan")
    med = float(np.median(powers))
    return float(powers.max() / med) if med > 0 else float("inf")


def summarize_time_ranges(times, bool_flags):
    """Collapse a per-snapshot boolean array into a list of contiguous
    (is_hex, t_start, t_end) ranges -- a complete 'what happened when'
    overview, correctly handling any number of transitions (not just a
    single clean one)."""
    times = list(times)
    bool_flags = list(bool_flags)
    if len(times) == 0:
        return []
    ranges = []
    start_idx = 0
    for i in range(1, len(bool_flags) + 1):
        if i == len(bool_flags) or bool_flags[i] != bool_flags[start_idx]:
            ranges.append((bool_flags[start_idx], times[start_idx], times[i - 1]))
            start_idx = i
    return ranges


def autodetect_domain_size(npz_path, seed):
    """Find this seed's own metrics JSON (same directory as the .npz,
    matching pattern *s{seed}*.json, excluding all_metrics files) and
    read its actual Lx/Ly/k_true directly, rather than trusting a
    CLI default or flag that might not match the domain actually being
    analyzed. This exists because verify_khpi_hexagonal_symmetry.py
    previously defaulted --Lx/--Ly to the hex_rect domain's values
    unconditionally, which silently produced a false "oscillates"
    result when run against khpi_square data without an explicit
    override -- confirmed via direct index-by-index verification
    (599/599 snapshots actually False when the correct square-domain
    Lx/Ly were used). Returns (k_true, Lx, Ly, source_file) or
    (None, None, None, None) if no metrics file is found."""
    npz_dir = os.path.dirname(npz_path)
    candidates = [f for f in os.listdir(npz_dir)
                  if f.endswith(".json") and f"s{seed}" in f
                  and "all_metrics" not in f]
    for fname in candidates:
        fpath = os.path.join(npz_dir, fname)
        try:
            with open(fpath) as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "Lx" in data and "Ly" in data and "k_true" in data:
                return data["k_true"], data["Lx"], data["Ly"], fpath
        except Exception:
            continue
    return None, None, None, None


def analyze_seed(npz_path, k_true, Lx, Ly):
    d = np.load(npz_path)
    t = d["t"]
    U = d["U"]

    status_per_snapshot = []
    for i in range(len(t)):
        is_hex, _ = find_hexagonal_peaks(U[i], k_true, Lx, Ly)
        status_per_snapshot.append(is_hex)

    status_per_snapshot = np.array(status_per_snapshot)
    ranges = summarize_time_ranges(t, status_per_snapshot)

    if status_per_snapshot.all():
        return "hold", None, t, status_per_snapshot, ranges
    if not status_per_snapshot.any():
        return "never_hexagonal", None, t, status_per_snapshot, ranges

    last_hex_idx = np.where(status_per_snapshot)[0][-1]
    if status_per_snapshot[: last_hex_idx + 1].all():
        transition_time = t[last_hex_idx + 1] if last_hex_idx + 1 < len(t) else t[-1]
        return "transition", transition_time, t, status_per_snapshot, ranges
    else:
        return "oscillates", None, t, status_per_snapshot, ranges


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 45, 49, 52])
    p.add_argument("--pattern", default="spectral_FFT_d2_K_Hpi_s{seed}_arrays.npz")
    p.add_argument("--k-true", type=float, default=None)
    p.add_argument("--Lx", type=float, default=None)
    p.add_argument("--Ly", type=float, default=None)
    p.add_argument("--peak-to-background", type=float,
                    default=PEAK_TO_BACKGROUND_THRESHOLD)
    p.add_argument("--save-to", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    file_lines = []

    def to_file(msg):
        file_lines.append(msg)
        if args.verbose:
            print(msg)

    def to_both(msg):
        file_lines.append(msg)
        print(msg)

    to_both(f"npz-dir: {args.npz_dir}")
    to_both(f"peak_to_background_threshold: {args.peak_to_background}")
    if args.k_true is not None or args.Lx is not None or args.Ly is not None:
        to_both(f"Using EXPLICIT overrides where given: "
                f"k_true={args.k_true}  Lx={args.Lx}  Ly={args.Ly} "
                f"(unset ones still auto-detected per seed)")
    else:
        to_both("k_true/Lx/Ly: auto-detecting per seed from each seed's own "
                "metrics JSON (safe default)")
    to_both("")

    results = {}
    for seed in args.seeds:
        fname = args.pattern.format(seed=seed)
        path = os.path.join(args.npz_dir, fname)
        to_both(f"{'='*60}")
        to_both(f"seed {seed}  ({path})")
        to_both(f"{'='*60}")
        if not os.path.exists(path):
            to_both("  [FILE MISSING]")
            to_both("")
            continue

        detected_k, detected_Lx, detected_Ly, source = autodetect_domain_size(path, seed)
        k_true = args.k_true if args.k_true is not None else detected_k
        Lx = args.Lx if args.Lx is not None else detected_Lx
        Ly = args.Ly if args.Ly is not None else detected_Ly

        if k_true is None or Lx is None or Ly is None:
            to_both(f"  [WARNING] Could not auto-detect domain size for seed {seed} "
                    f"-- SKIPPING. Pass --k-true/--Lx/--Ly explicitly if you "
                    f"are sure of the correct values.")
            results[seed] = "skipped (no domain size available)"
            to_both("")
            continue

        to_file(f"  domain size: k_true={k_true}  Lx={Lx}  Ly={Ly}"
                + (f"  (auto-detected from {source})" if source else "  (explicit override)"))

        status, transition_time, t, status_arr, ranges = analyze_seed(
            path, k_true, Lx, Ly)
        results[seed] = status

        to_file(f"  t range: [{t[0]:.3f}, {t[-1]:.3f}], {len(t)} snapshots")
        to_both("  TIME RANGE SUMMARY (complete overview):")
        for is_hex, t_start, t_end in ranges:
            label = "HEXAGONAL    " if is_hex else "NOT hexagonal"
            to_both(f"    {label}  t=[{t_start:.3f}, {t_end:.3f}]")
        to_both(f"  Overall status: {status}" +
                (f" (single transition at t={transition_time:.3f})" if transition_time else "") +
                (f" ({len(ranges)} periods -- flickers back and forth, inspect manually)"
                 if status == "oscillates" else ""))
        to_both("")

    to_both(f"{'='*60}")
    to_both("SUMMARY (all seeds)")
    to_both(f"{'='*60}")
    to_both(f"{'seed':<6} {'status':<16}")
    for seed, status in results.items():
        to_both(f"{seed:<6} {status:<16}")
    to_both(f"\nTotals: {sum(1 for s in results.values() if s == 'hold')} hold, "
            f"{sum(1 for s in results.values() if s == 'transition')} transition, "
            f"{sum(1 for s in results.values() if s not in ('hold', 'transition'))} other")

    next_steps = """
NEXT STEPS:
  1. Compare this result against K_HPI_HEX_STATUS in fig2_seed_outcome_grid.py.
  2. If it matches: update that dict's comment to reflect verification, and
     add a corresponding entry to PARAMETER_REGISTRY.md Section 13.
  3. If it DIFFERS: this needs an actual manuscript correction.
  4. NOTE: this version adds a peak-to-background check (top/median power
     in the k_true annulus, threshold 200 by default). Any verdict that
     flips relative to the previous version was borderline under the old
     classifier; treat those seeds as ambiguous until the threshold is
     independently calibrated via peak_to_background_ratio() traces.
"""
    to_both(next_steps)

    if args.save_to:
        with open(args.save_to, "w") as f:
            f.write("\n".join(file_lines) + "\n")
        print(f"[Saved full detail to {args.save_to}]")


if __name__ == "__main__":
    main()
