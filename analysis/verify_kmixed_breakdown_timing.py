"""
verify_kmixed_breakdown_timing.py

Purpose: check the manuscript's specific claim that two K_mixed Method B
seeds (spec_err 74.5% -> final u_amp 0.907, and spec_err ~collapsed ->
final u_amp 0.135) spent 89% and 94% of simulated time respectively in a
genuine, correctly-structured pattern before a late-stage collapse. This
figure is not recoverable from the summary all_metrics.json files (which
only record the final state) -- it requires the full saved trajectory,
which IS on disk in the _arrays.npz files (t, U, V at every window,
confirmed from the training script: snaps_t/snaps_u/snaps_v are appended
unconditionally every passing window, then saved via np.savez_compressed).

This script:
  1. Loads the trajectory (t, U, V) for each seed's .npz file.
  2. Computes amplitude (U.max()-U.min()) at every saved snapshot.
  3. Finds the peak amplitude and when it occurred.
  4. Identifies the breakdown point: the first time the amplitude drops
     below a threshold fraction of its own peak and does not recover
     above that threshold again for the remainder of the run (a simple,
     transparent rule -- not the original classifier, since that isn't
     available here, but a reasonable independent check of the same
     underlying claim).
  5. Reports what fraction of total simulated time (t) was spent before
     that breakdown point, directly comparable to the manuscript's
     89%/94% figures.

USAGE

  python verify_kmixed_breakdown_timing.py \
      --npz-dir results/kmixed_B \
      --seeds 45 49 \
      --pattern "autograd_d2_K_mixed_s{seed}_arrays.npz"

Adjust --pattern if your actual filename differs -- run with --list-only
first if you're not sure of the exact naming.
"""

import argparse
import glob
import os

import numpy as np


def load_trajectory(path):
    d = np.load(path)
    t = d["t"]
    U = d["U"]
    # amplitude at each saved snapshot: same definition as the metrics
    # pipeline uses for the final-state u_amp (max - min), computed here
    # for every timestep instead of just the last one
    amp = np.array([float(frame.max() - frame.min()) for frame in U])
    return t, amp


def find_breakdown_fraction(t, amp, drop_frac=0.5):
    """Find the first index after which amplitude never again exceeds
    drop_frac * peak_amplitude. Returns (breakdown_time_fraction,
    breakdown_index, peak_amp, peak_time, final_amp)."""
    peak_idx = int(np.argmax(amp))
    peak_amp = float(amp[peak_idx])
    peak_time = float(t[peak_idx])
    threshold = drop_frac * peak_amp

    above = amp >= threshold
    idx_after_peak = np.arange(peak_idx, len(amp))
    last_above = idx_after_peak[above[peak_idx:]]
    if len(last_above) == 0:
        breakdown_idx = peak_idx
    else:
        breakdown_idx = int(last_above[-1]) + 1
        breakdown_idx = min(breakdown_idx, len(t) - 1)

    breakdown_time = float(t[breakdown_idx])
    total_time = float(t[-1])
    frac_healthy = breakdown_time / total_time if total_time > 0 else float("nan")
    final_amp = float(amp[-1])
    return frac_healthy, breakdown_idx, peak_amp, peak_time, final_amp, breakdown_time, total_time


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-dir", required=True,
                    help="Directory containing the _arrays.npz files, e.g. results/kmixed_B")
    p.add_argument("--seeds", type=int, nargs="+", default=[45, 49],
                    help="Seeds to check (default: the two K_mixed breakdown seeds)")
    p.add_argument("--pattern", default="autograd_d2_K_mixed_s{seed}_arrays.npz",
                    help="Filename pattern with {seed} placeholder")
    p.add_argument("--drop-frac", type=float, default=0.5,
                    help="Fraction of peak amplitude below which the run is "
                         "considered 'broken down' (default 0.5 = 50%% of peak)")
    p.add_argument("--list-only", action="store_true",
                    help="Just list .npz files in --npz-dir and exit, to check naming")
    args = p.parse_args()

    if args.list_only:
        matches = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
        print(f"Found {len(matches)} .npz file(s) in {args.npz_dir}:")
        for m in matches:
            print(f"  {m}")
        return

    for seed in args.seeds:
        fname = args.pattern.format(seed=seed)
        path = os.path.join(args.npz_dir, fname)
        print(f"\n{'='*90}")
        print(f"seed={seed}  ({path})")
        print(f"{'='*90}")

        if not os.path.exists(path):
            print(f"  [MISSING] File not found. Run with --list-only to see "
                  f"what's actually in {args.npz_dir}, then adjust --pattern.")
            continue

        t, amp = load_trajectory(path)
        print(f"  {len(t)} saved snapshots, t range [{t[0]:.3f}, {t[-1]:.3f}]")
        print(f"  amplitude range [{amp.min():.4f}, {amp.max():.4f}]")

        frac_healthy, breakdown_idx, peak_amp, peak_time, final_amp, \
            breakdown_time, total_time = find_breakdown_fraction(t, amp, args.drop_frac)

        print(f"\n  Peak amplitude: {peak_amp:.4f} at t={peak_time:.3f}")
        print(f"  Final amplitude: {final_amp:.4f}")
        print(f"  Breakdown point (last time >= {args.drop_frac*100:.0f}% of peak, "
              f"snapshot {breakdown_idx}): t={breakdown_time:.3f}")
        print(f"  Total simulated time: {total_time:.3f}")
        print(f"  --> Fraction of simulated time spent 'healthy' before "
              f"breakdown: {frac_healthy*100:.1f}%")
        print(f"      (manuscript claims 89% for one seed, 94% for the other --"
              f" compare this seed's figure against whichever value it should be)")

        print(f"\n  Amplitude trajectory (every ~10th snapshot):")
        step = max(1, len(t) // 20)
        for i in range(0, len(t), step):
            marker = " <-- breakdown" if i == breakdown_idx else ""
            print(f"    t={t[i]:8.3f}  amp={amp[i]:8.4f}{marker}")

    print(f"""
{'='*90}
NOTES
{'='*90}
This uses a simple, transparent rule (first point after which amplitude
never again exceeds {args.drop_frac*100:.0f}% of its own peak) -- not
necessarily identical to whatever method originally produced the 89%/94%
figures, so don't expect an exact match. What matters is whether the
result is in the same ballpark (e.g. 85-95%) or genuinely different (e.g.
60% or 99%). Try --drop-frac 0.1 or --drop-frac 0.3 as well to see how
sensitive the figure is to this choice -- if the fraction stays roughly
stable across reasonable thresholds, that's good evidence the original
89%/94% figures were reasonable; if it swings wildly, the original method
mattered a lot and should be identified specifically before citing an
exact percentage in the manuscript.
""")


if __name__ == "__main__":
    main()
