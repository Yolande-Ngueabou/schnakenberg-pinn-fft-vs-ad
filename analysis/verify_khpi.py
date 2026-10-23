"""
Independent check of the K_Hpi hexagon hold/transition claims in Manuscript 2.tex.

Reads the saved field arrays directly (autograd_d2_K_Hpi_s{seed}_arrays.npz and
spectral_FFT_d2_K_Hpi_s{seed}_arrays.npz, keys t/U/V) and, for every snapshot:
  * tests the Fourier signature the manuscript uses for a hexagonal lattice:
    six peaks at the dominant radius, 60 degrees apart, of comparable power;
  * records the amplitude max(U)-min(U).

It then reports, per seed, when the signature is established, when (if ever) it
is lost, and the amplitude trajectory -- and compares with what the manuscript says:

    Method B: holds 42, 45, 49; seed 52 transitions at t = 7.465
    Method A: holds 45; seeds 42, 49, 52 transition at t = 8.105, 6.740, 7.910

IMPORTANT: the manuscript's own classifier thresholds are not in this script (they
live in your figure code). The thresholds here are explicit and tunable, and the
script prints a sensitivity grid. Held-vs-transitioned verdicts should not change
across reasonable settings; exact transition times may shift by a few tenths.

Usage (Kaggle cell):
    !python verify_khpi.py /kaggle/working/results
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/results"
SEEDS = [42, 45, 49, 52]
T_END = 15.0

# manuscript claims: None = holds for the whole window, float = transition time
REPORTED = {
    "autograd": {42: None, 45: None, 49: None, 52: 7.465},
    "spectral_FFT": {42: 8.105, 45: None, 49: 6.740, 52: 7.910},
}
TIME_TOL = 0.25          # tolerated difference in transition time (manuscript units)

# default thresholds of the hexagon test
DEFAULTS = dict(band=0.20,        # annulus half-width, as a fraction of the dominant radius
                ratio_min=0.35,   # weakest of the six peaks / strongest
                angle_tol=10.0,   # allowed deviation from 60 deg spacing, degrees
                persist=10)       # consecutive snapshots needed to call a state established / lost
GRID_RATIO = [0.2, 0.35, 0.5]
GRID_ANGLE = [5.0, 10.0, 15.0]


# ----------------------------------------------------------------------
def hex_signature(U, Lx, Ly, band, ratio_min, angle_tol):
    """True if the 2-D spectrum of U has six peaks at the dominant radius,
    ~60 degrees apart, of comparable power. U is indexed [ix, iy]."""
    Nx, Ny = U.shape
    P = np.abs(np.fft.fft2(U - U.mean())) ** 2
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=1.0 / Nx) / Lx
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=1.0 / Ny) / Ly
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kmag = np.hypot(KX, KY)
    P[0, 0] = 0.0
    if P.max() <= 0:
        return False
    k0 = kmag.flat[np.argmax(P)]
    ring = np.abs(kmag - k0) <= band * k0
    # local maxima on the periodic grid
    locmax = np.ones_like(P, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            locmax &= P >= np.roll(np.roll(P, dx, axis=0), dy, axis=1)
    cand = np.argwhere(locmax & ring & (P > 0))
    if len(cand) < 6:
        return False
    pw = P[cand[:, 0], cand[:, 1]]
    order = np.argsort(pw)[::-1][:6]
    top = cand[order]
    pw6 = pw[order]
    if pw6[-1] / pw6[0] < ratio_min:
        return False
    ang = np.degrees(np.arctan2(KY[top[:, 0], top[:, 1]], KX[top[:, 0], top[:, 1]])) % 360.0
    ang = np.sort(ang)
    diffs = np.diff(np.concatenate([ang, [ang[0] + 360.0]]))
    return bool(np.all(np.abs(diffs - 60.0) <= angle_tol))


def trajectory_flags(U_all, Lx, Ly, **kw):
    return np.array([hex_signature(U_all[i], Lx, Ly, **kw) for i in range(U_all.shape[0])])


def established_and_lost(sig, t, persist):
    """First time `persist` consecutive snapshots carry the signature, and the first time
    afterwards that `persist` consecutive snapshots lack it (None if it never does)."""
    n = len(sig)
    est = None
    for i in range(n - persist + 1):
        if sig[i:i + persist].all():
            est = i
            break
    if est is None:
        return None, None
    for i in range(est, n - persist + 1):
        if not sig[i:i + persist].any():
            return float(t[est]), float(t[i])
    return float(t[est]), None


def find_runs(method_label):
    """seed -> (arrays_path, Lx, Ly) for K_Hpi on the hex_rect domain."""
    out = {}
    for f in glob.glob(os.path.join(ROOT, "**", "*_all_metrics_s*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        for m in (d if isinstance(d, list) else [d]):
            if (m.get("label") == "K_Hpi" and m.get("method") == method_label
                    and m.get("domain") == "hex_rect" and bool(m.get("stab_reg_used"))):
                s = m.get("seed")
                arr = glob.glob(os.path.join(os.path.dirname(f), f"*K_Hpi_s{s}_arrays.npz"))
                if arr:
                    prev = out.get(s)
                    if prev is None or os.path.getmtime(arr[0]) > os.path.getmtime(prev[0]):
                        out[s] = (arr[0], float(m["Lx"]), float(m["Ly"]))
    return out


def describe(tl):
    return "holds" if tl is None else f"loses at t={tl:.3f}"


def main():
    print(f"Searching {ROOT} for K_Hpi hex_rect runs (stabilizer on)\n")
    summary = []
    for method in ("autograd", "spectral_FFT"):
        label = "Method B (autograd)" if method == "autograd" else "Method A (spectral)"
        runs = find_runs(method)
        print("=" * 78)
        print(f"{label}: seeds found {sorted(runs)}")
        if not runs:
            print("  nothing found")
            continue
        print(f"  {'seed':>4} {'snaps':>6} {'est_t':>7} {'verdict':>18} {'hex_frac':>9} "
              f"{'amp_peak':>9} {'amp_final':>9} {'final/peak':>10}   manuscript")
        for s in sorted(runs):
            path, Lx, Ly = runs[s]
            z = np.load(path)
            t, U = z["t"], z["U"]
            flat = U.reshape(U.shape[0], -1)
            amp = flat.max(axis=1) - flat.min(axis=1)
            sig = trajectory_flags(U, Lx, Ly, **{k: DEFAULTS[k] for k in ("band", "ratio_min", "angle_tol")})
            est, lost = established_and_lost(sig, t, DEFAULTS["persist"])
            after = sig[np.searchsorted(t, est):] if est is not None else sig
            frac = after.mean()
            rep = REPORTED[method].get(s, "n/a")
            rep_txt = "n/a" if rep == "n/a" else describe(rep)
            if est is None:
                status, verdict = "CHECK", "never established"
            else:
                verdict = describe(lost)
                if rep == "n/a":
                    status = "INFO"
                elif (lost is None) != (rep is None):
                    status = "FAIL"
                elif lost is not None and abs(lost - rep) > TIME_TOL:
                    status = "CHECK"
                else:
                    status = "PASS"
            ratio = amp[-1] / amp.max()
            print(f"  {s:>4} {len(t):>6} {est if est is not None else float('nan'):>7.3f} {verdict:>18} "
                  f"{frac:>9.3f} {amp.max():>9.3f} {amp[-1]:>9.3f} {ratio:>10.2f}   {rep_txt}   [{status}]")
            summary.append((status, label, s, verdict, rep_txt))
            if ratio < 0.6:
                print(f"       ^ final amplitude is only {ratio:.0%} of peak: check for late-stage amplitude decay")

        # sensitivity grid
        print(f"\n  Threshold sensitivity (verdict per seed; rows = ratio_min, cols = angle_tol)")
        for s in sorted(runs):
            path, Lx, Ly = runs[s]
            z = np.load(path)
            t, U = z["t"], z["U"]
            verdicts = set()
            cells = []
            for r in GRID_RATIO:
                row = []
                for a in GRID_ANGLE:
                    sg = trajectory_flags(U, Lx, Ly, band=DEFAULTS["band"], ratio_min=r, angle_tol=a)
                    _, lost = established_and_lost(sg, t, DEFAULTS["persist"])
                    verdicts.add(lost is None)
                    row.append("hold" if lost is None else f"{lost:.2f}")
                cells.append(row)
            sens = "STABLE verdict" if len(verdicts) == 1 else "THRESHOLD-SENSITIVE (held/lost flips)"
            print(f"   seed {s}: {sens}")
            print("          angle_tol:  " + "  ".join(f"{a:>6.0f}" for a in GRID_ANGLE))
            for r, row in zip(GRID_RATIO, cells):
                print(f"          ratio {r:>4.2f}:  " + "  ".join(f"{c:>6}" for c in row))
        print()

    print("=" * 78)
    from collections import Counter
    print("SUMMARY:", dict(Counter(s for s, *_ in summary)))
    for st in ("FAIL", "CHECK"):
        for status, label, s, verdict, rep in summary:
            if status == st:
                print(f"  {st}: {label} seed {s}: found {verdict}; manuscript says {rep}")


if __name__ == "__main__":
    main()
