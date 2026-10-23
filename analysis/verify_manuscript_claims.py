"""
Check the numbers reported in Manuscript 2.tex against the raw per-run files.

Covers:
  1. K_mixed, Method B (autograd), n=4
  2. K_mixed, Method A (spectral) baseline it is compared with
  3. K_stripes, Method A, stabilizer ON (baseline) and OFF (ablation)
  4. K_stripes, Method B (autograd) -- seeds not yet run are reported as PENDING
  5. Per-window cost ratio B/A (K_mixed, plus the other regimes quoted in the cost paragraph)
  6. K_mixed Method B seed 49: late-stage breakdown, read from the saved field arrays

Every value comes from the *_all_metrics_s*.json files (and, for item 6, the
*_arrays.npz files) that the training scripts wrote. Nothing is taken from the
crosscheck text file.

Usage (Kaggle cell):
    %run verify_manuscript_claims.py
or, to point at a different results folder:
    !python verify_manuscript_claims.py /kaggle/working/results
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/results"
SEEDS = [42, 45, 49, 52]

# ----------------------------------------------------------------------
# What the manuscript currently says. Edit here if the text changes.
# Tolerance = half a unit in the last digit printed in the manuscript.
# ----------------------------------------------------------------------
REPORTED = {
    # (regime, method, stab_reg_used): {metric: {seed: value}}
    ("K_mixed", "autograd", True): {
        "spec_err_vs_ktrue": {42: 10.3, 45: 10.3, 49: 10.3, 52: 6.7},
        "u_amp": {42: 5.248, 45: 4.343, 49: 0.185, 52: 4.428},
        "pattern": {42: "stripes", 45: "stripes", 49: "mixed/labyrinthine", 52: "stripes"},
        "collapsed": {s: False for s in SEEDS},
        "es_triggered": {s: False for s in SEEDS},
    },
    ("K_mixed", "spectral_FFT", True): {
        "spec_err_vs_ktrue": {42: 10.3, 45: 10.3, 49: 10.3, 52: 6.7},
        "pattern": {42: "mixed/labyrinthine", 45: "stripes", 49: "mixed/labyrinthine", 52: "stripes"},
        "collapsed": {s: False for s in SEEDS},
    },
    ("K_stripes", "spectral_FFT", True): {
        "spec_err_vs_ktrue": {42: 10.3, 45: 6.7, 49: 6.7, 52: 6.7},
        "pattern": {42: "mixed/labyrinthine", 45: "spots", 49: "spots", 52: "stripes"},
        "collapsed": {s: False for s in SEEDS},
        "es_triggered": {s: True for s in SEEDS},
    },
    ("K_stripes", "spectral_FFT", False): {
        "u_amp": {42: 0.0012, 45: 0.0007, 49: 1.72, 52: 1.43},
        "collapsed": {42: True, 45: True, 49: False, 52: False},
    },
    ("K_stripes", "autograd", True): {
        # only seed 42 is reported in the manuscript so far
        "spec_err_vs_ktrue": {42: 6.7},
        "u_amp": {42: 1.807},
        "pattern": {42: "spots"},
        "collapsed": {42: False},
        "es_triggered": {42: True},
    },
}
# decimals printed in the manuscript for each numeric metric -> tolerance
DECIMALS = {"spec_err_vs_ktrue": 1, "u_amp": 3}
# u_amp for the K_stripes stab-off seeds is printed with different precision
UAMP_DECIMALS_OVERRIDE = {
    ("K_stripes", "spectral_FFT", False): {42: 4, 45: 4, 49: 2, 52: 2},
}
# stabilizer-ON K_stripes amplitude *range* quoted in the ablation paragraph
KSTRIPES_ON_AMP_RANGE = (1.67, 1.84)

# cost ratios quoted in the cost paragraph: regime -> (reported ratio, domain of the metrics files)
COST_REPORTED = {
    "K_mixed": (8.92, "square"),
    "K_Hpi": (9.23, "hex_rect"),
    "T_spots": (4.35, "square"),
    "S_spots": (5.31, "square"),
    "M_spots": (5.27, "square"),
    "low_rho": (4.68, "square"),
    "min_rho": (5.24, "square"),
    "near_onset": (5.33, "square"),
}
COST_TOL = 0.01

# seed-49 breakdown numbers quoted in the K_mixed section
S49 = dict(peak_amp=5.689, final_amp=0.185, t_breakdown=18.805, window=20.0, frac=0.940)

# ----------------------------------------------------------------------
results = []  # (status, text)


def record(status, text):
    results.append((status, text))
    print(f"[{status:7s}] {text}")


def load_rows(root):
    rows = []
    files = glob.glob(os.path.join(root, "**", "*_all_metrics_s*.json"), recursive=True)
    for f in sorted(files):
        try:
            d = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"  (could not read {f}: {e})")
            continue
        for m in (d if isinstance(d, list) else [d]):
            m = dict(m)
            m["_file"] = f
            if "seed" not in m or m["seed"] is None:
                mm = re.search(r"_s(\d+)\.json$", f)
                m["seed"] = int(mm.group(1)) if mm else None
            rows.append(m)
    return rows


def select(rows, regime, method, stab, domain=None):
    """One row per seed. If several files claim the same seed, use the newest and warn."""
    cand = [r for r in rows
            if r.get("label") == regime and r.get("method") == method
            and bool(r.get("stab_reg_used")) == stab
            and (domain is None or r.get("domain") == domain)]
    by_seed = {}
    for r in cand:
        by_seed.setdefault(r["seed"], []).append(r)
    out = {}
    for s, lst in by_seed.items():
        if len(lst) > 1:
            lst.sort(key=lambda r: os.path.getmtime(r["_file"]))
            same = all(abs(float(x.get("u_amp", 0)) - float(lst[-1].get("u_amp", 0))) < 1e-9 for x in lst)
            record("WARN" if not same else "NOTE",
                   f"{regime} {method} stab={stab} seed {s}: {len(lst)} files found"
                   f"{' (identical)' if same else ' (DIFFERENT values; using newest)'}: "
                   + "; ".join(os.path.relpath(x['_file'], ROOT) for x in lst))
        out[s] = lst[-1]
    return out


def close(found, reported, decimals):
    return abs(float(found) - float(reported)) <= 0.5 * 10 ** (-decimals) + 1e-12


def check_group(rows, key):
    regime, method, stab = key
    spec = REPORTED[key]
    domain = "square"
    got = select(rows, regime, method, stab, domain)
    tag = f"{regime} | {method} | stab={'ON' if stab else 'OFF'}"
    if not got:
        record("MISSING", f"{tag}: no metrics files found under {ROOT}")
        return got
    seeds_found = sorted(got)
    record("INFO", f"{tag}: seeds found {seeds_found}  (n={len(seeds_found)})")
    # seeds present in files but not covered by the manuscript text
    extra = [s for s in seeds_found if all(s not in v for v in spec.values())]
    if extra:
        record("UPDATE", f"{tag}: seeds {extra} exist in the files but are NOT reported in the manuscript yet")
        for s in extra:
            r = got[s]
            print(f"           seed {s}: spec_err={r['spec_err_vs_ktrue']:.2f}%  u_amp={r['u_amp']:.4f}  "
                  f"pattern={r['pattern']}  es_triggered={r['es_triggered']}  collapsed={r['collapsed']}")
    for metric, per_seed in spec.items():
        for s, rep in per_seed.items():
            if s not in got:
                record("PENDING", f"{tag} seed {s} {metric}: reported {rep}, no file for this seed")
                continue
            val = got[s].get(metric)
            if metric in DECIMALS:
                dec = UAMP_DECIMALS_OVERRIDE.get(key, {}).get(s, DECIMALS[metric]) if metric == "u_amp" else DECIMALS[metric]
                ok = close(val, rep, dec)
                record("PASS" if ok else "FAIL",
                       f"{tag} seed {s} {metric}: reported {rep}  found {float(val):.5f}")
            else:
                ok = (val == rep)
                record("PASS" if ok else "FAIL",
                       f"{tag} seed {s} {metric}: reported {rep}  found {val}")
    return got


def check_stripes_on_range(got):
    if not got:
        return
    amps = [float(r["u_amp"]) for r in got.values()]
    lo, hi = min(amps), max(amps)
    ok = close(lo, KSTRIPES_ON_AMP_RANGE[0], 2) and close(hi, KSTRIPES_ON_AMP_RANGE[1], 2)
    record("PASS" if ok else "FAIL",
           f"K_stripes | spectral_FFT | stab=ON amplitude range: reported {KSTRIPES_ON_AMP_RANGE}  "
           f"found ({lo:.4f}, {hi:.4f})")


def check_stripes_off_collapse_rate(got):
    if not got:
        return
    n = len(got)
    k = sum(bool(r["collapsed"]) for r in got.values())
    record("PASS" if (n == 4 and k == 2) else "FAIL",
           f"K_stripes stabilizer-off collapse rate: reported 2/4 (50%)  found {k}/{n}")


def check_cost(rows):
    print("\n--- cost ratio, Method B / Method A, mean ms per window ---")
    for regime, (rep, dom) in COST_REPORTED.items():
        a = select(rows, regime, "spectral_FFT", True, dom)
        b = select(rows, regime, "autograd", True, dom)
        if not a or not b:
            record("MISSING", f"cost {regime}: need both methods (A seeds {sorted(a)}, B seeds {sorted(b)})")
            continue
        ma = np.mean([float(r["ms_per_window"]) for r in a.values()])
        mb = np.mean([float(r["ms_per_window"]) for r in b.values()])
        ratio = mb / ma
        ok = abs(ratio - rep) <= COST_TOL
        note = f"(A n={len(a)}, B n={len(b)})"
        record("PASS" if ok else "FAIL",
               f"cost {regime}: reported {rep:.2f}x  found {ratio:.3f}x  [{mb:.0f} / {ma:.0f} ms] {note}")


def check_seed49_breakdown(rows):
    print("\n--- K_mixed Method B seed 49: late-stage breakdown from the saved fields ---")
    got = select(rows, "K_mixed", "autograd", True, "square")
    if 49 not in got:
        record("MISSING", "K_mixed autograd seed 49 metrics file not found")
        return
    d = os.path.dirname(got[49]["_file"])
    cands = glob.glob(os.path.join(d, "*K_mixed_s49_arrays.npz"))
    if not cands:
        record("MISSING", f"no *K_mixed_s49_arrays.npz next to {os.path.relpath(got[49]['_file'], ROOT)}")
        return
    z = np.load(cands[0])
    t, U = z["t"], z["U"]
    amp = U.reshape(U.shape[0], -1).max(axis=1) - U.reshape(U.shape[0], -1).min(axis=1)
    peak_i = int(np.argmax(amp))
    peak, final = float(amp[peak_i]), float(amp[-1])
    after = np.where(amp[peak_i:] < 0.5 * peak)[0]
    t_half = float(t[peak_i + after[0]]) if len(after) else float("nan")
    record("INFO", f"seed 49 arrays: {len(t)} snapshots, t from {t[0]:.3f} to {t[-1]:.3f}")
    record("PASS" if close(peak, S49["peak_amp"], 3) else "FAIL",
           f"seed 49 peak amplitude: reported {S49['peak_amp']}  found {peak:.4f} at t={t[peak_i]:.3f}")
    record("PASS" if close(final, S49["final_amp"], 3) else "FAIL",
           f"seed 49 final amplitude (arrays): reported {S49['final_amp']}  found {final:.4f}")
    record("PASS" if close(final, float(got[49]['u_amp']), 4) else "FAIL",
           f"seed 49 final amplitude, arrays vs metrics JSON: {final:.5f} vs {float(got[49]['u_amp']):.5f}")
    if np.isfinite(t_half):
        frac = t_half / S49["window"]
        # The manuscript's 18.805 / 94.0% comes from per-snapshot pattern classification,
        # not an amplitude threshold, so this is a consistency check, not an exact match.
        record("PASS" if abs(t_half - S49["t_breakdown"]) <= 1.0 else "CHECK",
               f"seed 49 amplitude falls below 50% of peak at t={t_half:.3f} ({frac:.1%} of window); "
               f"manuscript's {S49['t_breakdown']} ({S49['frac']:.1%}) comes from pattern classification - compare by eye")
    # Same trajectory check for the other K_mixed Method B seeds
    print("\n--- K_mixed Method B: peak vs final amplitude for every seed (late-breakdown screen) ---")
    for s in sorted(got):
        dd = os.path.dirname(got[s]["_file"])
        c = glob.glob(os.path.join(dd, f"*K_mixed_s{s}_arrays.npz"))
        if not c:
            record("MISSING", f"K_mixed autograd seed {s}: arrays file not found")
            continue
        zz = np.load(c[0])
        a = zz["U"].reshape(zz["U"].shape[0], -1)
        a = a.max(axis=1) - a.min(axis=1)
        pk, fn = float(a.max()), float(a[-1])
        flag = "stable" if fn > 0.5 * pk else "LATE DROP"
        record("PASS" if (flag == "stable") == (s != 49) else "FAIL",
               f"K_mixed autograd seed {s}: peak {pk:.3f}, final {fn:.3f} -> {flag} "
               f"(manuscript: {'late breakdown' if s == 49 else 'stable'})")


def main():
    rows = load_rows(ROOT)
    print(f"Read {len(rows)} metric records under {ROOT}\n")
    if not rows:
        print("No metrics files found. Check the path (argument 1).")
        return
    got = {}
    for key in REPORTED:
        print(f"\n=== {key[0]} | {key[1]} | stab={'ON' if key[2] else 'OFF'} ===")
        got[key] = check_group(rows, key)
    check_stripes_on_range(got.get(("K_stripes", "spectral_FFT", True)))
    check_stripes_off_collapse_rate(got.get(("K_stripes", "spectral_FFT", False)))
    check_cost(rows)
    check_seed49_breakdown(rows)

    print("\n" + "=" * 70)
    from collections import Counter
    c = Counter(s for s, _ in results)
    print("SUMMARY:", dict(c))
    for status in ("FAIL", "UPDATE", "WARN", "CHECK"):
        items = [t for s, t in results if s == status]
        if items:
            print(f"\n{status} ({len(items)}):")
            for t in items:
                print("  -", t)


if __name__ == "__main__":
    main()
