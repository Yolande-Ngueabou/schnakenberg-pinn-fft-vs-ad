#!/usr/bin/env python3
"""
verify_table1_spectral_error.py
--------------------------------
Independently recomputes the "Spectral error" column of Table 1
(tab:sigma_order) directly from the raw per-seed metrics JSON files on disk,
so the manuscript's numbers can be cross-checked against source data rather
than against the (hand-copied) crosscheck .txt dump.

Table 1's caption states: "Spectral error is against k_true under the
spectral (Method A) implementation." This script therefore:
  1. Walks --results-dir recursively for files matching *_d2_all_metrics_s*.json
  2. Reads each file's spec_err_vs_ktrue field (searched flexibly in case of
     nesting) plus domain / stab_reg_used / seed for bookkeeping
  3. Keeps only Method A (spectral_FFT) records
  4. Maps each file's parent directory name to a canonical regime name
     (+ domain, for K_Hpi, which has two domains; + stab condition, for
     near_onset, which has stab-on and stab-off Method A runs)
  5. Computes mean +/- sample std (ddof=1) per (regime, domain, stab) group
  6. Prints every group found, flags anything NOT expected, and diffs
     against the values currently written into the manuscript's Table 1

Usage (run this in the same Kaggle working directory that holds `results/`):
    python verify_table1_spectral_error.py --results-dir /kaggle/working/results
    python verify_table1_spectral_error.py --results-dir /kaggle/working/results --verbose

If your directory names differ from the mapping below, add them to
DIR_TO_REGIME (see the comment inside) -- the script will list any
unrecognized directories it skips so you can extend the mapping rather than
silently mis-attribute data.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from collections import defaultdict

# ----------------------------------------------------------------------
# Directory-name -> (canonical regime, domain, stab_condition) mapping.
# stab_condition is only meaningful for near_onset (the one regime with
# both a stab-on and a stab-off Method A run under Table 1's "default"
# condition being stab-on); everything else is always run with the
# stabilizer active, so stab_condition is fixed at "on" for them.
#
# Extend this dict if your Kaggle output directories are named
# differently -- match on the directory name exactly as it appears one
# level above the metrics json file.
# ----------------------------------------------------------------------
DIR_TO_REGIME = {
    "S_spots_A":            ("S_spots",    "square",   "on"),
    "S_spots_B":            ("S_spots",    "square",   "on"),
    "M_spots_A":            ("M_spots",    "square",   "on"),
    "M_spots_B":            ("M_spots",    "square",   "on"),
    "T_spots_A":            ("T_spots",    "square",   "on"),
    "T_spots_B":            ("T_spots",    "square",   "on"),
    "low_rho_A":            ("low_rho",    "square",   "on"),
    "low_rho_B":            ("low_rho",    "square",   "on"),
    "min_rho_A":            ("min_rho",    "square",   "on"),
    "min_rho_B":            ("min_rho",    "square",   "on"),
    "kstripes_A_stabon":    ("K_stripes",  "square",   "on"),
    "kstripes_A_staboff":   ("K_stripes",  "square",   "off"),
    "kstripes_B":           ("K_stripes",  "square",   "on"),
    "kmixed_A":             ("K_mixed",    "square",   "on"),
    "kmixed_B":             ("K_mixed",    "square",   "on"),
    "khpi_hexrect":         ("K_Hpi",      "hex_rect", "on"),
    "khpi_square":          ("K_Hpi",      "square",   "on"),
    "khpi_B":               ("K_Hpi",      "hex_rect", "on"),
    "nearonset_A_stab_on":  ("near_onset", "square",   "on"),
    "nearonset_A_staboff":  ("near_onset", "square",   "off"),
    "nearonset_A_stab_off": ("near_onset", "square",   "off"),
    "nearonset_B":          ("near_onset", "square",   "on"),
}

# Table 1's current spectral-error entries (regime -> (mean%, std%)),
# for the Method A / "default condition" (stab-on where applicable) row.
# K_Hpi's Table value is drawn from the hex_rect domain (the domain where
# hexagons actually form; see sec:domain_geometry_regime2). None = not
# yet in the manuscript (still "pending").
EXPECTED_TABLE1 = {
    "S_spots":   (6.57, 1.82),
    "M_spots":   (7.48, 0.0),
    "K_Hpi":     (10.0, 0.0),
    "K_mixed":   (9.40, 1.82),
    "low_rho":   (15.15, 9.24),
    "T_spots":   (12.32, 5.66),
    "min_rho":   (29.3, 10.8),
    "K_stripes": (7.57, 1.82),
    "near_onset":(6.66, 5.66),
}

METRIC_KEY = "spec_err_vs_ktrue"
FILENAME_RE = re.compile(r"^(?P<method>[A-Za-z_]+)_d2_all_metrics_s(?P<seed>\d+)\.json$")


def find_value(obj, key):
    """Search a (possibly nested) dict/list for the first occurrence of `key`."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_value(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_value(item, key)
            if found is not None:
                return found
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="Root directory to search (e.g. /kaggle/working/results)")
    ap.add_argument("--tol", type=float, default=0.05, help="Percentage-point tolerance before flagging a mismatch (default 0.05)")
    ap.add_argument("--verbose", action="store_true", help="Print every individual seed value, not just group summaries")
    args = ap.parse_args()

    root = Path(args.results_dir)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    # groups[(regime, domain, stab)] = list of (seed, value, filepath)
    groups = defaultdict(list)
    unrecognized_dirs = set()
    skipped_method_b = 0
    skipped_no_metric = []

    json_files = sorted(root.rglob("*_d2_all_metrics_s*.json"))
    if not json_files:
        print(f"No files matching '*_d2_all_metrics_s*.json' found under {root}")
        sys.exit(1)

    for fp in json_files:
        m = FILENAME_RE.match(fp.name)
        if not m:
            continue
        method_raw = m.group("method")
        seed = int(m.group("seed"))

        is_method_a = "spectral" in method_raw.lower()
        is_method_b = "autograd" in method_raw.lower()
        if is_method_b:
            skipped_method_b += 1
            continue
        if not is_method_a:
            # Unknown method prefix -- skip but report
            unrecognized_dirs.add(f"(unrecognized method prefix '{method_raw}' in {fp})")
            continue

        dirname = fp.parent.name
        mapping = DIR_TO_REGIME.get(dirname)
        if mapping is None:
            unrecognized_dirs.add(dirname)
            continue
        regime, domain, stab = mapping

        try:
            data = json.loads(fp.read_text())
        except Exception as e:
            print(f"WARNING: could not parse {fp}: {e}", file=sys.stderr)
            continue

        val = find_value(data, METRIC_KEY)
        if val is None:
            skipped_no_metric.append(str(fp))
            continue

        groups[(regime, domain, stab)].append((seed, float(val), str(fp)))

    # ---- report ----
    print("=" * 100)
    print(f"Loaded {len(json_files)} candidate files under {root}")
    print(f"  Method A (spectral_FFT) records kept : {sum(len(v) for v in groups.values())}")
    print(f"  Method B (autograd) records skipped   : {skipped_method_b}")
    if skipped_no_metric:
        print(f"  Files missing '{METRIC_KEY}' entirely : {len(skipped_no_metric)}")
        for f in skipped_no_metric:
            print(f"    - {f}")
    if unrecognized_dirs:
        print(f"  Unrecognized directory names (not in DIR_TO_REGIME, skipped):")
        for d in sorted(unrecognized_dirs):
            print(f"    - {d}")
    print("=" * 100)

    if not groups:
        print("Nothing to report -- no Method A records matched a known regime directory.")
        sys.exit(0)

    print(f"\n{'Regime':<12} {'Domain':<10} {'Stab':<5} {'n':<3} {'Mean%':>8} {'Std%':>8}   Table 1 says      Match?")
    print("-" * 100)

    any_mismatch = False
    for (regime, domain, stab), records in sorted(groups.items()):
        vals = [v for _, v, _ in records]
        n = len(vals)
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if n > 1 else 0.0

        # Only compare against Table 1 for the row that matches Table 1's
        # own condition: stab-on everywhere, and (for K_Hpi) hex_rect domain.
        is_table_row = (stab == "on") and not (regime == "K_Hpi" and domain != "hex_rect")
        expected = EXPECTED_TABLE1.get(regime) if is_table_row else None

        if expected is None:
            match_str = "(not in Table 1)" if is_table_row else "(not Table 1's row)"
        else:
            exp_mean, exp_std = expected
            mean_ok = abs(mean - exp_mean) <= args.tol
            std_ok = abs(std - exp_std) <= args.tol
            if mean_ok and std_ok:
                match_str = "OK"
            else:
                match_str = f"MISMATCH (expected {exp_mean:.2f}+/-{exp_std:.2f})"
                any_mismatch = True

        print(f"{regime:<12} {domain:<10} {stab:<5} {n:<3} {mean:>8.2f} {std:>8.2f}   "
              f"{'-' if expected is None else f'{expected[0]:.2f}+/-{expected[1]:.2f}':<16} {match_str}")

        if args.verbose:
            for seed, val, path in sorted(records):
                print(f"      seed={seed:<4} value={val:<8.4f} {path}")

    # Flag any Table-1 regime that produced NO records at all (still pending)
    found_regimes_on = {regime for (regime, domain, stab) in groups
                         if stab == "on" and not (regime == "K_Hpi" and domain != "hex_rect")}
    missing_from_disk = set(EXPECTED_TABLE1) - found_regimes_on
    if missing_from_disk:
        print(f"\nRegimes Table 1 reports a value for, but no Method A json found on disk for: {sorted(missing_from_disk)}")

    print()
    if any_mismatch:
        print("RESULT: one or more MISMATCHes found above -- Table 1 does not match the raw metrics files. Investigate before submitting.")
        sys.exit(2)
    else:
        print("RESULT: every Table 1 spectral-error entry with data on disk matches the raw metrics files within tolerance.")


if __name__ == "__main__":
    main()
