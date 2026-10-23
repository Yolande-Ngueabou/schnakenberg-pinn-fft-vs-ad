"""
Two checks, run on Kaggle where the results live.

PART 1  K_Hpi hexagon classification: threshold robustness
    Re-runs the manuscript's Fourier-peak criterion (find_hexagonal_peaks from
    verify_khpi_hexagonal_symmetry.py, the version with the peak-to-background test)
    on every saved K_Hpi hex_rect snapshot, for peak-to-background thresholds
    100, 200 (the value used in the paper) and 500, for Method A and Method B.
    It reports hold / transition and the transition time per seed, and whether the
    held-vs-transition verdict changes with the threshold.

PART 2  Total GPU time from the saved metrics files
    Sums wall_time_s over every completed run (newest file per regime, method,
    stabilizer setting, domain and seed) and prints hours per method and in total.
    Memory use and restart counts are NOT stored in the metrics files, so they are
    not reported here.

Setup (Kaggle): put verify_khpi_hexagonal_symmetry.py (the version printing
"peak_to_background_threshold") in the working directory, then
    %run robustness_khpi_and_gpu_hours.py
or  !python robustness_khpi_and_gpu_hours.py /kaggle/working/results
"""
import glob, json, os, sys
import numpy as np
from verify_khpi_hexagonal_symmetry import find_hexagonal_peaks, summarize_time_ranges

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/results"
THRESHOLDS = [100.0, 200.0, 500.0]
SEEDS = [42, 45, 49, 52]
PAPER = {"autograd": {42: None, 45: None, 49: None, 52: 7.465},          # None = holds
         "spectral_FFT": {42: 8.105, 45: None, 49: 6.740, 52: 7.910}}   # float = transition time


def all_records(root):
    out = []
    for f in glob.glob(os.path.join(root, "**", "*_all_metrics_s*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for m in (d if isinstance(d, list) else [d]):
            if isinstance(m, dict) and "label" in m:
                m = dict(m); m["_mtime"] = os.path.getmtime(f); m["_dir"] = os.path.dirname(f)
                out.append(m)
    return out


def newest(records):
    best = {}
    for m in records:
        key = (m.get("label"), m.get("method"), bool(m.get("stab_reg_used")), m.get("domain"), m.get("seed"))
        if key not in best or m["_mtime"] > best[key]["_mtime"]:
            best[key] = m
    return best


def classify_run(U, t, k_true, Lx, Ly, thr):
    flags = np.array([find_hexagonal_peaks(U[i], k_true, Lx, Ly, peak_to_background_threshold=thr)[0]
                      for i in range(len(t))])
    if flags.all():
        return "hold", None
    if not flags.any():
        return "never", None
    last = np.where(flags)[0][-1]
    if flags[: last + 1].all():
        return "transition", float(t[last + 1]) if last + 1 < len(t) else float(t[-1])
    return "oscillates", None


recs = newest(all_records(ROOT))

print("=" * 78)
print("PART 1: K_Hpi (hex_rect, stabilizer on) -- threshold robustness")
print("=" * 78)
changed = False
for method, name in (("autograd", "Method B"), ("spectral_FFT", "Method A")):
    print(f"\n{name}")
    print(f"  {'seed':>4} " + "".join(f"{'thr=' + str(int(t)):>24}" for t in THRESHOLDS) + "   paper")
    for s in SEEDS:
        m = recs.get(("K_Hpi", method, True, "hex_rect", s))
        if m is None:
            print(f"  {s:>4}  [no run found]")
            continue
        arr = glob.glob(os.path.join(m["_dir"], f"*K_Hpi_s{s}_arrays.npz"))
        if not arr:
            print(f"  {s:>4}  [arrays missing]")
            continue
        z = np.load(arr[0]); t, U = z["t"], z["U"]
        row, verdicts = [], set()
        for thr in THRESHOLDS:
            st, tt = classify_run(U, t, m["k_true"], m["Lx"], m["Ly"], thr)
            verdicts.add(st)
            row.append(st if tt is None else f"{st} t={tt:.3f}")
        rep = PAPER[method][s]
        paper = "hold" if rep is None else f"transition t={rep:.3f}"
        flag = "" if len(verdicts) == 1 else "   <-- VERDICT CHANGES WITH THRESHOLD"
        changed |= len(verdicts) > 1
        print(f"  {s:>4} " + "".join(f"{r:>24}" for r in row) + f"   {paper}{flag}")
print("\nSUMMARY:", "at least one seed's verdict changes with the threshold -- inspect it."
      if changed else "no seed's hold/transition verdict changes across thresholds 100-500.")

print("\n" + "=" * 78)
print("PART 2: total GPU time from saved metrics (wall_time_s)")
print("=" * 78)
tot = {}
n_runs = {}
for (label, method, stab, domain, seed), m in recs.items():
    w = m.get("wall_time_s")
    if w is None:
        continue
    tot[method] = tot.get(method, 0.0) + float(w)
    n_runs[method] = n_runs.get(method, 0) + 1
for method, w in sorted(tot.items()):
    print(f"  {method:14s} {n_runs[method]:3d} runs   {w / 3600:8.1f} GPU-hours")
print(f"  {'TOTAL':14s} {sum(n_runs.values()):3d} runs   {sum(tot.values()) / 3600:8.1f} GPU-hours")
print("  (newest file per regime/method/stabilizer/domain/seed; reruns that were overwritten are not counted)")
