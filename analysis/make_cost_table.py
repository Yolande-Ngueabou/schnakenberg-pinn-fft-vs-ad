"""Per-window cost table, Method A vs Method B, from saved metrics (ms_per_window).
Stabilizer-on runs only; K_Hpi uses the hex_rect domain, all others the square domain.
Newest file per (regime, method, stab, domain, seed).  Run:  !python make_cost_table.py
Prints a text table and a LaTeX tabular to paste/send back."""
import glob, json, os
import numpy as np
ROOT = "/kaggle/working"          # add a second root below if results live in /kaggle/input too
ROOTS = [ROOT, "/kaggle/input"]
ORDER = ["S_spots", "M_spots", "T_spots", "near_onset", "low_rho", "min_rho", "K_stripes", "K_mixed", "K_Hpi"]

best = {}
for r in ROOTS:
    for f in glob.glob(os.path.join(r, "**", "*_all_metrics_s*.json"), recursive=True):
        try: d = json.load(open(f))
        except Exception: continue
        for m in (d if isinstance(d, list) else [d]):
            if not (isinstance(m, dict) and "label" in m and m.get("ms_per_window") is not None): continue
            if not m.get("stab_reg_used"): continue
            dom = "hex_rect" if m["label"] == "K_Hpi" else "square"
            if m.get("domain") != dom: continue
            key = (m["label"], m["method"], m.get("seed"))
            mt = os.path.getmtime(f)
            if key not in best or mt > best[key][0]:
                best[key] = (mt, float(m["ms_per_window"]), float(m.get("wall_time_s", float("nan"))), float(m.get("n_windows", float("nan"))))

def runstats(label, method):
    """mean wall-clock hours per run, mean windows completed, over all saved runs of that regime/method"""
    v = [(x[2], x[3]) for k, x in best.items() if k[0] == label and k[1] == method]
    if not v: return None
    return np.nanmean([a for a, _ in v]) / 3600.0, np.nanmean([b for _, b in v]), len(v)

def stats(label, method):
    v = [x[1] for k, x in best.items() if k[0] == label and k[1] == method]
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else float("nan"), len(v)) if v else None

print(f"{'regime':<12}{'A ms/win (mean±sd, n)':>26}{'B ms/win (mean±sd, n)':>26}{'B/A':>8}")
tex = []
for lab in ORDER:
    a, b = stats(lab, "spectral_FFT"), stats(lab, "autograd")
    fa = f"{a[0]:.0f} ± {a[1]:.0f} (n={a[2]})" if a else "-"
    fb = f"{b[0]:.0f} ± {b[1]:.0f} (n={b[2]})" if b else "-"
    ra = f"{b[0]/a[0]:.2f}" if a and b else "-"
    print(f"{lab:<12}{fa:>26}{fb:>26}{ra:>8}")
    L = lab.replace("_", r"\_")
    ca = f"{a[0]:.0f} $\\pm$ {a[1]:.0f}" if a and a[2] > 1 else (f"{a[0]:.0f}" if a else "--")
    cb = f"{b[0]:.0f} $\\pm$ {b[1]:.0f}" if b and b[2] > 1 else (f"{b[0]:.0f}" if b else "--")
    tex.append(f"\\texttt{{{L}}} & {ca} & {a[2] if a else 0} & {cb} & {b[2] if b else 0} & {ra} \\\\")
print("\nLaTeX rows:\n" + "\n".join(tex))

print("\n" + "=" * 78)
print("WALL-CLOCK PER RUN (all saved runs, stabilizer on; includes runs that stopped early or failed)")
print("=" * 78)
print(f"{'regime':<12}{'A hours/run':>13}{'A windows':>11}{'B hours/run':>13}{'B windows':>11}{'B/A hours':>11}")
tex2 = []
for lab in ORDER:
    a, b = runstats(lab, "spectral_FFT"), runstats(lab, "autograd")
    ra = f"{b[0]/a[0]:.2f}" if a and b else "-"
    print(f"{lab:<12}{(f'{a[0]:.2f}' if a else '-'):>13}{(f'{a[1]:.0f}' if a else '-'):>11}{(f'{b[0]:.2f}' if b else '-'):>13}{(f'{b[1]:.0f}' if b else '-'):>11}{ra:>11}")
    L = lab.replace("_", r"\_")
    tex2.append(f"\\texttt{{{L}}} & {a[0]:.2f} & {a[1]:.0f} & {b[0]:.2f} & {b[1]:.0f} & {ra} \\\\" if a and b else f"\\texttt{{{L}}} & -- & -- & -- & -- & -- \\\\")
print("\nLaTeX rows (hours/run, mean windows):\n" + "\n".join(tex2))
