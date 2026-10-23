# ===== CELL 1: seed-outcome grid =====
import glob, json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = "/kaggle/working/results"
OUT = "/kaggle/working/figures"; os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 45, 49, 52]

# (row label, regime label, method key in JSON, stabilizer on?, domain filter or None)
A, B = "spectral_FFT", "autograd"
ROWS = [x for r in ["S_spots", "M_spots", "T_spots", "low_rho", "min_rho"]
        for x in [(f"{r} (A)", r, A, True, None), (f"{r} (B)", r, B, True, None)]]
ROWS += [("near_onset (A, stab on)", "near_onset", A, True, None),
         ("near_onset (A, stab off)", "near_onset", A, False, None),
         ("near_onset (B)", "near_onset", B, True, None),
         ("K_stripes (A, stab on)", "K_stripes", A, True, None),
         ("K_stripes (A, stab off)", "K_stripes", A, False, None),
         ("K_stripes (B)", "K_stripes", B, True, None),
         ("K_mixed (A)", "K_mixed", A, True, None),
         ("K_mixed (B)", "K_mixed", B, True, None),
         ("K_Hpi (A, hex_rect)", "K_Hpi", A, True, "hex_rect"),
         ("K_Hpi (A, square)", "K_Hpi", A, True, "square"),
         ("K_Hpi (B, hex_rect)", "K_Hpi", B, True, "hex_rect")]

# Verified by full-trajectory Fourier test (verify_khpi / verify_khpi_hexagonal_symmetry output)
KHPI_HEX = {(B, 42): "hex_hold", (B, 45): "hex_hold", (B, 49): "hex_hold", (B, 52): "hex_transition",
            (A, 42): "hex_transition", (A, 45): "hex_hold", (A, 49): "hex_transition", (A, 52): "hex_transition"}
OVERRIDE = {("K_mixed", B, 49): "late_breakdown"}          # verified from saved arrays (94.0%, t=18.805)
DAGGER = {("K_Hpi", B, 49)}                                  # holds symmetry but amplitude decays at the end
SAVE_ORDER_ARTIFACT = {"T_spots", "low_rho", "min_rho", "near_onset"}

STYLE = {  # Okabe-Ito colour-blind-safe palette + a letter in every cell
    "stable": ("#009E73", "S"), "collapse": ("#D55E00", "C"), "late_breakdown": ("#E69F00", "L"),
    "diverge": ("#CC79A7", "D"), "hex_hold": ("#0072B2", "H"), "hex_transition": ("#999999", "T"),
    "no_hex": ("#D9D9D9", "N"), "unknown": ("#F0E442", "?"), "missing": ("#FFFFFF", "–")}
ORDER = list(STYLE)

def load_rows(root):
    rows = []
    for f in glob.glob(os.path.join(root, "**", "*_all_metrics_s*.json"), recursive=True):
        try: d = json.load(open(f))
        except Exception: continue
        for m in (d if isinstance(d, list) else [d]):
            if isinstance(m, dict) and "label" in m:
                m = dict(m); m["_mtime"] = os.path.getmtime(f); m["_file"] = f; rows.append(m)
    return rows

def pick(rows, label, method, stab, domain, seed):
    c = [r for r in rows if r.get("label") == label and r.get("method") == method
         and bool(r.get("stab_reg_used")) == stab and r.get("seed") == seed
         and (domain is None or r.get("domain") == domain)]
    return max(c, key=lambda r: r["_mtime"]) if c else None

def classify(r, label, method, domain, seed):
    if r is None: return "missing"
    if (label, method, seed) in OVERRIDE: return OVERRIDE[(label, method, seed)]
    if label == "K_Hpi":
        return "no_hex" if domain == "square" else KHPI_HEX.get((method, seed), "unknown")
    u, se = r.get("u_amp"), r.get("spec_err_vs_ktrue")
    if u is None: return "unknown"
    if r.get("collapsed") is True or u < 0.3: return "collapse"
    if se is not None and se > 50: return "late_breakdown" if u < 2 else "diverge"
    if method == B and label in SAVE_ORDER_ARTIFACT and u > 20: return "diverge"
    return "stable"

rows = load_rows(ROOT)
print(f"{len(rows)} metric records found under {ROOT}\n")
grid, audit = [], []
for name, label, method, stab, dom in ROWS:
    line = []
    for s in SEEDS:
        r = pick(rows, label, method, stab, dom, s)
        o = classify(r, label, method, dom, s); line.append(o)
        audit.append((name, s, o, None if r is None else r.get("u_amp"),
                      None if r is None else r.get("spec_err_vs_ktrue"),
                      None if r is None else r.get("collapsed"), None if r is None else r.get("domain")))
    grid.append(line)

# ---- audit table: CHECK THIS against the manuscript's results table before using the figure ----
print(f"{'row':26s} {'seed':>4s} {'outcome':15s} {'u_amp':>8s} {'spec_err':>9s} {'collapsed':>9s} domain")
for n, s, o, u, se, c, d in audit:
    print(f"{n:26s} {s:>4d} {o:15s} {u if u is None else round(u,3)!s:>8s} "
          f"{se if se is None else round(se,1)!s:>9s} {c!s:>9s} {d}")

idx = np.array([[ORDER.index(o) for o in line] for line in grid])
fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(ROWS) + 1.8))
ax.imshow(idx, cmap=ListedColormap([STYLE[o][0] for o in ORDER]), vmin=0, vmax=len(ORDER) - 1, aspect="auto")
for i, (name, label, method, stab, dom) in enumerate(ROWS):
    for j, s in enumerate(SEEDS):
        t = STYLE[grid[i][j]][1] + ("†" if (label, method, s) in DAGGER else "")
        ax.text(j, i, t, ha="center", va="center", fontsize=8,
                color="white" if grid[i][j] in ("stable", "collapse", "hex_hold", "diverge") else "black")
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS], fontsize=7.5)
ax.set_xticks(range(len(SEEDS))); ax.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=8)
ax.set_xticks(np.arange(-.5, len(SEEDS)), minor=True); ax.set_yticks(np.arange(-.5, len(ROWS)), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5); ax.tick_params(which="minor", length=0)
names = {"stable": "stable", "collapse": "collapse", "late_breakdown": "late breakdown", "diverge": "divergence",
         "hex_hold": "hexagon holds", "hex_transition": "hexagon lost", "no_hex": "never hexagonal", "unknown": "unclassified"}
used = {o for line in grid for o in line}
names = {k: v for k, v in names.items() if k in used}
ax.legend([plt.Rectangle((0, 0), 1, 1, color=STYLE[o][0]) for o in names], list(names.values()),
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=7.5)
fig.text(0.01, 0.045, "† hexagonal symmetry persists but amplitude decays near the end of the window.",
         fontsize=7, va="top")
fig.savefig(f"{OUT}/fig_seed_outcome_grid.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_seed_outcome_grid.png", bbox_inches="tight", dpi=300)
print(f"\nsaved {OUT}/fig_seed_outcome_grid.pdf/.png; unclassified cells:", int((idx == ORDER.index('unknown')).sum()))
