# ===== CELL 2: late-breakdown trajectories =====
import glob, json, os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/kaggle/working/results"
OUT = "/kaggle/working/figures"; os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 45, 49, 52]
PANELS = [  # (title, regime label, method, domain filter, highlighted seed)
    ("K_mixed, Method B", "K_mixed", "autograd", None, 49),
    ("K_Hpi (hex_rect), Method B", "K_Hpi", "autograd", "hex_rect", 49)]

def find_array(label, method, domain, seed):
    best = None
    for f in glob.glob(os.path.join(ROOT, "**", "*_all_metrics_s*.json"), recursive=True):
        try: d = json.load(open(f))
        except Exception: continue
        for m in (d if isinstance(d, list) else [d]):
            if (isinstance(m, dict) and m.get("label") == label and m.get("method") == method
                    and m.get("seed") == seed and bool(m.get("stab_reg_used"))
                    and (domain is None or m.get("domain") == domain)):
                a = glob.glob(os.path.join(os.path.dirname(f), f"*{label}_s{seed}_arrays.npz"))
                if a and (best is None or os.path.getmtime(a[0]) > os.path.getmtime(best)): best = a[0]
    return best

def amplitude(path):
    z = np.load(path); U = z["U"]
    return z["t"], U.reshape(len(U), -1).max(1) - U.reshape(len(U), -1).min(1)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, (title, label, method, dom, hs) in zip(axes, PANELS):
    for s in SEEDS:
        p = find_array(label, method, dom, s)
        if p is None: print(f"[missing] {title} seed {s}"); continue
        t, a = amplitude(p)
        if s == hs:
            k = int(a.argmax()); T = t[-1]
            ax.plot(t, a, color="#D55E00", lw=2, label=f"seed {s}", zorder=3)
            ax.plot(t[k], a[k], "o", color="#D55E00", zorder=4)
            ax.annotate(f"peak {a[k]:.2f}\nt={t[k]:.2f}", (t[k], a[k]), textcoords="offset points",
                        xytext=(-45, -34), ha="right", fontsize=7.5,
                        arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7))
            below = np.where(a[k:] < 0.5 * a[k])[0]
            if len(below):
                tb = t[k + below[0]]; ax.axvline(tb, color="#D55E00", ls=":", lw=1)
                ax.text(tb - 0.15, 0.22, f"falls below 50%\nof peak at t={tb:.2f}", fontsize=7.5, ha="right",
                        va="center", transform=ax.get_xaxis_transform())
            ax.plot(t[-1], a[-1], "s", color="#D55E00", zorder=4)
            ax.annotate(f"final {a[-1]:.3f}\n({a[-1]/a[k]:.0%} of peak)", (t[-1], a[-1]), textcoords="offset points",
                        xytext=(-10, 0), ha="right", va="center", fontsize=7.5)
            ax.axhline(a[0], color="k", ls="--", lw=0.8)
            ax.text(t[0], a[0], "  initial (seeded) amplitude", fontsize=7, va="bottom")
            print(f"{title}: seed {s} peak {a[k]:.3f} at t={t[k]:.3f}, final {a[-1]:.3f} ({a[-1]/a[k]:.1%})")
        else:
            ax.plot(t, a, color="0.6", lw=1, label=f"seed {s}")
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel(r"$\max u - \min u$")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_late_breakdown.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_late_breakdown.png", bbox_inches="tight", dpi=300)
print("saved", f"{OUT}/fig_late_breakdown.pdf/.png")
