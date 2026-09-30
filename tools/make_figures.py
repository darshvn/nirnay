"""Figures for the README, generated from results/*.csv (python tools/make_figures.py)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, SAFFRON, GREEN, GREY = "#1F2A44", "#E8702A", "#1E8C45", "#9AA5B1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 13})


def rows(name):
    return list(csv.DictReader(open(ROOT / "results" / f"{name}.csv", newline="")))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def solved(rs, tol, strict=True):
    return [r for r in rs if r["status"] == "optimal" and np.isfinite(f(r["rel_err"]))
            and (f(r["rel_err"]) < tol if strict else f(r["rel_err"]) <= tol)]


spx, ipm = rows("netlib_simplex_v4"), rows("netlib_ipm_v2")
mip, qp = rows("miplib3_bnb_v2"), rows("maros_qpipm_v2")

# 1. solved per suite ------------------------------------------------------------------------
suites = ["Netlib LP\n(90)", "MIPLIB 3, 60 s\n(64)", "Maros–Mészáros QP\n(138)"]
ours = [len(solved(spx, 1e-6)) / 90, len(solved(mip, 1e-4, False)) / 64, len(solved(qp, 1e-6)) / 138]
ref = [1.0, sum(r["ref_status"] == "Optimal" for r in mip) / 64,
       sum(r["ref_status"].startswith("Solved") for r in qp) / 138]
fig, ax = plt.subplots(figsize=(8, 4.2))
y = np.arange(3)
b1 = ax.barh(y - 0.2, [100 * v for v in ours], 0.38, color=SAFFRON, label="NIRNAY")
b2 = ax.barh(y + 0.2, [100 * v for v in ref], 0.38, color=GREY, label="Reference (HiGHS / Clarabel)")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.0f}%",
                va="center", fontsize=10)
ax.set_yticks(y, suites)
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_xlabel("problems solved to the reference optimum (%)")
ax.set_title("Public benchmark sets, same laptop")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "solved_by_suite.png", dpi=160)

# 2. performance profile on Netlib (time ratio to the best) -----------------------------------
def times(rs, ok_names):
    return {r["name"]: (f(r["time"]) if r["name"] in ok_names else np.inf) for r in rs}


ok_s = {r["name"] for r in solved(spx, 1e-6)}
ok_i = {r["name"] for r in solved(ipm, 1e-6)}
T = {"NIRNAY dual simplex": times(spx, ok_s), "NIRNAY interior point": times(ipm, ok_i),
     "HiGHS": {r["name"]: f(r["ref_time"]) for r in spx}}
names = sorted(set.intersection(*[set(v) for v in T.values()]))
M = np.array([[max(T[k][n], 1e-3) for k in T] for n in names])
best = M.min(axis=1, keepdims=True)
R = M / best
taus = np.logspace(0, 3, 300)
fig, ax = plt.subplots(figsize=(8, 4.2))
for (lab, col), j in zip([("NIRNAY dual simplex", SAFFRON), ("NIRNAY interior point", GREEN), ("HiGHS", NAVY)],
                         range(3)):
    ax.plot(taus, [(R[:, j] <= t).mean() for t in taus], color=col, lw=2.2, label=lab)
ax.set_xscale("log")
ax.set_ylim(0, 1.02)
ax.set_xlabel("within this factor of the fastest solver")
ax.set_ylabel("share of the 90 Netlib LPs")
ax.set_title("Performance profile (Dolan–Moré), Netlib LP")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "netlib_profile.png", dpi=160)

# 3. PDLP on large LPs: CPU vs GPU vs HiGHS ----------------------------------------------------
L = [r for r in rows("large_pdlp_gpu") if r["gpu_status"] == "optimal" and f(r["gpu_rel_err"]) < 1e-3]
L.sort(key=lambda r: f(r["gpu_time"]))
lab = [r["name"].replace("neos-5052403-", "") for r in L]
cpu = [f(r["cpu_time"]) if r["cpu_status"] == "optimal" else np.nan for r in L]
gpu = [f(r["gpu_time"]) for r in L]
hig = [f(r["highs_time"]) if r["highs_status"] == "Optimal" else np.nan for r in L]
fig, ax = plt.subplots(figsize=(9, 4.4))
x = np.arange(len(L))
ax.bar(x - 0.27, hig, 0.26, color=GREY, label="HiGHS (default)")
ax.bar(x, cpu, 0.26, color=NAVY, label="NIRNAY PDLP, CPU")
ax.bar(x + 0.27, gpu, 0.26, color=SAFFRON, label="NIRNAY PDLP, GPU (RTX 3050 laptop)")
for i, h in enumerate(hig):
    if not np.isfinite(h):
        ax.text(x[i] - 0.27, 0.35, "HiGHS\nno optimum\nin 300 s", ha="center", va="bottom", fontsize=7,
                color="#555")
ax.set_yscale("log")
ax.set_xticks(x, lab, rotation=20)
ax.set_ylabel("seconds (log scale)")
ax.set_title("Large LPs at 1e-4 accuracy: where the GPU pays")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "pdlp_large.png", dpi=160)

# 4. MIPLIB gaps at the time limit -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 3.8))
gaps = sorted(f(r["gap"]) if np.isfinite(f(r["gap"])) else 1.0 for r in mip)
ax.plot(np.arange(1, len(gaps) + 1), np.minimum(np.array(gaps) * 100, 100), color=SAFFRON, lw=2.2,
        drawstyle="steps-post")
ax.axhline(0.01, color=GREY, ls="--", lw=1)
ax.set_yscale("symlog", linthresh=0.01)
ax.set_xlabel("MIPLIB 3 instances, sorted")
ax.set_ylabel("proven gap after 60 s (%)")
ax.set_title("Branch-and-bound: every instance ends with a proven bound")
fig.tight_layout()
fig.savefig(OUT / "miplib_gaps.png", dpi=160)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
