"""Generate every number, table and plot file the NIRNAY report uses.

Reads   ../../results/*.csv      (benchmark sweeps; files starting with '_' are ignored)
        ../../nirnay/**/*.py     (line counts for the module table)
        ../../data/cases/*.mps   (case-study model sizes)
        data/cases_runs.csv      (optional: NIRNAY on the case studies, written by run_cases.py)
Writes  data/macros.tex          \\newcommand macros for every quoted number
        data/*.dat               PGFPlots tables (performance profiles, scatters, ECDFs, bars)
        data/*.tex               generated table bodies

The report never hard-codes a benchmark number: rerun this script when a sweep finishes and
rebuild the PDF.

usage: python gen_data.py
"""
from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = REPO / "results"
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)

LP_TOL, MIP_TOL, QP_TOL = 1e-6, 1e-4, 1e-6
PDLP_TOL_LOOSE = 1e-3            # PDLP runs at a 1e-4 relative KKT tolerance; 1e-3 on the objective
SHIFT_A, SHIFT_B = 1.0, 10.0     # shifted geometric mean shifts (seconds)
TL_NETLIB, TL_MIP, TL_QP, TL_PDLP = 300.0, 60.0, 120.0, 120.0

macros: dict[str, str] = {}


def macro(name: str, value, fmt: str | None = None):
    """Store a LaTeX macro. Names must be letters only (TeX)."""
    assert re.fullmatch(r"[A-Za-z]+", name), name
    if isinstance(value, float):
        if fmt is None:
            fmt = "{:.3g}"
        s = fmt.format(value)
    else:
        s = str(value)
    macros[name] = s


def tex_escape(s) -> str:
    s = str(s)
    return (s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#"))


def sgm(t, shift):
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(t + shift))) - shift)


def read(name: str) -> pd.DataFrame | None:
    p = RES / f"{name}.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
    except pd.errors.EmptyDataError:        # a sweep that has just started writing
        return None
    if len(d) == 0:
        return None
    for c in ("time", "ref_time", "rel_err", "obj", "ref_obj", "iters", "nodes", "gap", "bound"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def fmt_time(t):
    if t is None or not np.isfinite(t):
        return "--"
    if t < 0.01:
        return f"{t:.3f}"
    if t < 10:
        return f"{t:.2f}"
    return f"{t:.1f}"


def fint(v):
    try:
        return str(int(v)) if np.isfinite(float(v)) else "--"
    except (TypeError, ValueError):
        return "--"


def fmt_err(e):
    if e is None or not np.isfinite(e):
        return "--"
    if e == 0:
        return "0"
    return f"\\num{{{e:.1e}}}"


def short_ref(s) -> str:
    """Reference status for a table cell; an annotated fallback reference gets a section mark."""
    s = str(s)
    if "NIRNAY reading" in s:
        return tex_escape(s.split(" (")[0]) + "$^{\\S}$"
    return tex_escape(s)


def status_short(s: str) -> str:
    s = str(s)
    if s.startswith("error"):
        return "error"
    return s.replace("_", "\\_")


def write_dat(name: str, header: list[str], rows):
    with open(OUT / name, "w", newline="") as fh:
        fh.write(" ".join(header) + "\n")
        for r in rows:
            fh.write(" ".join(str(v) for v in r) + "\n")


# =============================================================================================
# LP sweeps (Netlib)
# =============================================================================================

def lp_solved(d: pd.DataFrame, tol: float) -> pd.Series:
    return (d.status == "optimal") & (d.rel_err < tol)


def lp_summary(tag: str, d: pd.DataFrame, tol: float, timelimit: float):
    ok = lp_solved(d, tol)
    macro(f"n{tag}", int(ok.sum()))
    macro(f"n{tag}Tot", len(d))
    both = d[ok]
    macro(f"sgm{tag}", sgm(both.time, SHIFT_A), "{:.2f}")
    macro(f"sgm{tag}Ref", sgm(both.ref_time, SHIFT_A), "{:.3f}")
    ra = sgm(both.time, SHIFT_A) / sgm(both.ref_time, SHIFT_A)
    rb = sgm(both.time, SHIFT_B) / sgm(both.ref_time, SHIFT_B)
    macro(f"ratio{tag}", ra, "{:.1f}")
    macro(f"ratio{tag}TenShift", rb, "{:.1f}")
    # geometric mean of per-instance ratios (unshifted), and the median
    rr = (both.time / both.ref_time).replace([np.inf, -np.inf], np.nan).dropna()
    macro(f"gmr{tag}", float(np.exp(np.log(rr).mean())) if len(rr) else float("nan"), "{:.1f}")
    macro(f"medr{tag}", float(rr.median()) if len(rr) else float("nan"), "{:.1f}")
    macro(f"maxr{tag}", float(rr.max()) if len(rr) else float("nan"), "{:.0f}")
    macro(f"nFaster{tag}", int((rr < 1).sum()))
    macro(f"tot{tag}", float(both.time.sum()), "{:.0f}")
    macro(f"totRef{tag}", float(both.ref_time.sum()), "{:.1f}")
    macro(f"maxt{tag}", float(both.time.max()) if len(both) else float("nan"), "{:.1f}")
    macro(f"maxtName{tag}", tex_escape(both.loc[both.time.idxmax(), "name"]) if len(both) else "--")
    macro(f"medErr{tag}", float(both.rel_err.median()) if len(both) else float("nan"), "{:.1e}")
    macro(f"maxErr{tag}", float(both.rel_err.max()) if len(both) else float("nan"), "{:.1e}")
    if "iters" in d:
        macro(f"medIt{tag}", float(both.iters.median()) if len(both) else float("nan"), "{:.0f}")
        macro(f"maxIt{tag}", float(both.iters.max()) if len(both) else float("nan"), "{:.0f}")
    return ok


def ecdf_dat(name: str, values, floor=1e-17):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    v = np.maximum(v, floor)
    v.sort()
    n = len(v)
    rows = [(math.log10(v[i]), (i + 1) / n) for i in range(n)] if n else []
    write_dat(name, ["logerr", "frac"], rows)
    return n


def profile_dat(name: str, times: dict[str, np.ndarray], tau_max=None):
    """Dolan-Moré performance profile: times[solver] is an array over the same problems, with
    NaN/inf for failures. Writes one .dat per solver: columns tau rho (step data)."""
    T = np.vstack([times[s] for s in times])          # solvers x problems
    T = np.where(np.isfinite(T), T, np.inf)
    best = T.min(axis=0)
    ok = np.isfinite(best) & (best > 0)
    n = int(ok.sum())
    tmax = 1.0
    ratios = {}
    for k, s in enumerate(times):
        r = T[k, ok] / best[ok]
        ratios[s] = r
        fin = r[np.isfinite(r)]
        if len(fin):
            tmax = max(tmax, float(fin.max()))
    if tau_max is None:
        tau_max = tmax * 1.5
    for s, r in ratios.items():
        r = np.sort(r)
        rows = [(1.0, float((r <= 1.0).mean()))]
        for i, v in enumerate(r):
            if np.isfinite(v):
                rows.append((v, (i + 1) / n))
        rows.append((tau_max, rows[-1][1]))
        write_dat(f"{name}_{s}.dat", ["tau", "rho"], rows)
    return tau_max, n


def versions(prefix: str):
    """[(version, stem)] of results/<prefix>.csv and results/<prefix>_v<k>.csv, oldest first."""
    out = []
    for p in RES.glob(prefix + "*.csv"):
        if p.name.startswith("_"):
            continue
        m_ = re.fullmatch(re.escape(prefix) + r"(?:_v(\d+))?", p.stem)
        if m_:
            out.append((int(m_.group(1) or 0), p.stem))
    return sorted(out)


def n_files(sub: str, exts=(".mps", ".mps.gz", ".qps", ".QPS")) -> int:
    d = REPO / "data" / sub
    if not d.exists():
        return 0
    return len({p.name.lower() for p in d.iterdir() if p.name.lower().endswith(tuple(e.lower() for e in exts))})


N_NETLIB, N_MIPLIB, N_MAROS = n_files("netlib"), n_files("miplib"), n_files("qp/mm")

# newer sweeps that are still running: tag -> (stem, dataframe, version)
in_progress: dict = {}


def latest(prefix: str, tag: str, full: int = 0):
    """The sweep the report analyses: the newest version that covers the whole suite (`full`
    instances). A newer version that is still running is not used for the headline numbers,
    because a partial sweep would report e.g. "23/23 solved"; it is kept in `in_progress` and the
    report shows its progress separately. Once it completes it becomes the primary sweep.
    Records the file in the macros <tag>File and <tag>Ver."""
    vs = versions(prefix)
    if not vs:
        macro(tag + "File", "none")
        macro(tag + "Ver", "--")
        macro("have" + tag.capitalize() + "New", 0)
        return None, None, 0
    frames = {stem: read(stem) for _, stem in vs}
    vs = [(v, s) for v, s in vs if frames[s] is not None]
    if not vs:
        macro(tag + "File", "none")
        macro(tag + "Ver", "--")
        macro("have" + tag.capitalize() + "New", 0)
        return None, None, 0
    complete = [(v, s) for v, s in vs if full == 0 or len(frames[s]) >= full]
    v, stem = complete[-1] if complete else vs[-1]
    newest_v, newest = vs[-1]
    if newest != stem:
        in_progress[tag] = (newest, frames[newest], newest_v)
    macro("have" + tag.capitalize() + "New", 1 if newest != stem else 0)
    macro(tag + "File", tex_escape(stem + ".csv"))
    macro(tag + "Ver", f"v{v}" if v else "v1")
    macro(tag + "VerNum", v if v else 1)
    macro(tag + "Complete", 1 if (full == 0 or len(frames[stem]) >= full) else 0)
    return stem, frames[stem], v


macro("nNetlibFiles", N_NETLIB)
macro("nMiplibFiles", N_MIPLIB)
macro("nMarosFiles", N_MAROS)
spx_stem, spx, spx_v = latest("netlib_simplex", "spx", N_NETLIB)
ipm_stem, ipm, ipm_v = latest("netlib_ipm", "ipm", N_NETLIB)
pdlp_stem, pdlp, _ = latest("netlib_pdlp", "pdlp", N_NETLIB)
spx1 = read("netlib_simplex_v1")
spx2 = read("netlib_simplex_v2")
ipm1 = read("netlib_ipm_v1")
ipm2 = read("netlib_ipm_v2")

macro("haveSpx", 1 if spx is not None else 0)
macro("haveIpm", 1 if ipm is not None else 0)
macro("havePdlp", 1 if pdlp is not None else 0)
macro("tlNetlib", f"{TL_NETLIB:.0f}")
macro("tlMip", f"{TL_MIP:.0f}")
macro("tlQp", f"{TL_QP:.0f}")
macro("tlPdlp", f"{TL_PDLP:.0f}")
macro("shiftA", f"{SHIFT_A:.0f}")
macro("shiftB", f"{SHIFT_B:.0f}")

if spx is not None:
    ok_spx = lp_summary("Spx", spx, LP_TOL, TL_NETLIB)
    # scatter
    write_dat("scatter_spx.dat", ["time", "ref", "ok", "name"],
              [(max(r.time, 1e-4) if np.isfinite(r.time) else TL_NETLIB, max(r.ref_time, 1e-4),
                int(o), r["name"]) for (_, r), o in zip(spx.iterrows(), ok_spx)])
    ecdf_dat("ecdf_spx.dat", spx[ok_spx].rel_err)
    # iterations vs m+n
    write_dat("iters_spx.dat", ["mn", "iters", "name"],
              [(r.m + r.n, r.iters, r["name"]) for _, r in spx[ok_spx].iterrows() if np.isfinite(r.iters)])
    # size statistics of Netlib
    macro("netlibMinRows", int(spx.m.min()))
    macro("netlibMaxRows", int(spx.m.max()))
    macro("netlibMaxCols", int(spx.n.max()))
    macro("netlibMaxNnz", int(spx.nnz.max()))
    macro("netlibMaxName", tex_escape(spx.loc[spx.nnz.idxmax(), "name"]))
    macro("netlibTotNnz", int(spx.nnz.sum()))

if ipm is not None:
    ok_ipm = lp_summary("Ipm", ipm, LP_TOL, TL_NETLIB)
    write_dat("scatter_ipm.dat", ["time", "ref", "ok", "name"],
              [(max(r.time, 1e-4) if np.isfinite(r.time) else TL_NETLIB, max(r.ref_time, 1e-4),
                int(o), r["name"]) for (_, r), o in zip(ipm.iterrows(), ok_ipm)])
    ecdf_dat("ecdf_ipm.dat", ipm[ok_ipm].rel_err)
    fails = ipm[~ok_ipm]
    rows = []
    for _, r in fails.iterrows():
        rows.append(f"{tex_escape(r['name'])} & {r.m} & {r.n} & {status_short(r.status)} & "
                    f"{fmt_err(r.rel_err)} & {fmt_time(r.time)} & {r.iters if np.isfinite(r.iters) else '--'} \\\\")
    (OUT / "ipm_failures.tex").write_text("\n".join(rows) + "\n")
    macro("ipmFailNames", ", ".join(f"\\texttt{{{tex_escape(n)}}}" for n in fails["name"]) or "none")
    # instances where the IPM converges only to the loose tolerance (near misses)
    near = fails[(fails.rel_err < 1e-3)]
    macro("nIpmNear", len(near))

if spx is not None and ipm is not None:
    # both engines on the same problems: per-instance which is faster
    j = spx.merge(ipm, on="name", suffixes=("_s", "_i"))
    okb = lp_solved(j.rename(columns={"status_s": "status", "rel_err_s": "rel_err"}), LP_TOL) & \
        lp_solved(j.rename(columns={"status_i": "status", "rel_err_i": "rel_err"}), LP_TOL)
    # The two sweeps ran at different times, under different machine load. Each re-solved every
    # instance with HiGHS in the same process pool, so HiGHS's time in each sweep measures the load:
    # rescale each engine's time by (common HiGHS time / HiGHS time in its own sweep), with the common
    # HiGHS time the geometric mean of the two. Load differences between sweeps then cancel.
    ref_c = np.sqrt(j.ref_time_s.clip(lower=1e-4) * j.ref_time_i.clip(lower=1e-4))
    j = j.assign(tn_s=j.time_s * ref_c / j.ref_time_s.clip(lower=1e-4),
                 tn_i=j.time_i * ref_c / j.ref_time_i.clip(lower=1e-4), ref_c=ref_c)
    jb = j[okb]
    load = float(np.exp(np.mean(np.log(j.ref_time_s.clip(lower=1e-4) / j.ref_time_i.clip(lower=1e-4)))))
    macro("sweepLoadRatio", load, "{:.1f}")
    macro("nBothSolved", len(jb))
    macro("nIpmFasterThanSpx", int((jb.tn_i < jb.tn_s).sum()))
    macro("bestOfTwoRatio", sgm(np.minimum(jb.tn_i, jb.tn_s), SHIFT_A) / sgm(jb.ref_c, SHIFT_A), "{:.1f}")
    # performance profile on load-normalised times: simplex, ipm, highs (common reference time)
    ok_s = lp_solved(j.rename(columns={"status_s": "status", "rel_err_s": "rel_err"}), LP_TOL)
    ok_i = lp_solved(j.rename(columns={"status_i": "status", "rel_err_i": "rel_err"}), LP_TOL)
    times = {"spx": np.where(ok_s, j.tn_s, np.inf).astype(float),
             "ipm": np.where(ok_i, j.tn_i, np.inf).astype(float),
             "highs": j.ref_c.values.astype(float)}
    tau_max, nprof = profile_dat("profile", times)
    macro("profTauMax", tau_max, "{:.0f}")
    macro("nProf", nprof)
    # profile including the best-of-two "auto" choice
    times2 = dict(times)
    times2["best"] = np.minimum(times["spx"], times["ipm"])
    profile_dat("profileb", times2, tau_max)
    # at what tau does each reach 100%?
    for s in ("spx", "ipm"):
        r = times[s] / np.min(np.vstack(list(times.values())), axis=0)
        fin = r[np.isfinite(r)]
        macro("tauAll" + s.capitalize(), float(fin.max()) if len(fin) else float("nan"), "{:.0f}")
        macro("rhoOne" + s.capitalize(), float((r <= 1).mean()), "{:.2f}")
        macro("rhoTen" + s.capitalize(), float((r <= 10).mean()), "{:.2f}")
    rh = times["highs"] / np.min(np.vstack(list(times.values())), axis=0)
    macro("rhoOneHighs", float((rh <= 1).mean()), "{:.2f}")

if pdlp is not None:
    okp = pdlp.status == "optimal"
    macro("nPdlp", int(okp.sum()))
    macro("nPdlpTot", len(pdlp))
    macro("nPdlpLoose", int(((pdlp.rel_err < PDLP_TOL_LOOSE) & okp).sum()))
    macro("nPdlpTight", int(((pdlp.rel_err < 1e-4) & okp).sum()))
    macro("nPdlpStrict", int(((pdlp.rel_err < LP_TOL) & okp).sum()))
    macro("sgmPdlp", sgm(pdlp[okp].time, SHIFT_A), "{:.2f}")
    macro("sgmPdlpRef", sgm(pdlp[okp].ref_time, SHIFT_A), "{:.3f}")
    macro("ratioPdlp", sgm(pdlp[okp].time, SHIFT_A) / sgm(pdlp[okp].ref_time, SHIFT_A), "{:.1f}")
    macro("medItPdlp", float(pdlp[okp].iters.median()), "{:.0f}")
    macro("maxItPdlp", float(pdlp[okp].iters.max()), "{:.0f}")
    macro("medErrPdlp", float(pdlp[okp].rel_err.median()), "{:.1e}")
    macro("maxErrPdlp", float(pdlp[okp].rel_err.max()), "{:.1e}")
    macro("pdlpTimeLimitNames", ", ".join(f"\\texttt{{{tex_escape(n)}}}" for n in pdlp[pdlp.status == "time_limit"]["name"]) or "none")
    macro("pdlpIterLimitNames", ", ".join(f"\\texttt{{{tex_escape(n)}}}" for n in pdlp[pdlp.status == "iteration_limit"]["name"]) or "none")
    macro("nPdlpTimeLimit", int((pdlp.status == "time_limit").sum()))
    macro("nPdlpIterLimit", int((pdlp.status == "iteration_limit").sum()))
    ecdf_dat("ecdf_pdlp.dat", pdlp[okp].rel_err)
    write_dat("scatter_pdlp.dat", ["time", "ref", "ok", "name"],
              [(max(r.time, 1e-4) if np.isfinite(r.time) else TL_PDLP, max(r.ref_time, 1e-4),
                int(o), r["name"]) for (_, r), o in zip(pdlp.iterrows(), okp)])
    # relative error against iterations (does more work buy accuracy?)
    write_dat("pdlp_err_iters.dat", ["iters", "logerr", "name"],
              [(r.iters, math.log10(max(r.rel_err, 1e-17)), r["name"]) for _, r in pdlp[okp].iterrows()
               if np.isfinite(r.rel_err)])
    # the worst objective errors among the "optimal" runs
    worst = pdlp[okp].sort_values("rel_err", ascending=False).head(6)
    macro("pdlpWorst", ", ".join(f"\\texttt{{{tex_escape(r['name'])}}} (\\num{{{r.rel_err:.1e}}})" for _, r in worst.iterrows()))

# ---- what fixed what: v1 -> v2 ----
def fix_table(v1, v2, tol, fname, tagname):
    if v1 is None or v2 is None:
        (OUT / fname).write_text("")
        macro(tagname, 0)
        return
    ok1, ok2 = lp_solved(v1, tol), lp_solved(v2, tol)
    j = v1.merge(v2, on="name", suffixes=("_a", "_b"))
    rows = []
    fixed = 0
    for _, r in j.iterrows():
        a = (r.status_a == "optimal") and (r.rel_err_a < tol)
        b = (r.status_b == "optimal") and (r.rel_err_b < tol)
        if a and b:
            continue
        fixed += int(b and not a)
        rows.append(f"{tex_escape(r['name'])} & {status_short(r.status_a)} & {fmt_err(r.rel_err_a)} & "
                    f"{fmt_time(r.time_a)} & {status_short(r.status_b)} & {fmt_err(r.rel_err_b)} & {fmt_time(r.time_b)} \\\\")
    (OUT / fname).write_text("\n".join(rows) + "\n")
    macro(tagname, fixed)
    macro(tagname + "VOne", int(ok1.sum()))
    macro(tagname + "VOneTot", len(v1))
    macro(tagname + "VOneSgm", sgm(v1[ok1].time, SHIFT_A), "{:.2f}")
    macro(tagname + "VOneRatio", sgm(v1[ok1].time, SHIFT_A) / sgm(v1[ok1].ref_time, SHIFT_A), "{:.1f}")
    # speed-up on the instances both versions solved
    both = j[[(ra == "optimal") and (rb == "optimal") for ra, rb in zip(j.status_a, j.status_b)]]
    if len(both):
        # relative to HiGHS inside each sweep, so machine-load differences between the sweeps cancel
        macro(tagname + "Speedup", (sgm(both.time_a, SHIFT_A) / sgm(both.ref_time_a, SHIFT_A))
              / (sgm(both.time_b, SHIFT_A) / sgm(both.ref_time_b, SHIFT_A)), "{:.1f}")


fix_table(spx1, spx2, LP_TOL, "fixes_simplex.tex", "nFixedSpx")
fix_table(ipm1, ipm2, LP_TOL, "fixes_ipm.tex", "nFixedIpm")

# ---- effect of presolve: the newest simplex sweep (run with presolve) against v2 (without) ----
spx_pre = in_progress["spx"][1] if "spx" in in_progress else (spx if spx_v >= 3 else None)
spx_pre_stem = in_progress["spx"][0] if "spx" in in_progress else spx_stem
macro("haveSpxPresolve", 1 if (spx_pre is not None and spx2 is not None) else 0)
if spx_pre is not None and spx2 is not None:
    a, b = lp_solved(spx2, LP_TOL), lp_solved(spx_pre, LP_TOL)
    j = spx2.assign(ok=a).merge(spx_pre.assign(ok=b), on="name", suffixes=("_a", "_b"))
    jb = j[j.ok_a & j.ok_b]
    macro("presolveFile", tex_escape(spx_pre_stem + ".csv"))
    macro("presolveDone", len(spx_pre))
    macro("presolveSolved", int(b.sum()))
    macro("presolvePartial", 1 if len(spx_pre) < N_NETLIB else 0)
    macro("presolveSpeedup", sgm(jb.time_a, SHIFT_A) / sgm(jb.time_b, SHIFT_A), "{:.2f}")
    macro("presolveIterRatio", float(np.exp(np.mean(np.log((jb.iters_a + 1) / (jb.iters_b + 1))))), "{:.2f}")
    macro("nPresolveCommon", len(jb))
    macro("nSpxVtwo", int(a.sum()))
    # against HiGHS inside each sweep (controls for machine load), on the common solved problems
    macro("presolveRatioHighs", sgm(jb.time_b, SHIFT_A) / sgm(jb.ref_time_b, SHIFT_A), "{:.1f}")
    macro("presolveRatioHighsVtwo", sgm(jb.time_a, SHIFT_A) / sgm(jb.ref_time_a, SHIFT_A), "{:.1f}")
    r_with = sgm(jb.time_b, SHIFT_A) / sgm(jb.ref_time_b, SHIFT_A)
    r_without = sgm(jb.time_a, SHIFT_A) / sgm(jb.ref_time_a, SHIFT_A)
    # 1: presolve pays (>5% better), 0: roughly even, -1: presolve costs more than it saves
    macro("presolveFaster", 1 if r_with < 0.95 * r_without else (-1 if r_with > 1.05 * r_without else 0))
    macro("presolveLoad", float(np.exp(np.mean(np.log(jb.ref_time_b.clip(lower=1e-4) / jb.ref_time_a.clip(lower=1e-4))))), "{:.1f}")
    macro("presolveGmrHighsCommon", float(np.exp(np.log(jb.time_b / jb.ref_time_b).mean())), "{:.1f}")
    macro("presolveGmrHighsCommonVtwo", float(np.exp(np.log(jb.time_a / jb.ref_time_a).mean())), "{:.1f}")
    g_with = float(np.exp(np.log(jb.time_b / jb.ref_time_b).mean()))
    g_without = float(np.exp(np.log(jb.time_a / jb.ref_time_a).mean()))
    macro("presolveSmallWorse", 1 if g_with > 1.05 * g_without else 0)
    rr = (spx_pre[b].time / spx_pre[b].ref_time)
    macro("presolveGmrHighs", float(np.exp(np.log(rr).mean())) if len(rr) else float("nan"), "{:.1f}")
    fails = spx_pre[~b]
    macro("presolveFailNames", ", ".join(f"\\texttt{{{tex_escape(r['name'])}}} ({status_short(r.status)})" for _, r in fails.iterrows()) or "none")
    macro("nPresolveFail", len(fails))
else:
    macro("presolveSpeedup", "--")
    macro("presolveIterRatio", "--")
    macro("nPresolveCommon", 0)
    macro("nSpxVtwo", int(lp_solved(spx2, LP_TOL).sum()) if spx2 is not None else 0)

# ---- full Netlib table (appendix) ----
if spx is not None and ipm is not None:
    j = spx.merge(ipm, on="name", suffixes=("_s", "_i"), how="outer")
    if pdlp is not None:
        j = j.merge(pdlp[["name", "status", "rel_err", "time"]].rename(
            columns={"status": "status_p", "rel_err": "rel_err_p", "time": "time_p"}), on="name", how="left")
    rows = []
    for _, r in j.sort_values("name").iterrows():
        def cell(st, err, t, tol):
            ok = (st == "optimal") and np.isfinite(err) and err < tol
            mark = "" if ok else "$^{\\dagger}$"
            return f"{fmt_time(t)}{mark} & {fmt_err(err)}"
        p = cell(r.get("status_p"), r.get("rel_err_p", np.nan), r.get("time_p", np.nan), PDLP_TOL_LOOSE) if pdlp is not None else "-- & --"
        rows.append(f"{tex_escape(r['name'])} & {fint(r.m_s)} & {fint(r.n_s)} & {fint(r.nnz_s)} & "
                    f"{fmt_time(r.ref_time_s)} & {cell(r.status_s, r.rel_err_s, r.time_s, LP_TOL)} & "
                    f"{cell(r.status_i, r.rel_err_i, r.time_i, LP_TOL)} & {p} \\\\")
    (OUT / "netlib_full.tex").write_text("\n".join(rows) + "\n")

# =============================================================================================
# MIP sweeps (MIPLIB 3)
# =============================================================================================

def mip_solved(d):
    return (d.status == "optimal") & (d.rel_err <= MIP_TOL)


def mip_summary(tag, d):
    ok = mip_solved(d)
    macro(f"n{tag}", int(ok.sum()))
    macro(f"n{tag}Tot", len(d))
    ref_ok = d.ref_status.astype(str).str.startswith("Optimal")
    macro(f"n{tag}RefOpt", int(ref_ok.sum()))
    macro(f"n{tag}RefTL", int((~ref_ok).sum()))
    # NIRNAY found the optimal objective (rel_err <= tol) but could not prove it
    found = (d.rel_err <= MIP_TOL) & ~ok
    macro(f"n{tag}FoundNotProved", int(found.sum()))
    # incumbents of any quality at the limit
    macro(f"n{tag}Incumbent", int(((d.status == "time_limit") & np.isfinite(d.obj)).sum()))
    macro(f"n{tag}NoIncumbent", int(((d.status == "time_limit") & ~np.isfinite(d.obj)).sum()))
    macro(f"n{tag}WrongOpt", int(((d.status == "optimal") & (d.rel_err > MIP_TOL)).sum()))
    both = d[ok]
    macro(f"sgm{tag}", sgm(both.time, SHIFT_A), "{:.2f}")
    macro(f"sgm{tag}Ref", sgm(both.ref_time, SHIFT_A), "{:.2f}")
    macro(f"ratio{tag}", sgm(both.time, SHIFT_A) / sgm(both.ref_time, SHIFT_A) if len(both) else float("nan"), "{:.1f}")
    macro(f"medNodes{tag}", float(both.nodes.median()) if len(both) else float("nan"), "{:.0f}")
    macro(f"maxNodes{tag}", float(d.nodes.max()) if len(d) else float("nan"), "{:.0f}")
    macro(f"nFaster{tag}", int((both.time < both.ref_time).sum()))
    macro(f"faster{tag}Names", ", ".join(f"\\texttt{{{tex_escape(n)}}}" for n in both[both.time < both.ref_time]["name"]) or "none")
    # gap statistics of the unsolved ones with an incumbent
    tl = d[(d.status == "time_limit") & np.isfinite(d.gap)]
    macro(f"n{tag}GapBelowOnePct", int((tl.gap < 0.01).sum()))
    macro(f"n{tag}GapBelowTenPct", int((tl.gap < 0.10).sum()))
    macro(f"n{tag}SmallInt", int((d.nint <= 100).sum()))
    return ok


mip_stem, mip, mip_v = latest("miplib3_bnb", "mip", N_MIPLIB)
mip1 = read("miplib3_bnb_v1")
# comparison of v1 (no cuts, no presolve) with a newer sweep: either the primary one (if newer
# than v1 and complete) or one still running
mip_newer = in_progress["mip"][1] if "mip" in in_progress else (mip if mip_v > 1 else None)
mip_newer_stem = in_progress["mip"][0] if "mip" in in_progress else mip_stem
mip2 = mip1 if mip_newer is not None else None
macro("haveMip", 1 if mip is not None else 0)
macro("haveMipVtwo", 1 if mip2 is not None else 0)
miplist = REPO / "data" / "miplib3_list.txt"
macro("nMipList", len(list((REPO / "data" / "miplib").glob("*.mps*"))))


def mip_table(d, fname):
    rows = []
    for _, r in d.sort_values("name").iterrows():
        ok = (r.status == "optimal") and np.isfinite(r.rel_err) and r.rel_err <= MIP_TOL
        st = status_short(r.status)
        gap = "--" if not np.isfinite(r.gap) else ("0" if r.gap == 0 else f"\\num{{{r.gap:.1e}}}")
        ref = "opt" if str(r.ref_status).startswith("Optimal") else "TL"
        mark = "" if ok else "$^{\\dagger}$"
        rows.append(f"{tex_escape(r['name'])} & {fint(r.m)} & {fint(r.n)} & {fint(r.nint)} & {st}{mark} & "
                    f"{gap} & {fmt_err(r.rel_err)} & {fint(r.nodes)} & "
                    f"{fmt_time(r.time)} & {fmt_time(r.ref_time)} & {ref} \\\\")
    (OUT / fname).write_text("\n".join(rows) + "\n")


if mip is not None:
    ok_mip = mip_summary("Mip", mip)
    mip_table(mip, "miplib_full.tex")
    nm = lambda msk: ", ".join(f"\\texttt{{{tex_escape(n)}}}" for n in mip[msk]["name"]) or "none"
    macro("mipFoundNotProvedNames", nm((mip.rel_err <= MIP_TOL) & ~ok_mip))
    macro("mipNoIncumbentNames", nm((mip.status == "time_limit") & ~np.isfinite(mip.obj)))
    ref_opt = mip.ref_status.astype(str).str.startswith("Optimal")
    macro("nMipRefOnly", int((ref_opt & ~ok_mip).sum()))
    macro("nMipBothOpt", int((ref_opt & ok_mip).sum()))
    macro("nMipOnlyUs", int((~ref_opt & ok_mip).sum()))
    macro("mipOnlyUsNames", nm(~ref_opt & ok_mip))
    other = ~mip.status.isin(["optimal", "time_limit"])
    macro("nMipOtherStatus", int(other.sum()))
    macro("mipOtherStatusNames", ", ".join(f"\\texttt{{{tex_escape(r['name'])}}} ({status_short(r.status)})" for _, r in mip[other].iterrows()) or "none")
    write_dat("scatter_mip.dat", ["time", "ref", "ok", "name"],
              [(max(r.time, 1e-3) if np.isfinite(r.time) else TL_MIP, max(r.ref_time, 1e-3), int(o), r["name"])
               for (_, r), o in zip(mip.iterrows(), ok_mip)])
    ecdf_dat("ecdf_mip.dat", mip[ok_mip].rel_err)
    # gap at the limit
    tl = mip[(mip.status == "time_limit") & np.isfinite(mip.gap)].sort_values("gap")
    write_dat("mip_gaps.dat", ["idx", "gap", "name"], [(i, max(r.gap, 1e-6), tex_escape(r["name"])) for i, (_, r) in enumerate(tl.iterrows())])
    # nodes per second
    both = mip[ok_mip & (mip.time > 0)]
    macro("mipNodesPerSec", float((both.nodes / both.time).median()) if len(both) else float("nan"), "{:.0f}")
    macro("mipLpItersPerNode", float((both.iters / both.nodes.clip(lower=1)).median()) if len(both) else float("nan"), "{:.0f}")
if mip2 is not None:
    # the older sweep (v1: no cuts, no presolve) against the newest
    mip_summary("MipVone", mip2)
    macro("mipNewerFile", tex_escape(mip_newer_stem + ".csv"))
    macro("mipNewerDone", len(mip_newer))
    macro("mipNewerPartial", 1 if len(mip_newer) < N_MIPLIB else 0)
    macro("mipNewerSolved", int(((mip_newer.status == "optimal") & (mip_newer.rel_err <= MIP_TOL)).sum()))
    j = mip2.merge(mip_newer, on="name", suffixes=("_a", "_b"))
    a = (j.status_a == "optimal") & (j.rel_err_a <= MIP_TOL)
    b = (j.status_b == "optimal") & (j.rel_err_b <= MIP_TOL)
    macro("nMipCutsGained", int((b & ~a).sum()))
    macro("nMipCutsLost", int((a & ~b).sum()))
    macro("nMipCommon", len(j))
    macro("mipGainedNames", ", ".join(f"\\texttt{{{tex_escape(n)}}}" for n in j[b & ~a]["name"]) or "none")
    macro("mipLostNames", ", ".join(f"\\texttt{{{tex_escape(n)}}}" for n in j[a & ~b]["name"]) or "none")

# =============================================================================================
# QP sweep (Maros-Mészáros)
# =============================================================================================

qp_stem, qp, qp_v = latest("maros_qpipm", "qp", N_MAROS)
macro("haveQp", 1 if qp is not None else 0)
qpdir = REPO / "data" / "qp" / "mm"
macro("nQpList", len({p.name.lower() for p in qpdir.iterdir() if p.suffix.lower() == ".qps"}) if qpdir.exists() else 0)
if qp is not None:
    ref_ok = qp.ref_status.astype(str).str.startswith(("Solved", "AlmostSolved"))
    ok_qp = (qp.status == "optimal") & (qp.rel_err < QP_TOL) & ref_ok
    macro("nQp", int(ok_qp.sum()))
    macro("nQpTot", len(qp))
    macro("nQpRefFail", int((~ref_ok).sum()))
    macro("qpRefFailNames", ", ".join(f"\\texttt{{{tex_escape(n)}}}" for n in qp[~ref_ok]["name"]) or "none")
    macro("nQpOptimalStatus", int((qp.status == "optimal").sum()))
    macro("nQpWrong", int(((qp.status == "optimal") & (qp.rel_err >= QP_TOL) & ref_ok).sum()))
    macro("nQpFail", int(((qp.status != "optimal") & ref_ok).sum()))
    both = qp[ok_qp]
    macro("sgmQp", sgm(both.time, SHIFT_A), "{:.2f}")
    macro("sgmQpRef", sgm(both.ref_time, SHIFT_A), "{:.2f}")
    macro("ratioQp", sgm(both.time, SHIFT_A) / sgm(both.ref_time, SHIFT_A) if len(both) else float("nan"), "{:.1f}")
    macro("nFasterQp", int((both.time < both.ref_time).sum()))
    macro("medItQp", float(both.iters.median()) if len(both) else float("nan"), "{:.0f}")
    macro("maxItQp", float(both.iters.max()) if len(both) else float("nan"), "{:.0f}")
    macro("medErrQp", float(both.rel_err.median()) if len(both) else float("nan"), "{:.1e}")
    macro("qpMaxN", int(qp.n.max()))
    macro("qpMaxNName", tex_escape(qp.loc[qp.n.idxmax(), "name"]))
    write_dat("scatter_qp.dat", ["time", "ref", "ok", "name"],
              [(max(r.time, 1e-3) if np.isfinite(r.time) else TL_QP, max(r.ref_time, 1e-3), int(o), r["name"])
               for (_, r), o in zip(qp.iterrows(), ok_qp)])
    ecdf_dat("ecdf_qp.dat", qp[ok_qp].rel_err)
    rows = []
    for _, r in qp.sort_values("name").iterrows():
        ok = bool(ok_qp[r.name] if False else ((r.status == "optimal") and np.isfinite(r.rel_err) and r.rel_err < QP_TOL and str(r.ref_status).startswith(("Solved", "AlmostSolved"))))
        mark = "" if ok else "$^{\\dagger}$"
        rows.append(f"{tex_escape(r['name'])} & {fint(r.m)} & {fint(r.n)} & {fint(r.nnz)} & {status_short(r.status)}{mark} & "
                    f"{short_ref(r.ref_status)} & {fmt_err(r.rel_err)} & {fint(r.iters)} & "
                    f"{fmt_time(r.time)} & {fmt_time(r.ref_time)} \\\\")
    (OUT / "maros_full.tex").write_text("\n".join(rows) + "\n")
    # failure categories
    def names(mask):
        return ", ".join(f"\\texttt{{{tex_escape(n)}}}" for n in qp[mask]["name"]) or "none"
    opt = qp.status == "optimal"
    cats = {
        "QpRefBad": ~ref_ok,
        "QpLoose": ref_ok & opt & (qp.rel_err >= QP_TOL) & (qp.rel_err < 1e-3),
        "QpWrongBig": ref_ok & opt & (qp.rel_err >= 1e-3),
        "QpNumErr": ref_ok & (qp.status == "numerical_error"),
        "QpOther": ref_ok & ~opt & (qp.status != "numerical_error"),
        "QpAlmostRef": ref_ok & qp.ref_status.astype(str).str.startswith("AlmostSolved") & ~ok_qp,
    }
    for k, msk in cats.items():
        macro("n" + k, int(msk.sum()))
        macro(k[0].lower() + k[1:] + "Names", names(msk))
    macro("nQpLooseOrWrong", int((cats["QpLoose"] | cats["QpWrongBig"]).sum()))
    q1 = read("maros_qpipm_v1")
    if q1 is not None and qp_v > 1:
        r1 = q1.ref_status.astype(str).str.startswith(("Solved", "AlmostSolved"))
        macro("nQpVone", int(((q1.status == "optimal") & (q1.rel_err < QP_TOL) & r1).sum()))
        macro("nQpVoneTot", len(q1))
        macro("haveQpVone", 1)
    else:
        macro("haveQpVone", 0)
    macro("nQpRefAnnotated", int(qp.ref_status.astype(str).str.contains("NIRNAY reading").sum()))
    fails = qp[~ok_qp]
    rows = []
    for _, r in fails.iterrows():
        rows.append(f"{tex_escape(r['name'])} & {fint(r.n)} & {status_short(r.status)} & {short_ref(r.ref_status)} & "
                    f"{fmt_err(r.rel_err)} & {fmt_time(r.time)} & {fmt_time(r.ref_time)} \\\\")
    (OUT / "qp_failures.tex").write_text("\n".join(rows) + "\n")

# =============================================================================================
# Solved-count bars
# =============================================================================================
bars = []
def bar(label, tag):
    n = macros.get("n" + tag, "0")
    t = macros.get("n" + tag + "Tot", "0")
    bars.append((len(bars), label, n, t))
bar("Netlib/simplex", "Spx")
bar("Netlib/IPM", "Ipm")
bar("Netlib/PDLP", "Pdlp")
bar("MIPLIB3/BnB", "Mip")
bar("Maros/QP-IPM", "Qp")
write_dat("solved_bars.dat", ["x", "label", "solved", "total"], [(x, l.replace(" ", "~"), n, t) for x, l, n, t in bars])
macro("barLabels", ",".join(l for _, l, _, _ in bars))
macro("barCount", len(bars))

# =============================================================================================
# Comparator solvers
# =============================================================================================

def comparators(fname, tol, tag, nirnay_sources):
    p = RES / f"{fname}.csv"
    if not p.exists():
        macro("have" + tag, 0)
        (OUT / f"comp_{tag.lower()}.tex").write_text("")
        (OUT / f"comp_{tag.lower()}_short.tex").write_text("")
        return
    macro("have" + tag, 1)
    d = pd.read_csv(p)
    d["time"] = pd.to_numeric(d.time, errors="coerce")
    d["rel_err"] = pd.to_numeric(d.rel_err, errors="coerce")
    insts = sorted(d.instance.unique())
    macro(tag + "Instances", ", ".join(f"\\texttt{{{tex_escape(i)}}}" for i in insts))
    macro("n" + tag + "Inst", len(insts))
    macro("n" + tag + "Solvers", d.solver.nunique())
    rows = []
    stats = []
    for s, g in d.groupby("solver"):
        first_order = any(k in s for k in ("pdlp", "osqp"))
        ok = (g.status == "optimal") & (g.rel_err < (PDLP_TOL_LOOSE if first_order else tol))
        platform = g.platform.iloc[0]
        t = g[ok].time
        stats.append((s, platform, int(ok.sum()), len(g), sgm(t, SHIFT_A), float(t.max()) if len(t) else float("nan")))
    # NIRNAY rows from its own sweeps, restricted to the same instances
    for label, src, tol_s in nirnay_sources:
        dd = read(src) if src else None
        if dd is None:
            continue
        sub = dd[dd["name"].isin(insts)]
        ok = (sub.status == "optimal") & (sub.rel_err < tol_s)
        t = sub[ok].time
        stats.append((label, "Windows 11 (CPU)", int(ok.sum()), len(sub), sgm(t, SHIFT_A), float(t.max()) if len(t) else float("nan")))
    stats.sort(key=lambda r: (-r[2], r[4] if np.isfinite(r[4]) else 1e9))
    short = []
    for s, platform, n, tot, g1, tmax in stats:
        nir = s.startswith("NIRNAY")
        name = ("\\textbf{" + tex_escape(s) + "}") if nir else f"\\texttt{{{tex_escape(s)}}}"
        gpu = "$^{\\ast}$" if "GPU" in platform else ""
        rows.append(f"{name} & {tex_escape(platform)} & {n}/{tot} & {fmt_time(g1)} & {fmt_time(tmax)} \\\\")
        short.append(f"{name}{gpu} & {n}/{tot} & {fmt_time(g1)} & {fmt_time(tmax)} \\\\")
    (OUT / f"comp_{tag.lower()}.tex").write_text("\n".join(rows) + "\n")
    (OUT / f"comp_{tag.lower()}_short.tex").write_text("\n".join(short) + "\n")


comparators("comparators_lp", LP_TOL, "CompLp",
            [("NIRNAY dual simplex", spx_stem, LP_TOL), ("NIRNAY IPM", ipm_stem, LP_TOL),
             ("NIRNAY PDLP (1e-4)", pdlp_stem, PDLP_TOL_LOOSE)])
comparators("comparators_mip", MIP_TOL * 1.0000001, "CompMip", [("NIRNAY branch-and-bound", mip_stem, MIP_TOL * 1.0000001)])
comparators("comparators_qp", LP_TOL, "CompQp", [("NIRNAY QP-IPM", qp_stem, QP_TOL)])

# =============================================================================================
# Module line counts
# =============================================================================================
ROLES = {
    "model.py": "problem form, CSC matrix, result object",
    "io/mps.py": "MPS/QPS reader (fixed and free format, gzip, RANGES, MARKER, QUADOBJ)",
    "io/mps_write.py": "MPS writer (round-trip checked)",
    "presolve/scaling.py": "geometric-mean and equilibration scaling in powers of two",
    "linalg/lu.py": "Gilbert--Peierls sparse LU, threshold pivoting, eta file, basis repair",
    "linalg/cholesky.py": "up-looking sparse Cholesky, elimination tree, pivot guard",
    "linalg/ldl.py": "quasi-definite $LDL^{\\mathsf T}$ with dynamic regularisation",
    "linalg/ordering.py": "minimum degree with mass elimination and dense-row deferral",
    "linalg/normal.py": "normal-equations assembly on a fixed pattern",
    "linalg/sparse_kernels.py": "Numba SpMV and per-line norms",
    "lp/simplex.py": "bounded dual simplex, phase 1, perturbation, primal cleanup",
    "lp/_simplex_kernels.py": "pivot row, DSE pricing, bound-flipping ratio test, updates",
    "lp/ipm.py": "Mehrotra predictor--corrector on normal equations",
    "lp/pdlp.py": "restarted PDHG, CPU (Numba) and GPU (CuPy, CUDA graphs)",
    "mip/bnb.py": "branch-and-bound, reliability branching, heuristics, node heap",
    "mip/cuts.py": "Gomory mixed-integer cuts with safeguards",
    "mip/propagate.py": "activity-based bound propagation",
    "qp/ipm.py": "QP interior point on the regularised augmented system",
}
rows = []
total = 0
for rel, role in ROLES.items():
    p = REPO / "nirnay" / rel
    if not p.exists():
        continue
    n = sum(1 for _ in open(p, encoding="utf-8"))
    total += n
    rows.append(f"\\texttt{{{tex_escape(rel)}}} & {n} & {role} \\\\")
(OUT / "modules.tex").write_text("\n".join(rows) + "\n")
macro("locCore", total)
cases_loc = sum(sum(1 for _ in open(p, encoding="utf-8")) for p in (REPO / "nirnay" / "cases").glob("*.py"))
macro("locCases", cases_loc)
macro("locAll", sum(sum(1 for _ in open(p, encoding="utf-8")) for p in (REPO / "nirnay").rglob("*.py")))
macro("locBench", sum(sum(1 for _ in open(p, encoding="utf-8")) for p in (REPO / "bench").glob("*.py")))
macro("locTests", sum(sum(1 for _ in open(p, encoding="utf-8")) for p in (REPO / "tests").glob("*.py")))
ntests = 0
for p in (REPO / "tests").glob("test_*.py"):
    ntests += len(re.findall(r"^def test_", p.read_text(encoding="utf-8"), flags=re.M))
macro("nTestFunctions", ntests)

# =============================================================================================
# Case studies: sizes from the MPS files, optional NIRNAY runs
# =============================================================================================
CASES = [
    ("refinery_planning", "Refinery crude selection and blending", "LP"),
    ("refinery_multiperiod", "Twelve-month refinery production plan", "MILP"),
    ("unit_commitment", "Thermal unit commitment (RTS-GMLC, 48 h)", "MILP"),
    ("economic_dispatch", "Economic dispatch with DC network (2000 buses)", "QP"),
    ("facility_location", "Capacitated facility location (capb)", "MILP"),
    ("product_distribution", "BS-VI petrol and diesel distribution", "LP"),
    ("crude_scheduling", "Crude unloading and blending schedule", "MILP"),
]
sys.path.insert(0, str(REPO))
sizes = {}
try:
    from nirnay.io.mps import read_mps  # read-only use of the solver's reader
    for key, _, _ in CASES:
        p = REPO / "data" / "cases" / f"{key}.mps"
        if p.exists():
            m = read_mps(p)
            sizes[key] = (m.m, m.n, int(m.integer.sum()), m.A.nnz, (m.Q.nnz if m.Q is not None else 0))
except Exception as e:  # noqa: BLE001
    print("case sizes skipped:", e, file=sys.stderr)
# fall back to the sizes the case-study document states, for cases without an MPS file
doc = REPO / "docs" / "CASE_STUDIES.md"
if doc.exists():
    for line in doc.read_text(encoding="utf-8").splitlines():
        m_ = re.match(r"\|\s*(.+?)\s*\|\s*`(\w+)`\s*\|\s*(LP|MILP|QP)\s*\|\s*([\d ]+)\s*\|\s*([\d ]+)(?:\s*\(([\d ]+)\))?\s*\|\s*([\d ]+)(?:\s*\(Q:\s*([\d ]+)\))?\s*\|", line)
        if m_ and m_.group(2) not in sizes:
            g = lambda k: int(m_.group(k).replace(" ", "")) if m_.group(k) else 0
            sizes[m_.group(2)] = (g(4), g(5), g(6), g(7), g(8))
# HiGHS reference results for the cases (data/cases/reference.csv) and NIRNAY's own runs, taken
# from the "NIRNAY on the cases" table of docs/CASE_STUDIES.md (the case-study record).
ref = {}
ref_p = REPO / "data" / "cases" / "reference.csv"
if ref_p.exists():
    for r in csv.DictReader(open(ref_p)):
        ref[r["case"]] = r
rows = []
for key, title, cls in CASES:
    if key not in sizes:
        continue
    m, n, ni, nnz, qnz = sizes[key]
    ints = f" ({ni})" if ni else ""
    q = f" (Q: {qnz})" if qnz else ""
    rows.append(f"{title} & \\texttt{{{tex_escape(key)}}} & {cls} & {m} & {n}{ints} & {nnz}{q} \\\\")
(OUT / "cases_sizes.tex").write_text("\n".join(rows) + "\n")
macro("nCases", len(rows))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


rows = []
for key, title, cls in CASES:
    r = ref.get(key)
    if r is None:
        continue
    gap = _f(r.get("mip_gap"))
    gaps = "--" if not np.isfinite(gap) else ("0" if gap == 0 else f"\\num{{{gap:.1e}}}")
    rows.append(f"\\texttt{{{tex_escape(key)}}} & {cls} & {tex_escape(r['highs_status'])} & "
                f"\\num{{{_f(r['objective']):.10g}}} & {gaps} & {fmt_time(_f(r['highs_time_s']))} \\\\")
(OUT / "cases_ref.tex").write_text("\n".join(rows) + "\n")
if "unit_commitment" in ref:
    macro("ucGap", 100 * _f(ref["unit_commitment"]["mip_gap"]), "{:.2f}")
    macro("ucGapTarget", 100 * _f(ref["unit_commitment"]["mip_rel_gap_target"]), "{:.0f}")
if "facility_location" in ref:
    macro("flTime", _f(ref["facility_location"]["highs_time_s"]), "{:.0f}")
runs = []
doc = REPO / "docs" / "CASE_STUDIES.md"
if doc.exists():
    txt = doc.read_text(encoding="utf-8")
    k = txt.find("## NIRNAY on the cases")
    if k >= 0:
        for line in txt[k:].splitlines():
            if not line.strip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 6 and cells[0] != "Case" and not cells[0].startswith("---"):
                runs.append(cells)
macro("haveCaseRuns", 1 if runs else 0)
macro("nCaseRuns", len(runs))
macro("nCaseRunsOptimal", sum(1 for r in runs if r[2].strip() == "optimal"))
rows = []
for case, method, status, obj, diff, t in runs:
    st = status.split(",")[0].replace("time limit", "time\\_limit")
    if len(st) > 24:
        st = "no result"
    ov, dv, tv = _f(obj), _f(diff), _f(t)
    rows.append(f"\\texttt{{{tex_escape(case.strip('`'))}}} & \\texttt{{{tex_escape(method.strip('`'))}}} & "
                f"{st} & {('--' if not np.isfinite(ov) else f'\\num{{{ov:.10g}}}')} & {fmt_err(dv)} & {fmt_time(tv)} \\\\")
(OUT / "cases_runs.tex").write_text("\n".join(rows) + "\n")

# =============================================================================================
# Large LPs: PDLP on CPU and GPU (bench/run_pdlp_gpu.py)
# =============================================================================================
large_p = RES / "large_pdlp_gpu.csv"
macro("haveLarge", 0)
(OUT / "large_pdlp.tex").write_text("")
if large_p.exists():
    L = pd.read_csv(large_p)
    for col in L.columns:
        if col.endswith(("_time", "_obj", "_rel_err", "_iters")) or col in ("speedup_gpu_vs_cpu", "tol"):
            L[col] = pd.to_numeric(L[col], errors="coerce")
    if len(L):
        macro("haveLarge", 1)
        macro("nLarge", len(L))
        dl = REPO / "data" / "large"
        macro("nLargeList", len(list(dl.glob("*.mps*"))) if dl.exists() else len(L))
        macro("largeTol", f"$10^{{{int(round(math.log10(L.tol.iloc[0])))}}}$")
        both_ok = (L.cpu_status == "optimal") & (L.gpu_status == "optimal")
        gpu_only = (L.cpu_status != "optimal") & (L.gpu_status == "optimal")
        macro("nLargeBothOk", int(both_ok.sum()))
        macro("nLargeGpuOnly", int(gpu_only.sum()))
        macro("largeGpuOnlyNames", ", ".join(f"\\texttt{{{tex_escape(n)}}}" for n in L[gpu_only]["name"]) or "none")
        macro("nLargeNeither", int(((L.cpu_status != "optimal") & (L.gpu_status != "optimal")).sum()))
        lim = None
        logp = RES / "large_pdlp_gpu.log"
        if logp.exists():
            mm_ = re.search(r"limit\s+([\d.]+)\s*s", logp.read_text(errors="replace"))
            lim = float(mm_.group(1)) if mm_ else None
        macro("largeLimit", f"{lim:.0f}" if lim else "--")
        macro("largeCpuHw", tex_escape(str(L.cpu_hw.iloc[0]).split(",")[0]))
        macro("largeGpuHw", tex_escape(str(L.gpu_hw.iloc[0]).split(",")[0]))
        sp = L.speedup_gpu_vs_cpu[both_ok].dropna()   # only where both backends finished
        macro("largeSpeedMax", float(sp.max()) if len(sp) else float("nan"), "{:.1f}")
        macro("largeSpeedMin", float(sp.min()) if len(sp) else float("nan"), "{:.1f}")
        macro("largeSpeedGm", float(np.exp(np.log(sp).mean())) if len(sp) else float("nan"), "{:.1f}")
        ok = L.gpu_status == "optimal"
        vs_h = (L.highs_time / L.gpu_time)[ok & (L.highs_status == "Optimal")]
        macro("largeVsHighsMax", float(vs_h.max()) if len(vs_h) else float("nan"), "{:.0f}")
        macro("largeVsHighsMin", float(vs_h.min()) if len(vs_h) else float("nan"), "{:.1f}")
        rows = []
        for _, r in L.iterrows():
            hs = "" if str(r.highs_status) == "Optimal" else "$^{\\ddagger}$"
            cok, gok = r.cpu_status == "optimal", r.gpu_status == "optimal"
            if cok and gok and np.isfinite(r.speedup_gpu_vs_cpu):
                spd = f"{r.speedup_gpu_vs_cpu:.1f}"
            elif gok and not cok and np.isfinite(r.cpu_time) and r.gpu_time > 0:
                spd = f"$>${r.cpu_time / r.gpu_time:.1f}"          # CPU stopped at the limit
            else:
                spd = "--"
            ct = fmt_time(r.cpu_time) + ("" if cok else "$^{\\dagger}$")
            gt = fmt_time(r.gpu_time) + ("" if gok else "$^{\\dagger}$")
            rows.append(f"\\texttt{{{tex_escape(r['name'])}}} & {fint(r.m)} & {fint(r.n)} & {fint(r.nnz)} & "
                        f"{ct} & {gt} & {spd} & "
                        f"{fmt_err(r.gpu_rel_err)} & {fmt_time(r.highs_time)}{hs} & {fmt_time(r.get('highs-ipm_time', np.nan))} \\\\")
        (OUT / "large_pdlp.tex").write_text("\n".join(rows) + "\n")
        write_dat("large_pdlp.dat", ["x", "cpu", "gpu", "highs", "name"],
                  [(i, r.cpu_time, r.gpu_time, r.highs_time, r["name"]) for i, (_, r) in enumerate(L.iterrows())])
        macro("largeNames", ",".join(tex_escape(n) for n in L["name"]))

# =============================================================================================
# Dual feasibility check (tools/dual_check.py output, captured in data/dual_check.txt)
# =============================================================================================
dc = OUT / "dual_check.txt"
macro("haveDualCheck", 0)
if dc.exists():
    lines = dc.read_text(encoding="utf-8", errors="replace").splitlines()
    ok_l = [l for l in lines if l.startswith("ok  ")]
    bad_l = [l for l in lines if l.startswith("BAD ")]
    st_l = [l for l in lines if " status " in l and not l.startswith(("ok", "BAD"))]
    if ok_l or bad_l:
        macro("haveDualCheck", 1)
        macro("nDualOk", len(ok_l))
        macro("nDualBad", len(bad_l))
        macro("nDualStatus", len(st_l))
        macro("nDualTot", len(ok_l) + len(bad_l) + len(st_l))
        z = [float(re.search(r"zrel (\S+)", l).group(1)) for l in ok_l + bad_l]
        y = [float(re.search(r"yrel (\S+)", l).group(1)) for l in ok_l + bad_l]
        macro("dualMaxZ", f"\\num{{{max(z):.1e}}}" if z else "--")
        macro("dualMaxY", f"\\num{{{max(y):.1e}}}" if y else "--")
        macro("dualBadNames", ", ".join("\\texttt{" + tex_escape(l.split()[1].replace(".mps.gz", "")) + "}" for l in bad_l) or "none")
        macro("dualStatusNames", ", ".join("\\texttt{" + tex_escape(Path(l.split()[0]).name.replace(".mps.gz", "")) + "} (" + tex_escape(l.split()[-1]) + ")" for l in st_l) or "none")

# =============================================================================================
# Presolve census: what the reductions remove on Netlib (runs nirnay.presolve, read-only use)
# =============================================================================================
cen = OUT / "presolve_census.csv"
pre_src = REPO / "nirnay" / "presolve" / "presolve.py"
if pre_src.exists() and (not cen.exists() or cen.stat().st_mtime < pre_src.stat().st_mtime or "--census" in sys.argv):
    try:
        import time as _t
        from nirnay.presolve.presolve import presolve as _presolve
        recs = []
        for f in sorted((REPO / "data" / "netlib").glob("*.mps*")):
            try:
                mdl = read_mps(f)
            except Exception:  # noqa: BLE001
                continue
            t0 = _t.perf_counter()
            P = _presolve(mdl)
            dt = _t.perf_counter() - t0
            mr = P.model.m if P.model is not None else 0
            nr = P.model.n if P.model is not None else 0
            zr = P.model.A.nnz if P.model is not None else 0
            recs.append(dict(name=f.name.split(".")[0], m=mdl.m, n=mdl.n, nnz=mdl.A.nnz, m2=mr, n2=nr, nnz2=zr,
                             status=P.status, time=dt, **{k: v for k, v in P.stats.items() if k != "c0"}))
        pd.DataFrame(recs).to_csv(cen, index=False)
    except Exception as e:  # noqa: BLE001
        print("presolve census skipped:", e, file=sys.stderr)
macro("haveCensus", 0)
(OUT / "presolve_rules.tex").write_text("")
if cen.exists():
    C = pd.read_csv(cen)
    macro("haveCensus", 1)
    macro("nCensus", len(C))
    macro("censusRowsPct", 100 * (1 - C.m2.sum() / C.m.sum()), "{:.1f}")
    macro("censusColsPct", 100 * (1 - C.n2.sum() / C.n.sum()), "{:.1f}")
    macro("censusNnzPct", 100 * (1 - C.nnz2.sum() / C.nnz.sum()), "{:.1f}")
    macro("censusMedRowsPct", float(np.median(100 * (1 - C.m2 / C.m))), "{:.1f}")
    macro("censusMaxRowsPct", float(np.max(100 * (1 - C.m2 / C.m))), "{:.0f}")
    macro("censusMaxRowsName", tex_escape(C.loc[(1 - C.m2 / C.m).idxmax(), "name"]))
    macro("censusTime", float(C.time.sum()), "{:.1f}")
    macro("censusMaxTime", float(C.time.max()), "{:.2f}")
    macro("censusMaxTimeName", tex_escape(C.loc[C.time.idxmax(), "name"]))
    macro("censusSolved", int((C.status == "solved").sum()))
    macro("censusUntouched", int(((C.m2 == C.m) & (C.n2 == C.n)).sum()))
    rules = [("singleton_row", "singleton row $\\to$ bound"), ("fixed_col", "fixed column"),
             ("empty_row", "empty row"), ("empty_col", "empty column"), ("redundant_row", "redundant row"),
             ("free_singleton", "free column singleton")]
    rows = []
    for k, lab in rules:
        if k not in C:
            continue
        rows.append(f"{lab} & {int(C[k].sum())} & {int((C[k] > 0).sum())} \\\\")
    (OUT / "presolve_rules.tex").write_text("\n".join(rows) + "\n")
    write_dat("presolve_reduction.dat", ["idx", "rowpct", "colpct", "name"],
              [(i, 100 * (1 - r.m2 / r.m), 100 * (1 - r.n2 / r.n), r["name"])
               for i, (_, r) in enumerate(C.assign(k=1 - C.m2 / C.m).sort_values("k").iterrows())])

# =============================================================================================
# Sweeps still running (newer than the version the report analyses)
# =============================================================================================
suite_n = {"spx": N_NETLIB, "ipm": N_NETLIB, "pdlp": N_NETLIB, "mip": N_MIPLIB, "qp": N_MAROS}
items = [f"\\file{{{tex_escape(stem + '.csv')}}} ({len(df)} of {suite_n[tag]})"
         for tag, (stem, df, _) in sorted(in_progress.items())]
macro("haveInProgress", 1 if items else 0)
macro("inProgressList", ", ".join(items) if items else "none")
macro("spxHasPresolve", 1 if spx_v >= 3 else 0)

# =============================================================================================
# Environment
# =============================================================================================
try:
    import numba, numpy, highspy  # noqa: E401
    macro("verNumpy", numpy.__version__)
    macro("verNumba", numba.__version__)
    macro("verHighs", highspy.Highs().version())
except Exception:  # noqa: BLE001
    pass
try:
    import clarabel
    macro("verClarabel", clarabel.__version__)
except Exception:  # noqa: BLE001
    macro("verClarabel", "0.11")
macro("verPython", "%d.%d.%d" % sys.version_info[:3])
try:
    import cupy
    macro("verCupy", cupy.__version__)
except Exception:  # noqa: BLE001
    macro("verCupy", "n/a")
import datetime
macro("genDate", datetime.date.today().isoformat())

with open(OUT / "macros.tex", "w", encoding="utf-8") as fh:
    fh.write("% generated by gen_data.py; do not edit\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros and {len(list(OUT.iterdir()))} files to {OUT}")
