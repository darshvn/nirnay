"""Uniform front end to established LP / MILP / QP solvers, used as comparators and references.

NOTHING HERE MAY BE IMPORTED BY THE SOLVER PACKAGE `nirnay/`. This file lives in bench/ and only
exists so NIRNAY's results can be checked against, and timed against, other solvers.

    solve_with(solver_name, mps_path, time_limit=60, method=None, verbose=False) -> dict
        status      one of optimal / infeasible / unbounded / inf_or_unb / time_limit /
                    iteration_limit / feasible / numerical / error / not_run / other
        objective   primal objective in the file's own sense (None if no point is available)
        bound       dual bound (MIP) or dual objective (LP) when the solver reports one
        time        wall seconds spent inside the solve call (file reading excluded)
        iterations  simplex / IPM / PDLP iterations the solver reports (None if not exposed)
        nodes       branch-and-bound nodes (MIP only)
        gap         relative MIP gap |primal - bound| / max(|primal|, 1e-10), as the solver reports
        raw_status  the solver's own status string

Solver names (a "-suffix" is shorthand for `method`):
    highs[-simplex|-ipm|-pdlp|-hipdlp]  highspy (Windows)
    scip                            PySCIPOpt / SCIP + SoPlex (Windows)
    ortools-glop | ortools-pdlp | ortools-clp | ortools-scip | ortools-cbc | ortools-cpsat
    glpk[-interior|-cuts]           swiglpk (GLPK 5.0)
    clarabel, osqp                  convex solvers; LP/QP read through highspy (relaxation for MIPs)
    cbc                             python-mip + CBC (runs in WSL: Windows Smart App Control
                                    blocks the unsigned cbcbox DLLs on this machine)
    cuopt[-pdlp|-dualsimplex|-barrier|-concurrent]
                                    NVIDIA cuOpt, runs in WSL on the GPU through cuopt_cli

Anything marked WSL is dispatched through `wsl.exe` to tools/wsl_single.sh, which runs this same
file inside the WSL venv /opt/cuopt/venv and hands back JSON.

CLI:
    python bench/comparators.py data/netlib/afiro.mps.gz --solvers highs,scip,ortools-pdlp,cbc
    python bench/comparators.py data/netlib/*.mps.gz --solvers highs-ipm,ortools-glop \
        --time-limit 60 --csv out.csv --log-dir logs/
Each (instance, solver) pair runs in its own subprocess with a hard kill at 1.5 x limit + 30 s,
so a solver that ignores its time limit cannot stall a batch.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IS_WINDOWS = platform.system() == "Windows"
WSL_DISTRO = "Ubuntu-24.04"
WSL_SCRIPT = "/mnt/c/Users/darsh/nirnay/tools/wsl_single.sh"
MARK = "@@COMPARATOR_RESULT@@"

ALIASES = {
    "highs-simplex": ("highs", "simplex"), "highs-ipm": ("highs", "ipm"),
    "highs-pdlp": ("highs", "pdlp"), "highs-hipdlp": ("highs", "hipdlp"),
    "highs-ipx": ("highs", "ipx"),
    "glpk-interior": ("glpk", "interior"), "glpk-cuts": ("glpk", "cuts"),
    "cuopt-pdlp": ("cuopt", "pdlp"), "cuopt-dualsimplex": ("cuopt", "dualsimplex"),
    "cuopt-barrier": ("cuopt", "barrier"), "cuopt-concurrent": ("cuopt", "concurrent"),
    "cuopt-pdlp-tight": ("cuopt", "pdlp-tight"), "ortools-pdlp-tight": ("ortools-pdlp", "tight"),
    "osqp-tight": ("osqp", "tight"),
    "scip-lp-dual": ("scip", "d"), "scip-lp-primal": ("scip", "p"),
}
WSL_SOLVERS = {"cuopt", "cbc"} if IS_WINDOWS else set()   # inside WSL: run natively


# ----------------------------------------------------------------------------------------- utils
def _result(**kw):
    base = dict(status="error", objective=None, bound=None, time=None, iterations=None,
                nodes=None, gap=None, raw_status="", solver="", method=None, instance="")
    base.update(kw)
    return base


_TMP = Path(tempfile.gettempdir()) / "nirnay_comparators"


def plain_mps(path) -> str:
    """Path to an uncompressed copy of `path` (HiGHS on Windows, GLPK and CBC cannot read .gz)."""
    path = Path(path)
    gz = path.suffix.lower() == ".gz"
    stem = path.name[:-3] if gz else path.name
    if not gz and path.suffix == ".mps":
        return str(path)
    # readers pick the format from the extension, so .QPS / .MPS / .qps become <stem>.mps
    _TMP.mkdir(parents=True, exist_ok=True)
    out = _TMP / (Path(stem).stem + ".mps")
    if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
        with (gzip.open(path, "rb") if gz else open(path, "rb")) as a, open(out, "wb") as b:
            shutil.copyfileobj(a, b)
    return str(out)


def _rel_gap(p, d):
    if p is None or d is None or not math.isfinite(p) or not math.isfinite(d):
        return None
    return abs(p - d) / max(abs(p), 1e-10)


# ----------------------------------------------------------------------------------------- HiGHS
def _highs(path, tl, method, verbose):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", bool(verbose))
    # HiGHS 1.15.1 + cuPDLP-C (solver=pdlp) on Windows stops after ~0.07 s with "Time limit
    # reached" whenever time_limit is finite (25fv47: 965 iterations, then objective 0). With no
    # limit it solves (63240 iterations, 3.7 s). So for pdlp we leave the limit unset and rely
    # on the batch driver's hard kill.
    if method != "pdlp":
        h.setOptionValue("time_limit", float(tl))
    h.readModel(plain_mps(path))
    is_mip = any(t != highspy.HighsVarType.kContinuous for t in h.getLp().integrality_)
    if method and not is_mip:
        h.setOptionValue("solver", method)
    t0 = time.perf_counter()
    h.run()
    t = time.perf_counter() - t0
    ms = h.getModelStatus()
    info = h.getInfo()
    raw = h.modelStatusToString(ms)
    status = {"Optimal": "optimal", "Infeasible": "infeasible", "Unbounded": "unbounded",
              "Primal infeasible or unbounded": "inf_or_unb", "Time limit reached": "time_limit",
              "Iteration limit reached": "iteration_limit"}.get(raw, "other")
    has_sol = info.primal_solution_status >= 1 or (is_mip and info.mip_node_count >= 0 and
                                                    math.isfinite(info.objective_function_value))
    obj = info.objective_function_value if has_sol else None
    iters = {"simplex": info.simplex_iteration_count, "ipm": info.ipm_iteration_count,
             "pdlp": info.pdlp_iteration_count, "crossover": info.crossover_iteration_count,
             "qp": getattr(info, "qp_iteration_count", 0)}
    it = sum(v for v in iters.values() if v and v > 0) or 0
    r = _result(status=status, objective=obj, time=t, iterations=it, raw_status=raw)
    if is_mip:
        r.update(nodes=int(info.mip_node_count), bound=info.mip_dual_bound, gap=info.mip_gap)
        if status == "time_limit" and obj is not None:
            r["status"] = "time_limit"
    r["detail"] = {k: v for k, v in iters.items() if v and v > 0}
    return r


# ------------------------------------------------------------------------------------------ SCIP
def _scip(path, tl, method, verbose):
    import pyscipopt
    m = pyscipopt.Model()
    if not verbose:
        m.hideOutput()
    m.readProblem(plain_mps(path))
    m.setParam("limits/time", float(tl))
    if method in ("p", "d", "s", "b", "c"):
        m.setParam("lp/initalgorithm", method)
        m.setParam("lp/resolvealgorithm", method)
    t0 = time.perf_counter()
    m.optimize()
    t = time.perf_counter() - t0
    raw = m.getStatus()
    status = {"optimal": "optimal", "infeasible": "infeasible", "unbounded": "unbounded",
              "inforunbd": "inf_or_unb", "timelimit": "time_limit"}.get(raw, "other")
    obj = m.getObjVal() if m.getNSols() > 0 else None
    r = _result(status=status, objective=obj, time=t, iterations=m.getNLPIterations(),
                nodes=m.getNNodes(), bound=m.getDualbound(), raw_status=raw)
    r["gap"] = m.getGap() if obj is not None else None
    if verbose:
        m.printStatistics()
        sys.stdout.flush()
    return r


# -------------------------------------------------------------------------------------- OR-Tools
def _ortools_proto(path):
    from ortools.linear_solver.python import model_builder as mb
    m = mb.Model()
    if not m.import_from_mps_file(plain_mps(path)):
        raise RuntimeError("OR-Tools could not import the MPS file")
    return m.export_to_proto()


def _ortools_wraplp(backend, path, tl, method, verbose):
    from ortools.linear_solver import pywraplp
    proto = _ortools_proto(path)
    s = pywraplp.Solver.CreateSolver(backend)
    if s is None:
        return _result(status="not_run", raw_status=f"{backend} backend not available")
    err = s.LoadModelFromProto(proto)
    if err:
        return _result(status="error", raw_status=err)
    s.SetTimeLimit(int(tl * 1000))
    if verbose:
        s.EnableOutput()
        if backend == "CP_SAT":
            s.SetSolverSpecificParametersAsString("log_search_progress:true")
    is_mip = any(v.is_integer for v in proto.variable)
    t0 = time.perf_counter()
    st = s.Solve()
    t = time.perf_counter() - t0
    names = {s.OPTIMAL: "OPTIMAL", s.FEASIBLE: "FEASIBLE", s.INFEASIBLE: "INFEASIBLE",
             s.UNBOUNDED: "UNBOUNDED", s.ABNORMAL: "ABNORMAL", s.NOT_SOLVED: "NOT_SOLVED",
             s.MODEL_INVALID: "MODEL_INVALID"}
    raw = names.get(st, str(st))
    status = {"OPTIMAL": "optimal", "INFEASIBLE": "infeasible", "UNBOUNDED": "unbounded",
              "FEASIBLE": "time_limit" if t >= 0.95 * tl else "feasible",
              "NOT_SOLVED": "time_limit" if t >= 0.95 * tl else "other",
              "ABNORMAL": "numerical", "MODEL_INVALID": "error"}.get(raw, "other")
    obj = s.Objective().Value() if st in (s.OPTIMAL, s.FEASIBLE) else None
    r = _result(status=status, objective=obj, time=t, iterations=s.iterations(), raw_status=raw)
    if is_mip:
        bd = s.Objective().BestBound()
        r.update(nodes=s.nodes(), bound=bd, gap=_rel_gap(obj, bd))
    return r


def _ortools_pdlp(path, tl, method, verbose):
    from ortools.pdlp import solve_log_pb2, solvers_pb2
    from ortools.pdlp.python import pdlp
    proto = _ortools_proto(path)
    qp = pdlp.qp_from_mpmodel_proto(proto, relax_integer_variables=True)
    params = solvers_pb2.PrimalDualHybridGradientParams()
    params.termination_criteria.time_sec_limit = float(tl)
    params.verbosity_level = 2 if verbose else 0
    if method == "tight":          # match simplex-level accuracy
        params.termination_criteria.simple_optimality_criteria.eps_optimal_relative = 1e-8
        params.termination_criteria.simple_optimality_criteria.eps_optimal_absolute = 1e-8
    elif method == "presolve":
        params.presolve_options.use_glop = True
    t0 = time.perf_counter()
    res = pdlp.primal_dual_hybrid_gradient(qp, params)
    t = time.perf_counter() - t0
    log = res.solve_log
    raw = solve_log_pb2.TerminationReason.Name(log.termination_reason)
    status = {"TERMINATION_REASON_OPTIMAL": "optimal",
              "TERMINATION_REASON_PRIMAL_INFEASIBLE": "infeasible",
              "TERMINATION_REASON_DUAL_INFEASIBLE": "unbounded",
              "TERMINATION_REASON_PRIMAL_OR_DUAL_INFEASIBLE": "inf_or_unb",
              "TERMINATION_REASON_TIME_LIMIT": "time_limit",
              "TERMINATION_REASON_ITERATION_LIMIT": "iteration_limit",
              "TERMINATION_REASON_NUMERICAL_ERROR": "numerical"}.get(raw, "other")
    obj = bound = None
    ci = log.solution_stats.convergence_information
    for c in ci:
        if c.candidate_type == log.solution_type:
            obj, bound = c.primal_objective, c.dual_objective
    if obj is None and ci:
        obj, bound = ci[0].primal_objective, ci[0].dual_objective
    r = _result(status=status, objective=obj, bound=bound, time=t,
                iterations=int(log.iteration_count), raw_status=raw)
    if ci:
        r["detail"] = {"rel_gap": _rel_gap(obj, bound),
                       "l_inf_primal_residual": ci[0].l_inf_primal_residual,
                       "l_inf_dual_residual": ci[0].l_inf_dual_residual}
    return r


# ------------------------------------------------------------------------------------------ GLPK
def _glpk(path, tl, method, verbose):
    import swiglpk as g
    g.glp_term_out(g.GLP_ON if verbose else g.GLP_OFF)
    lp = g.glp_create_prob()
    p = plain_mps(path)
    if g.glp_read_mps(lp, g.GLP_MPS_DECK, None, p) != 0:
        if g.glp_read_mps(lp, g.GLP_MPS_FILE, None, p) != 0:
            return _result(status="error", raw_status="GLPK could not read the MPS file")
    is_mip = g.glp_get_num_int(lp) > 0
    t0 = time.perf_counter()
    deadline_ms = int(tl * 1000)
    if method == "interior" and not is_mip:
        ip = g.glp_iptcp()
        g.glp_init_iptcp(ip)
        ret = g.glp_interior(lp, ip)
        t = time.perf_counter() - t0
        st = g.glp_ipt_status(lp)
        raw = f"ret={ret} ipt_status={st}"
        status = {g.GLP_OPT: "optimal", g.GLP_INFEAS: "infeasible", g.GLP_NOFEAS: "infeasible",
                  g.GLP_UNDEF: "numerical"}.get(st, "other")
        return _result(status=status, objective=g.glp_ipt_obj_val(lp) if st == g.GLP_OPT else None,
                       time=t, iterations=None, raw_status=raw)
    smcp = g.glp_smcp()
    g.glp_init_smcp(smcp)
    smcp.presolve = g.GLP_ON
    smcp.tm_lim = deadline_ms
    ret = g.glp_simplex(lp, smcp)
    st = g.glp_get_status(lp)
    iters = g.glp_get_it_cnt(lp)
    if not is_mip:
        t = time.perf_counter() - t0
        raw = f"ret={ret} status={st}"
        status = {g.GLP_OPT: "optimal", g.GLP_NOFEAS: "infeasible", g.GLP_UNBND: "unbounded",
                  g.GLP_INFEAS: "other", g.GLP_FEAS: "other"}.get(st, "other")
        if ret == g.GLP_ETMLIM:
            status = "time_limit"
        if ret == g.GLP_ENOPFS:
            status = "infeasible"
        if ret == g.GLP_ENODFS:
            status = "inf_or_unb"
        obj = g.glp_get_obj_val(lp) if st == g.GLP_OPT else None
        return _result(status=status, objective=obj, time=t, iterations=iters, raw_status=raw)
    iocp = g.glp_iocp()
    g.glp_init_iocp(iocp)
    remaining = max(1, deadline_ms - int((time.perf_counter() - t0) * 1000))
    iocp.tm_lim = remaining
    if st != g.GLP_OPT:
        iocp.presolve = g.GLP_ON
    if method == "cuts":
        iocp.gmi_cuts = iocp.mir_cuts = iocp.cov_cuts = iocp.clq_cuts = g.GLP_ON
        iocp.fp_heur = iocp.ps_heur = g.GLP_ON
    ret = g.glp_intopt(lp, iocp)
    t = time.perf_counter() - t0
    st = g.glp_mip_status(lp)
    raw = f"ret={ret} mip_status={st}"
    status = {g.GLP_OPT: "optimal", g.GLP_NOFEAS: "infeasible", g.GLP_FEAS: "feasible",
              g.GLP_UNDEF: "other"}.get(st, "other")
    if ret == g.GLP_ETMLIM:
        status = "time_limit"
    obj = g.glp_mip_obj_val(lp) if st in (g.GLP_OPT, g.GLP_FEAS) else None
    return _result(status=status, objective=obj, time=t, iterations=g.glp_get_it_cnt(lp),
                   raw_status=raw)


# --------------------------------------------------------------------------- conic / ADMM (LP, QP)
def _arrays(path):
    """Read an LP/QP through highspy and return plain arrays (objective in minimisation form)."""
    import highspy
    import numpy as np
    import scipy.sparse as sp
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(plain_mps(path))
    model = h.getModel()
    lp = model.lp_
    n, m = lp.num_col_, lp.num_row_
    a = lp.a_matrix_
    if a.format_ == highspy.MatrixFormat.kColwise:
        A = sp.csc_matrix((a.value_, a.index_, a.start_), shape=(m, n))
    else:
        A = sp.csr_matrix((a.value_, a.index_, a.start_), shape=(m, n)).tocsc()
    sign = -1.0 if lp.sense_ == highspy.ObjSense.kMaximize else 1.0
    c = sign * np.asarray(lp.col_cost_, float)
    P = sp.csc_matrix((n, n))
    hs = model.hessian_
    if hs.dim_ > 0:
        # HiGHS stores the lower triangle column-wise (or square)
        Ph = sp.csc_matrix((hs.value_, hs.index_, hs.start_), shape=(n, n))
        if hs.format_ == highspy.HessianFormat.kTriangular:
            Ph = Ph + sp.tril(Ph, -1).T
        P = sign * Ph.tocsc()
    return dict(n=n, m=m, A=A, c=c, P=P, off=sign * lp.offset_, sign=sign,
                lb=np.asarray(lp.col_lower_, float), ub=np.asarray(lp.col_upper_, float),
                rl=np.asarray(lp.row_lower_, float), ru=np.asarray(lp.row_upper_, float),
                is_mip=len(lp.integrality_) > 0 and any(
                    t != highspy.HighsVarType.kContinuous for t in lp.integrality_))


def _clarabel(path, tl, method, verbose):
    import clarabel
    import numpy as np
    import scipy.sparse as sp
    d = _arrays(path)
    n = d["n"]
    I = sp.identity(n, format="csc")
    big = 1e20
    blocks_eq, rhs_eq, blocks_in, rhs_in = [], [], [], []
    for M, lo, hi in ((d["A"], d["rl"], d["ru"]), (I, d["lb"], d["ub"])):
        lo = np.where(lo <= -big, -np.inf, lo)
        hi = np.where(hi >= big, np.inf, hi)
        eq = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
        up = np.isfinite(hi) & ~eq
        dn = np.isfinite(lo) & ~eq
        M = sp.csr_matrix(M)
        if eq.any():
            blocks_eq.append(M[eq]); rhs_eq.append(hi[eq])
        if up.any():
            blocks_in.append(M[up]); rhs_in.append(hi[up])
        if dn.any():
            blocks_in.append(-M[dn]); rhs_in.append(-lo[dn])
    Aeq = sp.vstack(blocks_eq) if blocks_eq else sp.csr_matrix((0, n))
    Ain = sp.vstack(blocks_in) if blocks_in else sp.csr_matrix((0, n))
    A = sp.vstack([Aeq, Ain]).tocsc()
    b = np.concatenate(rhs_eq + rhs_in) if (rhs_eq or rhs_in) else np.zeros(0)
    cones = []
    if Aeq.shape[0]:
        cones.append(clarabel.ZeroConeT(Aeq.shape[0]))
    if Ain.shape[0]:
        cones.append(clarabel.NonnegativeConeT(Ain.shape[0]))
    s = clarabel.DefaultSettings()
    s.verbose = bool(verbose)
    s.time_limit = float(tl)
    P = sp.triu(d["P"]).tocsc()
    t0 = time.perf_counter()
    sol = clarabel.DefaultSolver(P, d["c"], A, b, cones, s).solve()
    t = time.perf_counter() - t0
    raw = str(sol.status)
    status = {"Solved": "optimal", "PrimalInfeasible": "infeasible", "DualInfeasible": "unbounded",
              "MaxTime": "time_limit", "MaxIterations": "iteration_limit",
              "AlmostSolved": "optimal_inaccurate", "NumericalError": "numerical",
              "InsufficientProgress": "numerical"}.get(raw.split(".")[-1], "other")
    obj = d["sign"] * (sol.obj_val + d["off"]) if status in ("optimal", "optimal_inaccurate") else None
    r = _result(status=status, objective=obj, time=t, iterations=sol.iterations, raw_status=raw)
    if d["is_mip"]:
        r["raw_status"] += " (LP relaxation: integrality ignored)"
    return r


def _osqp(path, tl, method, verbose):
    import numpy as np
    import osqp
    import scipy.sparse as sp
    d = _arrays(path)
    n = d["n"]
    A = sp.vstack([d["A"], sp.identity(n)]).tocsc()
    lo = np.concatenate([d["rl"], d["lb"]])
    hi = np.concatenate([d["ru"], d["ub"]])
    lo = np.where(lo <= -1e20, -np.inf, lo)
    hi = np.where(hi >= 1e20, np.inf, hi)
    P = sp.triu(d["P"]).tocsc()
    prob = osqp.OSQP()
    kw = dict(verbose=bool(verbose), time_limit=float(tl))
    if method == "tight":
        kw.update(eps_abs=1e-6, eps_rel=1e-6, max_iter=1_000_000, polishing=True)
    prob.setup(P, d["c"], A, lo, hi, **kw)
    t0 = time.perf_counter()
    res = prob.solve(raise_error=False)
    t = time.perf_counter() - t0
    raw = str(res.info.status)
    low = raw.lower()
    status = ("optimal_inaccurate" if "inaccurate" in low else "optimal" if "solved" == low else
              "infeasible" if "primal infeasible" in low else "unbounded" if "dual infeasible" in low
              else "time_limit" if "time" in low else "iteration_limit" if "iteration" in low
              else "other")
    obj = d["sign"] * (res.info.obj_val + d["off"]) if status.startswith("optimal") else None
    r = _result(status=status, objective=obj, time=t, iterations=res.info.iter, raw_status=raw)
    if d["is_mip"]:
        r["raw_status"] += " (LP relaxation: integrality ignored)"
    return r


# ------------------------------------------------------------------------------ CBC (python-mip)
def _cbc(path, tl, method, verbose):
    import mip
    m = mip.Model(solver_name=mip.CBC)
    m.verbose = 1 if verbose else 0
    m.read(plain_mps(path))
    t0 = time.perf_counter()
    st = m.optimize(max_seconds=float(tl))
    t = time.perf_counter() - t0
    raw = st.name
    status = {"OPTIMAL": "optimal", "INFEASIBLE": "infeasible", "UNBOUNDED": "unbounded",
              "INT_INFEASIBLE": "infeasible", "FEASIBLE": "time_limit", "NO_SOLUTION_FOUND": "time_limit",
              "ERROR": "error"}.get(raw, "other")
    obj = m.objective_value if st in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE) else None
    is_mip = m.num_int > 0
    r = _result(status=status, objective=obj, time=t, raw_status=raw)
    if is_mip:
        bd = m.objective_bound
        r.update(bound=bd, gap=_rel_gap(obj, bd))
    return r


# ---------------------------------------------------------------------------------------- cuOpt
_CUOPT_CLI = "/opt/cuopt/venv/bin/cuopt_cli"


def _cuopt(path, tl, method, verbose):
    """Runs cuopt_cli (libcuopt wheel, cuOpt 26.8) on an MPS file and parses its log.

    method (LP only): concurrent (default, = cuOpt's own default: PDLP + dual simplex + barrier
    raced), pdlp, dualsimplex, barrier. MIP runs ignore it. Tolerances are cuOpt's defaults
    (1e-4 relative for PDLP) unless method ends in "-tight" -> 1e-8.
    """
    p = plain_mps(path)
    cli = os.environ.get("CUOPT_CLI", _CUOPT_CLI)
    base = (method or "concurrent").replace("-tight", "")
    code = {"concurrent": 0, "pdlp": 1, "dualsimplex": 2, "barrier": 3}.get(base, 0)
    cmd = [cli, p, "--time-limit", str(float(tl)), "--method", str(code)]
    if method and method.endswith("-tight"):
        for k in ("absolute-dual", "relative-dual", "absolute-primal", "relative-primal",
                  "absolute-gap", "relative-gap"):
            cmd += [f"--{k}-tolerance", "1e-8"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=tl * 1.5 + 60, cwd=tempfile.gettempdir())
        out = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        out += "\n[killed at hard timeout]"
    t = time.perf_counter() - t0
    if verbose:
        print(out)
    return _parse_cuopt_log(out, t)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_cuopt_log(out, t_wall):
    F = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|-inf|nan)"
    status, raw, obj, bound, iters, nodes, solve_t = "other", "", None, None, None, None, None
    is_mip = "B&B" in out or "Explored " in out or "MIP" in out and "Best objective" in out
    m = re.findall(r"Status:\s*([A-Za-z_ ]+?)\s+Objective:\s*" + F + r"\s+Iterations:\s*(\d+)\s+Time:\s*" + F + "s", out)
    if m:
        raw, o, it, tt = m[-1]
        obj, iters, solve_t = _num(o), int(it), _num(tt)
    else:
        mm = re.findall(r"Optimal solution found in (\d+) iterations and " + F + "s", out)
        if mm:
            raw = "Optimal"
            iters, solve_t = int(mm[-1][0]), _num(mm[-1][1])
            oo = re.findall(r"^Objective\s+" + F, out, re.M)
            obj = _num(oo[-1]) if oo else None
    if is_mip:
        bo = re.findall(r"Best objective\s+" + F + r", best bound\s+" + F + r", gap\s+" + F, out)
        if bo:
            obj, bound = _num(bo[-1][0]), _num(bo[-1][1])
        ex = re.findall(r"Explored (\d+) nodes \((\d+) simplex iterations\) in " + F + "s", out)
        if ex:
            nodes, iters, solve_t = int(ex[-1][0]), int(ex[-1][1]), _num(ex[-1][2])
        for line in out.splitlines()[::-1]:
            l = line.lower()
            if "optimal solution found" in l or "time limit" in l or "infeasible" in l or "unbounded" in l:
                raw = line.strip()
                break
    low = raw.lower()
    if "optimal" in low:
        status = "optimal"
    elif "time" in low and "limit" in low or "timelimit" in low:
        status = "time_limit"
    elif "infeasible" in low:
        status = "infeasible"
    elif "unbounded" in low:
        status = "unbounded"
    elif "feasible" in low:
        status = "feasible"
    if status == "other":
        if re.search(r"time limit", out, re.I):
            status = "time_limit"
        if "[killed at hard timeout]" in out:
            status, raw = "time_limit", "killed at hard timeout"
        if re.search(r"(?i)error|exception|out of memory", out) and obj is None and status == "other":
            status = "error"
            raw = (raw or out.strip().splitlines()[-1][:200]) if out.strip() else "no output"
    r = _result(status=status, objective=obj, bound=bound, time=solve_t if solve_t is not None else t_wall,
                iterations=iters, nodes=nodes, raw_status=raw[:200])
    if is_mip and obj is not None and bound is not None:
        r["gap"] = _rel_gap(obj, bound)
    r["detail"] = {"wall_incl_startup": t_wall}
    return r


# -------------------------------------------------------------------------------------- dispatch
BACKENDS = {
    "highs": _highs, "scip": _scip, "glpk": _glpk, "clarabel": _clarabel, "osqp": _osqp,
    "cbc": _cbc, "cuopt": _cuopt, "ortools-pdlp": _ortools_pdlp,
    "ortools-glop": lambda *a: _ortools_wraplp("GLOP", *a),
    "ortools-clp": lambda *a: _ortools_wraplp("CLP", *a),
    "ortools-scip": lambda *a: _ortools_wraplp("SCIP", *a),
    "ortools-cbc": lambda *a: _ortools_wraplp("CBC", *a),
    "ortools-cpsat": lambda *a: _ortools_wraplp("CP_SAT", *a),
}


def resolve(name, method=None):
    base, meth = ALIASES.get(name, (name, None))
    return base, method or meth


def _to_wsl_path(p: str) -> str:
    p = str(Path(p).resolve())
    if re.match(r"^[A-Za-z]:\\", p):
        return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")
    return p


def _via_wsl(name, path, tl, method, verbose):
    cmd = ["wsl.exe", "-d", WSL_DISTRO, "-u", "root", "-e", "bash", WSL_SCRIPT,
           name, _to_wsl_path(path), str(tl), method or "", "1" if verbose else "0"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=tl * 1.5 + 90)
    except subprocess.TimeoutExpired:
        return _result(status="time_limit", raw_status="killed (hard timeout)", time=tl)
    out = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")
    res = None
    log_lines = []
    for line in out.splitlines():
        if line.startswith(MARK):
            res = json.loads(line[len(MARK):])
        else:
            log_lines.append(line)
    if verbose:
        print("\n".join(log_lines))
    if res and name == "cbc":   # python-mip does not expose these; CBC prints them when verbose
        txt = "\n".join(log_lines)
        for key, pat in (("nodes", r"Enumerated nodes:\s*(\d+)"),
                         ("iterations", r"Total iterations:\s*(\d+)")):
            mm = re.findall(pat, txt)
            if mm and res.get(key) is None:
                res[key] = int(mm[-1])
    return res or _result(status="error", raw_status=out[-400:])


def solve_with(solver_name, mps_path, time_limit=60.0, method=None, verbose=False) -> dict:
    base, meth = resolve(solver_name, method)
    if base not in BACKENDS:
        raise ValueError(f"unknown solver {solver_name!r}; known: {sorted(BACKENDS) + sorted(ALIASES)}")
    if base in WSL_SOLVERS:
        r = _via_wsl(base, mps_path, time_limit, meth, verbose)
    else:
        try:
            r = BACKENDS[base](str(mps_path), float(time_limit), meth, verbose)
        except Exception as e:  # a comparator failing must not stop a benchmark sweep
            r = _result(status="error", raw_status=f"{type(e).__name__}: {e}"[:300])
    r.update(solver=solver_name, method=meth, instance=Path(mps_path).name.split(".")[0])
    return r


# ------------------------------------------------------------------------------------------- CLI
FIELDS = ["instance", "solver", "method", "status", "objective", "bound", "gap", "time",
          "iterations", "nodes", "raw_status"]


def _single_subprocess(solver, path, tl, method, log_path):
    """One solve in a child process, so a hung solver can be killed."""
    cmd = [sys.executable, __file__, "--single", solver, str(path), str(tl), method or "", "1"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=tl * 1.5 + 30 +
                              (60 if resolve(solver)[0] in WSL_SOLVERS else 0))
        out = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode("utf-8", "replace") + (e.stderr or b"").decode("utf-8", "replace")
               + "\n[killed at hard timeout]\n")
        r = _result(status="time_limit", raw_status="killed at hard timeout",
                    time=time.perf_counter() - t0)
        r.update(solver=solver, method=resolve(solver)[1],
                 instance=Path(path).name.split(".")[0])
        if log_path:
            Path(log_path).write_text(out, encoding="utf-8")
        return r
    res = None
    for line in out.splitlines():
        if line.startswith(MARK):
            res = json.loads(line[len(MARK):])
    if log_path:
        Path(log_path).write_text("\n".join(l for l in out.splitlines() if not l.startswith(MARK)),
                                  encoding="utf-8")
    if res is None:
        res = _result(status="error", raw_status=out[-300:], solver=solver,
                      instance=Path(path).name.split(".")[0])
    return res


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if argv and argv[0] == "--single":
        solver, path, tl, method, verbose = argv[1:6]
        r = solve_with(solver, path, float(tl), method or None, verbose == "1")
        sys.stdout.flush()
        print("\n" + MARK + json.dumps(r, default=str), flush=True)
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--solvers", default="highs,scip,ortools-pdlp,cbc")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--method", default=None, help="applied to every solver that takes one")
    ap.add_argument("--csv", default=None, help="append rows to this CSV")
    ap.add_argument("--log-dir", default=None, help="save each solver's full log here")
    ap.add_argument("--in-process", action="store_true", help="no subprocess isolation")
    a = ap.parse_args(argv)
    solvers = [s.strip() for s in a.solvers.split(",") if s.strip()]
    rows = []
    writer = None
    if a.csv:
        new = not Path(a.csv).exists()
        fh = open(a.csv, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
    print(f"{'instance':<12}{'solver':<20}{'status':<18}{'objective':>20}{'time':>10}{'iters':>9}{'nodes':>9}")
    for f in a.files:
        for s in solvers:
            log_path = None
            if a.log_dir:
                Path(a.log_dir).mkdir(parents=True, exist_ok=True)
                log_path = Path(a.log_dir) / f"{Path(f).name.split('.')[0]}__{s}.log"
            if a.in_process:
                r = solve_with(s, f, a.time_limit, a.method)
            else:
                r = _single_subprocess(s, f, a.time_limit, a.method, log_path)
            rows.append(r)
            obj = "" if r["objective"] is None else f"{r['objective']:.10g}"
            tm = "" if r["time"] is None else f"{r['time']:.3f}"
            print(f"{r['instance']:<12}{s:<20}{r['status']:<18}{obj:>20}{tm:>10}"
                  f"{_fmt(r['iterations']):>9}{_fmt(r['nodes']):>9}", flush=True)
            if writer:
                writer.writerow({k: _fmt(r.get(k)) for k in FIELDS})
                fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
