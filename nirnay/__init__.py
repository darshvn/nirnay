"""NIRNAY: a sovereign LP / MILP / QP optimisation solver, built from mathematical foundations."""
from __future__ import annotations

__version__ = "0.1.0"

from .model import CSC, Model, Result  # noqa: F401


def solve(model, method: str = "auto", presolve: bool = True, **options):
    """Solve a Model. method: auto | simplex | ipm | pdlp | pdlp-gpu | bnb | qp-ipm.

    With presolve the model is reduced first and the solution mapped back (nirnay.presolve)."""
    if method == "auto":
        method = "bnb" if model.is_mip else ("qp-ipm" if model.is_qp else "simplex")
    if presolve and not model.is_qp and method not in ("pdlp", "pdlp-gpu"):
        return _solve_presolved(model, method, **options)
    return _dispatch(model, method, **options)


def _dispatch(model, method, **options):
    if method == "ipm":
        from .lp import ipm
        return ipm.solve(model, **options)
    if method == "qp-ipm":
        from .qp import ipm as qipm
        return qipm.solve(model, **options)
    if method in ("pdlp", "pdlp-gpu"):
        from .lp import pdlp
        return pdlp.solve(model, gpu=method.endswith("gpu"), **options)
    if method == "simplex":
        from .lp import simplex
        return simplex.solve(model, **options)
    if method == "bnb":
        from .mip import bnb
        return bnb.solve(model, **options)
    raise ValueError(f"unknown method {method!r}")


def _solve_presolved(model, method, **options):
    import time
    import numpy as np
    from .presolve.presolve import postsolve, presolve
    t0 = time.perf_counter()
    P = presolve(model)
    if P.status in ("infeasible", "unbounded"):
        return Result(status=P.status, method=f"presolve", time=time.perf_counter() - t0, info={"presolve": P.stats})
    if P.status == "solved":
        x, y, z = postsolve(P, None, None, None)
        obj = model.user_objective(x)
        return Result(status="optimal", x=x, y=y * model.sense, z=z * model.sense, objective=obj, bound=obj,
                      method="presolve", time=time.perf_counter() - t0, info={"presolve": P.stats})
    r = _dispatch(P.model, method, **options)
    if r.x is None:
        r.info["presolve"] = P.stats
        return r
    y_red = None if r.y is None else r.y * model.sense
    z_red = None if r.z is None else r.z * model.sense
    x, y, z = postsolve(P, r.x, y_red, z_red)
    r.x, r.y, r.z = x, y * model.sense, z * model.sense
    if r.status in ("optimal", "feasible", "time_limit", "node_limit", "iteration_limit"):
        obj = model.user_objective(x)
        if np.isfinite(r.objective):
            r.objective = obj
        if r.nodes == 0 and r.status == "optimal":
            r.bound = obj
    r.info["presolve"] = P.stats
    r.time = time.perf_counter() - t0
    return r
