"""NIRNAY: a sovereign LP / MILP / QP optimisation solver, built from mathematical foundations."""
from __future__ import annotations

__version__ = "0.1.0"

from .model import CSC, Model, Result  # noqa: F401


def solve(model, method: str = "auto", **options):
    """Solve a Model. method: auto | ipm | simplex | pdlp | pdlp-gpu | bnb."""
    if method == "auto":
        method = "bnb" if model.is_mip else ("qp-ipm" if model.is_qp else "simplex")
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
