"""Regression tests for the solver core: each engine against published optimal values.

Published optima: Netlib (Koch, "The final NETLIB-LP results", ORL 32, 2004), MIPLIB 3 solution
file, Maros-Meszaros repository. Instances are skipped if the benchmark data is not downloaded.
"""
from pathlib import Path

import numpy as np
import pytest

from nirnay import solve
from nirnay.io.mps import read_mps
from nirnay.model import CSC, Model

DATA = Path(__file__).resolve().parent.parent / "data"


def need(path):
    p = DATA / path
    if not p.exists():
        pytest.skip(f"{path} not downloaded")
    return read_mps(p)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# ------------------------------------------------------------------ LP
NETLIB = {"afiro": -4.6475314286e02, "adlittle": 2.2549496316e05, "degen2": -1.4351780000e03,
          "perold": -9.3807552782e03, "pilotnov": -4.4972761882e03, "25fv47": 5.5018458883e03}


@pytest.mark.parametrize("name,opt", NETLIB.items())
def test_dual_simplex_netlib(name, opt):
    m = need(f"netlib/{name}.mps.gz")
    r = solve(m, method="simplex")
    assert r.status == "optimal"
    assert rel(r.objective, opt) < 1e-8
    v = m.violation(r.x)
    assert v["row"] < 1e-6 and v["bound"] < 1e-9


@pytest.mark.parametrize("name", ["afiro", "adlittle", "degen2", "25fv47"])
def test_ipm_netlib(name):
    m = need(f"netlib/{name}.mps.gz")
    r = solve(m, method="ipm")
    assert r.status == "optimal"
    assert rel(r.objective, NETLIB[name]) < 1e-6


def test_simplex_and_ipm_agree_on_random_lp():
    rng = np.random.default_rng(7)
    m_, n_ = 30, 50
    A = rng.standard_normal((m_, n_)) * (rng.random((m_, n_)) < 0.3)
    x0 = rng.random(n_)
    rows, cols = np.nonzero(A)
    model = Model(name="rand", c=rng.random(n_) + 0.1, A=CSC.from_triplets(m_, n_, rows, cols, A[rows, cols]),
                  rl=A @ x0 - 1.0, ru=A @ x0 + 1.0, lb=np.zeros(n_), ub=np.full(n_, 5.0),
                  integer=np.zeros(n_, dtype=bool))
    r1, r2 = solve(model, method="simplex"), solve(model, method="ipm")
    assert r1.status == r2.status == "optimal"
    assert rel(r1.objective, r2.objective) < 1e-6


def test_infeasible_lp_detected():
    A = CSC.from_triplets(2, 1, [0, 1], [0, 0], [1.0, 1.0])
    model = Model(name="inf", c=np.ones(1), A=A, rl=np.array([2.0, -np.inf]), ru=np.array([np.inf, 1.0]),
                  lb=np.zeros(1), ub=np.full(1, np.inf), integer=np.zeros(1, dtype=bool))
    assert solve(model, method="simplex").status == "infeasible"


# ------------------------------------------------------------------ MILP
MIPLIB = {"p0033": 3089.0, "flugpl": 1201500.0, "egout": 568.1007, "gt2": 21166.0}


@pytest.mark.parametrize("name,opt", MIPLIB.items())
def test_branch_and_bound_miplib(name, opt):
    m = need(f"miplib/{name}.mps.gz")
    r = solve(m, method="bnb", time_limit=120)
    assert r.status == "optimal"
    assert rel(r.objective, opt) < 1e-6
    v = m.violation(r.x)
    assert v["row"] < 1e-5 and v["integrality"] < 1e-6


def test_gomory_cuts_are_valid():
    """Every cut must keep the known optimal integer point feasible."""
    from nirnay.lp.simplex import SimplexLP
    from nirnay.mip.bnb import _relaxed
    from nirnay.mip.cuts import gmi_cuts
    m = need("miplib/p0033.mps.gz")
    best = solve(m, method="bnb", time_limit=60)
    lp = SimplexLP(_relaxed(m))
    lp.solve()
    cuts = gmi_cuts(lp, m)
    assert cuts
    for g, h, _ in cuts:
        assert g @ best.x >= h - 1e-6 * max(1.0, abs(h))


# ------------------------------------------------------------------ QP
MM = {"QAFIRO": -1.5907817939, "HS21": -99.96, "HS118": 664.82045, "CVXQP1_S": 11590.718,
      "DUAL1": 0.0350129659, "Q25FV47": 13744447.9}


@pytest.mark.parametrize("name,opt", MM.items())
def test_qp_ipm_maros_meszaros(name, opt):
    m = need(f"qp/mm/{name}.QPS")
    r = solve(m, method="qp-ipm")
    assert r.status == "optimal"
    assert rel(r.objective, opt) < 1e-6


# ------------------------------------------------------------------ kernels
def test_ldl_quasidefinite():
    from nirnay.linalg.ldl import LDL
    rng = np.random.default_rng(3)
    n, m = 25, 10
    Hd = rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.2)
    H = Hd @ Hd.T + np.eye(n)
    A = rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.3)
    K = np.block([[-H, A.T], [A, 1e-6 * np.eye(m)]])
    r, c = np.nonzero(K)
    order = np.lexsort((r, c))
    r, c = r[order], c[order]
    colptr = np.zeros(n + m + 1, dtype=np.int64)
    np.add.at(colptr, c + 1, 1)
    colptr = np.cumsum(colptr)
    f = LDL(n + m, colptr, r)
    f.factor(K[r, c], np.r_[-np.ones(n), np.ones(m)])
    b = rng.standard_normal(n + m)
    assert np.abs(K @ f.solve(b) - b).max() < 1e-8


def test_propagation_tightens_and_detects_infeasibility():
    from nirnay.mip.propagate import propagate
    # x + y <= 1 with x, y binary and x >= 1  =>  y <= 0
    Rp = np.array([0, 2]); Rj = np.array([0, 1]); Rv = np.array([1.0, 1.0])
    lb, ub = np.array([1.0, 0.0]), np.array([1.0, 1.0])
    st, _ = propagate(Rp, Rj, Rv, np.array([-np.inf]), np.array([1.0]), lb, ub, np.array([True, True]), 5, 1e-6)
    assert st == 0 and ub[1] == 0.0
    lb2, ub2 = np.array([1.0, 1.0]), np.array([1.0, 1.0])
    st, _ = propagate(Rp, Rj, Rv, np.array([-np.inf]), np.array([1.0]), lb2, ub2, np.array([True, True]), 5, 1e-6)
    assert st == 1
