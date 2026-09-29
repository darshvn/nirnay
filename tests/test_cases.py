"""Case studies and the MPS writer.

* every case builds, passes Model.validate(), and has CASE_INFO with sources and licences
* the exported data/cases/<name>.mps is exactly the model build() returns
* HiGHS, reading that MPS file, reaches the optimum recorded in data/cases/reference.csv
* write_mps round-trips every file in data/netlib and data/miplib through read_mps unchanged

HiGHS (highspy) is the comparator only. Slow reference solves are marked `slow`; run them with
    pytest tests/test_cases.py --runslow        (or set NIRNAY_SLOW=1)
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from nirnay.cases import CASES, build, info
from nirnay.io.mps import read_mps
from nirnay.io.mps_write import write_mps

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "data" / "cases"
REF = {}
if (CASE_DIR / "reference.csv").exists():
    with open(CASE_DIR / "reference.csv", newline="") as fh:
        REF = {r["case"]: r for r in csv.DictReader(fh)}

SLOW = os.environ.get("NIRNAY_SLOW") == "1"
# cases whose HiGHS solve takes more than about a minute on a laptop
SLOW_CASES = {name for name, r in REF.items() if float(r.get("highs_time_s") or 0) > 60}


def same_model(a, b) -> bool:
    ok = (a.m == b.m and a.n == b.n and a.sense == b.sense and a.c0 == b.c0
          and np.array_equal(a.c, b.c) and np.array_equal(a.rl, b.rl) and np.array_equal(a.ru, b.ru)
          and np.array_equal(a.lb, b.lb) and np.array_equal(a.ub, b.ub)
          and np.array_equal(a.integer, b.integer)
          and np.array_equal(a.A.colptr, b.A.colptr) and np.array_equal(a.A.rowidx, b.A.rowidx)
          and np.array_equal(a.A.vals, b.A.vals))
    if a.is_qp or b.is_qp:
        ok = ok and a.is_qp and b.is_qp and np.array_equal(a.Q.colptr, b.Q.colptr) \
            and np.array_equal(a.Q.rowidx, b.Q.rowidx) and np.array_equal(a.Q.vals, b.Q.vals)
    return ok


# ---- case studies ------------------------------------------------------------------------------

@pytest.fixture(scope="module", params=CASES)
def case(request):
    return request.param, build(request.param)


def test_builds_and_validates(case):
    name, model = case
    model.validate()
    assert model.n > 0 and model.m > 0
    ci = info(name)
    assert ci["class"] in ("LP", "MILP", "QP")
    assert ci["class"] == ("QP" if model.is_qp else "MILP" if model.is_mip else "LP")
    assert ci["sources"] and all(s.get("url") and s.get("licence") for s in ci["sources"])
    assert "real" in ci and "assumed" in ci


def test_exported_mps_matches_build(case, tmp_path):
    name, model = case
    path = CASE_DIR / f"{name}.mps"
    assert path.exists(), f"run python -m nirnay.cases._reference {name}"
    assert same_model(model, read_mps(path)), "data/cases MPS is stale: rerun _reference"
    write_mps(model, tmp_path / "x.mps")
    assert same_model(model, read_mps(tmp_path / "x.mps"))


def test_highs_reference(case):
    name, model = case
    pytest.importorskip("highspy")
    if name in SLOW_CASES and not SLOW:
        pytest.skip("slow reference solve; set NIRNAY_SLOW=1")
    from nirnay.cases._reference import options, solve_mps
    ref = REF[name]
    opts = options(name)
    r = solve_mps(CASE_DIR / f"{name}.mps", time_limit=max(600.0, 3 * float(ref["highs_time_s"])),
                  mip_rel_gap=opts["mip_rel_gap"])
    assert r["status"] == "Optimal"
    obj = float(ref["objective"])
    # a MIP solved to relative gap g is only known to within g; an LP/QP to solver tolerance
    rel = max(opts["mip_rel_gap"], 1e-6) if model.is_mip else 1e-7
    assert abs(r["objective"] - obj) <= rel * max(1.0, abs(obj)), (r["objective"], obj)


def test_reference_csv_complete():
    assert set(REF) == set(CASES)
    assert all(r["highs_status"] == "Optimal" for r in REF.values())


# ---- checks against values published with the source data --------------------------------------

def _highs(model, tmp_path, **kw):
    pytest.importorskip("highspy")
    from nirnay.cases._reference import solve_mps
    write_mps(model, tmp_path / "m.mps")
    return solve_mps(tmp_path / "m.mps", **kw)


@pytest.mark.parametrize("instance", [1, 2, 3, 4])
def test_crude_scheduling_published(instance, tmp_path):
    """MILP optima in Table 1 of the minlp.org problem 117 session results."""
    from nirnay.cases import crude_scheduling as cs
    n, published = cs.PUBLISHED_MILP[instance]
    m = cs.build(instance, n)
    r = _highs(m, tmp_path, time_limit=300)
    assert r["status"] == "Optimal"
    assert abs(r["objective"] - published) <= 5e-4 * published      # published to 3 decimals
    # column counts of Table 3 include GAMS's objective variable
    assert m.n + 1 == {1: 536, 2: 1387, 3: 1281, 4: 1565}[instance]


def test_economic_dispatch_matches_pglib_baseline(tmp_path):
    """pglib-opf BASELINE.md: DC objective 9.4304e+05 $/h for pglib_opf_case2000_goc."""
    r = _highs(build("economic_dispatch"), tmp_path, time_limit=300)
    assert r["status"] == "Optimal"
    assert abs(r["objective"] - 9.4304e5) <= 0.5e1 + 5e-5 * 9.4304e5


@pytest.mark.parametrize("instance", ["cap41", "cap131"])
def test_facility_location_published(instance, tmp_path):
    """OR-Library capopt optimal values."""
    from nirnay.cases import facility_location as fl
    m = fl.build(instance, None)
    r = _highs(m, tmp_path, time_limit=300)
    assert r["status"] == "Optimal"
    assert abs(r["objective"] - fl.PUBLISHED[instance, None]) <= 1e-3


# ---- MPS writer round trip ---------------------------------------------------------------------

BENCH = sorted((ROOT / "data").glob("netlib/*.mps.gz")) + sorted((ROOT / "data").glob("miplib/*.mps.gz"))


@pytest.mark.parametrize("path", BENCH, ids=[p.name for p in BENCH])
def test_mps_roundtrip(path, tmp_path):
    try:
        m1 = read_mps(path)
    except Exception as e:                       # a reader limitation, not a writer one
        pytest.skip(f"read_mps cannot read the original: {e}")
    out = tmp_path / "w.mps"
    write_mps(m1, out)
    m2 = read_mps(out)
    assert same_model(m1, m2)
    # names survive unless free MPS cannot hold them (fixed-format names with spaces, e.g. forplan)
    if not any(ch.isspace() for s in m1.col_names + m1.row_names for ch in s):
        assert m1.col_names == m2.col_names and m1.row_names == m2.row_names


def test_writer_features(tmp_path):
    """Maximisation, objective constant, ranged rows, every bound type, integers and Q."""
    from nirnay.model import CSC, INF, Model
    A = CSC.from_triplets(4, 6, [0, 0, 1, 1, 2, 3, 3], [0, 1, 1, 2, 3, 4, 5],
                          [1.0, 2.0, -1.0, 0.1, 3.0, 1.0, 1.0])
    Q = CSC.from_triplets(6, 6, [0, 1, 0, 1], [0, 0, 1, 1], [2.0, 0.5, 0.5, 1.0])
    m = Model(name="features", c=np.array([1.0, -2.0, 0.0, 0.3, 1e-17, 7.0]), A=A,
              rl=np.array([0.1, -INF, 2.0, -3.0]), ru=np.array([0.7, 5.0, 2.0, 4.25]),
              lb=np.array([-INF, -INF, 1.5, 0.0, -2.0, 0.0]),
              ub=np.array([INF, -1.0, 1.5, 1.0, 9.0, INF]),
              integer=np.array([False, False, False, True, True, True]),
              Q=Q, c0=-3.25, sense=-1, col_names=list("abcdef"), row_names=["r1", "r2", "r3", "r4"])
    write_mps(m, tmp_path / "f.mps")
    m2 = read_mps(tmp_path / "f.mps")
    assert same_model(m, m2)
