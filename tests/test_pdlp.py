"""Tests for nirnay.lp.pdlp (restarted PDHG). Plain asserts: `python tests/test_pdlp.py`.

HiGHS supplies reference objectives only; the solver under test never imports it. The GPU
tests run when CuPy imports and a device is present, and are skipped (with a message) otherwise.
"""
from __future__ import annotations

import gzip
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nirnay import solve  # noqa: E402
from nirnay.io.mps import read_mps  # noqa: E402
from nirnay.lp import pdlp  # noqa: E402
from nirnay.model import CSC, Model  # noqa: E402

NETLIB = ROOT / "data" / "netlib"
SMALL = ["afiro", "adlittle", "blend", "sc50a", "share2b"]
INF = np.inf
_REF: dict = {}


def highs_objective(name: str) -> float:
    if name not in _REF:
        import highspy
        tmp = Path(tempfile.gettempdir()) / f"{name}.mps"
        with gzip.open(NETLIB / f"{name}.mps.gz", "rb") as a, open(tmp, "wb") as b:
            shutil.copyfileobj(a, b)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(str(tmp))
        h.run()
        assert h.modelStatusToString(h.getModelStatus()) == "Optimal"
        _REF[name] = h.getInfo().objective_function_value
    return _REF[name]


def toy(A, c, rl, ru, lb, ub, sense=1, integer=None) -> Model:
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    r, k = np.nonzero(A)
    return Model("toy", np.asarray(c, float), CSC.from_triplets(m, n, r, k, A[r, k]),
                 np.asarray(rl, float), np.asarray(ru, float), np.asarray(lb, float),
                 np.asarray(ub, float),
                 np.zeros(n, bool) if integer is None else np.asarray(integer, bool), sense=sense)


def gpu_available() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception as e:                                   # noqa: BLE001
        print(f"  (GPU tests skipped: {type(e).__name__}: {str(e).splitlines()[0][:100]})")
        return False


# ---------------------------------------------------------------------------------------------

def check_netlib(tol: float, gpu: bool = False):
    """Objective within 10 tol (relative) of HiGHS, and the KKT conditions met at tol."""
    for name in SMALL:
        m = read_mps(NETLIB / f"{name}.mps.gz")
        r = pdlp.solve(m, tol=tol, gpu=gpu, time_limit=120)
        ref = highs_objective(name)
        err = abs(r.objective - ref) / max(1.0, abs(ref))
        i = r.info
        print(f"  {name:9s} tol {tol:.0e} {'gpu' if gpu else 'cpu'}: {r.status}, obj {r.objective:+.10e} "
              f"(HiGHS {ref:+.10e}, rel err {err:.1e}), {r.iterations} it, {r.time:.2f}s")
        assert r.status == "optimal", (name, r.status)
        assert max(i["rel_primal"], i["rel_dual"], i["rel_gap"]) <= tol, (name, i)
        assert err <= 10 * tol, (name, err)
        # the iterate is projected onto the box every step, so bounds hold exactly; the row
        # residual is recomputed here from the returned x against PDLP's relative criterion
        v = m.violation(r.x)
        assert v["bound"] <= 1e-12, (name, v)
        Ax = m.A.matvec(r.x)
        rp = np.linalg.norm(Ax - np.clip(Ax, m.rl, m.ru))
        bn = np.linalg.norm(pdlp._combined_bounds(m.rl, m.ru))
        assert rp <= tol * (1 + bn) * (1 + 1e-9), (name, rp, tol * (1 + bn))


def test_netlib_1e4():
    check_netlib(1e-4)


def test_netlib_1e6():
    check_netlib(1e-6)


def test_netlib_1e8():
    check_netlib(1e-8)


def test_dispatcher_and_sense():
    # max 3x + 2y  s.t.  x + y <= 4,  x + 3y <= 7,  0 <= x <= 3, y >= 0   ->  x = 3, y = 1, obj 11
    m = toy([[1, 1], [1, 3]], [-3, -2], [-INF, -INF], [4, 7], [0, 0], [3, INF], sense=-1)
    r = solve(m, method="pdlp", tol=1e-8)
    assert r.status == "optimal" and abs(r.objective - 11.0) < 1e-6, r
    assert np.allclose(r.x, [3, 1], atol=1e-6), r.x
    # duals in the user's (maximisation) sense: row 1 binds with multiplier 2, row 2 slack
    assert np.allclose(r.y, [2, 0], atol=1e-6), r.y


def test_ranges_free_rows_equalities():
    # min -x - 2y  s.t.  x + y <= 4,  -1 <= x - y <= 1,  3x + y free row,  0 <= x <= 3,  y >= 0
    m = toy([[1, 1], [1, -1], [3, 1]], [-1, -2], [-INF, -1, -INF], [4, 1, INF], [0, 0], [3, INF])
    r = pdlp.solve(m, tol=1e-9)
    assert r.status == "optimal" and abs(r.objective + 6.5) < 1e-7, r
    assert np.allclose(r.x, [1.5, 2.5], atol=1e-6)
    # equality row plus a free column
    m = toy([[1, 1, 0], [0, 1, -1]], [1, 1, 1], [2, 0], [2, 0], [0, -INF, 0], [INF, INF, INF])
    r = pdlp.solve(m, tol=1e-9)
    assert r.status == "optimal" and abs(r.objective - 2.0) < 1e-7, r


def test_mip_relaxation():
    # max x + y  s.t.  2x + 2y <= 3, x, y integer in [0, 1]: the LP relaxation has value 1.5
    m = toy([[2, 2]], [-1, -1], [-INF], [3], [0, 0], [1, 1], sense=-1, integer=[1, 1])
    r = pdlp.solve(m, tol=1e-8)
    assert r.status == "optimal" and abs(r.objective - 1.5) < 1e-6, r


def test_infeasible():
    # x + y <= 1 and x + y >= 3, x, y >= 0
    m = toy([[1, 1], [1, 1]], [1, 1], [-INF, 3], [1, INF], [0, 0], [INF, INF])
    r = pdlp.solve(m)
    assert r.status == "infeasible", r
    y = r.info["ray"]                                        # Farkas ray: y >= 0 on row 2, <= 0 on row 1
    assert y[0] < 0 < y[1]
    # infeasible through the bounds: x + y = 5 with 0 <= x, y <= 2
    r = pdlp.solve(toy([[1, 1]], [1, 2], [5], [5], [0, 0], [2, 2]))
    assert r.status == "infeasible", r


def test_unbounded():
    # min -x - y  s.t.  x - y <= 1, x, y >= 0: the ray (1, 1)
    r = pdlp.solve(toy([[1, -1]], [-1, -1], [-INF], [1], [0, 0], [INF, INF]))
    assert r.status == "unbounded", r
    d = r.info["ray"] / np.abs(r.info["ray"]).max()
    assert np.allclose(d, [1, 1], atol=1e-6), d
    # free variables: min x  s.t.  x - y = 0
    r = pdlp.solve(toy([[1, -1]], [1, 0], [0], [0], [-INF, -INF], [INF, INF]))
    assert r.status == "unbounded", r


def test_limits():
    m = read_mps(NETLIB / "share2b.mps.gz")
    r = pdlp.solve(m, tol=1e-10, max_iter=640)
    assert r.status == "iteration_limit" and r.iterations == 640, r
    assert r.x is not None and np.isfinite(r.objective)
    t = time.perf_counter()
    r = pdlp.solve(read_mps(NETLIB / "pilot87.mps.gz"), tol=1e-10, time_limit=1.0)
    assert r.status == "time_limit" and time.perf_counter() - t < 20, r


def test_parallel_matches_serial():
    m = read_mps(NETLIB / "adlittle.mps.gz")
    a = pdlp.solve(m, tol=1e-6, parallel=False)
    b = pdlp.solve(m, tol=1e-6, parallel=True)
    assert a.status == b.status == "optimal"
    assert abs(a.objective - b.objective) <= 1e-5 * max(1, abs(a.objective))


def test_gpu():
    if not gpu_available():
        return
    check_netlib(1e-4, gpu=True)
    check_netlib(1e-8, gpu=True)
    r = pdlp.solve(toy([[1, 1], [1, 1]], [1, 1], [-INF, 3], [1, INF], [0, 0], [INF, INF]), gpu=True)
    assert r.status == "infeasible", r
    r = pdlp.solve(toy([[1, -1]], [-1, -1], [-INF], [1], [0, 0], [INF, INF]), gpu=True)
    assert r.status == "unbounded", r
    # CPU and GPU run the same algorithm: same answer on the same model
    m = read_mps(NETLIB / "blend.mps.gz")
    c = pdlp.solve(m, tol=1e-8)
    g = pdlp.solve(m, tol=1e-8, gpu=True)
    assert g.info["backend"].startswith("gpu")
    assert abs(c.objective - g.objective) <= 1e-7 * max(1, abs(c.objective))


if __name__ == "__main__":
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({time.perf_counter() - t:.1f}s)", flush=True)
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e!r}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
