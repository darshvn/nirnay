"""NIRNAY Studio: a local web front end for the solver.

    python demo/studio/server.py            ->  http://127.0.0.1:8119

Pick a model, pick an engine, press Solve: the solver runs in this process and its own progress
log streams to the page (server-sent events), followed by the result, a model-specific reading
of the solution, and an independent re-solve by HiGHS for comparison. HiGHS is used here only as
that reference, exactly as in bench/; nothing in nirnay/ imports it.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import queue
import re
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nirnay import solve  # noqa: E402
from nirnay.cases import build  # noqa: E402
from nirnay.io.mps import read_mps  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="NIRNAY Studio")

# ---------------------------------------------------------------------------------------------
# the model shelf
MODELS = {
    "refinery": dict(
        group="MRPL scenario", title="Refinery crude plan", kind="LP",
        subtitle="Crude selection, CDU cuts, FCC and reformer, BS-VI blending",
        source="US DOE crude assays · BIS BS-VI specs · PPAC 2024-25 prices · MRPL CDU capacities",
        method="simplex", build=lambda: build("refinery_planning")),
    "plan4": dict(
        group="MRPL scenario", title="4-month refinery plan", kind="MILP",
        subtitle="Crude parcels per month, FCC mode changeovers, product inventories",
        source="As the crude plan, plus PPAC monthly demand and MRPL storage",
        method="bnb", build=lambda: build("refinery_multiperiod", months=4)),
    "unload": dict(
        group="MRPL scenario", title="Crude unloading schedule", kind="MILP",
        subtitle="Vessels → storage tanks → charging tanks → CDU, sulphur specs",
        source="Lee, Pinto, Grossmann & Park, Ind. Eng. Chem. Res. 35, 1996 (COSP1) · published optimum 79.75",
        method="bnb", published=79.75, build=lambda: build("crude_scheduling", instance=1)),
    "datt256": dict(
        group="Scale", title="datt256 · large LP", kind="LP",
        subtitle="11,077 rows · 262,144 columns · 1.5 M nonzeros (MIPLIB 2017 relaxation)",
        source="MIPLIB 2017 · reference times measured in results/large_pdlp_gpu.csv",
        method="pdlp-gpu", build=lambda: _mps("data/large/datt256.mps.gz")),
    "pilot": dict(
        group="Benchmarks", title="Netlib pilot", kind="LP",
        subtitle="Notoriously ill-conditioned economic planning model",
        source="Netlib LP collection", method="simplex", build=lambda: _mps("data/netlib/pilot.mps.gz")),
    "p0201": dict(
        group="Benchmarks", title="MIPLIB p0201", kind="MILP",
        subtitle="201 binary variables, classic MIPLIB 3 instance",
        source="MIPLIB 3", method="bnb", build=lambda: _mps("data/miplib/p0201.mps.gz")),
    "qship": dict(
        group="Benchmarks", title="Maros–Mészáros QSHIP04S", kind="QP",
        subtitle="Convex quadratic program",
        source="Maros–Mészáros QP test set", method="qp-ipm", build=lambda: _mps("data/qp/mm/QSHIP04S.QPS")),
}
ENGINES = {"simplex": "Dual simplex", "ipm": "Interior point", "pdlp": "PDLP · CPU",
           "pdlp-gpu": "PDLP · GPU", "bnb": "Branch-and-bound", "qp-ipm": "QP interior point"}
_cache: dict = {}
_lock = threading.Lock()


def _mps(rel):
    return read_mps(ROOT / rel)


def model(key):
    with _lock:
        if key not in _cache:
            _cache[key] = MODELS[key]["build"]()
        return _cache[key]


def meta(key, load=True):
    """Shelf entry; sizes are filled once the model is loaded (large ones load in the background)."""
    d = {k: v for k, v in MODELS[key].items() if k != "build"}
    d.update(key=key, engine=ENGINES[d["method"]], loaded=False)
    if load or key in _cache:
        m = model(key)
        d.update(rows=m.m, cols=m.n, nnz=int(m.A.nnz), integers=int(m.integer.sum()),
                 qnnz=int(m.Q.nnz) if m.Q is not None else 0, loaded=True)
    return d


# ---------------------------------------------------------------------------------------------
# turning the solver's own progress lines into events
PATTERNS = [
    ("simplex", re.compile(r"^\s*dual\s+(\d+)\s+obj\s+(\S+)\s+pinf\s+(\S+)")),
    ("bnb", re.compile(r"^\s*nodes\s+(\d+)\s+open\s+(\d+)\s+bound\s+(\S+)\s+incumbent\s+(\S+)")),
    ("incumbent", re.compile(r"^\s*incumbent\s+(\S+)\s+from\s+(.+?)\s+at node\s+(\d+)")),
    ("cuts", re.compile(r"^\s*cut round\s+(\d+):\s+(\d+)\s+cuts,\s+bound\s+(\S+)")),
    ("root", re.compile(r"^\s*root LP\s+(\S+),\s+(\d+)\s+fractional")),
    ("pdlp", re.compile(r"^\s*(\d+)\s+([\d.]+)s\s+obj\s+(\S+)\s+rel p\s+(\S+)\s+d\s+(\S+)\s+g\s+(\S+)")),
    ("ipm", re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+p\s+(\S+)\s+d\s+(\S+)\s+g\s+(\S+)\s+mu\s+(\S+)")),
]


def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def parse(line):
    for kind, rx in PATTERNS:
        mt = rx.match(line)
        if mt:
            g = mt.groups()
            if kind == "simplex":
                return {"type": "simplex", "it": int(g[0]), "obj": num(g[1]), "pinf": num(g[2])}
            if kind == "bnb":
                return {"type": "bnb", "nodes": int(g[0]), "open": int(g[1]), "bound": num(g[2]),
                        "incumbent": num(g[3])}
            if kind == "incumbent":
                return {"type": "incumbent", "value": num(g[0]), "source": g[1], "node": int(g[2])}
            if kind == "cuts":
                return {"type": "cuts", "round": int(g[0]), "cuts": int(g[1]), "bound": num(g[2])}
            if kind == "root":
                return {"type": "root", "value": num(g[0]), "fractional": int(g[1])}
            if kind == "pdlp":
                return {"type": "pdlp", "it": int(g[0]), "t": num(g[1]), "obj": num(g[2]),
                        "p": num(g[3]), "d": num(g[4]), "g": num(g[5])}
            if kind == "ipm":
                return {"type": "ipm", "it": int(g[0]), "pobj": num(g[1]), "dobj": num(g[2]),
                        "p": num(g[3]), "d": num(g[4]), "g": num(g[5]), "mu": num(g[6])}
    return None


class _Pipe(io.TextIOBase):
    """stdout replacement that hands complete lines to a queue."""

    def __init__(self, q):
        self.q, self.buf = q, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.q.put(("line", line))
        return len(s)


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


# ---------------------------------------------------------------------------------------------
# reading a solution the way a planner would
def insights(key, m, r):
    x = r.x
    if x is None:
        return None
    names = m.col_names
    if key == "refinery":
        crude = defaultdict(float)
        product = defaultdict(float)
        for n, v in zip(names, x):
            if v <= 1e-6:
                continue
            if n.startswith("crude_"):
                crude[n[6:].rsplit("_CDU", 1)[0].replace("_", " ")] += v
            elif n.startswith("sales_") or "_to_" in n:
                tgt = n.rsplit("_to_", 1)[-1] if "_to_" in n else n[6:]
                if tgt in ("LPG", "NAPHTHA", "MS91", "MS95", "ATF_A", "ATF_B", "HSD", "FO"):
                    product[tgt.replace("_", " ")] += v
        y = r.y * m.sense if r.y is not None else np.zeros(m.m)
        labels = {"cdu_cap_CDU1": "CDU-1 capacity", "cdu_cap_CDU2": "CDU-2 capacity",
                  "cdu_cap_CDU3": "CDU-3 capacity", "market_MS91": "MS 91 market", "market_MS95": "MS 95 market",
                  "market_HSD": "HSD market", "market_LPG": "LPG market", "market_ATF": "ATF market",
                  "ron_MS91": "MS 91 octane spec", "ron_MS95": "MS 95 octane spec", "cetane_HSD": "HSD cetane spec",
                  "reformer_in": "Reformer capacity", "fcc_low_in": "FCC capacity (low)", "fcc_high_in": "FCC capacity (high)"}
        duals = sorted(((labels[n], abs(v)) for n, v in zip(m.row_names, y) if n in labels and abs(v) > 1e-6),
                       key=lambda t: -t[1])
        return {"kind": "refinery",
                "crude": sorted(crude.items(), key=lambda t: -t[1]),
                "product": sorted(product.items(), key=lambda t: -t[1]),
                "duals": duals[:6], "margin": -r.objective}
    if key == "plan4":
        months = []
        for mo in ("APR", "MAY", "JUN", "JUL"):
            parcels = {n.split(f"parcels_{mo}_", 1)[1].replace("_", " "): int(round(v))
                       for n, v in zip(names, x) if n.startswith(f"parcels_{mo}_") and v > 0.5}
            mode = next((n.rsplit("_", 1)[1] for n, v in zip(names, x) if n.startswith(f"fcc_mode_{mo}_") and v > 0.5), "")
            months.append({"month": mo, "parcels": parcels, "fcc": mode})
        return {"kind": "plan", "months": months, "margin": -r.objective}
    if key == "unload":
        info = {"v1": ("Vessel 1", "unload → Tank 1"), "v2": ("Vessel 2", "unload → Tank 2"),
                "v3": ("Tank 1", "→ Charging tank A"), "v4": ("Tank 1", "→ Charging tank B"),
                "v5": ("Tank 2", "→ Charging tank A"), "v6": ("Tank 2", "→ Charging tank B"),
                "v7": ("CDU", "runs on blend A"), "v8": ("CDU", "runs on blend B")}
        val = dict(zip(names, x))
        bars = []
        for n, v in val.items():
            mt = re.match(r"Z_(\d+)_(v\d+)$", n)
            if mt and v > 0.5:
                i, op = mt.groups()
                s, e = val.get(f"S_{i}_{op}", 0.0), val.get(f"E_{i}_{op}", 0.0)
                vol = val.get(f"VT_{i}_{op}", 0.0)
                if e - s > 1e-6:
                    lane, what = info[op]
                    bars.append({"lane": lane, "what": what, "op": op, "start": round(s, 3), "end": round(e, 3),
                                 "volume": round(vol, 2)})
        bars.sort(key=lambda b: (["Vessel 1", "Vessel 2", "Tank 1", "Tank 2", "CDU"].index(b["lane"]), b["start"]))
        return {"kind": "gantt", "horizon": 8, "bars": bars, "published": 79.75}
    return None


def highs_reference(key, m):
    """HiGHS on the same model, written to MPS by NIRNAY's writer. For the large LP the value
    measured in the benchmark (results/large_pdlp_gpu.csv) is returned instead of a 4-minute run."""
    if key == "datt256":
        for row in csv.DictReader(open(ROOT / "results" / "large_pdlp_gpu.csv")):
            if row["name"] == "datt256":
                return {"objective": float(row["highs_obj"]), "time": float(row["highs_time"]),
                        "status": row["highs_status"], "recorded": True,
                        "cpu_time": float(row["cpu_time"]), "gpu_time_bench": float(row["gpu_time"])}
    import highspy
    from nirnay.io.mps_write import write_mps
    tmp = Path(tempfile.gettempdir()) / f"studio_{key}.mps"
    write_mps(m, tmp)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", 120.0)
    h.readModel(str(tmp))
    t = time.perf_counter()
    h.run()
    el = time.perf_counter() - t
    return {"objective": h.getInfo().objective_function_value, "time": el,
            "status": h.modelStatusToString(h.getModelStatus()), "recorded": False}


# ---------------------------------------------------------------------------------------------
@app.get("/api/models")
def api_models():
    return JSONResponse([meta(k, load=False) for k in MODELS])


@app.get("/api/model")
def api_model(model_key: str = Query(..., alias="model")):
    return JSONResponse(meta(model_key))


@app.get("/api/solve")
def api_solve(model_key: str = Query(..., alias="model"), method: str = "auto"):
    m = model(model_key)
    method = MODELS[model_key]["method"] if method == "auto" else method
    q: queue.Queue = queue.Queue()

    def run():
        pipe = _Pipe(q)
        try:
            opts = dict(verbose=2, time_limit=120)
            if method.startswith("pdlp"):
                opts["tol"] = 1e-4
            t = time.perf_counter()
            with contextlib.redirect_stdout(pipe):
                r = solve(m, method=method, **opts)
            el = time.perf_counter() - t
            v = m.violation(r.x) if r.x is not None else {"row": None, "bound": None, "integrality": None}
            q.put(("result", {
                "type": "result", "status": r.status, "objective": r.objective,
                "bound": r.bound if np.isfinite(r.bound) else None, "time": el,
                "iterations": int(r.iterations), "nodes": int(r.nodes), "method": r.method,
                "engine": ENGINES.get(method, method),
                "row_viol": v["row"], "bound_viol": v["bound"],
                "presolve": r.info.get("presolve"), "cuts": r.info.get("cuts"),
                "insights": insights(model_key, m, r)}))
        except Exception as e:  # shown on the page, not swallowed
            q.put(("result", {"type": "error", "message": f"{type(e).__name__}: {e}"}))
        q.put(("end", None))

    def stream():
        yield sse({"type": "start", "model": model_key, "method": method, "engine": ENGINES.get(method, method)})
        threading.Thread(target=run, daemon=True).start()
        while True:
            kind, payload = q.get()
            if kind == "line":
                ev = parse(payload) or {}
                ev.update(type=ev.get("type", "log"), line=payload.strip())
                yield sse(ev)
            elif kind == "result":
                yield sse(payload)
            else:
                break
        yield sse({"type": "end"})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/verify")
def api_verify(model_key: str = Query(..., alias="model")):
    m = model(model_key)
    return JSONResponse(highs_reference(model_key, m))


@app.get("/api/benchmarks")
def api_benchmarks():
    def rows(n):
        return list(csv.DictReader(open(ROOT / "results" / f"{n}.csv", newline="")))

    def ok(r, tol, strict=True):
        e = r.get("rel_err", "")
        return r["status"] == "optimal" and e not in ("", "nan") and (float(e) < tol if strict else float(e) <= tol)

    spx, ipm, mip, qp = rows("netlib_simplex_v5"), rows("netlib_ipm_v2"), rows("miplib3_bnb_v2"), rows("maros_qpipm_v2")
    sg = lambda v: math.exp(sum(math.log(x + 1) for x in v) / len(v)) - 1  # noqa: E731
    okl = [r for r in spx if ok(r, 1e-6)]
    return JSONResponse({
        "suites": [
            {"name": "Netlib LP", "what": "the classic hard LP collection", "total": 90,
             "ours": len(okl), "ref": 90, "ref_name": "HiGHS"},
            {"name": "MIPLIB 3", "what": "mixed-integer programs, 60 s each", "total": 64,
             "ours": sum(ok(r, 1e-4, False) for r in mip), "ref": sum(r["ref_status"] == "Optimal" for r in mip),
             "ref_name": "HiGHS"},
            {"name": "Maros–Mészáros", "what": "convex quadratic programs", "total": 138,
             "ours": sum(ok(r, 1e-6) for r in qp), "ref": sum(r["ref_status"].startswith("Solved") for r in qp),
             "ref_name": "Clarabel"},
        ],
        "ipm": sum(ok(r, 1e-6) for r in ipm),
        "wrong_mip": sum(r["status"] == "optimal" and not ok(r, 1e-4, False) for r in mip),
        "speed_ratio": sg([float(r["time"]) for r in okl]) / sg([float(r["ref_time"]) for r in okl]),
    })


@app.get("/api/source")
def api_source():
    pkg = ROOT / "nirnay"
    mods, imports = [], defaultdict(set)
    for p in sorted(pkg.rglob("*.py")):
        rel = p.relative_to(pkg).as_posix()
        text = p.read_text(encoding="utf-8")
        mods.append({"path": rel, "lines": text.count("\n") + 1})
        for mt in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", text, flags=re.M):
            name = (mt.group(1) or mt.group(2)).split(".")[0]
            if name and not name.startswith("_") and name not in ("nirnay",):
                imports[name].add(rel)
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    third = {k: sorted(v) for k, v in imports.items() if k not in stdlib and k != ""}
    solvers = ["highspy", "scipy", "cvxpy", "gurobipy", "cplex", "xpress", "pyscipopt", "ortools",
               "pulp", "mosek", "clarabel", "osqp", "cylp", "swiglpk"]
    kernel = (pkg / "lp" / "_simplex_kernels.py").read_text(encoding="utf-8")
    start = kernel.index("@njit(cache=True)\ndef dual_run")
    return JSONResponse({"modules": mods, "total_lines": sum(m["lines"] for m in mods),
                         "third_party": {k: len(v) for k, v in sorted(third.items())},
                         "solver_imports": {s: sorted(imports.get(s, [])) for s in solvers},
                         "snippet": kernel[start:start + 2600]})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    # warm the model cache and the compiled kernels so the first click is not a compile
    def warm():
        for k in MODELS:
            try:
                model(k)
            except Exception as e:  # a missing benchmark file only hides that entry
                print("skip", k, e)
        solve(model("refinery"), verbose=0)
        try:  # the first GPU solve in a process compiles the CUDA kernels; pay that here, not on screen
            solve(model("datt256"), method="pdlp-gpu", tol=1e-4)
        except Exception as e:
            print("gpu warm-up skipped:", e)
        print("warm", flush=True)
    threading.Thread(target=warm, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8119, log_level="warning")
