"""PDLP on the CPU (Numba) against PDLP on the GPU (CuPy) against HiGHS, on large LPs.

For every instance: NIRNAY PDLP on the CPU, NIRNAY PDLP on the GPU, both at relative KKT
tolerance --tol, and HiGHS as the comparator in up to three modes:
  * highs      HiGHS's default LP solve (simplex / IPM with crossover): the reference objective
  * highs-ipm  HiGHS IPM without crossover at the same tolerance (the nearest like-for-like run)
  * highs-pdlp HiGHS's own PDLP (a CPU port of cuPDLP-C) at the same tolerance (optional)
HiGHS is used here only as a comparator; the solver never imports it.

Each run is a separate process with a wall-clock limit, so a stalled run cannot stop the sweep.
Both PDLP backends are warmed up first on a tiny LP so that JIT compilation (Numba, NVRTC) is not
charged to the first large instance; the reported PDLP time includes preconditioning and the
host-to-device transfer.

usage: python bench/run_pdlp_gpu.py data/large --tol 1e-4 --limit 600 --out results/large_pdlp_gpu.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import multiprocessing as mp
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def _warm(gpu):
    from nirnay.lp import pdlp
    from nirnay.io.mps import read_mps
    here = Path(__file__).resolve().parent.parent / "data" / "netlib" / "afiro.mps.gz"
    if here.exists():
        pdlp.solve(read_mps(here), gpu=gpu, tol=1e-4)


def _pdlp(path, gpu, tol, limit, q):
    try:
        from nirnay.io.mps import read_mps
        from nirnay.lp import pdlp
        _warm(gpu)
        m = read_mps(path)
        r = pdlp.solve(m, gpu=gpu, tol=tol, time_limit=limit, max_iter=10**9)
        v = m.violation(r.x) if r.x is not None else {"row": np.nan}
        q.put({"status": r.status, "obj": r.objective, "time": r.time, "iters": r.iterations,
               "m": m.m, "n": m.n, "nnz": m.A.nnz, "backend": r.info.get("backend", ""),
               "rel_p": r.info.get("rel_primal"), "rel_d": r.info.get("rel_dual"),
               "rel_g": r.info.get("rel_gap"), "restarts": r.info.get("restarts"),
               "row_viol": v["row"]})
    except Exception as e:                                      # report, don't crash the sweep
        q.put({"status": f"error: {type(e).__name__}: {e}"[:200]})


def _highs(path, mode, tol, limit, q):
    try:
        import highspy
        tmp = Path(tempfile.gettempdir()) / (Path(path).name.replace(".gz", ""))
        if str(path).endswith(".gz"):
            with gzip.open(path, "rb") as a, open(tmp, "wb") as b:
                shutil.copyfileobj(a, b)
        else:
            tmp = Path(path)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(limit))
        if mode == "highs-ipm":
            h.setOptionValue("solver", "ipm")
            h.setOptionValue("run_crossover", "off")
            h.setOptionValue("ipm_optimality_tolerance", tol)
            h.setOptionValue("primal_feasibility_tolerance", tol)
            h.setOptionValue("dual_feasibility_tolerance", tol)
        elif mode == "highs-pdlp":
            h.setOptionValue("solver", "pdlp")
            h.setOptionValue("pdlp_d_gap_tol", tol)
            h.setOptionValue("primal_feasibility_tolerance", tol)
            h.setOptionValue("dual_feasibility_tolerance", tol)
        h.readModel(str(tmp))
        t = time.perf_counter()
        h.run()
        el = time.perf_counter() - t
        q.put({"status": h.modelStatusToString(h.getModelStatus()),
               "obj": h.getInfo().objective_function_value, "time": el})
    except Exception as e:
        q.put({"status": f"error: {type(e).__name__}: {e}"[:200]})


def _in_process(target, args, limit):
    q = mp.Queue()
    p = mp.Process(target=target, args=args + (q,))
    t = time.perf_counter()
    p.start()
    p.join(limit + 120)
    if p.is_alive():
        p.terminate()
        return {"status": "timeout", "time": time.perf_counter() - t}
    return q.get() if not q.empty() else {"status": "crashed"}


def _hardware():
    cpu = platform.processor() or platform.machine()
    try:
        import os
        cpu += f", {os.cpu_count()} logical cores"
    except Exception:
        pass
    gpu = ""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        gpu = out.stdout.strip()
    except Exception:
        pass
    return cpu, gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--limit", type=float, default=600)
    ap.add_argument("--out", default="results/large_pdlp_gpu.csv")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[], help="runs to skip: cpu gpu highs highs-ipm highs-pdlp")
    a = ap.parse_args()
    files = sorted(Path(a.folder).glob("*.mps*"))
    if a.only:
        files = [f for f in files if f.name.split(".")[0] in a.only]
    cpu, gpu = _hardware()
    print(f"CPU: {cpu}\nGPU: {gpu}\ntol {a.tol:g}, limit {a.limit:g}s", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    modes = [mo for mo in ("cpu", "gpu", "highs", "highs-ipm", "highs-pdlp") if mo not in a.skip]
    fields = ["name", "m", "n", "nnz", "tol"]
    for mo in modes:
        fields += [f"{mo}_status", f"{mo}_obj", f"{mo}_time"]
        if mo in ("cpu", "gpu"):
            fields += [f"{mo}_iters", f"{mo}_rel_err", f"{mo}_rel_p", f"{mo}_rel_d", f"{mo}_rel_g"]
    fields += ["speedup_gpu_vs_cpu", "cpu_hw", "gpu_hw"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for f in files:
            name = f.name.split(".")[0]
            row = {"name": name, "tol": a.tol, "cpu_hw": cpu, "gpu_hw": gpu}
            res = {}
            for mo in modes:
                if mo in ("cpu", "gpu"):
                    r = _in_process(_pdlp, (str(f), mo == "gpu", a.tol, a.limit), a.limit)
                else:
                    r = _in_process(_highs, (str(f), mo, a.tol, a.limit), a.limit)
                res[mo] = r
                for key in ("m", "n", "nnz"):
                    if key in r:
                        row[key] = r[key]
                row[f"{mo}_status"] = r["status"]
                row[f"{mo}_obj"] = r.get("obj", np.nan)
                row[f"{mo}_time"] = r.get("time", np.nan)
                print(f"  {name:14s} {mo:10s} {r['status']:16s} obj {r.get('obj', np.nan):+.10e} "
                      f"t {r.get('time', np.nan):8.2f}s", flush=True)
            ref = res.get("highs", {}).get("obj", np.nan)
            for mo in ("cpu", "gpu"):
                if mo in res:
                    r = res[mo]
                    obj = r.get("obj", np.nan)
                    row[f"{mo}_iters"] = r.get("iters")
                    row[f"{mo}_rel_err"] = abs(obj - ref) / max(1.0, abs(ref)) \
                        if np.isfinite(obj) and np.isfinite(ref) else np.nan
                    row[f"{mo}_rel_p"], row[f"{mo}_rel_d"], row[f"{mo}_rel_g"] = \
                        r.get("rel_p"), r.get("rel_d"), r.get("rel_g")
            if "cpu" in res and "gpu" in res:
                tc, tg = res["cpu"].get("time", np.nan), res["gpu"].get("time", np.nan)
                both = res["cpu"]["status"] == res["gpu"]["status"] == "optimal"
                row["speedup_gpu_vs_cpu"] = tc / tg if both and tg > 0 else np.nan
            w.writerow(row)
            fh.flush()


if __name__ == "__main__":
    main()
