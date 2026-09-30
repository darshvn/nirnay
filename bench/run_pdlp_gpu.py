"""PDLP on the CPU (Numba) against PDLP on the GPU (CuPy) against HiGHS, on large LPs.

For every instance: NIRNAY PDLP on the CPU, NIRNAY PDLP on the GPU, both at relative KKT
tolerance --tol, and HiGHS as the comparator in up to three modes:
  * highs      HiGHS interior point without crossover at its default (1e-8) tolerances: the
               reference objective and HiGHS's high-accuracy time. (HiGHS's default dual simplex
               and IPM+crossover need far longer than the limits used here on several of these
               degenerate instances, e.g. qap15, so they are not the reference.)
  * highs-ipm  HiGHS IPM without crossover at the same tolerance (the nearest like-for-like run)
  * highs-pdlp HiGHS's own PDLP (a CPU port of cuPDLP-C) at the same tolerance
HiGHS is used here only as a comparator; the solver never imports it.

MIP instances (MIPLIB 2017) are run as their LP relaxations by every solver: NIRNAY PDLP drops
integrality itself and HiGHS gets solve_relaxation = true.

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


def _pdlp(path, gpu, tol, limit, threads, q):
    try:
        from nirnay.io.mps import read_mps
        from nirnay.lp import pdlp
        _warm(gpu)
        m = read_mps(path)
        r = pdlp.solve(m, gpu=gpu, tol=tol, time_limit=limit, max_iter=10**9, threads=threads)
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

        def setopt(k, v):                        # a rejected option must not pass silently
            if h.setOptionValue(k, v) != highspy.HighsStatus.kOk:
                raise ValueError(f"HiGHS rejected option {k}={v}")
        setopt("output_flag", False)
        if mode != "highs-pdlp":
            # HiGHS 1.15 PDLP stops after a handful of iterations with "Time limit reached" as
            # soon as any time_limit is set; the wall-clock limit is enforced by the caller instead
            setopt("time_limit", float(limit))
        setopt("solve_relaxation", True)          # the MIP files are benchmarked as LP relaxations
        if mode == "highs":
            setopt("solver", "ipm")
            setopt("run_crossover", "off")
        elif mode == "highs-ipm":
            setopt("solver", "ipm")
            setopt("run_crossover", "off")
            setopt("ipm_optimality_tolerance", tol)
            setopt("primal_feasibility_tolerance", tol)
            setopt("dual_feasibility_tolerance", tol)
        elif mode == "highs-pdlp":
            setopt("solver", "pdlp")
            setopt("pdlp_optimality_tolerance", tol)
            setopt("primal_feasibility_tolerance", tol)
            setopt("dual_feasibility_tolerance", tol)
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
    p.join(limit + 60)
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
    ap.add_argument("--threads", type=int, default=None, help="Numba threads for CPU PDLP")
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
                    r = _in_process(_pdlp, (str(f), mo == "gpu", a.tol, a.limit, a.threads), a.limit)
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
            ref = np.nan
            for mo in ("highs", "highs-ipm"):          # the most accurate finished HiGHS run
                if res.get(mo, {}).get("status") in ("Optimal", "Unknown") and np.isfinite(res[mo].get("obj", np.nan)):
                    ref = res[mo]["obj"]
                    break
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
