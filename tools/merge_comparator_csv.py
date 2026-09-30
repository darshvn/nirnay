"""Merge raw comparator CSVs into results/comparators_{lp,mip,qp}.csv with reference objectives.

    python tools/merge_comparator_csv.py OUT.csv REF_SET raw1.csv [raw2.csv ...]
Later files override earlier rows with the same (instance, solver). REF_SET is lp, mip or qp.
Reference optima: Netlib / MIPLIB 3 / Maros-Meszaros published optimal values.
"""
import csv
import sys

REF = {
    "lp": {"afiro": -464.75314286, "adlittle": 225494.96316, "blend": -30.812149846,
           "sc50a": -64.575077059, "share2b": -415.73224074, "25fv47": 5501.8458883,
           "degen2": -1435.178, "pilot4": -2581.1392589, "greenbea": -72555248.130,
           "stocfor1": -41131.976219},
    "mip": {"p0033": 3089, "p0201": 7615, "p0282": 258411, "lseu": 1120, "stein27": 18,
            "egout": 568.1007, "mod008": 307, "bell5": 8966406.49, "flugpl": 1201500, "misc03": 3360},
    "qp": {"HS21": -99.96, "HS35": 0.1111111111, "QAFIRO": -1.5907817894, "QADLITTL": 480318.86,
           "CVXQP1_S": 11590.718, "DUALC1": 6155.2508, "PRIMALC1": -6155.2508, "QSC205": -0.0058139535,
           "QSHARE2B": 11703.692, "QPCBLEND": -0.0078425409},
}
PLATFORM = {"cbc": "WSL2 (CPU)", "cuopt": "WSL2 + RTX 3050 GPU"}
DEFAULT_TOL = {"highs-pdlp": "1e-7 rel (cuPDLP-C)", "highs-hipdlp": "1e-7 rel", "ortools-pdlp": "1e-6 rel",
               "ortools-pdlp-tight": "1e-8 rel", "cuopt-pdlp": "1e-4 rel", "osqp": "1e-3 abs/rel",
               "osqp-tight": "1e-6 abs/rel", "clarabel": "1e-8", "glpk-interior": "1e-8 (approx)"}


def main():
    out, refset, *raw = sys.argv[1:]
    rows = {}
    order = []
    for f in raw:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            key = (r["instance"], r["solver"])
            if key not in rows:
                order.append(key)
            rows[key] = r
    ref = REF[refset]
    insts = list(dict.fromkeys(k[0] for k in order))
    order.sort(key=lambda k: (insts.index(k[0]), k[1]))
    fields = ["instance", "solver", "platform", "status", "objective", "ref_objective", "rel_err",
              "bound", "gap", "time", "iterations", "nodes", "tolerance_note", "raw_status"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields, extrasaction="ignore")
        w.writeheader()
        for key in order:
            r = dict(rows[key])
            base = r["solver"].split("-")[0]
            r["platform"] = PLATFORM.get(base, "Windows 11 (CPU)")
            rv = ref.get(r["instance"])
            r["ref_objective"] = "" if rv is None else f"{rv:.10g}"
            try:
                o = float(r["objective"])
                r["rel_err"] = f"{abs(o - rv) / max(abs(rv), 1.0):.2e}" if rv is not None else ""
            except (TypeError, ValueError):
                r["rel_err"] = ""
            r["tolerance_note"] = DEFAULT_TOL.get(r["solver"], "")
            if r["solver"] == "highs-pdlp":
                r["tolerance_note"] += "; run with no time_limit (HiGHS 1.15.1 cuPDLP-C timer bug), hard kill 120 s"
            if r["raw_status"].startswith("Traceback") or len(r["raw_status"]) > 120:
                r["raw_status"] = "no result (process killed at hard timeout)" if r["status"] in ("error", "time_limit") else r["raw_status"][:120]
                if r["status"] == "error":
                    r["status"] = "time_limit"
            w.writerow(r)
    print(f"wrote {out}: {len(order)} rows")


if __name__ == "__main__":
    main()
