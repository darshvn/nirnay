"""Crude-oil operations scheduling (unloading, tank transfers, CDU charging), MILP.

The four crude-oil scheduling problems of Lee, Pinto, Grossmann and Park (1996), COSP1-COSP4, as
posed by Mouret and Grossmann on the CMU-IBM MINLP library (minlp.org, problem 117, "Crude-oil
Operations Scheduling"). The instance data below are transcribed line by line from the GAMS files
LeeCrudeOil1.gms ... LeeCrudeOil4.gms in the CrudeOil.zip archive of that problem; the model is
MOSMILP in Scheduler.gms of the same archive: the Multi-Operation Sequencing (MOS) priority-slot
formulation of Mouret, Grossmann and Pestiaux (2009) with the nonlinear composition constraint
(CompositionCst, a bilinear equality) removed. That MILP is the first step of the authors'
two-step MILP-NLP method; its optimal values are published in the session results (Table 1):

    COSP1 n>=5: 79.750    COSP2 n>=6: 101.175    COSP3 n>=5: 87.400    COSP4 n>=4: 132.548

Units are those of the source (volumes in the same unit as vessel contents, time in days as in
Lee et al. 1996, gross margin per unit volume). Constraint names follow Scheduler.gms.

Sets: slots i = 1..n, operations v (unloading, transfer, distillation), crudes p, resources r
(vessels, storage tanks, charging tanks, CDUs), properties k, operation sets vs.
Variables: Z(i,v) binary; S, D, E (start, duration, end); VT(i,v) total volume, VP(i,v,p) volume of
crude p; LT(i,r) level before slot i, LP(i,r,p) level of crude p.

    max  sum_{i,v,p} val(v,p) VP(i,v,p)
    s.t. MOSAssignment   sum_{v in clique} Z(i,v) <= 1
         MinCard/MaxCard number of assignments of each operation set within [minN, maxN]
         MinCard1/MaxCard1 the same for slot 1
         SetReqPrec1Time / SSTSetReqPrec1   vessels unload in arrival order
         Time E = S + D;  MinStart S >= minS Z;  MaxEnd E <= H Z
         Min/MaxVolumeTotal  minVT Z <= VT <= maxVT Z;  VolumeCompo VT = sum_p VP
         level bounds, LevelCompo, LevelTotalDef, LevelProdDef (inventory balances over slots)
         Min/MaxFlowrate  minFR D <= VT <= maxFR D
         TotalDuration   each CDU runs the whole horizon
         Min/MaxDemand   volume charged from each charging tank
         Min/MaxProperty linear blending of crude properties (e.g. sulphur) for CDU feeds
         end-of-horizon level bounds
         NoOverlapClique operations in a clique do not overlap in time
         SBClique2       symmetry breaking
"""
from __future__ import annotations

from itertools import product

from ..model import INF
from ._builder import Builder

CASE_INFO = {
    "title": "Crude-oil unloading, blending and CDU charging schedule (Lee et al. 1996, COSP2)",
    "sector": "Refinery scheduling",
    "class": "MILP",
    "sources": [
        {"what": "Instance data COSP1-COSP4 (vessels, tanks, CDUs, flow limits, crude properties, "
                 "specifications, demands, margins) and the MOS model (Scheduler.gms)",
         "name": "minlp.org problem 117, Crude-oil Operations Scheduling, S. Mouret and "
                 "I.E. Grossmann, model CrudeOil MOS Model, CrudeOil.zip",
         "url": "https://www.minlp.org/library/problem/mod/download.php?file=CrudeOil.zip&location=292/input/CrudeOil.zip",
         "licence": "no licence stated; published in the CMU-IBM open MINLP library (NSF grant "
                    "OCI-0750826) for research use"},
        {"what": "Problem page (problem statement, model description)",
         "name": "minlp.org problem 117",
         "url": "https://www.minlp.org/library/problem/index.php?i=117",
         "licence": "as above"},
        {"what": "Published MILP-relaxation optimum for each instance and slot count (Table 1) "
                 "and model sizes (Table 3)",
         "name": "minlp.org problem 117 session results (PDF)",
         "url": "https://www.minlp.org/problems/ver/152/results/SessionResults.pdf",
         "licence": "as above"},
        {"what": "Original problem and data",
         "name": "H. Lee, J.M. Pinto, I.E. Grossmann, S. Park, Mixed-integer linear programming "
                 "model for refinery short-term scheduling of crude oil unloading with inventory "
                 "management, Ind. Eng. Chem. Res. 35(5):1630-1641, 1996",
         "url": "https://doi.org/10.1021/ie950519h", "licence": "citation only (paywalled)"},
        {"what": "MOS priority-slot formulation",
         "name": "S. Mouret, I.E. Grossmann, P. Pestiaux, A novel priority-slot based "
                 "continuous-time formulation for crude-oil scheduling problems, Ind. Eng. Chem. "
                 "Res. 48(18):8515-8528, 2009",
         "url": "https://doi.org/10.1021/ie8019592", "licence": "citation only"},
    ],
    "real": ["all instance data as published on minlp.org (transcribed from the GAMS files, "
             "kept in data/cases/raw/scheduling/)",
             "the model equations of MOSMILP in Scheduler.gms"],
    "assumed": ["the number of priority slots n is a modelling parameter (default 8, above the "
                "minimum of 6 that already reaches the published optimum 101.175 for COSP2)",
                "this is the MILP relaxation: the bilinear composition equality is dropped, as in "
                "the first step of the published method, so tank-outlet compositions are not "
                "forced to equal tank compositions"],
}


def _instance(num: int) -> dict:
    """Returns the data of LeeCrudeOil<num>.gms. Names (v1, r3, p2, ...) are those of the files."""
    V = lambda a, b: [f"v{k}" for k in range(a, b + 1)]
    if num == 1:
        d = dict(
            H=8, nv=8, nr=7, P=["p1", "p2", "p3", "p4"], K=["k1"],
            vessel=["r1", "r2"], stor=["r3", "r4"], charg=["r5", "r6"], cdu=["r7"],
            unload=V(1, 2), transf=V(3, 6), distil=V(7, 8),
            IN={"v1": "r3", "v2": "r4", "v3": "r5", "v4": "r6", "v5": "r5", "v6": "r6", "v7": "r7", "v8": "r7"},
            OUT={"v1": "r1", "v2": "r2", "v3": "r3", "v4": "r3", "v5": "r4", "v6": "r4", "v7": "r5", "v8": "r6"},
            VSET={"vs1": ["v1"], "vs2": ["v2"], "vs7": ["v7"], "vs8": ["v8"],
                  "vs12": ["v1", "v2"], "vs13": ["v1", "v3"], "vs14": ["v1", "v4"], "vs25": ["v2", "v5"],
                  "vs26": ["v2", "v6"], "vs37": ["v3", "v7"], "vs48": ["v4", "v8"], "vs57": ["v5", "v7"],
                  "vs68": ["v6", "v8"], "vs78": ["v7", "v8"]},
            prec=[("vs12", "v1", "v2")],
            CLIQUE=["vs12", "vs13", "vs14", "vs25", "vs26", "vs37", "vs48", "vs57", "vs68", "vs78"],
            minN={"vs1": 1, "vs2": 1, "vs78": 3, "vs7": 1, "vs8": 1},
            maxN={"vs1": 1, "vs2": 1, "vs78": 3, "vs7": 2, "vs8": 2},
            minN1={"vs78": 1}, maxN1={"vs78": 1},
            iniCP={("r1", "p1"): 100, ("r2", "p2"): 100, ("r3", "p1"): 25, ("r4", "p2"): 75,
                   ("r5", "p3"): 50, ("r6", "p4"): 50},
            minCT={}, maxCT_default=100, maxCT={}, vessel_max_is_initial=False,
            minS={"v2": 4}, maxVT_default=100, maxVT={}, unload_maxVT_is_initial=False,
            minFR_distil=5, maxFR=50,
            totD={"vs78": 8}, dem={"vs7": 100, "vs8": 100},
            val={"p1": 0.1, "p2": 0.6, "p3": 0.2, "p4": 0.5},
            prop={("p1", "k1"): 0.1, ("p2", "k1"): 0.6, ("p3", "k1"): 0.2, ("p4", "k1"): 0.5},
            spec={("r5", "k1"): (0.15, 0.25), ("r6", "k1"): (0.45, 0.55)},
        )
    elif num in (2, 3):
        common = dict(
            nv=14, nr=11, vessel=["r1", "r2", "r3"], stor=["r4", "r5", "r6"],
            charg=["r7", "r8", "r9"], cdu=["r10", "r11"],
            unload=V(1, 3), transf=V(4, 10), distil=V(11, 14),
            IN={"v1": "r4", "v2": "r5", "v3": "r6", "v4": "r7", "v5": "r8", "v6": "r7", "v7": "r8",
                "v8": "r9", "v9": "r8", "v10": "r9", "v11": "r10", "v12": "r10", "v13": "r11", "v14": "r11"},
            OUT={"v1": "r1", "v2": "r2", "v3": "r3", "v4": "r4", "v5": "r4", "v6": "r5", "v7": "r5",
                 "v8": "r5", "v9": "r6", "v10": "r6", "v11": "r7", "v12": "r8", "v13": "r8", "v14": "r9"},
            VSET={"vs1": ["v1"], "vs2": ["v2"], "vs3": ["v3"], "vs11": ["v11"], "vs12": ["v12"],
                  "vs13": ["v13"], "vs14": ["v14"],
                  "vs1_2": ["v1", "v2"], "vs1_3": ["v1", "v3"], "vs1_4": ["v1", "v4"],
                  "vs1_5": ["v1", "v5"], "vs2_3": ["v2", "v3"], "vs2_6": ["v2", "v6"],
                  "vs2_7": ["v2", "v7"], "vs2_8": ["v2", "v8"], "vs3_9": ["v3", "v9"],
                  "vs3_10": ["v3", "v10"], "vs4_11": ["v4", "v11"], "vs6_11": ["v6", "v11"],
                  "vs8_14": ["v8", "v14"], "vs10_14": ["v10", "v14"], "vs11_12": ["v11", "v12"],
                  "vs12_13": ["v12", "v13"], "vs13_14": ["v13", "v14"],
                  "vs1-3": ["v1", "v2", "v3"], "vs5_12_13": ["v5", "v12", "v13"],
                  "vs7_12_13": ["v7", "v12", "v13"], "vs9_12_13": ["v9", "v12", "v13"],
                  "vs11-14": ["v11", "v12", "v13", "v14"]},
            prec=[("vs1_2", "v1", "v2"), ("vs2_3", "v2", "v3")],
            CLIQUE=["vs1_4", "vs1_5", "vs2_6", "vs2_7", "vs2_8", "vs3_9", "vs3_10", "vs4_11",
                    "vs6_11", "vs8_14", "vs10_14", "vs11_12", "vs13_14", "vs1-3", "vs5_12_13",
                    "vs7_12_13", "vs9_12_13"],
            minN={"vs1": 1, "vs2": 1, "vs3": 1, "vs11-14": 5, "vs11": 1, "vs12": 1, "vs13": 1, "vs14": 1},
            maxN={"vs1": 1, "vs2": 1, "vs3": 1, "vs11-14": 5, "vs11": 2, "vs12": 1, "vs13": 1, "vs14": 2},
            minN1={"vs11_12": 1, "vs13_14": 1}, maxN1={"vs11_12": 1, "vs13_14": 1},
            minCT={}, maxCT_default=100, maxCT={}, maxFR=50, minFR_distil=5,
        )
        d = dict(common)
        if num == 2:
            d.update(
                H=10, P=[f"p{k}" for k in range(1, 7)], K=["k1", "k2"],
                iniCP={("r1", "p1"): 100, ("r2", "p2"): 100, ("r3", "p3"): 100, ("r4", "p1"): 20,
                       ("r5", "p2"): 50, ("r6", "p3"): 70, ("r7", "p4"): 30, ("r8", "p5"): 50,
                       ("r9", "p6"): 30},
                vessel_max_is_initial=False, minS={"v2": 3, "v3": 6},
                maxVT_default=100, maxVT={}, unload_maxVT_is_initial=False,
                totD={"vs11_12": 10, "vs13_14": 10},
                dem={"vs11": 100, "vs12_13": 100, "vs14": 100},
                val={"p1": 0.1, "p2": 0.3, "p3": 0.5, "p4": 0.167, "p5": 0.3, "p6": 0.433},
                prop={("p1", "k1"): 0.1, ("p2", "k1"): 0.3, ("p3", "k1"): 0.5, ("p4", "k1"): 0.167,
                      ("p5", "k1"): 0.3, ("p6", "k1"): 0.433, ("p1", "k2"): 0.4, ("p2", "k2"): 0.2,
                      ("p3", "k2"): 0.1, ("p4", "k2"): 0.333, ("p5", "k2"): 0.23, ("p6", "k2"): 0.133},
                spec={("r7", "k1"): (0.1, 0.2), ("r8", "k1"): (0.25, 0.35), ("r9", "k1"): (0.4, 0.48),
                      ("r7", "k2"): (0.3, 0.38), ("r8", "k2"): (0.18, 0.27), ("r9", "k2"): (0.1, 0.18)},
            )
        else:
            d.update(
                H=12, P=[f"p{k}" for k in range(1, 8)], K=["k1"],
                iniCP={("r1", "p1"): 50, ("r2", "p2"): 50, ("r3", "p3"): 50, ("r4", "p4"): 20,
                       ("r5", "p5"): 20, ("r6", "p6"): 20, ("r7", "p7"): 30, ("r8", "p5"): 50,
                       ("r9", "p6"): 30},
                vessel_max_is_initial=True, minS={"v1": 0.001, "v2": 4.001, "v3": 8.001},
                maxVT_default=100, maxVT={v: 50 for v in V(11, 14)}, unload_maxVT_is_initial=True,
                totD={"vs11_12": 12, "vs13_14": 12},
                dem={"vs11": 50, "vs12_13": 50, "vs14": 50},
                val={"p1": 0.1, "p2": 0.6, "p3": 0.85, "p4": 0.2, "p5": 0.5, "p6": 0.8, "p7": 0.3},
                prop={("p1", "k1"): 0.1, ("p2", "k1"): 0.85, ("p3", "k1"): 0.6, ("p4", "k1"): 0.2,
                      ("p5", "k1"): 0.5, ("p6", "k1"): 0.8, ("p7", "k1"): 0.3},
                spec={("r7", "k1"): (0.25, 0.35), ("r8", "k1"): (0.45, 0.65), ("r9", "k1"): (0.75, 0.85)},
            )
    elif num == 4:
        maxVT = {v: 80 for v in V(4, 13)}
        maxVT.update({v: 100 for v in V(5, 10)})
        maxVT.update({v: 60 for v in V(14, 19)})
        d = dict(
            H=15, nv=19, nr=16, P=[f"p{k}" for k in range(1, 9)], K=["k1"],
            vessel=["r1", "r2", "r3"], stor=[f"r{k}" for k in range(4, 10)],
            charg=[f"r{k}" for k in range(10, 14)], cdu=["r14", "r15", "r16"],
            unload=V(1, 3), transf=V(4, 13), distil=V(14, 19),
            IN={"v1": "r5", "v2": "r6", "v3": "r7", "v4": "r10", "v5": "r10", "v6": "r11", "v7": "r11",
                "v8": "r12", "v9": "r11", "v10": "r12", "v11": "r12", "v12": "r13", "v13": "r13",
                "v14": "r14", "v15": "r14", "v16": "r15", "v17": "r15", "v18": "r16", "v19": "r16"},
            OUT={"v1": "r1", "v2": "r2", "v3": "r3", "v4": "r4", "v5": "r5", "v6": "r5", "v7": "r6",
                 "v8": "r6", "v9": "r7", "v10": "r7", "v11": "r8", "v12": "r8", "v13": "r9",
                 "v14": "r10", "v15": "r11", "v16": "r11", "v17": "r12", "v18": "r12", "v19": "r13"},
            VSET={"vs1": ["v1"], "vs2": ["v2"], "vs3": ["v3"], "vs14": ["v14"], "vs15": ["v15"],
                  "vs16": ["v16"], "vs17": ["v17"], "vs18": ["v18"], "vs19": ["v19"],
                  "vs1_2": ["v1", "v2"], "vs1_3": ["v1", "v3"], "vs1_5": ["v1", "v5"],
                  "vs1_6": ["v1", "v6"], "vs2_3": ["v2", "v3"], "vs2_7": ["v2", "v7"],
                  "vs2_8": ["v2", "v8"], "vs3_9": ["v3", "v9"], "vs3_10": ["v3", "v10"],
                  "vs4_14": ["v4", "v14"], "vs5_14": ["v5", "v14"], "vs12_19": ["v12", "v19"],
                  "vs13_19": ["v13", "v19"], "vs14_15": ["v14", "v15"], "vs15_16": ["v15", "v16"],
                  "vs16_17": ["v16", "v17"], "vs17_18": ["v17", "v18"], "vs18_19": ["v18", "v19"],
                  "vs1-3": ["v1", "v2", "v3"], "vs6_15_16": ["v6", "v15", "v16"],
                  "vs7_15_16": ["v7", "v15", "v16"], "vs8_17_18": ["v8", "v17", "v18"],
                  "vs9_15_16": ["v9", "v15", "v16"], "vs10_17_18": ["v10", "v17", "v18"],
                  "vs11_17_18": ["v11", "v17", "v18"], "vs14-19": V(14, 19)},
            prec=[("vs1_2", "v1", "v2"), ("vs2_3", "v2", "v3")],
            CLIQUE=["vs1_5", "vs1_6", "vs2_7", "vs2_8", "vs3_9", "vs3_10", "vs4_14", "vs5_14",
                    "vs12_19", "vs13_19", "vs1-3", "vs6_15_16", "vs7_15_16", "vs8_17_18",
                    "vs9_15_16", "vs10_17_18", "vs11_17_18", "vs14_15", "vs16_17", "vs18_19"],
            minN={"vs1": 1, "vs2": 1, "vs3": 1, "vs14-19": 7, "vs14": 1, "vs15": 1, "vs16": 1,
                  "vs17": 1, "vs18": 1, "vs19": 1},
            maxN={"vs1": 1, "vs2": 1, "vs3": 1, "vs14-19": 7, "vs14": 2, "vs15": 1, "vs16": 1,
                  "vs17": 1, "vs18": 1, "vs19": 2},
            minN1={"vs14_15": 1, "vs16_17": 1, "vs18_19": 1},
            maxN1={"vs14_15": 1, "vs16_17": 1, "vs18_19": 1},
            iniCP={("r1", "p1"): 60, ("r2", "p2"): 60, ("r3", "p3"): 60, ("r4", "p4"): 60,
                   ("r5", "p1"): 10, ("r6", "p2"): 50, ("r7", "p3"): 40, ("r8", "p5"): 30,
                   ("r9", "p5"): 60, ("r10", "p6"): 5, ("r11", "p7"): 30, ("r12", "p8"): 30,
                   ("r13", "p5"): 30},
            minCT={f"r{k}": 10 for k in range(4, 10)},
            maxCT_default=None,
            maxCT={**{f"r{k}": 90 for k in range(4, 10)}, **{f"r{k}": 110 for k in range(5, 8)},
                   **{f"r{k}": 80 for k in range(10, 14)}},
            vessel_max_is_initial=True, minS={"v2": 5, "v3": 10},
            maxVT_default=None, maxVT=maxVT, unload_maxVT_is_initial=True,
            minFR_distil=2, maxFR=50,
            totD={"vs14_15": 15, "vs16_17": 15, "vs18_19": 15},
            dem={"vs14": 60, "vs15_16": 60, "vs17_18": 60, "vs19": 60},
            val={"p1": 0.3, "p2": 0.5, "p3": 0.65, "p4": 0.31, "p5": 0.75, "p6": 0.317,
                 "p7": 0.483, "p8": 0.633},
            prop={("p1", "k1"): 0.3, ("p2", "k1"): 0.5, ("p3", "k1"): 0.65, ("p4", "k1"): 0.31,
                  ("p5", "k1"): 0.75, ("p6", "k1"): 0.317, ("p7", "k1"): 0.483, ("p8", "k1"): 0.633},
            spec={("r10", "k1"): (0.3, 0.35), ("r11", "k1"): (0.43, 0.5), ("r12", "k1"): (0.6, 0.65),
                  ("r13", "k1"): (0.71, 0.8)},
        )
    else:
        raise ValueError("instance must be 1, 2, 3 or 4")
    return d


# published MILP optimum (minlp.org problem 117, session results, Table 1) and the smallest n
PUBLISHED_MILP = {1: (5, 79.750), 2: (6, 101.175), 3: (5, 87.400), 4: (4, 132.548)}


def build(instance: int = 2, slots: int = 8):
    d = _instance(instance)
    H, P, K = d["H"], d["P"], d["K"]
    Vall = [f"v{k}" for k in range(1, d["nv"] + 1)]
    Rall = [f"r{k}" for k in range(1, d["nr"] + 1)]
    I = list(range(1, slots + 1))

    iniCP = {(r, p): float(d["iniCP"].get((r, p), 0.0)) for r in Rall for p in P}
    iniCT = {r: sum(iniCP[r, p] for p in P) for r in Rall}
    minCT = {r: float(d["minCT"].get(r, 0.0)) for r in Rall}
    maxCT = {}
    for r in Rall:
        if r in d["cdu"]:
            maxCT[r] = INF
        elif r in d["maxCT"]:
            maxCT[r] = float(d["maxCT"][r])
        elif r in d["vessel"] and d["vessel_max_is_initial"]:
            maxCT[r] = iniCT[r]
        else:
            maxCT[r] = float(d["maxCT_default"])
    minS = {v: float(d["minS"].get(v, 0.0)) for v in Vall}
    maxE = {v: float(H) for v in Vall}
    minVT = {v: (iniCT[d["OUT"][v]] if v in d["unload"] else 0.0) for v in Vall}
    maxVT = {}
    for v in Vall:
        if v in d["maxVT"]:
            maxVT[v] = float(d["maxVT"][v])
        elif v in d["unload"] and d["unload_maxVT_is_initial"]:
            maxVT[v] = iniCT[d["OUT"][v]]
        else:
            maxVT[v] = float(d["maxVT_default"])
    minFR = {v: (float(d["minFR_distil"]) if v in d["distil"] else 0.0) for v in Vall}
    maxFR = {v: float(d["maxFR"]) for v in Vall}
    val = {(v, p): (d["val"][p] if v in d["distil"] else 0.0) for v in Vall for p in P}
    prop = {(p, k): float(d["prop"].get((p, k), 0.0)) for p in P for k in K}
    minProp, maxProp = {}, {}
    for v in d["distil"]:
        for k in K:
            if (d["OUT"][v], k) in d["spec"]:
                minProp[v, k], maxProp[v, k] = d["spec"][d["OUT"][v], k]
    IN = {r: [v for v in Vall if d["IN"][v] == r] for r in Rall}
    OUT = {r: [v for v in Vall if d["OUT"][v] == r] for r in Rall}
    VSET = d["VSET"]
    CLIQUE2 = {v: set() for v in Vall}
    for vs in d["CLIQUE"]:
        for a in VSET[vs]:
            for b in VSET[vs]:
                if a != b:
                    CLIQUE2[a].add(b)

    B = Builder(f"cosp{instance}_n{slots}", sense=-1)
    Z, S, D, E, VT, VP, LT, LP = {}, {}, {}, {}, {}, {}, {}, {}
    for i in I:
        for v in Vall:
            Z[i, v] = B.binary(f"Z_{i}_{v}")
            S[i, v] = B.var(f"S_{i}_{v}")
            D[i, v] = B.var(f"D_{i}_{v}")
            E[i, v] = B.var(f"E_{i}_{v}")
            VT[i, v] = B.var(f"VT_{i}_{v}")
            for p in P:
                VP[i, v, p] = B.var(f"VP_{i}_{v}_{p}", cost=val[v, p])
        for r in Rall:
            LT[i, r] = B.var(f"LT_{i}_{r}")
            for p in P:
                LP[i, r, p] = B.var(f"LP_{i}_{r}_{p}")

    for i in I:                                                   # MOSAssignment
        for vs in d["CLIQUE"]:
            B.le(f"MOSAssignment_{i}_{vs}", [Z[i, v] for v in VSET[vs]], 1.0, 1.0)
    for vs, n_ in d["minN"].items():                              # MinCard
        if n_ > 0:
            B.ge(f"MinCard_{vs}", [Z[i, v] for i in I for v in VSET[vs]], 1.0, n_)
    for vs, n_ in d["maxN"].items():                              # MaxCard
        B.le(f"MaxCard_{vs}", [Z[i, v] for i in I for v in VSET[vs]], 1.0, n_)
    for vs, n_ in d["minN1"].items():                             # MinCard1
        if n_ > 0:
            B.ge(f"MinCard1_{vs}", [Z[1, v] for v in VSET[vs]], 1.0, n_)
    for vs, n_ in d["maxN1"].items():                             # MaxCard1
        B.le(f"MaxCard1_{vs}", [Z[1, v] for v in VSET[vs]], 1.0, n_)
    for vs, a, b in d["prec"]:
        B.le(f"SetReqPrec1Time_{vs}", [E[i, a] for i in I] + [S[i, b] for i in I],
             [1.0] * len(I) + [-1.0] * len(I), 0.0)
        for i in I:                                               # sum_{j<i} Z(j,a) >= sum_{j<=i} Z(j,b)
            B.ge(f"SSTSetReqPrec1_{i}_{vs}", [Z[j, a] for j in I if j < i] + [Z[j, b] for j in I if j <= i],
                 [1.0] * (i - 1) + [-1.0] * i, 0.0)

    for i in I:
        for v in Vall:
            B.eq(f"Time_{i}_{v}", [E[i, v], S[i, v], D[i, v]], [1.0, -1.0, -1.0], 0.0)
            B.ge(f"MinStart_{i}_{v}", [S[i, v], Z[i, v]], [1.0, -minS[v]], 0.0)
            B.le(f"MaxEnd_{i}_{v}", [E[i, v], Z[i, v]], [1.0, -maxE[v]], 0.0)
            if minVT[v] > 0:
                B.ge(f"MinVolumeTotal_{i}_{v}", [VT[i, v], Z[i, v]], [1.0, -minVT[v]], 0.0)
            B.le(f"MaxVolumeTotal_{i}_{v}", [VT[i, v], Z[i, v]], [1.0, -maxVT[v]], 0.0)
            B.eq(f"VolumeCompo_{i}_{v}", [VT[i, v]] + [VP[i, v, p] for p in P], [1.0] + [-1.0] * len(P), 0.0)
            B.ge(f"MinFlowrate_{i}_{v}", [VT[i, v], D[i, v]], [1.0, -minFR[v]], 0.0)
            B.le(f"MaxFlowrate_{i}_{v}", [VT[i, v], D[i, v]], [1.0, -maxFR[v]], 0.0)
            for k in K:
                if (v, k) in minProp:
                    B.ge(f"MinProperty_{i}_{v}_{k}", [VP[i, v, p] for p in P] + [VT[i, v]],
                         [prop[p, k] for p in P] + [-minProp[v, k]], 0.0)
                    B.le(f"MaxProperty_{i}_{v}_{k}", [VP[i, v, p] for p in P] + [VT[i, v]],
                         [prop[p, k] for p in P] + [-maxProp[v, k]], 0.0)
        for r in Rall:
            if minCT[r] > 0:
                B.set_bounds(LT[i, r], lb=minCT[r])               # MinLevelTotal
            if maxCT[r] < INF:
                B.set_bounds(LT[i, r], ub=maxCT[r])               # MaxLevelTotal
                for p in P:
                    B.set_bounds(LP[i, r, p], ub=maxCT[r])        # MaxLevelProduct
            B.eq(f"LevelCompo_{i}_{r}", [LT[i, r]] + [LP[i, r, p] for p in P], [1.0] + [-1.0] * len(P), 0.0)
            prev = [j for j in I if j < i]
            B.eq(f"LevelTotalDef_{i}_{r}",
                 [LT[i, r]] + [VT[j, v] for j in prev for v in IN[r]] + [VT[j, v] for j in prev for v in OUT[r]],
                 [1.0] + [-1.0] * (len(prev) * len(IN[r])) + [1.0] * (len(prev) * len(OUT[r])), iniCT[r])
            for p in P:
                B.eq(f"LevelProdDef_{i}_{r}_{p}",
                     [LP[i, r, p]] + [VP[j, v, p] for j in prev for v in IN[r]]
                     + [VP[j, v, p] for j in prev for v in OUT[r]],
                     [1.0] + [-1.0] * (len(prev) * len(IN[r])) + [1.0] * (len(prev) * len(OUT[r])), iniCP[r, p])

    for vs, tot in d["totD"].items():                             # TotalDuration
        B.eq(f"TotalDuration_{vs}", [D[i, v] for i in I for v in VSET[vs]], 1.0, float(tot))
    for vs, dem in d["dem"].items():                              # Min/MaxDemand (equal here)
        B.row(f"Demand_{vs}", [VT[i, v] for i in I for v in VSET[vs]], 1.0, float(dem), float(dem))

    for r in Rall:                                                # end-of-horizon levels
        cols = [VT[i, v] for i in I for v in IN[r]] + [VT[i, v] for i in I for v in OUT[r]]
        coef = [1.0] * (len(I) * len(IN[r])) + [-1.0] * (len(I) * len(OUT[r]))
        B.row(f"EndLevelTotal_{r}", cols, coef, minCT[r] - iniCT[r],
              maxCT[r] - iniCT[r] if maxCT[r] < INF else INF)
        for p in P:
            cols = [VP[i, v, p] for i in I for v in IN[r]] + [VP[i, v, p] for i in I for v in OUT[r]]
            B.row(f"EndLevelProduct_{r}_{p}", cols, coef, -iniCP[r, p],
                  maxCT[r] - iniCP[r, p] if maxCT[r] < INF else INF)

    for i1, i2 in product(I, I):                                  # NoOverlapClique
        if i1 >= i2:
            continue
        for vs in d["CLIQUE"]:
            ops = VSET[vs]
            M = max(maxE[v] for v in ops)
            cols = [E[i1, v] for v in ops] + [D[i, v] for i in I if i1 < i < i2 for v in ops] \
                + [S[i2, v] for v in ops] + [Z[i2, v] for v in ops]
            coef = [1.0] * len(ops) + [1.0] * (len(ops) * max(0, i2 - i1 - 1)) \
                + [-1.0] * len(ops) + [M] * len(ops)
            B.le(f"NoOverlapClique_{i1}_{i2}_{vs}", cols, coef, M)

    for i in I:                                                   # SBClique2
        if i == 1:
            continue
        for v in Vall:
            ws = sorted(CLIQUE2[v], key=lambda s: int(s[1:]))
            B.le(f"SBClique2_{i - 1}_{v}", [Z[i, v]] + [Z[i - 1, w] for w in ws], [1.0] + [-1.0] * len(ws), 0.0)
    return B.to_model()
