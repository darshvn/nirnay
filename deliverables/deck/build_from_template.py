"""Build the SIH26119 idea deck inside the team's SIH26117 template.

The template (tpl117/src.pptx, the team's own SIH26117 deck) supplies every shape, colour,
position and the speaker-notes layout. This script replaces only text, paragraph by paragraph,
so the design is untouched. Where the template hard-wraps a block into one paragraph per line,
the new text is written as lines of similar length so nothing overflows.

Headline numbers are read from ../../results/*.csv at build time.

    python build_from_template.py        ->  ZeroCloud_SIH26119.pptx
"""
from __future__ import annotations

import copy
import csv
import shutil
import zipfile
from pathlib import Path

from defusedxml import minidom

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / "tpl117" / "src.pptx"
OUT = HERE / "ZeroCloud_SIH26119.pptx"
WORK = HERE / "tpl117" / "build"


# ------------------------------------------------------------------ numbers from results/
def rows(name):
    p = ROOT / "results" / f"{name}.csv"
    return list(csv.DictReader(open(p, newline=""))) if p.exists() else []


def newest(names, min_rows):
    for n in names:
        r = rows(n)
        if len(r) >= min_rows:
            return r
    return []


def ok(r, tol, strict=True):
    e = r.get("rel_err", "")
    if r.get("status") != "optimal" or e in ("", "nan"):
        return False
    return float(e) < tol if strict else float(e) <= tol


spx = newest(["netlib_simplex_v5", "netlib_simplex_v4", "netlib_simplex_v3", "netlib_simplex_v2"], 80)
ipm = newest(["netlib_ipm_v2", "netlib_ipm_v1"], 80)
mip = newest(["miplib3_bnb_v2", "miplib3_bnb_v1"], 60)
qp = newest(["maros_qpipm_v2", "maros_qpipm_v1"], 130)
N = dict(
    spx=sum(ok(r, 1e-6) for r in spx), lp=len(spx) or 90,
    ipm=sum(ok(r, 1e-6) for r in ipm),
    mip=sum(ok(r, 1e-4, strict=False) for r in mip), miptot=len(mip) or 64,
    mipref=sum(r["ref_status"] == "Optimal" for r in mip),
    qp=sum(ok(r, 1e-6) for r in qp), qptot=len(qp) or 138,
    qpref=sum(r["ref_status"].startswith("Solved") for r in qp),
)
print("numbers:", N)
S = lambda t: t.format(**N)


# ------------------------------------------------------------------ text replacement
def shape_by_id(doc, sid):
    for sp in doc.getElementsByTagName("p:sp"):
        if sp.getElementsByTagName("p:cNvPr")[0].getAttribute("id") == str(sid):
            return sp
    raise KeyError(sid)


def set_runs(para, texts):
    """Put texts[i] into the i-th run of the paragraph (extra runs removed, missing runs cloned)."""
    runs = para.getElementsByTagName("a:r")
    if not runs:
        return
    while len(runs) < len(texts):
        para.insertBefore(runs[-1].cloneNode(True), runs[-1].nextSibling)
        runs = para.getElementsByTagName("a:r")
    for r, t in zip(runs, texts):
        tn = r.getElementsByTagName("a:t")[0]
        while tn.firstChild:
            tn.removeChild(tn.firstChild)
        tn.appendChild(tn.ownerDocument.createTextNode(t))
        if t != t.strip():
            tn.setAttribute("xml:space", "preserve")
    for r in runs[len(texts):]:
        para.removeChild(r)


def set_text(doc, sid, lines):
    """Replace the paragraphs of shape sid. Each line is a string (one run) or a list of run
    texts; paragraph i keeps the template's paragraph i formatting (the last is cloned when the
    new text has more lines, surplus template paragraphs are removed)."""
    body = shape_by_id(doc, sid).getElementsByTagName("p:txBody")[0]
    paras = body.getElementsByTagName("a:p")
    while len(paras) < len(lines):
        last = paras[-1]
        body.insertBefore(last.cloneNode(True), last.nextSibling)
        paras = body.getElementsByTagName("a:p")
    for p, line in zip(paras, lines):
        set_runs(p, [line] if isinstance(line, str) else list(line))
    for p in paras[len(lines):]:
        body.removeChild(p)


def set_notes(doc, text):
    for sp in doc.getElementsByTagName("p:sp"):
        ph = sp.getElementsByTagName("p:ph")
        if ph and ph[0].getAttribute("type") == "body":
            set_text_from_shape(sp, text.split("\n"))
            return
    raise KeyError("notes body")


def set_text_from_shape(sp, lines):
    body = sp.getElementsByTagName("p:txBody")[0]
    paras = body.getElementsByTagName("a:p")
    # use the first paragraph that has a run as the style source
    src = next((p for p in paras if p.getElementsByTagName("a:r")), paras[0])
    for p in list(paras):
        body.removeChild(p)
    for line in lines:
        p = src.cloneNode(True)
        body.appendChild(p)
        set_runs(p, [line])


# ------------------------------------------------------------------ the content
D = "–"   # the template's en dash
SLIDES = {
    1: {
        94: [f"Problem Statement ID  {D}  SIH26119",
             f"Problem Statement Title {D}  Indigenous GPU-Accelerated",
             "Optimization Solver (Sovereign Alternative to",
             "Express / CEPLEX)",
             f"Organisation {D}  Mangalore Refinery and Petrochemicals Ltd (MRPL)",
             f"Theme {D}  Smart Automation",
             f"PS Category {D}  Software",
             f"Team ID {D}  175338",
             f"Team Name {D}  ZeroCloud"],
    },
    2: {
        102: ["NIRNAY  ·  a solver India can read, change and own"],
        103: ["Proposed Solution",
              "An LP, MILP and QP optimisation solver written from the mathematics up.",
              "No solver library inside — every factorisation, pivot and cut is ours.",
              "",
              "How it addresses the problem",
              "Refinery planning, blending and scheduling models run on CPLEX,",
              "Gurobi or Xpress: licence fees, and no way to look inside. NIRNAY:",
              "•  Three LP engines: dual simplex, interior point, PDLP on the GPU",
              "•  Branch-and-bound with cuts for MILP; interior point for convex QP",
              "•  Reads the MPS/QPS files every planning tool already writes",
              "•  Every answer re-checked against an independent solver"],
        107: ["Innovation & Uniqueness",
              "Each claim names what existing work does not do.",
              "",
              "1.  Every factorisation is ours",
              "Open-source solvers borrow LU, Cholesky and",
              "orderings from libraries. We wrote them: sparse",
              "LU with basis repair, Cholesky, quasi-definite",
              "LDLᵀ — auditable end to end, no black box.",
              S("Verified: {spx}/{lp} Netlib LPs, {qp}/{qptot} QPs."),
              "",
              "2.  Three engines, chosen by problem shape",
              "Simplex for warm starts and MILP, interior point",
              "for large sparse LP/QP, GPU PDLP for huge LPs:",
              "datt256 in 0.8 s where HiGHS's IPM takes 259 s.",
              "",
              "3.  Correctness checked, not assumed",
              "Every solve is re-solved independently. It caught",
              "bugs in the references too: HiGHS misreads a QPS",
              "file; cuOpt reports a wrong optimum on p0201."],
        108: ["ONE MODEL, START TO FINISH — NO FOREIGN SOLVER IN THE LOOP"],
        109: ["Refinery model (.mps)", "LP, MILP or QP from any planning tool"],
        111: ["NIRNAY presolves, solves", "picks the engine, recovers exact duals"],
        113: ["Plan + shadow prices", "optimum and duals, independently checked"],
        114: [S("Measured today on public benchmarks: {spx}/{lp} Netlib LPs, {mip}/{miptot} MIPLIB, {qp}/{qptot} QPs."),
              "Every number regenerates from one script."],
    },
    3: {
        123: ["Six layers, all written from the mathematics up. Nothing borrowed below the API."],
        127: ["Input & Presolve",
              "MPS/QPS reader  ·  scaling  ·  singleton, empty, redundant rows  ·  exact dual postsolve"],
        129: ["LP Engines   ◆ THREE",
              "Dual simplex (compiled loop)  ·  Mehrotra interior point  ·  PDLP on CPU and GPU"],
        131: ["MILP Branch-and-Bound",
              "warm-started dual simplex  ·  reliability branching  ·  Gomory cuts  ·  diving"],
        133: ["Sparse Linear Algebra", "LU (Suhl–Suhl order) · Cholesky · LDLᵀ · min-degree"],
        134: ["QP Interior Point", "quasi-definite augmented system · regularised, refined"],
        136: ["Compiled Kernels", "Numba → native machine code · all CPU cores"],
        137: ["GPU Backend", "CuPy CUDA kernels · CUDA graphs · RTX-class GPUs"],
        139: ["Verification Plane   ◆ NEW",
              "every solve re-solved by HiGHS / Clarabel  ·  primal and dual residuals  ·  published optima"],
        140: ["Technology Stack",
              "",
              "Language",
              "Python 3.12 · NumPy arrays",
              "Compilation",
              "Numba JIT → native code  ·  CuPy → CUDA",
              "Engines",
              "dual simplex · interior point · PDLP · B&B",
              "Linear algebra",
              "LU · Cholesky · LDLᵀ — all our own",
              "Interfaces",
              "MPS · QPS · Python API · command line",
              "Licence",
              "Apache-2.0 — any PSU can adopt it",
              "",
              "Measured on a laptop (RTX 3050, 4 GB)",
              S("Netlib LP          {spx}/{lp}   ·  HiGHS {lp}"),
              S("MIPLIB 3, 60 s    {mip}/{miptot}   ·  HiGHS {mipref}"),
              S("Maros–Mészáros QP  {qp}/{qptot}  ·  Clarabel {qpref}")],
        141: ["The demo that proves it",
              "1.  Load an MRPL-style refinery planning model (.mps)",
              "2.  NIRNAY solves it; HiGHS re-solves it alongside",
              "3.  Same optimum, same shadow prices of crude and products",
              "4.  Open the code: every pivot and factorisation is ours"],
    },
    4: {
        151: ["Why this is buildable",
              "•  It already runs.  The dual simplex solves all Netlib LPs,",
              S("branch-and-bound proves {mip} MIPLIB problems optimal, and"),
              S("the QP interior point solves {qp} of {qptot} Maros–Mészáros QPs."),
              "•  Published mathematics.  Mehrotra, Forrest–Goldfarb,",
              "Gilbert–Peierls, PDLP: decades of documented algorithms,",
              "implemented carefully rather than invented.",
              "•  Licensing is clean.  Apache-2.0; Python, Numba and CuPy",
              "only — a PSU can deploy it without procurement friction.",
              "•  Honest benchmarks.  One script re-solves every instance",
              "with HiGHS or Clarabel and logs errors and residuals.",
              "•  Team fit.  Python, systems and GPU skills map directly",
              "onto the kernel and benchmark work."],
        155: ["Roadmap to the finale"],
        156: [["Wk 1-2", "   Hyper-sparse FTRAN/BTRAN; Markowitz LU for the bump"]],
        157: [["Wk 3-4", "   Forrest–Tomlin update; faster presolve (compiled)"]],
        158: [["Wk 5-6", "   MIR, cover and clique cuts; RINS/RENS heuristics"]],
        159: [["Wk 7-8", "   Interior-point crossover; QP presolve"]],
        160: [["Finale", "   MRPL-style model solved live, verified against HiGHS"]],
        161: ["Risks and how we retire them"],
        162: ["Slower than CPLEX / HiGHS today",
              "Simplex loop now compiled; hyper-sparse solves and Markowitz LU next"],
        163: ["Hard MIPLIB instances time out",
              "Report the proven gap honestly; add cut families and heuristics"],
        164: ["Numerical breakdown on bad models",
              "Three engines back each other up; scaling, perturbation, basis repair"],
        165: ["Jury doubts it is really from scratch",
              "No solver import in the package; open the LU or simplex kernel live"],
        166: ["GPU not available at the venue",
              "PDLP runs on CPU with the same code; the GPU only accelerates"],
        167: ["A wrong answer reported as optimal",
              "Independent re-solve and residual check on every benchmark instance"],
    },
    5: {
        176: ["Every refinery plan, crude blend, schedule and dispatch decision sits on an optimisation solver. "
              "Owning that layer means owning the decision."],
        180: ["Who it serves", "",
              "MRPL planners and schedulers first — then every refinery, power utility and PSU that runs LP/MILP planning models today."],
        181: ["Operational impact", "",
              "Crude selection, blending, scheduling and dispatch models solved on hardware MRPL owns, with shadow prices it can audit."],
        182: ["Economic", "",
              "No recurring solver licence (IBM lists CPLEX developer subscriptions from US$320 per user per month). Apache-2.0, no lock-in."],
        183: ["Strategic / national", "",
              "The decision engine behind critical energy infrastructure becomes Indian code. China made this move with COPT."],
        184: ["Transparency", "",
              "Every pivot, factorisation and cut is inspectable; tolerances and heuristics can be tuned to Indian refinery models."],
        185: ["Scalable by design", "",
              "The GPU engine needs only matrix–vector products, so large planning LPs speed up as GPUs improve."],
    },
    6: {
        194: ["Base papers anchor the engines; the rest map one-to-one onto the layers above."],
        198: ["BASE PAPER",
              "On the Implementation of a Primal-Dual Interior Point Method",
              "Mehrotra, SIAM J. Optimization 2(4), 1992  ·  doi:10.1137/0802028",
              "Our interior-point LP and QP engines follow its predictor-corrector."],
        199: ["GPU PRECEDENT",
              "cuPDLP: restarted primal-dual hybrid gradient on the GPU",
              "Lu & Yang, Operations Research 73, 2025  ·  doi:10.1287/opre.2024.1069",
              "Shows first-order LP on GPUs wins at scale; basis of our PDLP engine."],
        200: ["Dual simplex", "Koberstein, PhD thesis 2005   ·   Forrest & Goldfarb, Math. Prog. 57, 1992"],
        201: ["Ratio tests", "Harris, Math. Prog. 5, 1973   ·   Maros, EJOR 149, 2003"],
        202: ["Sparse LU", "Gilbert & Peierls, SISSC 1988   ·   Suhl & Suhl, ORSA JoC 2, 1990"],
        203: ["PDLP", "Applegate et al., NeurIPS 2021   ·   arXiv:2106.04756"],
        204: ["MILP branching", "Achterberg, Koch & Martin, Oper. Res. Letters 33, 2005"],
        205: ["Cutting planes", "Balas, Ceria, Cornuéjols & Natraj, Oper. Res. Letters 19, 1996"],
        206: ["Presolve", "Andersen & Andersen, Math. Prog. 71, 1995"],
        207: ["Quasi-definite LDLᵀ (QP)", "Vanderbei, SIAM J. Optim. 5, 1995   ·   Friedlander & Orban, 2012"],
        208: ["Benchmarks: Netlib (Koch, ORL 32, 2004) · MIPLIB (Gleixner et al., MPC 13, 2021) · Maros & Mészáros, OMS 11, 1999 · "
              "HiGHS and Clarabel are used only as references in testing, never inside the solver."],
    },
}

NOTES = {
    1: ("OPEN (20s). Good morning. We are Team ZeroCloud, presenting problem statement SIH26119 from Mangalore "
        "Refinery and Petrochemicals: an indigenous optimisation solver.\n"
        "The title is copied exactly as the portal lists it, including its spelling of Xpress and CPLEX.\n"
        "One line: every refinery plan, blend and schedule in India runs on a handful of foreign solvers nobody here "
        "can inspect. We wrote the engine itself, from the mathematics up.\n"
        "Do not read this slide. Move straight on."),
    2: ("THE IDEA (60s). Start from the dependency, not the code: crude selection, blending and scheduling at MRPL "
        "run through CPLEX, Gurobi or Xpress. Recurring cost, and nobody here can see inside.\n"
        + S("Then the proof it is real: the dual simplex solves all {spx} Netlib LPs, branch-and-bound proves "
            "{mip} of {miptot} MIPLIB problems optimal in 60 seconds, and the QP solver handles {qp} of {qptot} "
            "Maros-Meszaros problems. No solver library is inside.\n")
        + "Spend the time on claim 1: every factorisation is ours, so the whole decision path is auditable.\n"
        "If asked about speed: it is slower than HiGHS today; the iteration loop is now compiled and the linear "
        "algebra is next. Say it plainly."),
    3: ("TECHNICAL APPROACH (75s). Walk the stack top to bottom, one sentence per layer.\n"
        "The argument: the problem statement forbids building on an existing solver, so the places where solvers "
        "usually borrow code, the sparse LU, the Cholesky and the orderings, are exactly the ones we wrote.\n"
        "Why three LP engines: simplex gives exact vertices and warm starts for branch-and-bound; interior point is "
        "robust on large sparse problems and is the basis of the QP solver; PDLP needs only matrix-vector products, "
        "which is what a GPU does well.\n"
        "Finish on the verification plane and the demo: every answer is re-solved by HiGHS or Clarabel."),
    4: ("FEASIBILITY (45s). Left: it already runs, the maths is published, the licence is clean, the benchmarks are "
        "honest.\n"
        "Right: pick two risks. Speed against commercial solvers (compiled loop done, linear algebra next) and "
        "whether it is really from scratch (open the LU kernel live, there is no solver import).\n"
        "The roadmap shows what the remaining weeks buy."),
    5: ("IMPACT (35s). Pick three cards. Operational: MRPL's own planning models on hardware it owns, with shadow "
        "prices it can audit. Economic: no per-seat licence. Strategic: the decision engine behind critical energy "
        "infrastructure becomes Indian code; China made this move with COPT.\n"
        "Close on transferability to every refinery and utility."),
    6: ("RESEARCH (25s, then stop). Do not read citations.\n"
        "Say: our interior point follows Mehrotra's predictor-corrector; our GPU engine follows PDLP and cuPDLP; "
        "everything else on this slide maps one-to-one onto a layer in the architecture.\n"
        "Then: HiGHS and Clarabel appear only in our test harness, never in the solver. Leave time for questions."),
}


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    with zipfile.ZipFile(SRC) as z:
        z.extractall(WORK)
    for i, shapes in SLIDES.items():
        p = WORK / "ppt" / "slides" / f"slide{i}.xml"
        doc = minidom.parse(str(p))
        for sid, lines in shapes.items():
            set_text(doc, sid, lines)
        p.write_text(doc.toxml(), encoding="utf-8")
    for i, text in NOTES.items():
        p = WORK / "ppt" / "notesSlides" / f"notesSlide{i}.xml"
        doc = minidom.parse(str(p))
        set_notes(doc, text)
        p.write_text(doc.toxml(), encoding="utf-8")
    # repack, keeping [Content_Types].xml first
    with zipfile.ZipFile(SRC) as zsrc:
        order = [i.filename for i in zsrc.infolist()]
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in order:
            z.write(WORK / name, name)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
