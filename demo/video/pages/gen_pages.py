"""The designed (non-Studio) scenes of the walkthrough: hook, architecture, roadmap, end card.
Same palette and type as NIRNAY Studio. python gen_pages.py -> c1_hook.html ... c7_end.html"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

CSS = """
@font-face { font-family: Inter; font-weight: 400; src: url(fonts/inter-latin-400-normal.woff2) format("woff2"); }
@font-face { font-family: Inter; font-weight: 500; src: url(fonts/inter-latin-500-normal.woff2) format("woff2"); }
@font-face { font-family: Inter; font-weight: 600; src: url(fonts/inter-latin-600-normal.woff2) format("woff2"); }
@font-face { font-family: Inter; font-weight: 700; src: url(fonts/inter-latin-700-normal.woff2) format("woff2"); }
@font-face { font-family: Inter; font-weight: 800; src: url(fonts/inter-latin-800-normal.woff2) format("woff2"); }
@font-face { font-family: "JetBrains Mono"; font-weight: 400; src: url(fonts/jetbrains-mono-latin-400-normal.woff2) format("woff2"); }
:root { --bg:#0B1020; --bg2:#0F1629; --card:#131B31; --card2:#172039; --line:#243052; --line2:#2E3B63; --ink:#E8ECF6;
  --ink2:#AEB8D0; --muted:#7C88A6; --saffron:#F28C38; --saffron2:#FFB066; --green:#2FBF71; --green2:#7BE0A6;
  --blue:#5B8CFF; --violet:#A27BFF; }
* { box-sizing: border-box; }
html, body { margin: 0; width: 1920px; height: 1080px; overflow: hidden; }
body { background: radial-gradient(1400px 800px at 80% -10%, #1B2A5A 0%, rgba(27,42,90,0) 60%),
  radial-gradient(1000px 700px at -10% 110%, #2A1E3E 0%, rgba(42,30,62,0) 55%), var(--bg);
  color: var(--ink); font: 400 28px/1.35 Inter, sans-serif; -webkit-font-smoothing: antialiased; }
.stage { position: absolute; inset: 0; }
.grid { position: absolute; inset: -200px; background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
  linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px); background-size: 64px 64px; }
.mark { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.mark span { border-radius: 8px; background: var(--line2); }
.mark span:nth-child(1) { background: var(--saffron); } .mark span:nth-child(4) { background: var(--green); }
.card { background: linear-gradient(180deg, var(--card) 0%, #111831 100%); border: 1px solid var(--line); border-radius: 22px; }
.kicker { font: 700 22px Inter; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
.saffron { color: var(--saffron2); } .green { color: var(--green2); } .muted { color: var(--muted); }
"""


def page(title, body, extra_css=""):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>{CSS}{extra_css}</style></head><body><div class="stage"><div class="grid"></div>{body}</div></body></html>"""


HOOK = page("Hook", """
<div id="a" class="scene">
  <div class="big" id="l1"><span>Every</span> <span>barrel</span> <span>a</span> <span>refinery</span> <span>processes</span></div>
  <div class="big dim" id="l2">is decided by an <b class="saffron">optimisation solver.</b></div>
  <div class="chips">
    <div class="chip" id="c1"><i style="background:var(--saffron)"></i>Which crudes to buy</div>
    <div class="chip" id="c2"><i style="background:var(--green)"></i>How to blend them</div>
    <div class="chip" id="c3"><i style="background:var(--blue)"></i>When to run each unit</div>
  </div>
</div>
<div id="b" class="scene">
  <div class="kicker">In India, those solvers are</div>
  <div class="words"><span id="w1">foreign.</span><span id="w2">closed.</span><span id="w3">licensed by the seat.</span></div>
  <div class="brands" id="brands">CPLEX&nbsp;&nbsp;·&nbsp;&nbsp;Gurobi&nbsp;&nbsp;·&nbsp;&nbsp;Xpress</div>
</div>
<div id="c" class="scene">
  <div class="built" id="built">So we built one.</div>
  <div class="wordmark" id="wm"><div class="mark m"><span></span><span></span><span></span><span></span></div>
    <div><div class="name">NIRNAY</div><div class="deva">निर्णय · decision</div></div></div>
  <div class="tagline" id="tag">An indigenous LP · MILP · QP solver, written from the mathematics up</div>
  <div class="sih" id="sih">Smart India Hackathon 2026 · SIH26119 · MRPL</div>
</div>""", """
.scene { position: absolute; left: 170px; right: 170px; top: 0; bottom: 0; display: flex; flex-direction: column; justify-content: center; }
.big { font: 800 92px/1.08 Inter; letter-spacing: -.02em; }
.big.dim { color: var(--ink2); font-weight: 700; margin-top: 18px; }
.big b { font-weight: 800; }
.chips { display: flex; gap: 22px; margin-top: 70px; }
.chip { display: flex; align-items: center; gap: 16px; font: 600 34px Inter; padding: 22px 30px; border-radius: 18px;
  background: var(--card); border: 1px solid var(--line2); box-shadow: 0 20px 50px rgba(0,0,0,.35); }
.chip i { width: 16px; height: 16px; border-radius: 50%; display: block; }
.words { display: flex; flex-direction: column; gap: 6px; margin-top: 24px; }
.words span { font: 800 120px/1.05 Inter; letter-spacing: -.03em; }
.words span:nth-child(3) { color: var(--saffron2); }
.brands { margin-top: 44px; font: 600 34px Inter; color: var(--muted); letter-spacing: .04em; }
.built { font: 700 58px Inter; color: var(--ink2); }
.wordmark { display: flex; align-items: center; gap: 40px; margin-top: 40px; }
.mark.m { width: 150px; height: 150px; gap: 14px; } .mark.m span { border-radius: 14px; }
.name { font: 800 150px/1 Inter; letter-spacing: .06em; }
.deva { font: 500 38px Inter; color: var(--muted); margin-top: 8px; }
.tagline { margin-top: 44px; font: 600 40px Inter; color: var(--ink); }
.sih { margin-top: 16px; font: 500 30px Inter; color: var(--muted); }
""")

ARCH = page("Architecture", """
<div class="wrap">
  <div class="kicker" id="k">Under the hood</div>
  <div class="row" id="rIn"><div class="blk slim">MPS · QPS files</div><div class="blk slim">Python API</div><div class="blk slim">Command line</div></div>
  <div class="row" id="rPre"><div class="blk wide pre">Presolve with exact dual postsolve · power-of-two scaling</div></div>
  <div class="row" id="rTop">
    <div class="blk top" id="milp"><b>MILP</b><span>branch-and-bound · reliability branching · Gomory cuts · diving</span></div>
    <div class="blk top qp" id="qp"><b>QP</b><span>interior point on the quasi-definite system</span></div>
  </div>
  <div class="row engines" id="rEng">
    <div class="eng" id="e1"><div class="en">Dual simplex</div><div class="why">exact answers · warm starts for branch-and-bound</div><div class="tag">default</div></div>
    <div class="eng" id="e2"><div class="en">Interior point</div><div class="why">large sparse models · Mehrotra predictor–corrector</div><div class="tag">robust</div></div>
    <div class="eng gpu" id="e3"><div class="en">PDLP</div><div class="why">very large LPs · only matrix–vector products</div><div class="tag">GPU · CUDA</div></div>
  </div>
  <div class="row" id="rLa">
    <div class="la-h">Linear algebra</div>
    <div class="la" id="la1">Sparse LU</div><div class="la" id="la2">Cholesky</div><div class="la" id="la3">LDLᵀ</div><div class="la" id="la4">Min-degree ordering</div>
  </div>
  <div class="row" id="rK"><div class="blk wide kern">Compiled to native code with Numba · CUDA kernels via CuPy</div></div>
  <div class="stamp" id="stamp">written from scratch · no solver library inside</div>
</div>""", """
.wrap { position: absolute; left: 160px; right: 160px; top: 70px; bottom: 60px; display: flex; flex-direction: column; gap: 18px; }
.row { display: flex; gap: 18px; }
.blk { flex: 1; background: var(--card); border: 1px solid var(--line2); border-radius: 16px; padding: 18px 24px; font: 600 28px Inter; text-align: center; }
.blk.slim { color: var(--ink2); }
.blk.pre { background: rgba(242,140,56,.08); border-color: rgba(242,140,56,.35); color: var(--saffron2); }
.blk.top { display: flex; align-items: center; gap: 18px; text-align: left; background: rgba(47,191,113,.08); border-color: rgba(47,191,113,.4); }
.blk.top b { font: 800 34px Inter; color: var(--green2); }
.blk.top span { font: 500 24px Inter; color: var(--ink2); }
.blk.qp { background: rgba(162,123,255,.08); border-color: rgba(162,123,255,.4); } .blk.qp b { color: #C9B3FF; }
.engines .eng { flex: 1; position: relative; background: linear-gradient(180deg, #1A2442, #131B31); border: 1px solid var(--line2);
  border-radius: 20px; padding: 26px 28px 24px; box-shadow: 0 20px 50px rgba(0,0,0,.35); }
.eng .en { font: 800 44px Inter; }
.eng .why { margin-top: 10px; font: 500 25px/1.35 Inter; color: var(--ink2); }
.eng .tag { position: absolute; top: 24px; right: 24px; font: 700 18px Inter; letter-spacing: .08em; text-transform: uppercase;
  color: var(--muted); border: 1px solid var(--line2); padding: 6px 12px; border-radius: 999px; }
.eng.gpu { border-color: rgba(242,140,56,.55); } .eng.gpu .tag { color: var(--saffron2); border-color: rgba(242,140,56,.5); }
.la-h { font: 700 22px Inter; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); align-self: center; width: 250px; }
.la { flex: 1; text-align: center; font: 700 30px Inter; padding: 18px; border-radius: 16px; background: var(--bg2); border: 1px solid var(--line2); color: var(--ink2); }
.blk.kern { color: var(--muted); font-weight: 500; }
.stamp { position: absolute; right: 0; bottom: -8px; font: 700 26px Inter; color: var(--green2); letter-spacing: .02em; }
""")

NEXT = page("Roadmap", """
<div class="cols">
  <div class="left">
    <div class="kicker" id="k1">Next</div>
    <div class="step" id="s1"><div class="n">1</div><div><div class="t">Close the speed gap</div>
      <div class="d">hyper-sparse linear algebra · a Markowitz LU factorisation</div><div class="now">today: 2.6× slower than HiGHS on Netlib</div></div></div>
    <div class="step" id="s2"><div class="n">2</div><div><div class="t">Harder integer problems</div>
      <div class="d">MIR, cover and clique cuts · RINS / RENS heuristics</div><div class="now">today: 33 of 64 MIPLIB proven optimal</div></div></div>
    <div class="step" id="s3"><div class="n">3</div><div><div class="t">Live at the finale</div>
      <div class="d">an MRPL planning model, solved and verified against HiGHS on stage</div></div></div>
  </div>
  <div class="right card" id="mrpl">
    <div class="kicker">For MRPL</div>
    <div class="head">Planning software it owns</div>
    <div class="ben" id="b1"><i></i><div><b>No per-seat licence</b><span>Apache-2.0, runs on hardware MRPL already has</span></div></div>
    <div class="ben" id="b2"><i></i><div><b>Code it can audit</b><span>every pivot, factorisation and cut is readable</span></div></div>
    <div class="ben" id="b3"><i></i><div><b>Shadow prices it can trust</b><span>exact duals, checked against an independent solver</span></div></div>
  </div>
</div>""", """
.cols { position: absolute; left: 150px; right: 150px; top: 0; bottom: 0; display: grid; grid-template-columns: 1.1fr 1fr; gap: 80px; align-items: center; }
.step { display: flex; gap: 28px; margin-top: 44px; }
.step .n { width: 64px; height: 64px; flex: none; border-radius: 18px; display: grid; place-items: center; font: 800 32px Inter;
  background: rgba(242,140,56,.12); border: 1px solid rgba(242,140,56,.45); color: var(--saffron2); }
.step .t { font: 800 44px Inter; }
.step .d { font: 500 28px/1.35 Inter; color: var(--ink2); margin-top: 6px; }
.step .now { font: 600 22px Inter; color: var(--muted); margin-top: 8px; }
.right { padding: 48px 50px 40px; box-shadow: 0 30px 80px rgba(0,0,0,.4); }
.right .head { font: 800 50px/1.1 Inter; margin: 14px 0 22px; }
.ben { display: flex; gap: 22px; align-items: flex-start; padding: 22px 0; border-top: 1px dashed var(--line2); }
.ben i { width: 18px; height: 18px; border-radius: 50%; background: var(--green); margin-top: 12px; flex: none; box-shadow: 0 0 16px var(--green); }
.ben b { display: block; font: 700 36px Inter; }
.ben span { display: block; font: 500 25px Inter; color: var(--ink2); margin-top: 4px; }
""")

END = page("End", """
<div class="center">
  <div class="wordmark" id="wm"><div class="mark m"><span></span><span></span><span></span><span></span></div><div class="name">NIRNAY</div></div>
  <div class="line" id="l1">Smart India Hackathon 2026 · SIH26119 · MRPL</div>
  <div class="line strong" id="l2">Team ZeroCloud · Team ID 175338</div>
  <div class="repo" id="l3"><span class="gh">github.com/darshvn/nirnay</span></div>
  <div class="small" id="l4">Solver, benchmarks, 59-page report and case studies — open under Apache-2.0</div>
</div>
<div class="credit" id="cr">Voice: elevenlabs.io</div>""", """
.center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
.wordmark { display: flex; align-items: center; gap: 36px; }
.mark.m { width: 120px; height: 120px; gap: 12px; } .mark.m span { border-radius: 12px; }
.name { font: 800 140px/1 Inter; letter-spacing: .06em; }
.line { margin-top: 44px; font: 600 36px Inter; color: var(--ink2); }
.line.strong { margin-top: 14px; color: var(--ink); font-weight: 700; }
.repo { margin-top: 48px; }
.gh { font: 600 40px "JetBrains Mono", monospace; color: var(--saffron2); padding: 18px 34px; border-radius: 18px;
  background: rgba(242,140,56,.08); border: 1px solid rgba(242,140,56,.45); }
.small { margin-top: 34px; font: 500 26px Inter; color: var(--muted); }
.credit { position: absolute; right: 60px; bottom: 44px; font: 500 20px Inter; color: var(--muted); }
""")

for name, html in [("c1_hook", HOOK), ("c5b_arch", ARCH), ("c6b_next", NEXT), ("c7_end", END)]:
    (HERE / f"{name}.html").write_text(html, encoding="utf-8")
    print("wrote", name)
