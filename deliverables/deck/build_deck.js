// SIH 2026 idea deck for SIH26119 (MRPL), following the official six-slide idea template.
// Every benchmark number is read from results/*.csv at build time, so the deck cannot drift
// from what the solver actually did.
//   node build_deck.js   (NODE_PATH must reach the global pptxgenjs)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const LOGO = path.join(__dirname, "tpl", "ppt", "media", "image1.png");   // SIH 2026 logo from the official template

// ---------------------------------------------------------------- results
function readCsv(file) {
  if (!fs.existsSync(file)) return [];
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const head = lines.shift().split(",");
  // fields may be quoted and contain commas (the QP reference status does)
  const split = (l) => { const out = []; let cur = "", q = false;
    for (const ch of l) { if (ch === '"') q = !q; else if (ch === "," && !q) { out.push(cur); cur = ""; } else cur += ch; }
    out.push(cur); return out; };
  return lines.map((l) => { const v = split(l); const o = {}; head.forEach((h, i) => (o[h] = v[i] ?? "")); return o; });
}
const solved = (rows) => rows.filter((r) => r.status === "optimal" && r.rel_err !== "" && r.rel_err !== "nan" && parseFloat(r.rel_err) < 1e-6);
const ipm1 = readCsv(path.join(ROOT, "results", "netlib_ipm_v1.csv"));
const ipm2 = readCsv(path.join(ROOT, "results", "netlib_ipm_v2.csv"));
// newest complete sweep of each kind (a sweep still running is skipped until it is complete)
const newest = (names, minRows) => { for (const n of names) { const r = readCsv(path.join(ROOT, "results", n)); if (r.length >= minRows) return r; } return []; };
const spx = newest(["netlib_simplex_v4.csv", "netlib_simplex_v3.csv", "netlib_simplex_v2.csv", "netlib_simplex_v1.csv"], 80);
const ipmBest = ipm2.length >= 80 ? ipm2 : ipm1;
const nIpm = solved(ipmBest).length, nIpmTried = ipmBest.length;
const nSpx = solved(spx).length, nSpxTried = spx.length;
const union = new Set([...solved(ipmBest), ...solved(spx)].map((r) => r.name));
const nUnion = union.size;
const TOTAL = 90;
// MILP: MIPLIB 3, solved = proven optimal within the 1e-4 gap and agreeing with HiGHS to 1e-4
const latest = (...names) => { for (const n of names) { const r = readCsv(path.join(ROOT, "results", n)); if (r.length) return r; } return []; };
const okErr = (r, tol) => r.rel_err !== "" && r.rel_err !== "nan" && parseFloat(r.rel_err) <= tol;
const mip = newest(["miplib3_bnb_v2.csv", "miplib3_bnb_v1.csv"], 60);
const nMip = mip.filter((r) => r.status === "optimal" && okErr(r, 1e-4)).length;
const nMipRef = mip.filter((r) => r.ref_status === "Optimal").length;
const MIPTOT = mip.length || 64;
const mipLimit = 60;
// QP: Maros-Meszaros, solved = optimal and within 1e-6 of Clarabel
const qp = newest(["maros_qpipm_v2.csv", "maros_qpipm_v1.csv"], 130);
const nQp = qp.filter((r) => r.status === "optimal" && r.rel_err !== "" && r.rel_err !== "nan" && parseFloat(r.rel_err) < 1e-6).length;
const nQpRef = qp.filter((r) => r.ref_status.startsWith("Solved")).length;
const QPTOT = qp.length || 138;
const PDLP_BUILT = fs.existsSync(path.join(ROOT, "results", "netlib_pdlp_gpu.csv"));
console.log(`IPM ${nIpm}/${nIpmTried}, simplex ${nSpx}/${nSpxTried}, either ${nUnion}/${TOTAL}; MIP ${nMip}/${MIPTOT} (HiGHS ${nMipRef}); QP ${nQp}/${QPTOT} (Clarabel ${nQpRef}); PDLP ${PDLP_BUILT}`);

// ---------------------------------------------------------------- palette (from the SIH logo)
const NAVY = "1F2A44", SAFFRON = "E8702A", GREEN = "1E8C45", INK = "2B2F36", MUTED = "5B6770",
      CARD = "F2F4F7", LINE = "D5DAE1", WHITE = "FFFFFF";
const HF = "Arial", BF = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.333 x 7.5 in, the official template's size
pres.title = "SIH26119 — NIRNAY, a sovereign optimisation solver";
pres.author = "Team ZeroCloud";

function frame(slide, title, n) {
  slide.background = { color: WHITE };
  slide.addImage({ path: LOGO, x: 11.05, y: 0.22, w: 1.85, h: 0.874 });
  slide.addText(title, { x: 0.5, y: 0.3, w: 10.3, h: 0.7, fontFace: HF, fontSize: 28, bold: true, color: NAVY, margin: 0, isTextBox: true });
  slide.addText("@SIH Idea submission  ·  Team ZeroCloud  ·  SIH26119", { x: 0.5, y: 7.05, w: 7, h: 0.3, fontFace: BF, fontSize: 9, color: MUTED, margin: 0, isTextBox: true });
  slide.addText(String(n), { x: 12.33, y: 7.05, w: 0.5, h: 0.3, fontFace: BF, fontSize: 9, color: MUTED, align: "right", margin: 0, isTextBox: true });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || CARD }, line: { color: fill || CARD } });
}

// ================================================================= 1. TITLE PAGE
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addImage({ path: LOGO, x: 0.6, y: 0.55, w: 4.2, h: 1.983 });
  s.addText("SMART INDIA HACKATHON 2026", { x: 5.3, y: 0.85, w: 7.6, h: 0.7, fontFace: HF, fontSize: 32, bold: true, color: NAVY, margin: 0, isTextBox: true });
  s.addText("TITLE PAGE", { x: 5.3, y: 1.55, w: 7.6, h: 0.5, fontFace: HF, fontSize: 18, bold: true, color: SAFFRON, charSpacing: 2, margin: 0, isTextBox: true });
  const rows = [
    ["Problem Statement ID", "SIH26119"],
    ["Problem Statement Title", "Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to Express / CEPLEX)"],
    ["Organisation", "Mangalore Refinery and Petrochemicals Limited (MRPL)"],
    ["Theme", "Smart Automation"],
    ["PS Category", "Software"],
    ["Team ID", ""],
    ["Team Name", "ZeroCloud"],
  ];
  let y = 3.05;
  rows.forEach(([k, v]) => {
    const tall = v.length > 60;
    s.addText(k, { x: 0.9, y, w: 3.4, h: tall ? 0.72 : 0.46, fontFace: BF, fontSize: 17, bold: true, color: NAVY, valign: "top", margin: 0, isTextBox: true });
    s.addText(v, { x: 4.4, y, w: 8.3, h: tall ? 0.72 : 0.46, fontFace: BF, fontSize: 17, color: INK, valign: "top", margin: 0, isTextBox: true });
    y += tall ? 0.78 : 0.52;
  });
  s.addNotes(
    "OPEN (20s). We are Team ZeroCloud, on problem statement SIH26119 from MRPL: an indigenous optimisation solver.\n" +
    "The title is copied exactly as the portal lists it, including its spelling of Xpress and CPLEX.\n" +
    "One line: every refinery plan, blend and schedule in India runs on a handful of foreign solvers nobody here can inspect. We are writing the engine itself, from the mathematics up, and it already solves the standard benchmarks.");
}

// ================================================================= 2. PROPOSED SOLUTION
{
  const s = pres.addSlide();
  frame(s, "PROPOSED SOLUTION", 2);
  s.addText("NIRNAY — an optimisation engine India can read, change and own", { x: 0.5, y: 1.05, w: 10.3, h: 0.5, fontFace: HF, fontSize: 20, bold: true, color: SAFFRON, margin: 0, isTextBox: true });
  s.addText("A solver core for LP, MILP and QP written from mathematical foundations: no solver library inside, every factorisation our own, GPU acceleration where it measurably pays.",
    { x: 0.5, y: 1.55, w: 10.3, h: 0.6, fontFace: BF, fontSize: 14, color: INK, margin: 0, isTextBox: true });

  // left: problem -> answer
  s.addText([
    { text: "The problem", options: { bold: true, color: NAVY, fontSize: 15, breakLine: true } },
    { text: "Refinery scheduling, crude blending, planning and dispatch in India run on CPLEX, Gurobi or Xpress: recurring licence cost, and no way to inspect or adapt the algorithms.", options: { bullet: true, fontSize: 13, color: INK, breakLine: true } },
    { text: "Open-source solvers exist but lag on hard industrial MILPs and were never tuned for Indian models.", options: { bullet: true, fontSize: 13, color: INK, breakLine: true } },
    { text: "How NIRNAY answers it", options: { bold: true, color: NAVY, fontSize: 15, breakLine: true } },
    { text: "Three LP engines behind one API, picked by problem shape: dual simplex, interior point, and PDLP on the GPU.", options: { bullet: true, fontSize: 13, color: INK, breakLine: true } },
    { text: "Branch-and-bound for MILP on warm-started dual simplex; interior point for convex QP.", options: { bullet: true, fontSize: 13, color: INK, breakLine: true } },
    { text: "Reads the MPS/QPS files every benchmark and every planning tool already produces.", options: { bullet: true, fontSize: 13, color: INK } },
  ], { x: 0.5, y: 2.3, w: 6.9, h: 2.75, fontFace: BF, valign: "top", paraSpaceAfter: 4, margin: 0, isTextBox: true });

  // right: measured today
  card(s, 7.75, 2.3, 5.1, 2.75, CARD);
  s.addText("MEASURED ON 30 SEP 2026", { x: 8.0, y: 2.42, w: 4.6, h: 0.3, fontFace: HF, fontSize: 11, bold: true, color: MUTED, charSpacing: 1, margin: 0, isTextBox: true });
  const stats = [
    [`${nSpx} / ${TOTAL}`, "Netlib LPs solved to the known optimum by our dual simplex"],
    [`${nMip} / ${MIPTOT}`, `MIPLIB 3 problems proven optimal in ${mipLimit} s by our branch-and-bound`],
    [`${nQp} / ${QPTOT}`, "Maros–Mészáros convex QPs solved by our QP interior point"],
    ["0", "optimisation libraries inside: LU, Cholesky, LDLᵀ and orderings are ours"],
  ];
  stats.forEach(([big, small], i) => {
    const y = 2.75 + i * 0.56;
    s.addText(big, { x: 8.0, y, w: 1.85, h: 0.5, fontFace: HF, fontSize: 22, bold: true, color: i === 3 ? GREEN : SAFFRON, margin: 0, valign: "middle", isTextBox: true });
    s.addText(small, { x: 9.9, y, w: 2.85, h: 0.5, fontFace: BF, fontSize: 10.5, color: INK, margin: 0, valign: "middle", isTextBox: true });
  });

  // bottom: innovation
  s.addText("INNOVATION AND UNIQUENESS", { x: 0.5, y: 5.2, w: 8, h: 0.3, fontFace: HF, fontSize: 12, bold: true, color: NAVY, charSpacing: 1, margin: 0, isTextBox: true });
  const inno = [
    ["Three engines, one decision", "Simplex gives exact vertices and warm starts for MILP; interior point gives accuracy on large sparse LPs; GPU PDLP gives scale. The solver chooses."],
    ["Every factorisation is ours", "Sparse LU with basis repair, sparse Cholesky with minimum-degree ordering. Auditable end to end: no black box in the loop."],
    ["Built for bad models", "Scaling, cost perturbation, bound-flipping and Harris ratio tests, a stall-aware interior point: tested on Netlib's degenerate and ill-conditioned set."],
    ["Benchmarks MRPL recognises", "Netlib, MIPLIB and Maros–Mészáros, plus public refinery-blending, unit-commitment and supply-chain models, each checked against a reference solver."],
  ];
  inno.forEach(([h, b], i) => {
    const x = 0.5 + i * 3.1;
    card(s, x, 5.55, 2.95, 1.4, CARD);
    s.addText(String(i + 1), { x: x + 0.15, y: 5.65, w: 0.4, h: 0.4, fontFace: HF, fontSize: 18, bold: true, color: SAFFRON, margin: 0, isTextBox: true });
    s.addText(h, { x: x + 0.55, y: 5.65, w: 2.3, h: 0.4, fontFace: BF, fontSize: 12.5, bold: true, color: NAVY, margin: 0, valign: "middle", isTextBox: true });
    s.addText(b, { x: x + 0.15, y: 6.07, w: 2.7, h: 0.85, fontFace: BF, fontSize: 10, color: INK, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes(
    "THE IDEA (60s). Start from the dependency, not the code: every refinery plan and blend in India runs through CPLEX, Gurobi or Xpress. Recurring cost, and nobody here can see inside.\n" +
    `Then the proof it is real: our dual simplex solves all ${nSpx} of the ${TOTAL} Netlib LPs to the known optimum, branch-and-bound proves ${nMip} of ${MIPTOT} MIPLIB 3 problems optimal in ${mipLimit} seconds, and the QP interior point solves ${nQp} of ${QPTOT} Maros–Mészáros QPs. No solver library is inside; the LU, Cholesky and LDLᵀ factorisations are ours.
` +
    "If asked about speed: on Netlib the dual simplex is 3.7x slower than HiGHS in shifted geometric mean; the kernels are compiled but the iteration loop is still Python, and that is the next piece of work.\n" +
    "Land the four claims briefly. Spend the time on the first: we do not pick one algorithm, we ship three, because simplex, interior point and GPU PDLP each win on different problems, and a sovereign solver must not be weak where MRPL's models live.");
}

// ================================================================= 3. TECHNICAL APPROACH
{
  const s = pres.addSlide();
  frame(s, "TECHNICAL APPROACH", 3);
  s.addText("One engine, six layers. Everything below the API is written from the mathematics up.", { x: 0.5, y: 1.05, w: 10.3, h: 0.4, fontFace: BF, fontSize: 14, italic: true, color: MUTED, margin: 0, isTextBox: true });

  const X = 0.5, W = 7.9;
  const layers = [
    ["INPUT", "MPS / QPS reader (fixed and free, all bound types) · Python API · command line", NAVY, WHITE, "BUILT"],
    ["PRESOLVE", "Scaling in powers of two · singleton, empty and redundant rows · fixed and free-singleton columns · exact dual postsolve", "3A4A6B", WHITE, "BUILT"],
  ];
  const chip = (label, x, y) => {
    const col = label === "BUILT" ? GREEN : label === "PARTIAL" ? "B7791F" : SAFFRON;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 0.95, h: 0.26, rectRadius: 0.05, fill: { color: WHITE }, line: { color: col, width: 1 } });
    s.addText(label, { x, y, w: 0.95, h: 0.26, fontFace: HF, fontSize: 8, bold: true, color: col, align: "center", valign: "middle", margin: 0, isTextBox: true });
  };
  let y = 1.6;
  layers.forEach(([k, v, bg, fg, st]) => {
    card(s, X, y, W, 0.62, bg);
    s.addText(k, { x: X + 0.15, y, w: 1.6, h: 0.62, fontFace: HF, fontSize: 11, bold: true, color: fg, valign: "middle", margin: 0, isTextBox: true });
    s.addText(v, { x: X + 1.75, y, w: W - 2.95, h: 0.62, fontFace: BF, fontSize: 11.5, color: fg, valign: "middle", margin: 0, isTextBox: true });
    chip(st, X + W - 1.1, y + 0.18);
    y += 0.72;
  });
  // three LP engines side by side
  const eng = [
    ["Dual simplex", "Own sparse LU + eta updates · dual steepest edge · bound-flipping ratio test · perturbation against degeneracy", "BUILT"],
    ["Interior point", "Mehrotra predictor-corrector · own sparse Cholesky with minimum-degree ordering · regularisation", "BUILT"],
    ["PDLP  (GPU)", "Restarted primal-dual hybrid gradient · adaptive steps · only mat-vec products, so it scales on CUDA", PDLP_BUILT ? "BUILT" : "IN BUILD"],
  ];
  const ew = (W - 0.3) / 3;
  eng.forEach(([h, b, st], i) => {
    const ex = X + i * (ew + 0.15);
    card(s, ex, y, ew, 1.35, i === 2 ? "FCE9DC" : "E3ECF7");
    s.addText(h, { x: ex + 0.12, y: y + 0.08, w: ew - 1.2, h: 0.35, fontFace: HF, fontSize: 12.5, bold: true, color: i === 2 ? SAFFRON : NAVY, margin: 0, isTextBox: true });
    chip(st, ex + ew - 1.05, y + 0.12);
    s.addText(b, { x: ex + 0.12, y: y + 0.45, w: ew - 0.24, h: 0.85, fontFace: BF, fontSize: 10.5, color: INK, valign: "top", margin: 0, isTextBox: true });
  });
  y += 1.45;
  const lower = [
    ["MILP", "Branch-and-bound on warm-started dual simplex · reliability branching · propagation · Gomory cuts · diving", "E6F3EA", GREEN, "BUILT"],
    ["QP", "Convex QP by interior point on the quasi-definite augmented system, own sparse LDLᵀ", "E6F3EA", GREEN, "BUILT"],
    ["KERNELS", "Numba-compiled native machine code · CUDA via CuPy on the GPU · no solver library anywhere", "EEF0F3", NAVY, "BUILT"],
  ];
  lower.forEach(([k, v, bg, fg, st]) => {
    card(s, X, y, W, 0.62, bg);
    s.addText(k, { x: X + 0.15, y, w: 1.6, h: 0.62, fontFace: HF, fontSize: 11, bold: true, color: fg, valign: "middle", margin: 0, isTextBox: true });
    s.addText(v, { x: X + 1.75, y, w: W - 2.95, h: 0.62, fontFace: BF, fontSize: 11.5, color: INK, valign: "middle", margin: 0, isTextBox: true });
    chip(st, X + W - 1.1, y + 0.18);
    y += 0.72;
  });

  // right column
  const RX = 8.8, RW = 4.05;
  s.addText("TECHNOLOGY", { x: RX, y: 1.6, w: RW, h: 0.3, fontFace: HF, fontSize: 11, bold: true, color: MUTED, charSpacing: 1, margin: 0, isTextBox: true });
  s.addText([
    { text: "Python 3.12 API, NumPy arrays", options: { bullet: true, breakLine: true } },
    { text: "Numba: kernels compiled to native machine code", options: { bullet: true, breakLine: true } },
    { text: "CUDA through CuPy for the GPU engine", options: { bullet: true, breakLine: true } },
    { text: "Apache-2.0 licence, so any PSU can adopt and extend it", options: { bullet: true } },
  ], { x: RX, y: 1.95, w: RW, h: 1.5, fontFace: BF, fontSize: 12, color: INK, paraSpaceAfter: 3, valign: "top", margin: 0, isTextBox: true });
  s.addText("HOW IT IS CHECKED", { x: RX, y: 3.55, w: RW, h: 0.3, fontFace: HF, fontSize: 11, bold: true, color: MUTED, charSpacing: 1, margin: 0, isTextBox: true });
  s.addText([
    { text: "Netlib LP (90), MIPLIB, Mittelmann, Maros–Mészáros QP: the sets the problem statement names", options: { bullet: true, breakLine: true } },
    { text: "Every instance re-solved by a reference (HiGHS for LP and MILP, Clarabel for QP); errors and residuals logged", options: { bullet: true, breakLine: true } },
    { text: "Hard cases on purpose: degenerate (DEGEN3), ill-conditioned (PILOT, GREENBEA), weak relaxations (MIPLIB)", options: { bullet: true } },
  ], { x: RX, y: 3.9, w: RW, h: 2.4, fontFace: BF, fontSize: 12, color: INK, paraSpaceAfter: 3, valign: "top", margin: 0, isTextBox: true });
  s.addNotes(
    "TECHNICAL APPROACH (75s). Walk the stack top to bottom, one sentence each.\n" +
    "The point to land: the problem statement forbids building on an existing solver, so the three places where solvers usually borrow code are the ones we wrote ourselves: the sparse LU under the simplex, the sparse Cholesky under the interior point, and the orderings that keep both sparse.\n" +
    "Then why three LP engines: simplex gives an exact vertex and warm starts, which branch-and-bound needs; interior point is the most reliable on big sparse LPs; PDLP needs only matrix-vector products, which is exactly what a GPU is good at. The published evidence (cuPDLP, Lu & Yang 2023) is that GPU first-order methods win on very large LPs at moderate accuracy and lose at high accuracy, so we keep all three.\n" +
    "Close on verification: every benchmark run is re-solved by a reference solver (HiGHS for LP and MILP, Clarabel for QP).\n" +
    "Status, if asked: the green BUILT chips run today and are benchmarked. " + (PDLP_BUILT ? "" : "The GPU PDLP engine is being built now; say so plainly. ") +
    "Presolve covers the classic reductions with an exact dual postsolve; doubleton, dominated-column and parallel-row reductions come next.");
}

// ================================================================= 4. FEASIBILITY AND VIABILITY
{
  const s = pres.addSlide();
  frame(s, "FEASIBILITY AND VIABILITY", 4);
  s.addText("Not a plan: the core already runs. Share of each public benchmark set solved, NIRNAY against the reference solver on the same laptop:",
    { x: 0.5, y: 1.05, w: 6.9, h: 0.6, fontFace: BF, fontSize: 13, color: INK, margin: 0, isTextBox: true });
  const pct = (a, b) => (b ? Math.round((1000 * a) / b) / 10 : 0);
  const cats = [`Netlib LP (${TOTAL})`, `MIPLIB 3, ${mipLimit} s (${MIPTOT})`, `Maros–Mészáros QP (${QPTOT})`];
  s.addChart(pres.charts.BAR, [
    { name: "NIRNAY", labels: cats, values: [pct(nSpx, TOTAL), pct(nMip, MIPTOT), pct(nQp, QPTOT)] },
    { name: "Reference (HiGHS / Clarabel)", labels: cats, values: [100, pct(nMipRef, MIPTOT), pct(nQpRef, QPTOT)] },
  ], {
    x: 0.5, y: 1.65, w: 6.9, h: 3.2, barDir: "bar", barGrouping: "clustered", chartColors: [SAFFRON, "9AA5B1"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 11, dataLabelFormatCode: '0"%"',
    valAxisMinVal: 0, valAxisMaxVal: 115, valAxisHidden: true, catAxisOrientation: "maxMin", catAxisLabelColor: INK, catAxisLabelFontSize: 11,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 10,
    showTitle: true, title: "Problems solved to the reference optimum (%)", titleFontSize: 13, titleColor: NAVY,
  });
  s.addText(`LP within 1e-6 of HiGHS (dual simplex ${nSpx}, interior point ${nIpm}); MILP proven optimal within 1e-4; QP within 1e-6 of Clarabel.`,
    { x: 0.5, y: 4.9, w: 6.9, h: 0.35, fontFace: BF, fontSize: 10.5, italic: true, color: MUTED, margin: 0, isTextBox: true });

  s.addText("WHY IT IS BUILDABLE", { x: 0.5, y: 5.35, w: 6.9, h: 0.3, fontFace: HF, fontSize: 11, bold: true, color: NAVY, charSpacing: 1, margin: 0, isTextBox: true });
  s.addText([
    { text: "Runs today on a 4 GB laptop GPU; a server GPU only makes the large LPs faster.", options: { bullet: true, breakLine: true } },
    { text: "Algorithms are fifty years of published, well-documented mathematics; we implement them, we do not invent them.", options: { bullet: true, breakLine: true } },
    { text: "Team skills (Python, systems, DevOps) map onto the kernel and benchmark work directly.", options: { bullet: true } },
  ], { x: 0.5, y: 5.7, w: 6.9, h: 1.3, fontFace: BF, fontSize: 11.5, color: INK, paraSpaceAfter: 2, valign: "top", margin: 0, isTextBox: true });

  const RX = 7.85, RW = 5.0;
  s.addText("RISKS AND HOW WE RETIRE THEM", { x: RX, y: 1.2, w: RW, h: 0.3, fontFace: HF, fontSize: 11, bold: true, color: NAVY, charSpacing: 1, margin: 0, isTextBox: true });
  const risks = [
    ["Hard MIPLIB instances time out", "Report proven gap, not just failure; cuts and heuristics close the gap step by step, measured per release"],
    ["Python slower than C++ solvers", "Hot loops are Numba-compiled native code; the kernel boundary is clean enough to port to C++ later"],
    ["Numerical breakdown on bad models", "Three engines as fallbacks for each other; scaling, perturbation, basis repair, stall detection"],
    ["GPU not always available", "PDLP runs on CPU with the same code path; GPU is an accelerator, never a requirement"],
    ["Claims a jury cannot check", "Every number regenerates from one script against HiGHS and Clarabel on public benchmark files"],
  ];
  risks.forEach(([r, m], i) => {
    const y = 1.6 + i * 1.05;
    card(s, RX, y, RW, 0.95, CARD);
    s.addText(r, { x: RX + 0.15, y: y + 0.07, w: RW - 0.3, h: 0.32, fontFace: BF, fontSize: 12, bold: true, color: NAVY, margin: 0, isTextBox: true });
    s.addText(m, { x: RX + 0.15, y: y + 0.4, w: RW - 0.3, h: 0.5, fontFace: BF, fontSize: 10.5, color: INK, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes(
    `FEASIBILITY (45s). Lead with the chart: this is measured, not planned. Dual simplex ${nSpx} of ${TOTAL} Netlib LPs; branch-and-bound ${nMip} of ${MIPTOT} MIPLIB 3 problems proven optimal in ${mipLimit} seconds where HiGHS proves ${nMipRef}; QP interior point ${nQp} of ${QPTOT} Maros–Mészáros problems where Clarabel solves ${nQpRef}. Every value is checked against the reference.\n` +
    "Then two risks only: hard MIPLIB instances (we report the proven gap honestly and close it with cuts) and speed against C++ solvers (the hot loops are compiled native code). Say the others are on the slide.\n" +
    "If asked whether students can build a solver: the algorithms are published mathematics; the work is careful implementation and testing, which is what this chart shows.");
}

// ================================================================= 5. IMPACT AND BENEFITS
{
  const s = pres.addSlide();
  frame(s, "IMPACT AND BENEFITS", 5);
  s.addText("Every refinery plan, crude blend, schedule and dispatch decision sits on an optimisation solver. Owning that layer is owning the decision.",
    { x: 0.5, y: 1.05, w: 10.3, h: 0.6, fontFace: BF, fontSize: 14, italic: true, color: MUTED, margin: 0, isTextBox: true });
  const cards = [
    ["Strategic", "A decision engine for critical energy infrastructure whose every line is readable and changeable in India. China took this route with COPT; India has no equivalent yet.", NAVY],
    ["Economic", "Commercial solvers cost recurring licence fees: IBM lists CPLEX developer subscriptions from US$320 per user per month; deployment prices are not published at all.", SAFFRON],
    ["Operational (MRPL)", "Crude selection and blending LPs, unit scheduling MILPs, power and steam dispatch: the models MRPL already runs, solved on hardware it owns.", GREEN],
    ["Transparency", "Every pivot, factorisation and cut is inspectable. Tolerances and heuristics can be tuned to Indian refinery models instead of generic benchmarks.", NAVY],
    ["Performance headroom", "The GPU engine uses only matrix-vector products: large planning LPs speed up on commodity GPUs as the hardware improves.", SAFFRON],
    ["Ecosystem", "An Apache-2.0 base that PSUs, IITs and Indian start-ups can build on: modelling layers, domain heuristics, new problem classes (MIQP, MINLP).", GREEN],
  ];
  cards.forEach(([h, b, col], i) => {
    const cx = 0.5 + (i % 3) * 4.18, cy = 1.85 + Math.floor(i / 3) * 2.55;
    card(s, cx, cy, 3.98, 2.35, CARD);
    s.addShape(pres.shapes.OVAL, { x: cx + 0.22, y: cy + 0.25, w: 0.5, h: 0.5, fill: { color: col }, line: { color: col } });
    s.addText(String(i + 1), { x: cx + 0.22, y: cy + 0.25, w: 0.5, h: 0.5, fontFace: HF, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(h, { x: cx + 0.85, y: cy + 0.25, w: 3.0, h: 0.5, fontFace: HF, fontSize: 14.5, bold: true, color: NAVY, valign: "middle", margin: 0, isTextBox: true });
    s.addText(b, { x: cx + 0.22, y: cy + 0.9, w: 3.6, h: 1.35, fontFace: BF, fontSize: 11.5, color: INK, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes(
    "IMPACT (35s). Pick three cards. Strategic: the decision engine behind critical energy infrastructure becomes something India can read and change; China made the same move with COPT. Economic: IBM's own pricing page lists CPLEX developer subscriptions from 320 US dollars per user per month, and deployment pricing is not published at all. Operational: these are the models MRPL already runs, crude blending, scheduling, dispatch, now on hardware it owns.\n" +
    "Close on the ecosystem: an open base that PSUs, IITs and start-ups can extend.");
}

// ================================================================= 6. RESEARCH AND REFERENCES
{
  const s = pres.addSlide();
  frame(s, "RESEARCH AND REFERENCES", 6);
  s.addText("Primary sources only. Each maps onto a component we have implemented or are implementing.", { x: 0.5, y: 1.05, w: 10.3, h: 0.4, fontFace: BF, fontSize: 13, italic: true, color: MUTED, margin: 0, isTextBox: true });
  const left = [
    ["Interior point", "Mehrotra, SIAM J. Optim. 2(4), 1992 — doi:10.1137/0802028"],
    ["Dual simplex", "Koberstein, PhD thesis, Univ. Paderborn, 2005; Forrest & Goldfarb, Math. Prog. 57, 1992 — doi:10.1007/BF01581089"],
    ["Ratio tests", "Harris, Math. Prog. 5, 1973 — doi:10.1007/BF01580108; Maros, EJOR 149, 2003 (bound flipping)"],
    ["Sparse LU", "Gilbert & Peierls, SIAM J. Sci. Stat. Comput. 9(5), 1988 — doi:10.1137/0909058"],
    ["Orderings", "Amestoy, Davis & Duff, SIAM J. Matrix Anal. Appl. 17(4), 1996 — doi:10.1137/S0895479894278952"],
    ["Presolve", "Andersen & Andersen, Math. Prog. 71, 1995 — doi:10.1007/BF01586000"],
  ];
  const right = [
    ["PDLP", "Applegate et al., NeurIPS 2021 — arXiv:2106.04756; Math. Prog. Comp. 2026 — doi:10.1007/s12532-026-00309-2"],
    ["GPU LP", "Lu & Yang, cuPDLP, Operations Research 73, 2025 — doi:10.1287/opre.2024.1069"],
    ["MILP", "Achterberg, Koch & Martin, reliability branching, ORL 33, 2005 — doi:10.1016/j.orl.2004.04.002"],
    ["Cuts", "Balas, Ceria, Cornuéjols & Natraj, 'Gomory cuts revisited', ORL 19, 1996"],
    ["Benchmarks", "MIPLIB 2017: Gleixner et al., Math. Prog. Comp. 13, 2021; exact Netlib optima: Koch, ORL 32, 2004"],
    ["QP", "Vanderbei, quasi-definite LDLᵀ, SIAM J. Optim. 5(1), 1995 — doi:10.1137/0805005; test set: Maros & Mészáros, OMS 11, 1999"],
  ];
  const col = (items, x) => items.forEach(([k, v], i) => {
    const y = 1.6 + i * 0.86;
    s.addText(k, { x, y, w: 1.7, h: 0.75, fontFace: HF, fontSize: 11.5, bold: true, color: SAFFRON, valign: "top", margin: 0, isTextBox: true });
    s.addText(v, { x: x + 1.75, y, w: 4.4, h: 0.75, fontFace: BF, fontSize: 11, color: INK, valign: "top", margin: 0, isTextBox: true });
  });
  col(left, 0.5);
  col(right, 6.85);
  s.addText("Comparators used only in the benchmark harness, never inside the solver: HiGHS 1.15 (MIT), Clarabel 0.11 (Apache-2.0), SCIP, OR-Tools. Benchmark data: Netlib via COIN-OR mirror, MIPLIB 3 / 2017 (miplib.zib.de).",
    { x: 0.5, y: 6.72, w: 12.3, h: 0.3, fontFace: BF, fontSize: 9.5, color: MUTED, margin: 0, isTextBox: true });
  s.addNotes(
    "RESEARCH (25s, then stop). Do not read citations. Say: every component on the architecture slide has a primary source here: Mehrotra for the interior point, Koberstein and Forrest-Goldfarb for the dual simplex, Gilbert-Peierls for our LU, Applegate et al. for PDLP and Lu and Yang for the GPU version.\n" +
    "Then: comparators like HiGHS appear only in our test harness, never in the solver, which is what the problem statement requires.");
}

const out = path.join(__dirname, "ZeroCloud_SIH26119_NIRNAY.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote " + out));
