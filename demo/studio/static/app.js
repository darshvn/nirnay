// NIRNAY Studio front end: model shelf, live solve over server-sent events, result and plan views.
(() => {
  const $ = (q, r = document) => r.querySelector(q);
  const $$ = (q, r = document) => Array.from(r.querySelectorAll(q));
  const ENG_BY_KIND = {
    LP: [["simplex", "Dual simplex"], ["ipm", "Interior point"], ["pdlp", "PDLP · CPU"], ["pdlp-gpu", "PDLP · GPU"]],
    MILP: [["bnb", "Branch-and-bound"]],
    QP: [["qp-ipm", "QP interior point"]],
  };
  const S = { models: [], cur: null, engine: null, run: null, es: null };

  // ------------------------------------------------------------------ formatting
  const fmt = (v, d = 2) => v == null || !isFinite(v) ? "—" :
    Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const fint = (v) => v == null ? "—" : Number(v).toLocaleString("en-US");
  const fsec = (t) => t == null ? "—" : t < 1 ? `${(t * 1000).toFixed(0)} ms` : t < 100 ? `${t.toFixed(2)} s` : `${t.toFixed(0)} s`;
  const fexp = (v) => v == null ? "—" : v === 0 ? "0" : Number(v).toExponential(1).replace("e-", "e−");
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

  // how each model's objective reads to a planner
  const OBJ = {
    refinery: { lab: "Gross margin", show: (o) => `$ ${fmt(-o / 1000, 2)} M / yr`, flip: true },
    plan4: { lab: "Margin, Apr–Jul", show: (o) => `$ ${fmt(-o / 1000, 2)} M`, flip: true },
    unload: { lab: "Gross margin", show: (o) => fmt(o, 3) },
  };

  // ------------------------------------------------------------------ shelf
  async function loadShelf() {
    S.models = await (await fetch("/api/models")).json();
    const shelf = $("#shelf");
    shelf.innerHTML = "";
    let g = null;
    for (const m of S.models) {
      if (m.group !== g) { g = m.group; shelf.appendChild(el("div", "grp", g)); }
      const b = el("button", "item", `<div class="t"><span class="kind ${m.kind}">${m.kind}</span>${m.title}</div><div class="s">${m.subtitle}</div>`);
      b.dataset.key = m.key;
      b.onclick = () => select(m.key);
      shelf.appendChild(b);
    }
    const want = new URLSearchParams(location.search).get("model") || S.models[0].key;
    select(want);
  }

  async function select(key) {
    if (S.es) { S.es.close(); S.es = null; }
    $$(".item").forEach((b) => b.classList.toggle("active", b.dataset.key === key));
    const base = S.models.find((m) => m.key === key);
    S.cur = base; S.engine = base.method;
    fillCard(base);
    renderEngines(base);
    resetRun();
    const full = await (await fetch(`/api/model?model=${key}`)).json();
    if (S.cur.key === key) { S.cur = full; fillCard(full); }
  }

  function fillCard(m) {
    $("#mc-kind").textContent = m.kind; $("#mc-kind").className = `kind ${m.kind}`;
    $("#mc-group").textContent = m.group;
    $("#mc-title").textContent = m.title;
    $("#mc-sub").textContent = m.subtitle;
    $("#mc-src").textContent = m.source;
    const st = $("#mc-stats");
    const stats = [["Rows", m.rows], ["Columns", m.cols], ["Nonzeros", m.nnz]];
    stats.push(m.kind === "MILP" ? ["Integers", m.integers] : m.kind === "QP" ? ["Q nonzeros", m.qnnz] : ["Type", "LP"]);
    st.innerHTML = stats.map(([l, n]) => `<div class="stat"><div class="n">${typeof n === "number" ? fint(n) : (n ?? "…")}</div><div class="l">${l}</div></div>`).join("");
  }

  function renderEngines(m) {
    const box = $("#engines");
    box.innerHTML = "";
    for (const [k, lab] of ENG_BY_KIND[m.kind]) {
      const b = el("button", "eng" + (k === S.engine ? " active" : ""), lab);
      b.dataset.eng = k;
      b.onclick = () => { S.engine = k; $$(".eng").forEach((x) => x.classList.toggle("active", x === b)); };
      box.appendChild(b);
    }
  }

  // ------------------------------------------------------------------ run state
  function resetRun() {
    S.run = { kind: null, pts: [], log: [], t0: 0 };
    $("#log").innerHTML = "";
    $("#chart").innerHTML = "";
    $("#chart-legend").innerHTML = "";
    $("#chart-title").textContent = "Progress";
    const r = $("#result"); r.className = "result";
    $("#res-status").textContent = "READY";
    for (const id of ["#res-obj", "#res-time", "#res-it", "#res-viol"]) $(id).textContent = "—";
    $("#res-obj-lab").textContent = (OBJ[S.cur.key] || {}).lab || "Objective";
    $("#res-it-lab").textContent = S.cur.kind === "MILP" ? "Nodes" : "Iterations";
    $("#verify-out").innerHTML = "";
    $("#insight").innerHTML = `<div class="empty">Solve the model to read the plan.</div>`;
  }

  function logLine(text, cls = "") {
    const box = $("#log");
    box.appendChild(el("div", "l " + cls, text.replace(/&/g, "&amp;").replace(/</g, "&lt;")));
    while (box.children.length > 13) box.removeChild(box.firstChild);
  }

  function solve() {
    if (!S.cur) return;
    resetRun();
    const btn = $("#solve-btn");
    btn.classList.add("busy"); $("#solve-label").textContent = "Solving…";
    $("#res-status").textContent = "RUNNING";
    S.run.t0 = performance.now();
    const es = new EventSource(`/api/solve?model=${S.cur.key}&method=${S.engine}`);
    S.es = es;
    es.onmessage = (msg) => {
      const ev = JSON.parse(msg.data);
      if (ev.type === "start") {
        $("#log-engine").textContent = ev.engine;
        logLine(`$ nirnay solve ${S.cur.key} --method ${ev.method}`, "sys");
      } else if (ev.type === "result") {
        showResult(ev);
      } else if (ev.type === "error") {
        logLine(ev.message, "sys"); $("#res-status").textContent = "ERROR";
      } else if (ev.type === "end") {
        es.close(); S.es = null;
        btn.classList.remove("busy"); $("#solve-label").textContent = "Solve";
      } else {
        progress(ev);
      }
    };
    es.onerror = () => { es.close(); btn.classList.remove("busy"); $("#solve-label").textContent = "Solve"; };
  }

  function progress(ev) {
    const cls = ev.type === "incumbent" ? "inc" : "";
    if (ev.line) logLine(ev.line, cls);
    const P = S.run.pts;
    if (ev.type === "simplex") { S.run.kind = "simplex"; P.push({ x: ev.it, a: ev.obj, b: ev.pinf }); }
    else if (ev.type === "bnb") { S.run.kind = "bnb"; P.push({ x: ev.nodes, a: ev.incumbent, b: ev.bound }); }
    else if (ev.type === "cuts" || ev.type === "root") { S.run.kind = "bnb"; if (!P.length) P.push({ x: 0, a: null, b: ev.bound ?? ev.value }); else P[P.length - 1].b = ev.bound ?? ev.value; }
    else if (ev.type === "incumbent") { S.run.kind = "bnb"; P.push({ x: ev.node, a: ev.value, b: P.length ? P[P.length - 1].b : null }); }
    else if (ev.type === "pdlp") { S.run.kind = "pdlp"; P.push({ x: ev.it, a: ev.p, b: ev.d, c: ev.g }); }
    else if (ev.type === "ipm") { S.run.kind = "ipm"; P.push({ x: ev.it, a: ev.p, b: ev.d, c: ev.g }); }
    else return;
    drawChart();
  }

  // ------------------------------------------------------------------ chart (SVG, so it freezes cleanly)
  function drawChart(final) {
    const svg = $("#chart");
    const W = 640, H = 260, L = 58, R = 18, T = 14, B = 30;
    const P = S.run.pts.filter((p) => p.x != null);
    const kind = S.run.kind;
    if (!P.length) return;
    const xs = P.map((p) => p.x);
    let x0 = Math.min(...xs), x1 = Math.max(...xs); if (x1 === x0) x1 = x0 + 1;
    const X = (v) => L + (v - x0) / (x1 - x0) * (W - L - R);
    let series, legend, title, logScale;
    const flip = kind === "bnb" && (OBJ[S.cur.key] || {}).flip;
    if (flip) P.forEach((p) => { p._f = true; });
    const val = (p, k) => p[k] == null ? null : (flip ? -p[k] : p[k]);
    if (kind === "bnb") {
      title = flip ? "Branch-and-bound: margin" : "Branch-and-bound";
      series = [["a", "var(--green)", "Best plan found"], ["b", "var(--saffron)", "Best still possible"]];
      logScale = false;
    } else if (kind === "simplex") {
      title = "Dual simplex: primal infeasibility → 0";
      series = [["b", "var(--blue)", "Primal infeasibility"]];
      logScale = true;
    } else {
      title = kind === "pdlp" ? "PDLP: relative KKT error" : "Interior point: residuals";
      series = [["a", "var(--blue)", "Primal"], ["b", "var(--violet)", "Dual"], ["c", "var(--saffron)", "Gap"]];
      logScale = true;
    }
    $("#chart-title").textContent = title;
    $("#chart-legend").innerHTML = series.map(([, c, l]) => `<span><i style="background:${c}"></i>${l}</span>`).join("");
    const vals = [];
    for (const [k] of series) for (const p of P) { const v = val(p, k); if (v != null && (!logScale || v > 0)) vals.push(logScale ? Math.log10(v) : v); }
    if (!vals.length) return;
    let y0 = Math.min(...vals), y1 = Math.max(...vals);
    if (kind === "bnb") {
      // frame the convergence: from the first plan found to the final bound, not the early root spike
      // frame the second half of the search, where bound and best plan close in on each other
      const late = P.filter((p) => p.x >= x0 + 0.4 * (x1 - x0));
      const key = [];
      for (const p of late) for (const k of ["a", "b"]) { const v = val(p, k); if (v != null) key.push(v); }
      if (key.length) { y0 = Math.min(...key); y1 = Math.max(...key); const span = Math.max(y1 - y0, Math.abs(y1) * 2e-4, 1e-6); y0 -= span * 0.25; y1 += span * 0.35; }
    }
    if (logScale) { y0 = Math.floor(y0); y1 = Math.ceil(y1); if (y1 === y0) y1 = y0 + 1; }
    else if (kind !== "bnb") { const pad = (y1 - y0) * 0.08 || Math.abs(y1) * 0.01 || 1; y0 -= pad; y1 += pad; }
    const Y = (v) => Math.max(T, Math.min(H - B, T + (1 - ((logScale ? Math.log10(v) : v) - y0) / (y1 - y0)) * (H - T - B)));
    let g = "";
    // grid and labels
    const nt = logScale ? Math.min(y1 - y0, 6) : 4;
    for (let i = 0; i <= nt; i++) {
      const yv = y0 + (y1 - y0) * i / nt;
      const yy = T + (1 - i / nt) * (H - T - B);
      const span = y1 - y0;
      const lab = logScale ? `1e${Math.round(yv)}`.replace("e-", "e−")
        : Math.abs(yv) >= 1e4 && span < 1e3 ? (yv / 1e3).toFixed(span < 100 ? 3 : 2) + "k"
        : span < 10 ? yv.toFixed(2) : shortNum(yv);
      g += `<line x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}" stroke="#243052" stroke-width="1"/>`;
      g += `<text x="${L - 8}" y="${yy + 4}" fill="#7C88A6" font-size="11" text-anchor="end" font-family="Inter">${lab}</text>`;
    }
    g += `<text x="${W - R}" y="${H - 8}" fill="#7C88A6" font-size="11" text-anchor="end" font-family="Inter">${fint(x1)} ${kind === "bnb" ? "nodes" : "iterations"}</text>`;
    g += `<text x="${L}" y="${H - 8}" fill="#7C88A6" font-size="11" font-family="Inter">${fint(x0)}</text>`;
    g += `<defs><linearGradient id="ga" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F28C38" stop-opacity=".28"/><stop offset="1" stop-color="#F28C38" stop-opacity="0"/></linearGradient></defs>`;
    series.forEach(([k, col], i) => {
      const pts = P.filter((p) => val(p, k) != null && (!logScale || val(p, k) > 0)).map((p) => Object.assign({}, p, { [k]: val(p, k) }));
      if (!pts.length) return;
      const step = kind === "bnb";
      let d = "";
      pts.forEach((p, j) => {
        const px = X(p.x), py = Y(p[k]);
        if (j === 0) d += `M${px.toFixed(1)},${py.toFixed(1)}`;
        else if (step) d += `H${px.toFixed(1)}V${py.toFixed(1)}`;
        else d += `L${px.toFixed(1)},${py.toFixed(1)}`;
      });
      if (step) d += `H${(W - R).toFixed(1)}`;
      if (kind === "bnb" && k === "b") g += `<path d="${d}V${H - B}H${X(pts[0].x)}Z" fill="url(#ga)" class="area"/>`;
      g += `<path d="${d}" fill="none" stroke="${col}" stroke-width="${i === 0 ? 2.8 : 2.4}" stroke-linejoin="round" stroke-linecap="round" class="series s-${k}"/>`;
      const last = pts[pts.length - 1];
      g += `<circle cx="${step ? W - R : X(last.x)}" cy="${Y(last[k])}" r="4" fill="${col}" class="dot d-${k}"/>`;
    });
    if (kind === "simplex") {
      // phase 1 (a dual feasible start on auxiliary bounds) ends where infeasibility reaches ~0 and
      // phase 2 starts on the real bounds: mark it, or the jump reads like a glitch
      for (let j = 1; j < P.length; j++) {
        if (P[j - 1].b != null && P[j].b != null && P[j - 1].b < 1e-6 && P[j].b > 1e-3) {
          const px = X(P[j - 1].x);
          g += `<line x1="${px}" x2="${px}" y1="${T}" y2="${H - B}" stroke="#7C88A6" stroke-dasharray="4 4"/>`;
          g += `<text x="${px - 8}" y="${T + 12}" fill="#AEB8D0" font-size="11" text-anchor="end" font-family="Inter">phase 1</text>`;
          g += `<text x="${px + 8}" y="${T + 12}" fill="#AEB8D0" font-size="11" font-family="Inter">phase 2</text>`;
          break;
        }
      }
    }
    svg.innerHTML = g;
  }
  const shortNum = (v) => { const a = Math.abs(v); return a >= 1e6 ? (v / 1e6).toFixed(2) + "M" : a >= 1e4 ? (v / 1e3).toFixed(1) + "k" : a >= 100 ? v.toFixed(0) : v.toFixed(2); };

  // ------------------------------------------------------------------ result
  function showResult(ev) {
    const r = $("#result");
    const ok = ev.status === "optimal";
    r.className = "result " + (ok ? "optimal" : "limit");
    $("#res-status").textContent = ev.status.replace("_", " ").toUpperCase();
    const o = OBJ[S.cur.key];
    $("#res-obj").textContent = o ? o.show(ev.objective) : fmt(ev.objective, 4);
    $("#res-time").textContent = fsec(ev.time);
    $("#res-it").textContent = fint(S.cur.kind === "MILP" ? ev.nodes : ev.iterations);
    const v = Math.max(ev.row_viol ?? 0, ev.bound_viol ?? 0);
    $("#res-viol").textContent = fexp(v);
    logLine(`${ev.engine}: ${ev.status}, objective ${fmt(ev.objective, 6)} in ${fsec(ev.time)}`, "ok");
    if (S.run.kind) drawChart(true);
    renderInsight(ev);
  }

  async function verify() {
    if (!S.cur) return;
    const out = $("#verify-out");
    out.innerHTML = `<span class="muted">HiGHS is re-solving the same file…</span>`;
    const ref = await (await fetch(`/api/verify?model=${S.cur.key}`)).json();
    const mine = parseFloat(($("#res-obj").dataset.raw) || NaN);
    const ours = S.lastObjective;
    const rel = ours != null ? Math.abs(ours - ref.objective) / Math.max(1, Math.abs(ref.objective)) : null;
    const same = rel != null && rel < (S.cur.kind === "MILP" ? 1e-4 : S.engine.startsWith("pdlp") ? 1e-3 : 1e-6);
    out.innerHTML = `<div class="vh">HiGHS 1.15 · ${fsec(ref.time)}${ref.recorded ? " (benchmark)" : ""}</div>` +
      `<div class="v">${fmt(ref.objective, 4)}</div>` +
      (rel != null ? `<div class="${same ? "ok" : ""}">${same ? "✓ same optimum" : "differs"} · Δ ${fexp(rel)}</div>` : "");
    if (S.cur.key === "datt256" && S.lastResult) renderInsight(S.lastResult, ref);
  }

  // ------------------------------------------------------------------ plan views
  function renderInsight(ev, ref) {
    S.lastObjective = ev.objective; S.lastResult = ev;
    const box = $("#insight");
    const I = ev.insights;
    if (S.cur.key === "datt256") { box.innerHTML = raceView(ev, ref); return; }
    if (!I) { box.innerHTML = `<div class="empty">${ev.status === "optimal" ? "Solved. This benchmark has no plan view." : ""}</div>`; return; }
    if (I.kind === "refinery") box.innerHTML = refineryView(I);
    else if (I.kind === "plan") box.innerHTML = planView(I);
    else if (I.kind === "gantt") box.innerHTML = ganttView(I, ev);
  }

  function barList(items, cls, unit, max) {
    const m = max || Math.max(...items.map((i) => i[1]));
    return `<div class="bars ${cls}">` + items.map(([n, v]) =>
      `<div class="b"><span>${n}</span><div class="track"><div class="fill" style="width:${(100 * v / m).toFixed(1)}%"></div></div><span class="num">${fmt(v, 0)}${unit}</span></div>`).join("") + `</div>`;
  }

  function refineryView(I) {
    return `<div class="ins-grid">
      <div><div class="ins-h">Crude slate chosen</div>${barList(I.crude, "", "")}
        <div class="ins-note">thousand m³ per year, across the three CDUs</div></div>
      <div><div class="ins-h">Products made</div>${barList(I.product.slice(0, 7), "green", "")}</div>
      <div><div class="ins-h">What more capacity is worth (shadow prices)</div>
        <div class="duals">${I.duals.map(([n, v]) => `<div class="d"><span>${n}</span><b>$ ${fmt(v, 1)} k / unit</b></div>`).join("")}</div>
        <div class="ins-note">Row duals recovered exactly through presolve: the value of one more unit of each limit.</div></div>
    </div>`;
  }

  function planView(I) {
    return `<div class="ins-h">Crude parcels and FCC mode, month by month</div><div class="months">` +
      I.months.map((mo) => `<div class="month"><div class="mh"><span>${mo.month}</span><span class="fcc">FCC ${mo.fcc.toUpperCase()}</span></div>` +
        Object.entries(mo.parcels).map(([c, n]) => `<div class="parcel ${c.includes("Sweet") ? "sweet" : ""}"><span class="ships">${"<i></i>".repeat(n)}</span>${n} × ${c}</div>`).join("") +
        `</div>`).join("") + `</div><div class="ins-note">Each block is one crude parcel. Integer decisions: how many parcels of which crude, and which FCC mode, every month.</div>`;
  }

  function ganttView(I, ev) {
    const lanes = ["Vessel 1", "Vessel 2", "Tank 1", "Tank 2", "CDU"];
    const H = I.horizon;
    const ticks = Array.from({ length: H + 1 }, (_, d) => `<span style="left:${100 * d / H}%;transform:translateX(${d === 0 ? 0 : d === H ? -100 : -50}%)">day ${d}</span>`).join("");
    const cls = (op) => ["v1", "v2"].includes(op) ? "unload" : ["v7", "v8"].includes(op) ? "distil" : "transfer";
    const rows = lanes.map((ln) => `<div class="g-lane"><div class="ln">${ln}</div><div class="g-track">` +
      I.bars.filter((b) => b.lane === ln).map((b) => { const w = 100 * (b.end - b.start) / H;
        const lab = w > 18 ? `${b.what} · ${fmt(b.volume, 0)}` : w > 5 ? `${b.what.replace("Charging tank ", "CT-")} ${fmt(b.volume, 0)}` : "";
        return `<div class="g-bar ${cls(b.op)}" style="left:${100 * b.start / H}%;width:${w}%" title="${b.lane}: ${b.what} · ${fmt(b.volume, 0)}">${lab}</div>`; }).join("") +
      `</div></div>`).join("");
    const pub = Math.abs(ev.objective - I.published) < 1e-6 * Math.max(1, I.published);
    return `<div class="ins-h">Crude unloading and charging schedule (8 days)</div><div class="gantt">
      <div class="g-axis"><span></span><div class="g-ticks">${ticks}</div></div>${rows}</div>
      <div class="g-foot"><span><i style="background:#5B8CFF"></i>Unloading</span><span><i style="background:#F28C38"></i>Tank transfer</span><span><i style="background:#2FBF71"></i>CDU charging</span>
      <span style="margin-left:auto">${pub ? "✓" : ""} Published optimum (Lee et al. 1996): <b style="color:var(--green2)">${fmt(I.published, 2)}</b> · NIRNAY: <b style="color:var(--saffron2)">${fmt(ev.objective, 2)}</b></span></div>`;
  }

  function raceView(ev, ref) {
    const gpu = ev.time, cpu = ref ? ref.cpu_time : null, hig = ref ? ref.time : null;
    const mx = Math.max(gpu, cpu || 0, hig || 0);
    const row = (who, sub, t, col) => `<div class="r"><div class="who">${who}<small>${sub}</small></div><div class="track"><div class="fill" style="width:${t ? Math.max(1.2, 100 * Math.sqrt(t / mx)).toFixed(1) : 0}%;background:${col}"></div></div><div class="t">${t ? fsec(t) : "…"}</div></div>`;
    return `<div class="scale"><div><div class="ins-h">Same LP, same laptop · 1e-4 accuracy</div><div class="race">
      ${row("NIRNAY PDLP", "GPU · RTX 3050 laptop · this run", gpu, "linear-gradient(90deg,#F28C38,#FFB066)")}
      ${row("NIRNAY PDLP", "CPU · 6 cores · benchmark", cpu, "linear-gradient(90deg,#2E4A8F,#5B8CFF)")}
      ${row("HiGHS 1.15", "default · benchmark", hig, "linear-gradient(90deg,#5C667F,#9AA5B1)")}
      </div><div class="ins-note">Bar length on a square-root scale. Reference times from results/large_pdlp_gpu.csv.</div></div>
      <div class="speedup">${hig ? `<div class="x">${Math.round(hig / gpu)}×</div><p>faster than HiGHS on this LP</p>` : `<p class="muted">Press “Verify with HiGHS” for the reference times.</p>`}</div></div>`;
  }

  // ------------------------------------------------------------------ benchmarks and source
  async function loadBench() {
    const b = await (await fetch("/api/benchmarks")).json();
    $("#bench-cards").innerHTML = b.suites.map((s) => `<div class="card bc"><div class="nm">${s.name}</div><div class="wh">${s.what}</div>
      <div class="big">${s.ours}<small> / ${s.total}</small></div>
      <div class="cmp"><div class="row"><span>NIRNAY</span><div class="track"><div class="fill" style="width:${100 * s.ours / s.total}%;background:linear-gradient(90deg,#F28C38,#FFB066)"></div></div><b>${s.ours}</b></div>
      <div class="row"><span>${s.ref_name}</span><div class="track"><div class="fill" style="width:${100 * s.ref / s.total}%;background:#5C667F"></div></div><b>${s.ref}</b></div></div></div>`).join("");
    $("#speed-big").textContent = `${b.speed_ratio.toFixed(1)}×`;
  }

  async function loadSource() {
    const s = await (await fetch("/api/source")).json();
    $("#src-total").textContent = `${fint(s.total_lines)} lines`;
    $("#src-tree").innerHTML = s.modules.filter((m) => !m.path.endsWith("__init__.py")).map((m) => `<div class="m"><span>${m.path}</span><b>${m.lines}</b></div>`).join("");
    $("#src-third").innerHTML = Object.entries(s.third_party).map(([k, n]) => `<span class="chip">${k}<b>${n} files</b></span>`).join("");
    $("#src-solvers").innerHTML = Object.entries(s.solver_imports).map(([k, f]) => `<span class="chip">${f.length ? "✗" : "✓"} ${k} <b>${f.length}</b></span>`).join("");
    const esc = s.snippet.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    $("#src-code").innerHTML = esc
      .replace(/(#[^\n]*|"""[\s\S]*?""")/g, '<span class="cm">$1</span>')
      .replace(/\b(def|for|in|if|elif|else|return|while|and|or|not)\b/g, '<span class="kw">$1</span>')
      .replace(/(@njit\(cache=True\))/g, '<span class="nb">$1</span>')
      .replace(/\b(dual_run|_ftran_lu|_btran_lu_sparse|_eta_ftran|_eta_btran|pivot_row|dual_ratio_test|choose_leaving_head)\b/g, '<span class="fn">$1</span>');
  }

  // ------------------------------------------------------------------ wiring
  $$(".tab").forEach((t) => t.onclick = () => {
    $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
    $$(".view").forEach((v) => v.classList.toggle("hidden", v.id !== `view-${t.dataset.view}`));
    if (t.dataset.view === "bench") loadBench();
    if (t.dataset.view === "source") loadSource();
  });
  $("#solve-btn").onclick = solve;
  $("#verify-btn").onclick = verify;
  window.studio = { select, solve, verify, state: S };
  loadShelf();
})();
