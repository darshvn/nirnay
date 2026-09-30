// Shared helpers for the NIRNAY walkthrough timelines (prepended to every scene script).
// Everything here is seek-safe: pure functions of the timeline time, no Date.now or random.
window.K = (function () {
  const $ = (q, r) => (r || document).querySelector(q);
  const $$ = (q, r) => Array.from((r || document).querySelectorAll(q));
  const W = 1920, H = 1080;

  function init() {
    const cam = $("#camera");
    cam.style.transform = "none";
    const PAGE = Math.max(H, cam.scrollHeight, document.body.scrollHeight);
    // measure in untransformed page coordinates, whatever the camera is doing at the time
    const rect = (el) => {
      const prev = cam.style.transform;
      cam.style.transform = "none";
      const cr = cam.getBoundingClientRect();
      const r = el.getBoundingClientRect();
      cam.style.transform = prev;
      return { x: r.left - cr.left, y: r.top - cr.top, w: r.width, h: r.height,
               cx: r.left - cr.left + r.width / 2, cy: r.top - cr.top + r.height / 2 };
    };
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
    const at = (r, s) => ({ scale: s, x: clamp(W / 2 - r.cx * s, W - W * s, 0), y: clamp(H / 2 - r.cy * s, H - PAGE * s, 0) });
    // the scale that shows the whole element with a margin, capped
    // close-ups stay inside a region (default: the app's working area) so no half-cut words show
    let within = null;
    const view = (r, o) => {
      o = Object.assign({ pad: 90, max: 2.2, min: 1 }, o || {});
      const s = clamp(Math.min((W - 2 * o.pad) / Math.max(r.w, 1), (H - 2 * o.pad) / Math.max(r.h, 1)), o.min, o.max);
      const f = at(r, s);
      const box = o.within === undefined ? within : o.within;
      if (box && s > 1.02 && W / s <= box.w + 40) {
        const left = -f.x / s, right = left + W / s;
        if (left < box.x - 20) f.x = -(box.x - 20) * s;
        else if (right > box.x + box.w + 20) f.x = -(box.x + box.w + 20 - W / s) * s;
        f.x = clamp(f.x, W - W * s, 0);
      }
      return f;
    };
    const setWithin = (el) => { within = el ? rect(el) : null; };
    // union of several rects
    const union = (...rs) => {
      const x0 = Math.min(...rs.map((r) => r.x)), y0 = Math.min(...rs.map((r) => r.y));
      const x1 = Math.max(...rs.map((r) => r.x + r.w)), y1 = Math.max(...rs.map((r) => r.y + r.h));
      return { x: x0, y: y0, w: x1 - x0, h: y1 - y0, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 };
    };
    const tl = gsap.timeline({ paused: true });
    const spring = MotionKit.springEase(MotionKit.RECORDLY.cameraSpring, 350);
    // spring-eased camera move (Recordly constants) to a framing
    const move = (frame, t, dur) => {
      if (dur) tl.to(cam, Object.assign({ duration: dur, ease: "power3.inOut" }, frame), t);
      else tl.to(cam, Object.assign({ duration: spring.duration, ease: spring.ease }, frame), t);
    };
    // slow push-in so a reading hold never looks frozen
    const drift = (frame, t, dur, k) => {
      const f = Object.assign({}, frame);
      const s2 = f.scale * (k || 1.05);
      const cx = (W / 2 - f.x) / f.scale, cy = (H / 2 - f.y) / f.scale;
      tl.to(cam, { scale: s2, x: clamp(W / 2 - cx * s2, W - W * s2, 0), y: clamp(H / 2 - cy * s2, H - PAGE * s2, 0),
                   duration: dur, ease: "sine.inOut" }, t);
    };
    gsap.set(cam, { transformOrigin: "0 0" });
    gsap.set(["#hf-cursor", "#hf-ripple"], { opacity: 0 });
    const work = $(".work");
    if (work && work.offsetParent) setWithin(work);
    return { cam, rect, at, view, union, tl, move, drift, PAGE, clamp, setWithin };
  }

  // cursor through a list of targets: [{el|x,y, at, click, dur}]
  function cursor(ctx, start, moves) {
    const mv = moves.map((m) => {
      const r = m.el ? ctx.rect(m.el) : { cx: m.x, cy: m.y, w: 40 };
      return { x: r.cx + (m.dx || 0), y: r.cy + (m.dy || 0), click: m.click, at: m.at, dur: m.dur, targetW: r.w, settle: m.settle };
    });
    const path = MotionKit.buildCursorPath({ start, sway: 0.18, moves: mv });
    MotionKit.addCursor(ctx.tl, path, { el: $("#hf-cursor"), hotspot: { x: 4, y: 2.5 }, ripple: $("#hf-ripple"),
                                        cursorH: 26, rippleR: 26, stageW: W, at: 0 });
    return path;
  }
  function press(ctx, el, t) { MotionKit.addPress(ctx.tl, el, t, { scale: 0.95 }); }

  // ----- replaying a Studio run from its frozen final state -----
  // result bar: stacked "READY"/"RUNNING" twins over the real (solved) one
  function resultTwins() {
    const res = $("#result");
    const wrap = document.createElement("div");
    wrap.style.position = "relative";
    res.parentElement.insertBefore(wrap, res);
    wrap.appendChild(res);
    const twin = (label) => {
      const n = res.cloneNode(true);
      n.removeAttribute("id");
      n.className = "result";
      n.style.position = "absolute"; n.style.inset = "0"; n.style.margin = "0";
      $(".res-status", n).textContent = label;
      $$(".res-cell .val", n).forEach((v) => (v.textContent = "—"));
      const vo = $(".ver-out", n); if (vo) vo.innerHTML = "";
      wrap.appendChild(n);
      return n;
    };
    return { real: res, ready: twin("READY"), running: twin("RUNNING") };
  }

  function prepareRun() {
    const T = resultTwins();
    const series = $$("#chart path.series");
    series.forEach((p) => { const L = p.getTotalLength(); p.style.strokeDasharray = L; p.style.strokeDashoffset = L; p.dataset.len = L; });
    const extras = $$("#chart path.area, #chart circle.dot, #chart line[stroke-dasharray], #chart text");
    const chartText = $$("#chart text");
    const log = $$("#log .l");
    const verify = $$("#verify-out > div");
    const insight = $("#insight");
    // the plan card shows the app's own empty state until the plan is read out
    const iw = document.createElement("div");
    iw.style.position = "relative";
    insight.parentElement.insertBefore(iw, insight);
    iw.appendChild(insight);
    const empty = insight.cloneNode(false);
    empty.removeAttribute("id");
    empty.style.position = "absolute"; empty.style.inset = "0";
    const msg = document.createElement("div"); msg.className = "empty"; msg.textContent = "Solve the model to read the plan.";
    empty.appendChild(msg);
    iw.appendChild(empty);
    gsap.set([T.running, T.real], { opacity: 0 });
    gsap.set(log, { opacity: 0 });
    gsap.set(verify, { opacity: 0, y: 6 });
    gsap.set($$("#chart path.area, #chart circle.dot, #chart line[stroke-dasharray]"), { opacity: 0 });
    gsap.set(chartText.filter((t) => /phase/.test(t.textContent)), { opacity: 0 });
    gsap.set(insight, { opacity: 0 });
    return { T, series, log, verify, insight, empty, chartText };
  }

  // the run itself: log streams, curves draw, then the status flips
  function playRun(ctx, R, t0, dur) {
    const tl = ctx.tl;
    tl.set(R.T.ready, { opacity: 0 }, t0);
    tl.set(R.T.running, { opacity: 1 }, t0);
    const n = R.log.length;
    R.log.forEach((l, k) => tl.to(l, { opacity: 1, duration: 0.12 }, t0 + 0.15 + (dur - 0.3) * (k / Math.max(1, n - 1))));
    R.series.forEach((p) => tl.to(p, { strokeDashoffset: 0, duration: dur, ease: "sine.inOut" }, t0 + 0.1));
    tl.to($$("#chart path.area"), { opacity: 1, duration: dur * 0.6 }, t0 + dur * 0.4);
    tl.to($$("#chart line[stroke-dasharray]"), { opacity: 1, duration: 0.3 }, t0 + dur * 0.35);
    tl.to(R.chartText.filter((t) => /phase/.test(t.textContent)), { opacity: 1, duration: 0.3 }, t0 + dur * 0.35);
    tl.to($$("#chart circle.dot"), { opacity: 1, duration: 0.25 }, t0 + dur);
    const tDone = t0 + dur + 0.15;
    tl.to(R.T.running, { opacity: 0, duration: 0.2 }, tDone);
    tl.to(R.T.real, { opacity: 1, duration: 0.2 }, tDone);
    tl.fromTo($("#res-status"), { filter: "brightness(1.6)" }, { filter: "brightness(1)", duration: 0.6 }, tDone);
    return tDone;
  }

  function showInsight(ctx, R, t) {
    ctx.tl.to(R.empty, { opacity: 0, duration: 0.25 }, t);
    ctx.tl.to(R.insight, { opacity: 1, duration: 0.35 }, t);
  }
  // bars that grow from zero, staggered (refinery bars, race bars, Gantt bars)
  function growBars(ctx, els, t, each, dur) {
    els.forEach((e, k) => {
      gsap.set(e, { scaleX: 0, transformOrigin: "0% 50%" });
      ctx.tl.to(e, { scaleX: 1, duration: dur || 0.6, ease: "power2.out" }, t + k * (each || 0.08));
    });
  }
  function fadeIn(ctx, els, t, each, dy) {
    els.forEach((e, k) => {
      gsap.set(e, { opacity: 0, y: dy == null ? 8 : dy });
      ctx.tl.to(e, { opacity: 1, y: 0, duration: 0.4, ease: "power2.out" }, t + k * (each || 0.08));
    });
  }
  function revealVerify(ctx, R, t) {
    R.verify.forEach((e, k) => ctx.tl.to(e, { opacity: 1, y: 0, duration: 0.35, ease: "power2.out" }, t + k * 0.25));
  }
  // number that counts up (onUpdate is re-run on every seek, so it is deterministic)
  function countUp(ctx, el, to, t, dur, fmt) {
    const o = { v: 0 };
    const f = fmt || ((v) => Math.round(v).toLocaleString("en-US"));
    el.textContent = f(0);
    ctx.tl.to(o, { v: to, duration: dur, ease: "power2.out", onUpdate: () => { el.textContent = f(o.v); } }, t);
  }
  function washIn(ctx) { gsap.set("#hf-wash", { opacity: 1 }); ctx.tl.to("#hf-wash", { opacity: 0, duration: 0.45, ease: "power1.out" }, 0); }
  function washOut(ctx, t) { ctx.tl.to("#hf-wash", { opacity: 1, duration: 0.4, ease: "power1.in" }, t); }

  return { $, $$, W, H, init, cursor, press, prepareRun, playRun, showInsight, growBars, fadeIn, revealVerify, countUp, washIn, washOut };
})();
