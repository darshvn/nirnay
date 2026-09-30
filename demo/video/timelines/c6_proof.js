// Clip 6 — proof. f1 0.40–2.34, f2 2.34–16.68 (90/90 ~2.4, 33 MIPLIB ~4.7, 121/138 ~8.7,
// independent re-check ~12.8), f3 16.68–20.92 (mistakes in the references), f4 20.92–23.60 (slower
// than HiGHS), f5 23.60–29.90 (no solver library inside … every pivot our own code).
(function () {
  window.__timelines = window.__timelines || {};
  const { $, $$ } = K;
  const C = K.init();
  const tl = C.tl, rc = C.rect;
  C.setWithin(null); // the benchmark and source views use the full width
  const vSolve = $("#view-solve"), vBench = $("#view-bench"), vSrc = $("#view-source");
  [vBench, vSrc].forEach((v) => v.classList.remove("hidden"));
  const show = (v) => [vSolve, vBench, vSrc].forEach((x) => (x.style.display = x === v ? "" : "none"));
  const tabs = $$(".tab"), tBench = tabs[1], tSrc = tabs[2];
  // measure each view while it is laid out
  show(vBench);
  const cards = $$(".bench-cards .bc"), found = $(".found"), speedCard = $(".speed");
  const R = { head: rc($(".bench-head")), cards: cards.map(rc), all: rc($(".bench-cards")), found: rc(found), speed: rc(speedCard),
              benchAll: C.union(rc($(".bench-head")), rc($(".bench-cards"))) };
  show(vSrc);
  R.imports = rc($(".imports")); R.scan = rc($("#src-solvers")); R.code = rc($(".code")); R.tree = rc($("#src-tree"));
  show(vSolve);
  const W0 = { x: 0, y: 0, w: 1920, h: 1080, cx: 960, cy: 540 };
  gsap.set(C.cam, C.at(W0, 1));
  K.washIn(C);
  // cursor: Benchmarks tab, later Source tab
  const path = K.cursor(C, { x: 1500, y: 700 }, [
    { el: tBench, at: 0.2, dur: 0.75, click: true, settle: 0.22 },
    { el: tSrc, at: 23.2, dur: 0.7, click: true, settle: 0.22 }]);
  tl.to("#hf-cursor", { opacity: 1, duration: 0.2 }, 0.15);
  const c1 = path.clicks[0].t, c2 = path.clicks[1].t;
  K.press(C, tBench, c1); K.press(C, tSrc, c2);
  const setView = (v, t) => {
    tl.set(vSolve, { display: v === vSolve ? "" : "none" }, t);
    tl.set(vBench, { display: v === vBench ? "" : "none" }, t);
    tl.set(vSrc, { display: v === vSrc ? "" : "none" }, t);
  };
  const setTab = (tab, t) => tabs.forEach((x) => tl.set(x, { attr: { class: x === tab ? "tab active" : "tab" } }, t));
  gsap.set(vBench, { display: "none" }); gsap.set(vSrc, { display: "none" });
  setView(vBench, c1 + 0.05); setTab(tBench, c1 + 0.05);
  tl.to("#hf-cursor", { opacity: 0, duration: 0.3 }, c1 + 0.6);
  // numbers count up with the voice
  const nums = cards.map((c) => {
    const big = $(".big", c);
    const span = document.createElement("span");
    span.textContent = big.firstChild.textContent.trim();
    big.replaceChild(span, big.firstChild);
    return span;
  });
  const vals = nums.map((s) => parseInt(s.textContent, 10));
  const fills = cards.map((c) => $$(".fill", c));
  fills.forEach((f) => f.forEach((e) => gsap.set(e, { scaleX: 0, transformOrigin: "0% 50%" })));
  // cards not yet talked about sit back until their turn
  cards.forEach((c, k) => { if (k) gsap.set(c, { opacity: 0.35 }); });
  [[1, 4.6], [2, 8.6]].forEach(([k, t]) => tl.to(cards[k], { opacity: 1, duration: 0.4 }, t - 0.2));
  C.move(C.view(R.benchAll, { max: 1.25, pad: 60 }), c1 + 0.1);
  [[0, 2.35], [1, 4.6], [2, 8.6]].forEach(([k, t]) => {
    C.move(C.view(R.cards[k], { max: 2.0, pad: 140 }), t - 0.15);
    K.countUp(C, nums[k], vals[k], t + 0.1, 1.3);
    fills[k].forEach((e, j) => tl.to(e, { scaleX: 1, duration: 0.8, ease: "power2.out" }, t + 0.3 + j * 0.25));
  });
  C.drift(C.view(R.cards[1], { max: 2.0, pad: 140 }), 5.6, 2.8, 1.04);
  C.drift(C.view(R.cards[2], { max: 2.0, pad: 140 }), 9.6, 2.8, 1.04);
  C.move(C.view(R.all, { max: 1.4, pad: 60 }), 12.6);
  C.drift(C.view(R.all, { max: 1.4, pad: 60 }), 13.6, 3.0, 1.04);
  // what checking found in the references
  const rows = $$(".found-t tr");
  rows.forEach((r) => gsap.set(r, { opacity: 0 }));
  C.move(C.view(R.found, { max: 1.7, pad: 80 }), 16.6);
  rows.forEach((r, k) => tl.to(r, { opacity: 1, duration: 0.35 }, 17.0 + k * 0.45));
  C.drift(C.view(R.found, { max: 1.7, pad: 80 }), 17.6, 3.2, 1.04);
  // honest about speed
  const sb = $("#speed-big");
  const sv = parseFloat(sb.textContent);
  gsap.set(sb, { opacity: 0 });
  tl.to(sb, { opacity: 1, duration: 0.2 }, 20.95);
  C.move(C.view(R.speed, { max: 2.0, pad: 120 }), 20.85);
  K.countUp(C, sb, sv, 21.0, 1.2, (v) => `${v.toFixed(1)}×`);
  // no solver library inside: the import scan, then the code
  setView(vSrc, c2 + 0.05); setTab(tSrc, c2 + 0.05);
  tl.to("#hf-cursor", { opacity: 1, duration: 0.2 }, 23.1);
  tl.to("#hf-cursor", { opacity: 0, duration: 0.3 }, c2 + 0.6);
  const chips = $$("#src-solvers .chip");
  chips.forEach((c) => gsap.set(c, { opacity: 0, scale: 0.8 }));
  C.move(C.view(R.imports, { max: 1.7, pad: 80 }), c2 + 0.1);
  chips.forEach((c, k) => tl.to(c, { opacity: 1, scale: 1, duration: 0.3, ease: "back.out(2)" }, c2 + 0.5 + k * 0.1));
  C.move(C.view(R.scan, { max: 2.1, pad: 100 }), 25.9);
  C.move(C.view(R.code, { max: 1.5, pad: 60 }), 27.7);
  C.drift(C.view(R.code, { max: 1.5, pad: 60 }), 28.7, 2.3, 1.05);
  K.washOut(C, 31.0);
  window.__beats = { c1, c2, end: 31.4 };
  window.__timelines["main"] = tl;
})();
