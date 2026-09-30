// Clip 5b — architecture. a1 0.50–5.56 (three engines by problem shape), a2 5.56–19.22 (simplex ~5.6,
// interior point ~11.0, PDLP ~13.5 … GPU pays ~16), a3 19.22–29.52 (MILP & QP on top ~19.3–22.6,
// factorisations underneath ~24.5: LU ~25.8, Cholesky ~26.8, LDLᵀ ~27.9, "we wrote ourselves" ~29).
(function () {
  window.__timelines = window.__timelines || {};
  const { $, $$ } = K;
  const C = K.init();
  const tl = C.tl, rc = C.rect;
  const rows = ["#rIn", "#rPre", "#rTop", "#rEng", "#rLa", "#rK"].map((q) => $(q));
  const e1 = $("#e1"), e2 = $("#e2"), e3 = $("#e3");
  const all = { x: 0, y: 0, w: 1920, h: 1080, cx: 960, cy: 540 };
  gsap.set(C.cam, C.at(all, 1));
  K.washIn(C);
  gsap.set(["#k", ...rows, "#stamp"], { opacity: 0, y: 18 });
  tl.to("#k", { opacity: 1, y: 0, duration: 0.4 }, 0.3);
  tl.to(["#rIn", "#rPre"], { opacity: 1, y: 0, duration: 0.5, stagger: 0.25 }, 0.6);
  // three engines
  gsap.set([e1, e2, e3], { opacity: 0, y: 30 });
  tl.to("#rEng", { opacity: 1, y: 0, duration: 0.01 }, 2.2);
  tl.to([e1, e2, e3], { opacity: 1, y: 0, duration: 0.55, stagger: 0.3, ease: "back.out(1.4)" }, 2.3);
  // each engine and why
  const glow = (el, t0, t1, col) => {
    tl.to(el, { boxShadow: `0 0 0 3px ${col}, 0 30px 70px rgba(0,0,0,.45)`, duration: 0.35 }, t0);
    tl.to(el, { boxShadow: "0 20px 50px rgba(0,0,0,.35)", duration: 0.35 }, t1);
  };
  C.move(C.view(rc(e1), { max: 1.8, pad: 200, within: null }), 5.5);
  glow(e1, 5.6, 10.8, "rgba(91,140,255,.8)");
  C.drift(C.view(rc(e1), { max: 1.8, pad: 200, within: null }), 6.6, 4.0, 1.05);
  C.move(C.view(rc(e2), { max: 1.8, pad: 200, within: null }), 10.8);
  glow(e2, 10.9, 13.3, "rgba(162,123,255,.8)");
  C.move(C.view(rc(e3), { max: 1.8, pad: 200, within: null }), 13.3);
  glow(e3, 13.4, 18.6, "rgba(242,140,56,.9)");
  C.drift(C.view(rc(e3), { max: 1.8, pad: 200, within: null }), 14.4, 3.8, 1.05);
  C.move(C.at(all, 1), 18.6);
  // MILP and QP on top
  tl.to("#rTop", { opacity: 1, y: 0, duration: 0.6, ease: "back.out(1.3)" }, 19.4);
  C.move(C.view(C.union(rc($("#rTop")), rc($("#rEng"))), { max: 1.35, pad: 60, within: null }), 19.6);
  // the factorisations underneath, lit one by one
  tl.to(["#rLa", "#rK"], { opacity: 1, y: 0, duration: 0.5, stagger: 0.2 }, 23.4);
  C.move(C.view(C.union(rc($("#rLa")), rc($("#rK"))), { max: 1.6, pad: 80, within: null }), 24.2);
  [["#la1", 25.7], ["#la2", 26.7], ["#la3", 27.8], ["#la4", 28.4]].forEach(([id, t]) => {
    tl.to(id, { color: "#FFFFFF", borderColor: "rgba(47,191,113,.7)", backgroundColor: "rgba(47,191,113,.14)", duration: 0.3 }, t);
  });
  tl.to("#stamp", { opacity: 1, y: 0, duration: 0.5 }, 28.8);
  C.move(C.at(all, 1), 28.7, 1.4);
  K.washOut(C, 30.8);
  window.__beats = { end: 31.2 };
  window.__timelines["main"] = tl;
})();
