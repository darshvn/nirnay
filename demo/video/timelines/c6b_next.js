// Clip 6b — roadmap and what it means for MRPL. n1 0.50–13.10 (speed gap ~1–6, cutting planes ~7,
// live MRPL model ~10.4), n2 13.10–22.20 (owns ~13.1, no per-seat licence ~15.8, audit ~17.3, trust ~18.7).
(function () {
  window.__timelines = window.__timelines || {};
  const { $, $$ } = K;
  const C = K.init();
  const tl = C.tl, rc = C.rect;
  const W0 = { x: 0, y: 0, w: 1920, h: 1080, cx: 960, cy: 540 };
  gsap.set(C.cam, C.at(W0, 1.0));
  K.washIn(C);
  const left = $(".left"), right = $("#mrpl");
  gsap.set(["#k1", "#s1", "#s2", "#s3", right, "#b1", "#b2", "#b3"], { opacity: 0, y: 24 });
  tl.to("#k1", { opacity: 1, y: 0, duration: 0.4 }, 0.3);
  C.move(C.view(rc(left), { max: 1.35, pad: 70, within: null }), 0.4);
  [["#s1", 0.8], ["#s2", 6.9], ["#s3", 10.3]].forEach(([id, t]) => {
    tl.to(id, { opacity: 1, y: 0, duration: 0.55, ease: "power3.out" }, t);
    tl.fromTo(`${id} .n`, { scale: 0.6 }, { scale: 1, duration: 0.5, ease: "back.out(2)" }, t);
  });
  C.drift(C.view(rc(left), { max: 1.35, pad: 70, within: null }), 1.4, 11.0, 1.06);
  // for MRPL
  tl.to(right, { opacity: 1, y: 0, duration: 0.6, ease: "power3.out" }, 12.9);
  C.move(C.view(rc(right), { max: 1.45, pad: 80, within: null }), 13.0);
  [["#b1", 15.6], ["#b2", 17.1], ["#b3", 18.5]].forEach(([id, t]) =>
    tl.to(id, { opacity: 1, y: 0, duration: 0.45, ease: "power3.out" }, t));
  C.drift(C.view(rc(right), { max: 1.45, pad: 80, within: null }), 14.0, 6.5, 1.05);
  C.move(C.at(W0, 1.0), 21.0, 1.4);
  K.washOut(C, 23.6);
  window.__beats = { end: 24.0 };
  window.__timelines["main"] = tl;
})();
