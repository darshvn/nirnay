// Clip 3 — 4-month plan (MILP). p1 0.40–7.00 (whole decisions: parcels, cracker mode),
// p2 7.00–14.82 (branch-and-bound closes the gap … proves nothing better exists).
(function () {
  window.__timelines = window.__timelines || {};
  const { $, $$ } = K;
  const C = K.init();
  const R = K.prepareRun();
  const rc = C.rect;
  const card = $(".model-card"), stats = $("#mc-stats"), solve = $("#solve-btn");
  const chart = $(".chart-card"), live = $(".live"), months = $(".months");
  const intsStat = $$("#mc-stats .stat")[3] || stats;

  gsap.set(C.cam, C.view(rc(card), { max: 1.6 }));
  K.washIn(C);
  // p1: the integer decisions in this model
  C.move(C.view(rc(intsStat), { max: 2.4, pad: 160 }), 0.5);
  C.drift(C.view(rc(intsStat), { max: 2.4, pad: 160 }), 1.4, 1.4, 1.03);
  C.move(C.view(C.union(rc(card), rc(live)), { max: 1.3, pad: 40 }), 2.5);
  const path = K.cursor(C, { x: 1750, y: 950 }, [{ el: solve, at: 2.5, dur: 0.9, click: true, settle: 0.26 }]);
  C.tl.to("#hf-cursor", { opacity: 1, duration: 0.2 }, 2.4);
  const tClick = path.clicks[0].t;
  K.press(C, solve, tClick);
  C.tl.to("#hf-cursor", { opacity: 0, duration: 0.3 }, tClick + 0.7);
  // the search: long enough to watch the two lines meet
  C.move(C.view(rc(live), { max: 1.45, pad: 50 }), tClick + 0.3);
  const tDone = K.playRun(C, R, tClick + 0.2, 9.0);
  C.move(C.view(rc(chart), { max: 1.9, pad: 60 }), 7.1);
  C.drift(C.view(rc(chart), { max: 1.9, pad: 60 }), 8.3, 3.6, 1.05);
  // proven optimal
  const resFrame = C.view(C.union(rc($("#res-status")), rc($("#res-obj")), rc($("#res-time")), rc($("#res-it"))), { max: 2.0, pad: 80 });
  C.move(resFrame, tDone + 0.05);
  C.tl.to(R.empty, { opacity: 0, duration: 0.3 }, tDone + 0.1); // solved: the 'solve first' hint goes
  // the plan itself
  const tPlan = tDone + 1.4;
  K.showInsight(C, R, tPlan - 0.2);
  C.move(C.view(rc(months), { max: 1.5, pad: 60 }), tPlan);
  K.fadeIn(C, $$(".month", months), tPlan + 0.15, 0.12, 14);
  $$(".parcel .ships i").forEach((s, k) => { gsap.set(s, { scale: 0 });
    C.tl.to(s, { scale: 1, duration: 0.25, ease: "back.out(2)" }, tPlan + 0.35 + k * 0.03); });
  C.drift(C.view(rc(months), { max: 1.5, pad: 60 }), tPlan + 1.0, 2.0, 1.05);
  const end = Math.max(16.9, tPlan + 3.0);
  K.washOut(C, end - 0.4);
  window.__beats = { tClick, tDone, tPlan, end };
  window.__timelines["main"] = C.tl;
})();
