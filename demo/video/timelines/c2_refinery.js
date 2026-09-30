// Clip 2 — the refinery crude plan (LP). Narration anchors: r1 0.40–10.94, r2 10.94 "Press solve",
// r3 12.40–23.14 (run, optimal, slate, products, shadow prices), r4 23.14–27.44 (verify, same answer).
(function () {
  window.__timelines = window.__timelines || {};
  const { $, $$ } = K;
  const C = K.init();
  const R = K.prepareRun();
  const card = $(".model-card"), src = $("#mc-src"), stats = $("#mc-stats"), runbar = $(".runbar");
  const solve = $("#solve-btn"), live = $(".live"), result = $("#result"), verifyBtn = $("#verify-btn");
  const cols = $$("#insight .ins-grid > div");
  const [cCrude, cProd, cDual] = cols;
  const rc = C.rect;

  gsap.set(C.cam, C.view(rc(card), { max: 1.7 }));
  K.washIn(C);
  // r1: the model and where its data come from
  C.drift(C.view(rc(card), { max: 1.7 }), 0.2, 2.4, 1.04);
  // "public Indian data": the sources line, then the model's size
  C.move(C.view(rc(src), { max: 2.3, pad: 120 }), 2.6);
  C.drift(C.view(rc(src), { max: 2.3, pad: 120 }), 3.6, 2.2, 1.04);
  C.move(C.view(rc(stats), { max: 2.2, pad: 120 }), 6.0);
  C.drift(C.view(rc(stats), { max: 2.2, pad: 120 }), 7.0, 1.4, 1.04);
  // pull back to the run bar as the cursor comes in
  const wide = C.view(C.union(rc(card), rc(runbar), rc(live)), { max: 1.25, pad: 40 });
  C.move(wide, 8.5);
  const path = K.cursor(C, { x: 1780, y: 980 }, [
    { el: solve, at: 9.7, dur: 1.0, click: true, settle: 0.3 },
    { el: verifyBtn, at: 23.3, dur: 0.9, click: true, settle: 0.28 }]);
  C.tl.to("#hf-cursor", { opacity: 1, duration: 0.25 }, 9.3);
  const tClick = path.clicks[0].t;
  K.press(C, solve, tClick);
  // r3: the run
  C.move(C.view(rc(live), { max: 1.42, pad: 50 }), tClick + 0.25);
  const tDone = K.playRun(C, R, tClick + 0.15, 5.0);
  C.tl.to("#hf-cursor", { opacity: 0, duration: 0.3 }, tClick + 0.8);
  // optimal in milliseconds
  const resFrame = C.view(C.union(rc($("#res-status")), rc($("#res-obj")), rc($("#res-time")), rc($("#res-it"))), { max: 2.0, pad: 80 });
  C.move(resFrame, tDone + 0.1);
  C.drift(resFrame, tDone + 0.8, 1.2, 1.04);
  // the crude slate, what to make, what capacity is worth
  const tIns = 17.3;
  K.showInsight(C, R, tIns - 0.2);
  C.move(C.view(C.union(rc(cCrude), rc(cProd)), { max: 1.55, pad: 60 }), tIns);
  K.growBars(C, $$(".bars .fill", cCrude), tIns + 0.2, 0.12, 0.7);
  K.growBars(C, $$(".bars .fill", cProd), tIns + 1.2, 0.08, 0.6);
  C.move(C.view(rc(cDual), { max: 1.75, pad: 60 }), 19.9);
  K.fadeIn(C, $$(".duals .d", cDual), 20.1, 0.14);
  C.drift(C.view(rc(cDual), { max: 1.75, pad: 60 }), 21.0, 2.0, 1.04);
  // r4: independent check
  C.move(C.view(C.union(rc(result)), { max: 1.3, pad: 40 }), 23.0);
  C.tl.to("#hf-cursor", { opacity: 1, duration: 0.2 }, 23.2);
  const tV = path.clicks[1].t;
  K.press(C, verifyBtn, tV);
  K.revealVerify(C, R, tV + 0.7);
  const vFrame = C.view(C.union(rc(verifyBtn), rc($("#verify-out"))), { max: 2.1, pad: 60 });
  C.move(vFrame, tV + 0.5);
  C.tl.to("#hf-cursor", { opacity: 0, duration: 0.3 }, tV + 0.9);
  C.drift(vFrame, tV + 1.5, 3.2, 1.05);
  K.washOut(C, 28.9);
  window.__beats = { tClick, tDone, tV, end: 29.3 };
  window.__timelines["main"] = C.tl;
})();
