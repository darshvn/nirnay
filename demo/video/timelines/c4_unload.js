// Clip 4 — crude unloading schedule. u1 0.40–7.64 (published benchmark: tankers, tanks, CDU),
// u2 7.64–12.26 (reproduces the published optimum exactly ~10.2, draws the schedule ~11.3).
(function () {
  window.__timelines = window.__timelines || {};
  const { $, $$ } = K;
  const C = K.init();
  const R = K.prepareRun();
  const rc = C.rect;
  const card = $(".model-card"), src = $("#mc-src"), solve = $("#solve-btn"), live = $(".live");
  const gantt = $("#insight .gantt"), foot = $("#insight .g-foot");

  gsap.set(C.cam, C.view(rc(card), { max: 1.6 }));
  K.washIn(C);
  C.move(C.view(rc(src), { max: 2.2, pad: 120 }), 0.5);
  C.drift(C.view(rc(src), { max: 2.2, pad: 120 }), 1.5, 1.6, 1.04);
  C.move(C.view(C.union(rc(card), rc(live)), { max: 1.3, pad: 40 }), 3.1);
  const path = K.cursor(C, { x: 1750, y: 950 }, [{ el: solve, at: 3.2, dur: 0.85, click: true, settle: 0.26 }]);
  C.tl.to("#hf-cursor", { opacity: 1, duration: 0.2 }, 3.1);
  const tClick = path.clicks[0].t;
  K.press(C, solve, tClick);
  C.tl.to("#hf-cursor", { opacity: 0, duration: 0.3 }, tClick + 0.7);
  C.move(C.view(rc(live), { max: 1.45, pad: 50 }), tClick + 0.3);
  const tDone = K.playRun(C, R, tClick + 0.2, 3.2);
  const resFrame = C.view(C.union(rc($("#res-status")), rc($("#res-obj")), rc($("#res-time")), rc($("#res-it"))), { max: 2.0, pad: 80 });
  C.move(resFrame, tDone + 0.05);
  C.tl.to(R.empty, { opacity: 0, duration: 0.3 }, tDone + 0.1); // solved: the 'solve first' hint goes
  C.drift(resFrame, tDone + 1.0, 1.1, 1.04);
  // the schedule draws itself, then the published value
  const tG = Math.max(9.4, tDone + 2.0);
  K.showInsight(C, R, tG - 0.2);
  C.move(C.view(rc($("#insight")), { max: 1.35, pad: 40 }), tG);
  K.growBars(C, $$(".g-bar", gantt), tG + 0.25, 0.13, 0.55);
  C.move(C.view(rc(foot), { max: 2.0, pad: 120 }), tG + 2.1);
  C.drift(C.view(rc(foot), { max: 2.0, pad: 120 }), tG + 3.0, 1.5, 1.04);
  const end = tG + 4.4;
  K.washOut(C, end - 0.4);
  window.__beats = { tClick, tDone, tG, end };
  window.__timelines["main"] = C.tl;
})();
