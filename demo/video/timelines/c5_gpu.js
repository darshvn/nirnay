// Clip 5 — GPU. g1 0.40–3.80 (GPU engine), g2 3.80–12.82 (1.5 M nonzeros ~5.6, about a second ~7.3,
// laptop graphics card ~8.6, HiGHS over four minutes ~12.8).
(function () {
  window.__timelines = window.__timelines || {};
  const { $, $$ } = K;
  const C = K.init();
  const R = K.prepareRun();
  const rc = C.rect;
  const card = $(".model-card"), stats = $("#mc-stats"), engines = $("#engines"), solve = $("#solve-btn"), live = $(".live");
  const race = $("#insight .race"), speed = $("#insight .speedup .x");

  gsap.set(C.cam, C.view(rc(card), { max: 1.6 }));
  K.washIn(C);
  // g1: the GPU engine is one of the choices
  C.move(C.view(rc(engines), { max: 2.3, pad: 140 }), 0.6);
  C.drift(C.view(rc(engines), { max: 2.3, pad: 140 }), 1.6, 2.0, 1.04);
  // "one and a half million nonzeros"
  C.move(C.view(rc(stats), { max: 2.3, pad: 140 }), 3.7);
  C.drift(C.view(rc(stats), { max: 2.3, pad: 140 }), 4.6, 1.0, 1.04);
  C.move(C.view(C.union(rc(card), rc(live)), { max: 1.3, pad: 40 }), 5.5);
  const path = K.cursor(C, { x: 1750, y: 950 }, [{ el: solve, at: 5.5, dur: 0.8, click: true, settle: 0.25 }]);
  C.tl.to("#hf-cursor", { opacity: 1, duration: 0.2 }, 5.4);
  const tClick = path.clicks[0].t;
  K.press(C, solve, tClick);
  C.tl.to("#hf-cursor", { opacity: 0, duration: 0.3 }, tClick + 0.6);
  C.move(C.view(rc(live), { max: 1.45, pad: 50 }), tClick + 0.25);
  const tDone = K.playRun(C, R, tClick + 0.15, 1.1);
  const resFrame = C.view(C.union(rc($("#res-status")), rc($("#res-obj")), rc($("#res-time")), rc($("#res-it"))), { max: 2.0, pad: 80 });
  C.move(resFrame, tDone + 0.05);
  // the race against HiGHS on the same laptop
  const tR = Math.max(8.9, tDone + 1.3);
  K.showInsight(C, R, tR - 0.2);
  C.move(C.view(rc($("#insight")), { max: 1.45, pad: 40 }), tR);
  const fills = $$(".race .fill", race);
  K.growBars(C, [fills[0]], tR + 0.3, 0, 0.35);
  K.growBars(C, [fills[1]], tR + 0.6, 0, 0.5);
  K.growBars(C, [fills[2]], tR + 1.0, 0, 2.6);
  const fmtX = (v) => `${Math.round(v)}×`;
  const target = parseInt(speed.textContent, 10) || 0;
  gsap.set(speed, { opacity: 0 });
  C.tl.to(speed, { opacity: 1, duration: 0.2 }, tR + 3.3);
  K.countUp(C, speed, target, tR + 3.3, 1.0, fmtX);
  C.move(C.view(rc($("#insight")), { max: 1.75, pad: 30 }), tR + 3.2);
  C.drift(C.view(rc($("#insight")), { max: 1.75, pad: 30 }), tR + 4.2, 1.6, 1.04);
  const end = tR + 6.0;
  K.washOut(C, end - 0.4);
  window.__beats = { tClick, tDone, tR, end };
  window.__timelines["main"] = C.tl;
})();
