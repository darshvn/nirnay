// Clip 7 — end card. e1 0.50–7.70 ("NIRNAY" ~1.3, SIH 2026 ~2–4, Team ZeroCloud ~4–5.4, code open ~6.2).
(function () {
  window.__timelines = window.__timelines || {};
  const { $, $$ } = K;
  const C = K.init();
  const tl = C.tl;
  gsap.set(C.cam, { scale: 1.03, x: -29, y: -16 });
  tl.to(C.cam, { scale: 1, x: 0, y: 0, duration: 9.6, ease: "sine.out" }, 0);
  K.washIn(C);
  gsap.set(["#wm", "#l1", "#l2", "#l3", "#l4", "#cr"], { opacity: 0, y: 20 });
  tl.to("#wm", { opacity: 1, y: 0, duration: 0.7, ease: "power3.out" }, 0.45);
  $$("#wm .mark span").forEach((s, k) => { gsap.set(s, { scale: 0 });
    tl.to(s, { scale: 1, duration: 0.45, ease: "back.out(2)" }, 0.5 + k * 0.08); });
  tl.to("#l1", { opacity: 1, y: 0, duration: 0.5 }, 2.0);
  tl.to("#l2", { opacity: 1, y: 0, duration: 0.5 }, 4.0);
  tl.to("#l3", { opacity: 1, y: 0, duration: 0.5, ease: "back.out(1.6)" }, 5.9);
  tl.to("#l4", { opacity: 1, y: 0, duration: 0.5 }, 6.5);
  tl.to("#cr", { opacity: 1, y: 0, duration: 0.5 }, 1.0);
  tl.to("#hf-wash", { opacity: 1, duration: 0.8, ease: "power1.in" }, 9.0);
  window.__beats = { end: 9.8 };
  window.__timelines["main"] = tl;
})();
