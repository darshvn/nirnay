// Clip 1 — hook. h1 0.60–9.20 (barrel … solver: crudes 4.4, blend 5.8, units 7.2),
// h2 9.20–13.86 (foreign ~10.6, closed ~11.3, licensed ~12.2), h3 13.86–16.80 ("This is NIRNAY" ~15.3).
(function () {
  window.__timelines = window.__timelines || {};
  const { $, $$ } = K;
  const C = K.init();
  const tl = C.tl;
  gsap.set(C.cam, { scale: 1.04, x: -38, y: -21 });
  tl.to(C.cam, { scale: 1.0, x: 0, y: 0, duration: 18, ease: "none" }, 0);
  tl.to(".grid", { x: 64, y: 32, duration: 18, ease: "none" }, 0);
  gsap.set(["#b", "#c"], { opacity: 0 });
  // A: the stakes
  const words = $$("#l1 span");
  words.forEach((w, k) => { gsap.set(w, { opacity: 0, y: 26, display: "inline-block" });
    tl.to(w, { opacity: 1, y: 0, duration: 0.5, ease: "power3.out" }, 0.55 + k * 0.28); });
  gsap.set("#l2", { opacity: 0, y: 20 });
  tl.to("#l2", { opacity: 1, y: 0, duration: 0.6, ease: "power3.out" }, 2.6);
  [["#c1", 4.35], ["#c2", 5.75], ["#c3", 7.15]].forEach(([id, t]) => {
    gsap.set(id, { opacity: 0, y: 30, scale: 0.96 });
    tl.to(id, { opacity: 1, y: 0, scale: 1, duration: 0.55, ease: "back.out(1.6)" }, t);
  });
  tl.to("#a", { opacity: 0, y: -30, duration: 0.45, ease: "power2.in" }, 8.95);
  // B: the dependency
  tl.to("#b", { opacity: 1, duration: 0.4 }, 9.25);
  gsap.set(["#w1", "#w2", "#w3"], { opacity: 0, x: -30 });
  [["#w1", 10.4], ["#w2", 11.15], ["#w3", 11.95]].forEach(([id, t]) =>
    tl.to(id, { opacity: 1, x: 0, duration: 0.5, ease: "power3.out" }, t));
  gsap.set("#brands", { opacity: 0 });
  tl.to("#brands", { opacity: 1, duration: 0.6 }, 12.6);
  tl.to("#b", { opacity: 0, y: -30, duration: 0.45, ease: "power2.in" }, 13.6);
  // C: the answer
  tl.to("#c", { opacity: 1, duration: 0.3 }, 13.9);
  gsap.set("#built", { opacity: 0, y: 16 });
  tl.to("#built", { opacity: 1, y: 0, duration: 0.5, ease: "power3.out" }, 13.95);
  gsap.set("#wm", { opacity: 0, scale: 0.92, transformOrigin: "0% 50%" });
  tl.to("#wm", { opacity: 1, scale: 1, duration: 0.7, ease: "back.out(1.4)" }, 15.1);
  $$("#wm .mark span").forEach((s, k) => { gsap.set(s, { scale: 0 });
    tl.to(s, { scale: 1, duration: 0.45, ease: "back.out(2)" }, 15.1 + k * 0.07); });
  gsap.set(["#tag", "#sih"], { opacity: 0, y: 14 });
  tl.to("#tag", { opacity: 1, y: 0, duration: 0.5 }, 15.9);
  tl.to("#sih", { opacity: 1, y: 0, duration: 0.5 }, 16.3);
  K.washOut(C, 17.9);
  window.__beats = { end: 18.3 };
  window.__timelines["main"] = tl;
})();
