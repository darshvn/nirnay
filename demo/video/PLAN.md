# NIRNAY demo video — production plan

**Goal:** a 2.5–3 minute narrated walkthrough for YouTube and the SIH26119 submission that makes a
judge want to open the repository. **One message:** *India can own the solver behind its refinery
decisions — and this one already works, verifiably.*

**Format:** 1920×1080, 30 fps, H.264 + AAC, loudness −14 LUFS (YouTube), burned-in-free (captions
as a separate .srt), thumbnail 1280×720. Voice: ElevenLabs "Chris" (natural, calm, Indian-English
friendly pacing ~2.4 words/s), one continuous read per clip.

## Story (one MRPL scenario throughout)

| # | Clip | ~s | What the viewer sees | What the voice says (gist) |
|---|---|---|---|---|
| 1 | Hook | 14 | Dark title card: "Every barrel a refinery processes is decided by a solver." Logos fade: CPLEX · Gurobi · Xpress → "India doesn't own one." → NIRNAY wordmark | The dependency; the stakes; the promise |
| 2 | Crude plan (LP) | 36 | Studio: click *Refinery crude plan* → camera on the model card (93×188, sources) → click **Solve** → live log + infeasibility curve (phase 1 → phase 2) → zoom to **OPTIMAL · $290.64 M/yr** → pan to crude slate, products, shadow prices → click **Verify with HiGHS** → zoom on "✓ same optimum · Δ 2e-15" | A planner's question → answered in milliseconds → with the value of every constraint → and an independent solver agrees |
| 3 | Four months (MILP) | 28 | Click *4-month plan* → Solve → the green "best plan" climbs, the orange "best possible" falls, they meet → OPTIMAL → month cards with crude parcels & FCC mode | Real decisions are whole numbers; branch-and-bound proves no better plan exists |
| 4 | Crude unloading | 24 | *Crude unloading schedule* → Solve (time-lapse) → Gantt draws itself bar by bar → zoom on "Published optimum 79.75 · NIRNAY 79.75" | A published refinery benchmark, reproduced exactly |
| 5 | GPU | 22 | *datt256* (1.5 M nonzeros) → PDLP · GPU → residual curves drop → race bars: GPU ~1 s · CPU 2 s · HiGHS 259 s → "229×" | Where the GPU pays — and why it is one engine of three |
| 6 | Proof + close | 30 | Benchmarks tab: 90/90 · 33/64 · 121/138 count up → "What checking found in the references" → Source tab: every solver import ✓ 0 → end card: repo URL, SIH26119, Team ZeroCloud | Measured, not claimed; honest about speed; open code; invitation |

Total ≈ 154 s + transitions ≈ 2:45.

## Scenarios and how each is handled

| Risk | Handling |
|---|---|
| Live run timing varies (GPU first-call compile, OS load) | Render from **frozen DOM states** of real runs (HyperFrames/GSAP), not a live screen capture; the numbers come from the real run, the pacing is directed |
| The Studio changes after capture | Capture script regenerates all states in one pass (Playwright + Edge, which works here) |
| agent-browser broken here | Use Playwright for freezing pages (same `outerHTML` approach the skill uses) |
| HyperFrames fails to render on this machine | Fallback A: Playwright CDP screencast at 30–60 fps of a scripted run + ffmpeg zoom/pan (crop + scale with eased keyframes) and cursor overlay; Fallback B: Remotion |
| ElevenLabs key invalid / out of credit | Fallback voice: best local open TTS (Kokoro-82M) — decided by the research step; script stays identical |
| ElevenLabs free tier not licensed for commercial use | YouTube demo for a hackathon is non-commercial; if the account is free-tier, note it and offer a regenerate on a paid plan (a few thousand characters) |
| Narration overruns a clip | `place.py` flags overruns; shorten wording, never speed the voice past 1.05× |
| Mispronunciations (NIRNAY, HiGHS, MIPLIB, Maros–Mészáros, PDLP) | `say` map + a pronunciation test per word, verified with speech-to-text (`stt-check.py`) |
| Dead frames while a number is read | Slow push-ins, counters, cursor drift (skill rule: no still > 0.8 s) |
| Claims that the video does not show | Every sentence maps to a visible moment; numbers identical to README/report |
| Viewer on a phone | Zoom hard on key numbers (≥ 2×), large type in cards, captions file |
| Audio levels | Normalise to −14 LUFS, −1 dBTP, gentle room tone under silences |

## Narration draft (≈ 360 words)

1. **Hook.** "Every barrel a refinery processes is decided by an optimisation solver: which crudes to
   buy, how to blend them, when to run each unit. In India, those solvers are foreign, closed, and
   licensed by the seat. This is NIRNAY — a solver India can read, change, and own."
2. **Crude plan.** "Here's a refinery planning model built from public Indian data: MRPL-sized
   distillation units, BS-VI specifications, this year's prices. Press solve. The dual simplex
   drives infeasibility to zero, and the plan is optimal in milliseconds — the crude slate, what to
   make, and what every extra unit of capacity is worth. Then we ask HiGHS, an independent solver,
   to check. Same answer."
3. **MILP.** "Real plans are made of whole decisions: how many crude parcels each month, which mode
   the cracker runs in. Branch-and-bound closes the gap between the best plan found and the best
   still possible — and proves there's nothing better."
4. **Unloading.** "This is a published refinery benchmark: tankers, storage tanks, charging tanks
   and the distillation unit. NIRNAY reproduces the published optimum exactly, and draws the schedule."
5. **GPU.** "For very large models, NIRNAY has a GPU engine. One-and-a-half million nonzeros: about
   a second on a laptop graphics card. The same laptop running HiGHS takes over four minutes."
6. **Proof.** "None of this is a projection. Ninety of ninety classic LPs, thirty-three MIPLIB
   problems proven optimal, a hundred and twenty-one of a hundred and thirty-eight quadratic
   programs — each re-checked by an independent solver. We're slower than HiGHS today, and we say
   so. And there's no solver library inside: every factorisation, every pivot is our code. It's
   all open. NIRNAY — built for SIH twenty-twenty-six by Team ZeroCloud."

## Build steps

1. Capture: Playwright drives the Studio through each clip; freeze full page + each changing fragment.
2. Compose: one HyperFrames project per clip; GSAP timeline with camera, cursor (motion-kit), presses.
3. Check: `hyperframes check`, snapshots at beats, `check-stills.sh`.
4. Voice: `tts.py` (continuous read per clip) → `place.py` → `stt-check.py` → `mix.py`.
5. Join with 0.3 s crossfades; loudness normalise; export MP4 + SRT + thumbnail.
6. Review every clip at full speed before the next; send clips to the user as they finish.
