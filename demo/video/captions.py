"""Write captions for the joined video from script.json.

    python captions.py <fade> <out.srt> <transcript.txt>

Each narration line is split into short caption chunks at clause breaks and timed in
proportion to characters over the line's spoken length (the voice's end is found with
silencedetect). Clip offsets follow join60.sh: each clip starts (duration - fade) after the last.
"""
import json, re, subprocess, sys

ORDER = ["c1_hook", "c2_refinery", "c3_plan", "c4_unload", "c5_gpu", "c5b_arch", "c6_proof", "c6b_next", "c7_end"]
fade, srt, txt = float(sys.argv[1]), sys.argv[2], sys.argv[3]
d = json.load(open("script.json", encoding="utf-8"))
# the narration spells numbers out for the voice; captions use digits
WRITTEN = {"twenty twenty-six": "2026", "Ninety of ninety": "90 of 90", "Thirty-three": "33",
           "A hundred and twenty-one of a hundred and thirty-eight": "121 of 138",
           "One and a half million": "1.5 million"}


def written(t):
    for a, b in WRITTEN.items():
        t = t.replace(a, b)
    return t


def speech_end(f):
    r = subprocess.run(["ffmpeg", "-i", f, "-af", "silencedetect=n=-40dB:d=0.12", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    ss = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r)]
    se = re.findall(r"silence_end: ([\d.]+)", r)
    lead = float(se[0]) if se and ss and ss[0] < 0.05 else 0.0
    end = ss[-1] if ss and len(ss) > len(se) else float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f],
        capture_output=True, text=True).stdout)
    return lead, end


def chunks(text, limit=84):
    parts = re.split(r"(?<=[.,;:?!—])\s+", text.strip())
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > limit:
            out.append(cur); cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur: out.append(cur)
    # split anything still too long at the word nearest the middle
    res = []
    for c in out:
        while len(c) > limit:
            w = c.split(" "); k = len(w) // 2
            res.append(" ".join(w[:k])); c = " ".join(w[k:])
        res.append(c)
    return res


def ts(t):
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


cues, lines_txt, t0 = [], [], 0.0
for name in ORDER:
    v = d["videos"][name]
    for l in v["lines"]:
        lead, end = speech_end(f"cut/{l['id']}.wav")
        a, b = t0 + l["at"] + lead, t0 + l["at"] + end
        cs = chunks(written(l["text"])); total = sum(len(c) for c in cs)
        t = a
        for c in cs:
            dt = (b - a) * len(c) / total
            cues.append((t, t + dt, c)); t += dt
        lines_txt.append(written(l["text"]))
    t0 += v["duration"] - fade
with open(srt, "w", encoding="utf-8") as f:
    for k, (a, b, c) in enumerate(cues, 1):
        f.write(f"{k}\n{ts(a)} --> {ts(b)}\n{c}\n\n")
open(txt, "w", encoding="utf-8").write("\n\n".join(lines_txt) + "\n")
print(len(cues), "cues; video length", round(t0 + fade, 2), "s")
