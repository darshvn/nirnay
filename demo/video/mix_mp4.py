"""The skill's mix.py without the WebM/poster exports: narration under each silent render.

    python mix_mp4.py <silent_dir> <out_dir> [clip ...]
"""
import json, os, subprocess, sys

d = json.load(open("script.json", encoding="utf-8"))
SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
for name in sys.argv[3:] or d["videos"]:
    v = d["videos"][name]
    inputs, filt = [], []
    for i, l in enumerate(v["lines"]):
        inputs += ["-i", f"cut/{l['id']}.wav"]
        ms = int(l["at"] * 1000)
        filt.append(f"[{i + 1}:a]adelay={ms}|{ms}[a{i}]")
    n = len(v["lines"])
    filt.append("".join(f"[a{i}]" for i in range(n)) +
                f"amix=inputs={n}:normalize=0,apad,atrim=0:{v['duration']},loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[aout]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{SRC}/{name}.mp4", *inputs, "-filter_complex", ";".join(filt),
                    "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", f"{OUT}/{name}.mp4"], check=True)
    print("mixed", name, flush=True)
