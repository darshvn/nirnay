"""Crop numbered figures out of rendered report pages (used for README images)."""
import re, sys
from pathlib import Path
import numpy as np
from PIL import Image

SRC = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
JOBS = {11: (1, "architecture"), 18: (3, "fill_in"), 19: (4, "kkt_blocks"), 22: (5, "simplex_loop"),
        27: (7, "pdlp_gpu"), 30: (9, "bnb_tree")}
for page, (fig, name) in JOBS.items():
    html = (SRC / f"bb{page}.html").read_text(encoding="utf-8", errors="replace")
    pw = float(re.search(r'<page width="([\d.]+)"', html).group(1))
    words = re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', html)
    cap_y = None
    for i, w in enumerate(words):
        if w[4] == "Figure" and i + 1 < len(words) and words[i + 1][4] == f"{fig}:":
            cap_y = float(w[1]); break
    img = Image.open(SRC / f"pg{page}-{page}.png").convert("RGB")
    s = img.width / pw
    cy = int(cap_y * s) - 6
    a = np.asarray(img).astype(int)
    ink = (a < 235).any(axis=2)
    rows = ink.any(axis=1)
    # walk up from the caption: skip the gap, then take content until a gap of >= 28 px
    y = cy
    while y > 0 and not rows[y]: y -= 1
    bottom = y + 1
    gap = 0
    while y > 0:
        if rows[y]: gap = 0
        else:
            gap += 1
            if gap >= int(0.16 * s * 72 / 72 * 10): break
        y -= 1
    top = y + gap
    cols = ink[top:bottom].any(axis=0)
    xs = np.flatnonzero(cols)
    box = (max(xs[0] - 12, 0), max(top - 12, 0), min(xs[-1] + 12, img.width), bottom + 12)
    img.crop(box).save(OUT / f"fig_{name}.png")
    print(name, box)
