"""Build one HyperFrames composition for the walkthrough.

    python build_clip.py <clip> <page.html> <duration> [--title "..."]

Concatenates timelines/kit.js + timelines/<clip>.js, runs the skill's build-composition.py
at 1920x1080 against the running Studio (for its CSS), and copies the Studio fonts next to
the composition (the inlined CSS refers to them relatively).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL = Path(os.environ.get("WALKTHROUGH_KIT", Path.home() / ".claude" / "skills" / "ui-walkthrough-video"))
FONTS = HERE.parent / "studio" / "static" / "fonts"

ap = argparse.ArgumentParser()
ap.add_argument("clip")
ap.add_argument("page")
ap.add_argument("duration", type=float)
ap.add_argument("--title", default="NIRNAY walkthrough")
a = ap.parse_args()

tl_dir = HERE / "timelines"
combined = HERE / "renders" / f"{a.clip}.timeline.js"
combined.parent.mkdir(parents=True, exist_ok=True)
combined.write_text((tl_dir / "kit.js").read_text(encoding="utf-8") + "\n" +
                    (tl_dir / f"{a.clip}.js").read_text(encoding="utf-8"), encoding="utf-8")
out = HERE / "renders" / a.clip
cmd = [sys.executable, str(SKILL / "scripts" / "build-composition.py"), "--page", a.page, "--out", str(out),
       "--timeline", str(combined), "--duration", str(a.duration), "--title", a.title,
       "--origin", "http://127.0.0.1:8119", "--bg", "#0B1020", "--size", "1920x1080"]
subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
(out / "fonts").mkdir(exist_ok=True)
for f in FONTS.glob("*.woff2"):
    shutil.copy(f, out / "fonts" / f.name)
print("built", out)
