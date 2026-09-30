from pathlib import Path
from playwright.sync_api import sync_playwright
here = Path(__file__).resolve().parent
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1280, "height": 720})
    pg.goto((here / "thumb.html").as_uri()); pg.wait_for_timeout(600)
    pg.screenshot(path=str(here / "thumbnail.png"))
    b.close()
print("ok")
