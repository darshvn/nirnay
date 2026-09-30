"""Screenshots of NIRNAY Studio for the README (python demo/studio/capture.py, server running).

Drives the real app in the system Edge through Playwright: selects each model, presses Solve,
waits for the live run to finish, presses Verify, and captures the page at 1600x1000 (2x DPI).
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8119/"
OUT = Path(__file__).resolve().parents[2] / "docs" / "images" / "studio"
OUT.mkdir(parents=True, exist_ok=True)


def solve_and_verify(page, key, verify=True, timeout=180_000):
    page.click(f'.item[data-key="{key}"]')
    page.wait_for_function(f"window.studio.state.cur && window.studio.state.cur.key === '{key}' && window.studio.state.cur.loaded", timeout=timeout)
    page.click("#solve-btn")
    page.wait_for_function("document.querySelector('#res-status').textContent !== 'RUNNING' && "
                           "document.querySelector('#res-status').textContent !== 'READY'", timeout=timeout)
    page.wait_for_function("!document.querySelector('#solve-btn').classList.contains('busy')", timeout=timeout)
    if verify:
        page.click("#verify-btn")
        page.wait_for_function("document.querySelector('#verify-out').textContent.includes('HiGHS') && "
                               "!document.querySelector('#verify-out').textContent.includes('re-solving')", timeout=timeout)
    page.wait_for_timeout(600)


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    page = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
    page.goto(URL)
    page.wait_for_selector(".item")
    shots = [("refinery", "01_refinery_lp"), ("plan4", "02_plan_milp"), ("unload", "03_unloading_schedule"),
             ("datt256", "04_gpu_large_lp")]
    for key, name in shots:
        solve_and_verify(page, key)
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        print("captured", name)
    page.click('.tab[data-view="bench"]')
    page.wait_for_function("document.querySelector('#speed-big').textContent !== '—'")
    page.wait_for_timeout(500)
    page.screenshot(path=str(OUT / "05_benchmarks.png"), full_page=True)
    page.click('.tab[data-view="source"]')
    page.wait_for_function("document.querySelector('#src-code').textContent.length > 100")
    page.wait_for_timeout(500)
    page.screenshot(path=str(OUT / "06_source.png"), full_page=True)
    print("done")
    b.close()
