"""Freeze real NIRNAY Studio states for the video (server must be running on :8119).

For each model the script does what a user does: select, Solve, wait for the live run to finish,
Verify with HiGHS. Then it freezes the whole page (scripts stripped, CSS kept) into
capture/<key>.html. The last model's page is frozen after the Benchmarks and Source tabs have
also been loaded, so that one file holds all three views for the closing clip.
"""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8119/"
OUT = Path(__file__).resolve().parent / "capture"
OUT.mkdir(exist_ok=True)
STRIP = "script,noscript,iframe,template,audio"
FREEZE = (f"(()=>{{const d=document.documentElement.cloneNode(true);"
          f"d.querySelectorAll('{STRIP}').forEach(e=>e.remove());return '<!doctype html>'+d.outerHTML}})()")


def run(page, key):
    page.click(f'.item[data-key="{key}"]')
    page.wait_for_function(f"studio.state.cur && studio.state.cur.key === '{key}' && studio.state.cur.loaded", timeout=180_000)
    page.click("#solve-btn")
    page.wait_for_function("!['RUNNING','READY'].includes(document.querySelector('#res-status').textContent)", timeout=180_000)
    page.wait_for_function("!document.querySelector('#solve-btn').classList.contains('busy')", timeout=180_000)
    page.click("#verify-btn")
    page.wait_for_function("document.querySelector('#verify-out .v') !== null", timeout=180_000)
    page.wait_for_timeout(500)


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    page = b.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(URL)
    page.wait_for_selector(".item")
    facts = {}
    for key in ["refinery", "plan4", "unload", "datt256"]:
        run(page, key)
        if key == "datt256":
            page.click('.tab[data-view="bench"]')
            page.wait_for_function("document.querySelector('#speed-big').textContent !== '—'")
            page.click('.tab[data-view="source"]')
            page.wait_for_function("document.querySelector('#src-code').textContent.length > 100")
            page.click('.tab[data-view="solve"]')
            page.wait_for_timeout(400)
        html = page.evaluate(FREEZE)
        (OUT / f"{key}.html").write_text(html, encoding="utf-8")
        facts[key] = page.evaluate("""() => ({
            status: document.querySelector('#res-status').textContent,
            obj: document.querySelector('#res-obj').textContent,
            time: document.querySelector('#res-time').textContent,
            it: document.querySelector('#res-it').textContent,
            verify: document.querySelector('#verify-out').innerText,
            logLines: document.querySelectorAll('#log .l').length,
            paths: document.querySelectorAll('#chart path.series').length })""")
        print(key, json.dumps(facts[key]))
    (OUT / "facts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    b.close()
