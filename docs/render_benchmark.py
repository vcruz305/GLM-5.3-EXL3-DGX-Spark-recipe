#!/usr/bin/env python3
"""Render docs/index.html to a PNG (and report JavaScript errors or unexpected network requests).

  python -m pip install playwright && python -m playwright install chromium
  python docs/render_benchmark.py            # writes benchmark-renders/benchmark.png

No model, CUDA or DGX Spark is needed: the page only displays docs/benchmark-data.json. The script serves docs/ on a
local port, opens the page in headless Chromium and screenshots it.
"""
import functools
import http.server
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "benchmark-renders")


def main() -> int:
    from playwright.sync_api import sync_playwright

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=HERE)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/"
    errors, foreign = [], []
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1200, "height": 900})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: foreign.append(r.url) if not r.url.startswith(base) else None)
        page.goto(base + "index.html")
        page.wait_for_selector("#cards .card")
        out = os.path.join(OUT, "benchmark.png")
        page.screenshot(path=out, full_page=True)
        b.close()
    srv.shutdown()
    print(f"wrote {out}")
    for e in errors:
        print("JS error:", e)
    for u in foreign:
        print("unexpected request:", u)
    return 1 if errors or foreign else 0


if __name__ == "__main__":
    sys.exit(main())
