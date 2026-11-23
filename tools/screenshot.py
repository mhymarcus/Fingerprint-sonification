"""Screenshot the running app so visual changes can be reviewed.

Development helper only; not imported by the pipeline or the app.
"""

import argparse
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8801/")
    parser.add_argument("--out", default=".shots/shot.png")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1600)
    parser.add_argument(
        "--full", action="store_true", help="Capture the full scroll height"
    )
    parser.add_argument("--wait", type=float, default=1.5, help="Extra settle time")
    arguments = parser.parse_args(argv)

    out = Path(arguments.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel="chrome",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--hide-scrollbars"],
        )
        page = browser.new_page(
            viewport={"width": arguments.width, "height": arguments.height},
            device_scale_factor=1,
        )
        page.goto(arguments.url, wait_until="load", timeout=60_000)
        # Streamlit paints a skeleton until its websocket run finishes.
        try:
            page.wait_for_selector("h1", timeout=60_000)
        except Exception as error:  # noqa: BLE001
            print(f"warning: no h1 appeared ({error})", file=sys.stderr)
        page.wait_for_timeout(int(arguments.wait * 1000))
        page.screenshot(path=str(out), full_page=arguments.full)
        browser.close()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
