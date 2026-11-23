"""Drive the app through the sample flow and screenshot the result page.

Development helper for reviewing the visual design end to end.
"""

import argparse
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8803/")
    parser.add_argument("--out", default=".shots/flow.png")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=3600)
    parser.add_argument("--duration-steps", type=int, default=0)
    arguments = parser.parse_args(argv)

    out = Path(arguments.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel="chrome",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--hide-scrollbars"],
        )
        page = browser.new_page(viewport={"width": arguments.width, "height": arguments.height})
        page.goto(arguments.url, wait_until="load", timeout=60_000)
        page.wait_for_selector("h1", timeout=60_000)
        page.wait_for_timeout(1200)

        try:
            page.locator('[data-testid="stCheckbox"] input').first.check(force=True)
        except Exception as error:  # noqa: BLE001
            print(f"warning: could not tick sample ({error})", file=sys.stderr)
        page.wait_for_timeout(1200)

        button = page.locator('button:has-text("generate sound")').first
        button.click(timeout=20_000)
        page.wait_for_selector("audio", timeout=180_000)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(out), full_page=False)
        browser.close()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
