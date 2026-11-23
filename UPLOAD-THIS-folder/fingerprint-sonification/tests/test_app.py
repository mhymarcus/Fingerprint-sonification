"""Smoke tests for the Streamlit front end.

These run the real app script in Streamlit's test harness, so a broken widget
call or an import error fails the suite instead of surfacing in the browser.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_before_any_upload():
    app = AppTest.from_file(str(APP), default_timeout=120).run()

    assert not app.exception
    assert app.title[0].value.startswith("Fingerprint")
    labels = [button.label for button in app.button]
    assert "Generate sound" in labels
    # Nothing to render yet, but the app must not crash on the empty state.
    assert any("Pick an image" in info.value for info in app.info)


def test_generate_disabled_without_an_image():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    generate = next(button for button in app.button if button.label == "Generate sound")
    assert generate.disabled


def test_sample_print_runs_the_full_pipeline():
    app = AppTest.from_file(str(APP), default_timeout=180).run()

    app.slider[0].set_value(5).run()  # keep the render short
    sample = next(
        checkbox for checkbox in app.checkbox if "sample" in checkbox.label.lower()
    )
    sample.check().run()

    generate = next(button for button in app.button if button.label == "Generate sound")
    assert not generate.disabled
    generate.click().run()

    assert not app.exception
    assert app.session_state["result"] is not None
    summary = app.session_state["result"].summary
    assert summary["segments"] > 0
    assert summary["duration_seconds"] == 5.0

    labels = [metric.label for metric in app.metric]
    assert "Ridge segments" in labels
