"""Smoke tests for the Streamlit front end.

These run the real app script in Streamlit's test harness, so a broken widget
call or an import error fails the suite instead of surfacing in the browser.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def _app():
    return AppTest.from_file(str(APP), default_timeout=180).run()


def test_app_renders_before_any_upload():
    app = _app()

    assert not app.exception
    markdown = " ".join(element.value for element in app.markdown)
    assert "fingerprints" in markdown.lower()
    labels = [button.label for button in app.button]
    assert "generate sound" in labels
    assert any("choose an image" in info.value for info in app.info)


def test_hero_carries_the_artwork_title_and_a_lowkey_credit():
    app = _app()
    markdown = " ".join(element.value for element in app.markdown)

    assert "fingerprints / 001" in markdown
    assert "Marcus Mak" in markdown
    # The credit is a peripheral label, not a headline: no <h1> of the name.
    assert '<h1 class="title">fingerprints</h1>' in markdown
    assert "class=\"credit\"" in markdown


def test_generate_disabled_without_an_image():
    app = _app()
    generate = next(button for button in app.button if button.label == "generate sound")
    assert generate.disabled


def test_sample_print_runs_the_full_pipeline():
    app = _app()

    app.slider[0].set_value(5).run()  # keep the render short
    sample = next(
        checkbox for checkbox in app.checkbox if "sample" in checkbox.label.lower()
    )
    sample.check().run()

    generate = next(button for button in app.button if button.label == "generate sound")
    assert not generate.disabled
    generate.click().run()

    assert not app.exception
    assert app.session_state["result"] is not None
    summary = app.session_state["result"].summary
    assert summary["segments"] > 0
    assert summary["duration_seconds"] == 5.0

    markdown = " ".join(element.value for element in app.markdown)
    assert "segments" in markdown
    assert "03" in markdown and "output" in markdown.lower()
