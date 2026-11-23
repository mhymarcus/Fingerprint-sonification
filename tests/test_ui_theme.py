import re

from src import ui_theme


def test_style_block_is_a_single_style_element_with_fonts():
    block = ui_theme.style_block()
    assert block.startswith("<style>")
    assert block.endswith("</style>")
    assert "fonts.googleapis.com" in block
    assert "--mono" in block


def test_masthead_reads_as_a_plate():
    markup = ui_theme.masthead(
        title="fingerprints",
        credit="Marcus Mak",
        left="fingerprints / 001",
        right="sys.live · 100%",
        timecode="t:00:00:00",
        middle="ridge · flow · grain",
    )
    assert '<h1 class="title">fingerprints</h1>' in markup
    assert 'class="credit"' in markup
    for fragment in ("fingerprints / 001", "sys.live", "t:00:00:00", "ridge"):
        assert fragment in markup
    # Compact: one plate, not a full-viewport hero.
    assert "hero" not in markup
    assert markup.count('class="row"') == 2


def test_section_resets_numbering_layout():
    markup = ui_theme.section("03", "output")
    assert 'class="num">03<' in markup
    assert 'class="name">output<' in markup


def test_readouts_render_one_cell_per_item():
    markup = ui_theme.readouts([("segments", "31"), ("peak", "-3.0 dB")])
    assert markup.count('class="k"') == 2
    assert ">31<" in markup
    assert ">-3.0 dB<" in markup


def test_footer_contains_privacy_and_credit():
    markup = ui_theme.footer("not stored", "Fingerprints — Marcus Mak, 2026")
    assert "not stored" in markup
    assert "Marcus Mak" in markup
    assert re.search(r'class="foot"', markup)
