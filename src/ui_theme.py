"""Editorial monochrome styling for the web front end.

The visual language is borrowed from instrument panels and printed matter
rather than from dashboards: a compact masthead, hairline rules, letterspaced
monospace readouts, restrained display type, grain, and no colour at all.

This module is framework-free on purpose. It builds strings; ``app.py`` decides
where to put them. That keeps it testable without a browser.
"""

FONTS = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Inter:wght@300;400;500&family=IBM+Plex+Mono:wght@300;400;500&display=swap');"
)

CSS = """
:root {
  --bg: #070708;
  --fg: #ededed;
  --dim: #8b8b8f;
  --faint: #5c5c61;
  --line: rgba(255, 255, 255, 0.13);
  --sans: "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}

html, body {
  background: var(--bg) !important;
  font-family: var(--sans);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

[data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--bg) !important;
}

/* Strip every piece of Streamlit chrome that does not belong to the artwork. */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stSkipToContent"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
[data-testid="stAppIframeResizerAnchor"],
header[data-testid="stHeader"],
#MainMenu, footer {
  display: none !important;
}

[data-testid="stMainBlockContainer"] {
  max-width: 1180px;
  padding: 0 2rem 7rem 2rem;
}

/* Film grain over the whole viewport. */
[data-testid="stApp"]::after {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 9999;
  opacity: 0.05;
  mix-blend-mode: screen;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.82' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* ---------------------------------------------------------------- type --- */

p, li, span, label, input, textarea, .stMarkdown, [data-testid="stMarkdownContainer"] {
  font-family: var(--sans);
  font-weight: 300;
  color: var(--fg);
}

[data-testid="stMarkdownContainer"] p {
  font-size: 14.5px;
  line-height: 1.75;
  color: #cfcfd2;
}

[data-testid="stMarkdownContainer"] a {
  color: var(--fg);
  text-decoration: underline;
  text-underline-offset: 3px;
  text-decoration-color: var(--line);
}

/* ------------------------------------------------------------ masthead --- */

.masthead {
  position: relative;
  padding: 1.15rem 0 1.05rem 0;
  border-bottom: 1px solid var(--line);
  overflow: hidden;
}

.masthead .row {
  position: relative;
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 2rem;
  flex-wrap: wrap;
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.34em;
  text-transform: uppercase;
  color: var(--faint);
}

.masthead .row > span {
  white-space: nowrap;
}

.masthead .title-row {
  position: relative;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 2.5rem;
}

.masthead .title-note {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  padding-bottom: 1.15rem;
  text-align: right;
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--faint);
  white-space: nowrap;
}

.masthead .title {
  position: relative;
  margin: 0.75rem 0 0.55rem 0;
  font-family: var(--sans);
  font-weight: 300;
  font-size: clamp(46px, 7.4vw, 104px);
  line-height: 0.88;
  letter-spacing: -0.045em;
  color: #f5f5f6;
  text-transform: lowercase;
}

.masthead .credit {
  color: var(--faint);
}

/* ------------------------------------------------------------- section --- */

.sec {
  display: flex;
  align-items: center;
  gap: 1.1rem;
  margin: 3rem 0 1.5rem 0;
}

.sec .num {
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.3em;
  color: var(--faint);
}

.sec .name {
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.34em;
  text-transform: uppercase;
  color: var(--dim);
  white-space: nowrap;
}

.sec .line {
  flex: 1;
  height: 1px;
  background: var(--line);
}

.hr {
  height: 1px;
  background: var(--line);
  margin: 2.2rem 0;
}

.lede {
  margin: 1.6rem 0 0 0;
  max-width: 62ch;
  color: #cfcfd2;
  font-size: 15px;
  line-height: 1.8;
}

.micro-line {
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: var(--faint);
}

/* ------------------------------------------------------------ readouts --- */

.readouts {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  margin: 2.4rem 0 2rem 0;
}

.readouts > div {
  padding: 1.5rem 1.2rem 1.4rem 0;
  border-right: 1px solid var(--line);
}

.readouts > div:last-child { border-right: none; }

.readouts .k {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: var(--faint);
}

.readouts .v {
  margin-top: 0.7rem;
  font-size: 27px;
  font-weight: 300;
  letter-spacing: -0.025em;
  color: #f2f2f3;
  font-variant-numeric: tabular-nums;
}

/* ------------------------------------------------------------- controls --- */

[data-testid="stWidgetLabel"] p {
  font-family: var(--mono) !important;
  font-size: 9.5px !important;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: var(--dim) !important;
}

[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button {
  width: 100%;
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.22);
  border-radius: 0;
  color: var(--fg);
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  padding: 0.95rem 1.1rem;
  transition: background 0.18s ease, border-color 0.18s ease, color 0.18s ease;
}

[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
  background: rgba(255, 255, 255, 0.06);
  border-color: rgba(255, 255, 255, 0.5);
  color: #ffffff;
}

[data-testid="stBaseButton-primary"] {
  background: #f2f2f2 !important;
  border-color: #f2f2f2 !important;
  color: #08080a !important;
}

[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primary"] span,
[data-testid="stBaseButton-primary"] div {
  color: #08080a !important;
  opacity: 1 !important;
}

[data-testid="stButton"] button *,
[data-testid="stDownloadButton"] button * {
  font-family: inherit !important;
  font-size: inherit !important;
  letter-spacing: inherit !important;
  color: inherit !important;
  text-transform: inherit !important;
}

[data-testid="stBaseButton-primary"]:hover {
  background: #ffffff !important;
  border-color: #ffffff !important;
}

[data-testid="stBaseButton-primary"]:disabled,
[data-testid="stButton"] button:disabled {
  background: transparent !important;
  border-color: rgba(255, 255, 255, 0.1) !important;
  color: var(--faint) !important;
  opacity: 1;
}

[data-testid="stFileUploaderDropzone"] {
  background: transparent;
  border: 1px dashed rgba(255, 255, 255, 0.2);
  border-radius: 0;
  padding: 1.7rem 1.4rem;
}

[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(255, 255, 255, 0.42);
  background: rgba(255, 255, 255, 0.02);
}

/* Never restyle every span here: the upload icon is a Material Symbols
   ligature, and changing its font prints the literal word instead. */
[data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {
  display: none;
}

[data-testid="stFileUploaderDropzone"] button {
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.24);
  border-radius: 0;
  padding: 0.8rem 1.2rem;
}

[data-testid="stFileUploaderDropzone"] button:hover {
  background: rgba(255, 255, 255, 0.06);
  border-color: rgba(255, 255, 255, 0.5);
}

[data-testid="stFileUploaderDropzone"] button p {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--fg);
  margin: 0;
}

[data-testid="stFileUploaderDropzoneInstructions"] span {
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--dim);
}

[data-testid="stCheckbox"] label p,
[data-testid="stRadioOption"] label p {
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--dim);
}

[data-testid="stExpander"] details {
  border: 1px solid var(--line);
  border-radius: 0;
  background: transparent;
}

[data-testid="stExpander"] summary {
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  color: var(--dim);
}

[data-testid="stExpander"] summary:hover { color: var(--fg); }

[data-testid="stAlert"] {
  background: transparent;
  border: 1px solid var(--line);
  border-radius: 0;
  color: var(--dim);
}

[data-testid="stAlert"] p {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.14em;
  color: var(--dim);
}

[data-testid="stAudio"] audio {
  width: 100%;
  filter: grayscale(1) contrast(1.05);
}

[data-testid="stImage"] img {
  border: 1px solid var(--line);
}

[data-testid="stCaptionContainer"] p {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  line-height: 1.7;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--faint) !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
}

[data-testid="stCheckbox"],
[data-testid="stRadio"],
[data-testid="stSlider"] {
  filter: grayscale(1);
}

/* The theme's near-white primary also paints the tick white, which vanishes
   into the checked box. Force the tick to ink. */
[data-testid="stCheckbox"] svg {
  fill: none;
}

[data-testid="stCheckbox"] svg polyline {
  stroke: #08080a !important;
  stroke-width: 1.7;
}

[data-testid="stCheckbox"] label > div:first-of-type {
  border-radius: 0 !important;
}

[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.16em;
  color: var(--dim);
}

/* ---------------------------------------------------------------- footer --- */

.foot {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 3rem;
  border-top: 1px solid var(--line);
  margin-top: 6rem;
  padding-top: 1.4rem;
}

.foot p {
  font-family: var(--mono);
  font-size: 9.5px;
  line-height: 1.9;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--faint);
  max-width: 62ch;
  margin: 0;
}

@media (max-width: 860px) {
  [data-testid="stMainBlockContainer"] { padding: 0 1.1rem 5rem 1.1rem; }
  .masthead .row { justify-content: flex-start; gap: 0.9rem; }
  .masthead .title-note { display: none; }
  .readouts { grid-template-columns: repeat(2, 1fr); }
  .readouts > div:nth-child(2) { border-right: none; }
  .foot { flex-direction: column; gap: 1rem; }
}
"""


def style_block():
    return f"<style>{FONTS}{CSS}</style>"


def masthead(title, credit, left, right, timecode, middle=""):
    """Compact opening plate: telemetry row, title, credit row, hairline."""
    return f"""
<header class="masthead">
  <div class="row"><span>{left}</span><span>{right}</span></div>
  <div class="title-row">
    <h1 class="title">{title}</h1>
    <div class="title-note"><span>a study in ridge geometry</span><span>four layers · one wav</span></div>
  </div>
  <div class="row">
    <span class="credit">{credit}</span>
    <span>{middle}</span>
    <span>{timecode}</span>
  </div>
</header>
"""


def section(number, label):
    return (
        f'<div class="sec"><span class="num">{number}</span>'
        f'<span class="name">{label}</span><span class="line"></span></div>'
    )


def rule():
    return '<div class="hr"></div>'


def lede(text):
    return f'<p class="lede">{text}</p>'


def micro(text):
    return f'<span class="micro-line">{text}</span>'


def readouts(items):
    """A row of readouts, in the manner of instrument panel figures."""
    cells = "".join(
        f'<div><div class="k">{label}</div><div class="v">{value}</div></div>'
        for label, value in items
    )
    return f'<div class="readouts">{cells}</div>'


def footer(privacy, credit):
    return f"""
<div class="foot">
  <p>{privacy}</p>
  <p>{credit}</p>
</div>
"""
