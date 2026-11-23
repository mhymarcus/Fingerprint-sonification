# TouchDesigner patch for the queue installation

One visitor, one print, one piece. A patch that reads the visual package the
website exports, plays the sound, and drives four independent visual systems
from the geometry rather than from a generic FFT.

> This is a build recipe, not a `.toe` file — TouchDesigner projects are a
> binary format, so the operators below have to be created in TouchDesigner.
> Everything you paste is in the code blocks.

---

## 1. What the website gives you

The **visual .zip** download contains:

| File | Role |
| --- | --- |
| `*_mix.wav` | the piece, 44.1 kHz mono. Play this to the audience. |
| `*_controls.wav` | 6 channels at 1000 Hz, 0..1, aligned to sample 0 of the mix. |
| `*_events.json` | every discrete event with the ridge segment behind it. |
| `stems/*.wav` | the four layers separately, if you want to react to one alone. |

The six control channels, in order:

| Ch | Name | Drives |
| --- | --- | --- |
| 0 | `ridge_length` | structure size, decay length, how far things travel |
| 1 | `ridge_position` | timed events, spatial triggers |
| 2 | `ridge_orientation` | slow harmonic movement |
| 3 | `ridge_texture` | detail, turbulence, grain |
| 4 | `flow` | the print's ridge rotation, 0..1 |
| 5 | `energy` | overall envelope of the mix |

Channels 0–3 and 5 are envelopes (0..1, normalised per layer). Channel 4 is the
print's own rotation, unwrapped across the frame — the same curve that drives
the pitch glide, so the picture turns as the sound turns.

Because `controls.wav` and `mix.wav` are the same length and start together,
**the picture cannot drift out of sync with the sound**. No audio capture, no
device loopback, no latency to calibrate.

---

## 2. Network

```text
                      ┌──────────────────┐
                      │ audiofilein MIX  │─── audiodeviceout
                      └──────────────────┘
                      ┌──────────────────┐
                      │ audiofilein CTRL │──┬─ select ch0 ─► length
                      └──────────────────┘  ├─ select ch1 ─► position
                                            ├─ select ch2 ─► orientation
                                            ├─ select ch3 ─► texture
                                            ├─ select ch4 ─► flow
                                            └─ select ch5 ─► energy

  events.json ── python ── table EVENTS ──┬─ CHOP execute ─► pulse bursts
                                          └─ datum (x, y, angle)
```

---

## 3. Operators to create

Create them in this order inside a base COMP called `fingerprint`.

### 3.1 Audio

1. **Audio File In CHOP** → rename `MIX` → File: `…_mix.wav`
2. **Audio File In CHOP** → rename `CTRL` → File: `…_controls.wav`
3. Set both to start at frame 0. If your timeline is not starting at 0, set
   `Play` on both to the same trigger so they start together.
4. **Audio Device Out CHOP** ← `MIX`

> Do **not** resample or trim either file independently — their alignment is the
> whole point.

### 3.2 Control channels

For each channel, add a **Select CHOP** on `CTRL`:

| Select CHOP | Channel pattern |
| --- | --- |
| `sel_length` | `chan1` |
| `sel_position` | `chan2` |
| `sel_orientation` | `chan3` |
| `sel_texture` | `chan4` |
| `sel_flow` | `chan5` |
| `sel_energy` | `chan6` |

Audio File In numbers channels from 1. Verify with the CHOP's channel list.

Add a **Math CHOP** after `sel_flow` if you want it signed (`-1..1`), and after
`sel_energy` a **Math CHOP** with `Range` to shape the response curve.

### 3.3 Events

1. **Table DAT** → rename `EVENTS`
2. **Text DAT** → rename `load_events`, set its extension to `.py`, paste
   section 4.1
3. **Text DAT** → rename `fire_events`, extension `.py`, paste section 4.2
4. **CHOP Execute DAT** → rename `exec`, turn **Frame Start** on, set its
   `Frame Start` DAT to `fire_events`

---

## 4. Scripts

### 4.1 `load_events` — read the JSON once

Run this when a new package is loaded (button, or on start):

```python
import json

PACKAGE = "/path/to/print_events.json"   # set per visitor, or point at a fixed name

def onCook(dat):
    with open(PACKAGE, "r") as handle:
        document = json.load(handle)

    table = op("EVENTS")
    table.clear()
    table.appendRow(["t", "dur", "layer", "x", "y", "angle"])
    for event in document["events"]:
        table.appendRow([
            event["t"], event["dur"], event["layer"],
            event["x"], event["y"], event["angle"],
        ])
    table.appendRow(["duration", document["duration"]])

    # hand the summary to anything that wants it
    op("fingerprint").par.Source = document.get("source", "")
```

### 4.2 `fire_events` — one-shot bursts on the frame clock

```python
# CHOP Execute DAT, Frame Start

LAST = {"frame": -1, "index": 0}

def onFrameStart(chop):
    table = op("EVENTS")
    if table.numRows < 2:
        return

    # seconds since the mix started: keep this in step with the Audio File In
    now = absTime.seconds - op("fingerprint").par.StartSeconds

    rows = table.rows()[1:]
    for row in rows:
        if len(row) < 6:
            continue
        t = float(row[0].val)
        layer = row[2].val
        # fire once, in the frame the event lands
        key = (t, layer)
        if t <= now and key not in LAST and now - t < 0.25:
            LAST[key] = True
            _fire(layer, float(row[3].val), float(row[4].val), float(row[5].val))

def _fire(layer, x, y, angle):
    """Replace with your burst logic. One entry point per layer."""
    if layer == "ridge_position":
        # a point appears where the ridge actually was
        op("emitter_position").par.tx = (x - 0.5) * 2.0
        op("emitter_position").par.ty = (y - 0.5) * 2.0
        op("burst_position").par.trigger.pulse()
    elif layer == "ridge_texture":
        op("burst_texture").par.trigger.pulse()
    elif layer == "ridge_length":
        op("burst_length").par.trigger.pulse()
```

Keep `LAST` small — clear it when a new package loads.

### 4.3 Mapping the geometry

Two rules worth keeping, because they are what makes the piece *this person's*:

- **Position events place emitters at `(x, y)`.** Do not randomise placement —
  the point is that the mark came from a specific place on their print.
- **`flow` rotates the field.** Feed it into a Transform TOP's `rotate` or a
  Lookup CHOP shaping the noise offset, so the visual turns when their ridges
  turn.

---

## 5. Projection notes

You said wall **or** floor, audience in the centre. Two things that change the
design, so decide before building the visuals:

| | Floor | Wall |
| --- | --- | --- |
| Composition | radial, top-down, reads from above | frontal, centred, reads at eye level |
| Shadows | people cast strong shadows from a high projector | less shadowing, but bodies block each other |
| Brightness | needs a lot — floor is far and at a grazing angle | more forgiving |
| Safety | light comes up from below, avoid faces | avoid projecting into eyes |

**A composition that survives both:** centred and radially symmetric, so it
works as a pool of light on the floor and as a mandala on a wall. Build it that
way first; commit to a surface once you have a venue.

Projecting onto people directly is the hardest case: faces need roughly
3–5× the brightness of a wall, and moving bodies break any fixed mapping. If the
audience is the surface, treat them as a soft, moving canvas (large, slow,
high-contrast forms) rather than trying to land detail on a specific body.

---

## 6. Sync checklist before you open the doors

1. `mix.wav` and `controls.wav` are the same duration — check in the CHOP.
2. Both Audio File In CHOPs start on the same frame.
3. A test print plays start to finish with no drift.
4. Run the queue twice: the second print must reset `LAST`, `EVENTS` and every
   burst, or visitor two gets visitor one's ghosts.
5. Kill the network and restart it — the patch should come back with the
   timeline at zero, not mid-print.

---

## 7. What is not built yet

- The queue itself (who is next, whose print is on screen).
- Automatic package hand-off from the website to this machine.
- Any `.toe` file.

Those are the next three pieces; the data contract above is what they will be
built on, so it is worth getting right before the visuals get complicated.

---

## 8. The projector folder (one laptop)

When the site runs on the same machine as TouchDesigner, set a folder and the
app writes every finished piece into it:

```bash
RIDGE_QUEUE_DIR=~/fingerprint-queue streamlit run app.py
```

Then each generation writes:

```text
~/fingerprint-queue/
├── latest/
│   ├── mix.wav
│   ├── controls.wav
│   ├── events.json
│   ├── mask.png        white ridges on black
│   ├── roi.png         white print outline on black
│   └── meta.json       who it was, when
└── archive/20261123-2015/  a copy of each print, last 20 kept
```

TouchDesigner always loads `~/fingerprint-queue/latest/…`, so **the patch never
needs re-pathing between visitors**. To move to the next person, pulse a button
that reloads the three files:

```python
# Text DAT, "load next"
def onPulse(par):
    q = "/Users/you/fingerprint-queue/latest/"
    for name in ("mix", "controls"):
        op("audiofilein_" + name).par.file = q + name + ".wav"
        op("audiofilein_" + name).par.reload.pulse()
    op("load_events").cook(force=True)
    op("EVENTS").clearRows()          # forget the previous visitor
    op("fingerprint").par.StartSeconds = absTime.seconds
```

Without the environment variable nothing is written at all, which is what you
want on the public website.

---

## 9. Particles that gather and scatter

Chosen direction: dust that assembles into the print and lets go again.

### The idea

- All particles start **scattered** in a noise cloud.
- `ridge_length` rises → they **gather** onto the ridges of that print.
- `ridge_length` falls → they **drift apart** again.
- `flow` **rotates** the whole cloud, so it turns as the ridges turn.
- `texture` adds **turbulence**, proportional to how tightly packed the ridges are.
- `ridge_position` events send a **burst** from one spot on the print.

### The target positions

Sampling `mask.png` gives you points that lie exactly on the person's ridges.
A Text DAT (`make_targets`, 30 lines) can do it:

```python
# Text DAT: make_targets
import random

TARGETS = op("targets")          # Table DAT
MASK = op("mask")                # Movie File In TOP, pointing at mask.png
COUNT = 20000

def cook(dat):
    TARGETS.clear()
    TARGETS.appendRow(["tx", "ty", "tz"])
    random.seed(1)
    made = 0
    while made < COUNT:
        u, v = random.random(), random.random()
        r, g, b, a = MASK.sample(u, v)
        if r > 0.5:                      # a ridge pixel
            TARGETS.appendRow([(u - 0.5) * 2, (v - 0.5) * 2, random.random()])
            made += 1
```

`random.seed(1)` matters: the same print should always scatter the same way,
or the artwork is not reproducible.

### The network

| Operator | Type | Purpose |
| --- | --- | --- |
| `mask` | Movie File In TOP | `mask.png` |
| `targets` | Table DAT | positions on the ridges (script above) |
| `target_chop` | DAT to CHOP | turns the table into `tx`/`ty`/`tz` channels |
| `scatter` | Noise CHOP | drifting offsets for the scattered state |
| `gather` | Math CHOP | mixes scatter → targets, driven by `sel_length` |
| `spin` | Transform CHOP | rotates by `sel_flow` |
| `turb` | Noise CHOP | amplitude driven by `sel_texture` |
| `points` | Merge CHOP | final `tx`/`ty`/`tz`/`scale` per particle |
| `particle_geo` | Geo COMP | small circle SOP, **Instancing: on**, Instance CHOP = `points` |
| `cam` / `light` / `render` | Camera / Light / Render TOP | the picture |
| `bloom` → `grain` → `out` | TOPs | glow, film grain, output |

Two things to get right:

- **Instancing, not copies.** Set the Geo COMP's Instancing to CHOP mode with
  `tx`, `ty`, `tz`, `scale` channels — 20k instances stay real-time; 20k
  Copy SOPs will not.
- **Lag the gather.** Put a Lag CHOP after `sel_length` (rise ~0.6 s, fall
  ~1.8 s). Particles that snap into place read as a bug; particles that drift
  in read as intention.

### Framing it for wall *or* floor

Build it centred and radially symmetric around the origin: the cloud fills a
disc. On a wall it reads as a mandala; on a floor it reads as a pool of light
around the seated audience. That way the surface decision can wait until you
have the venue.
