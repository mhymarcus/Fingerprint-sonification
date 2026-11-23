# ML-Assisted Multi-Layer Fingerprint Sonification

> New user? Choose a manual: [English](USER_MANUAL.md) ·
> [繁體中文](USER_MANUAL_zh-HK.md). Both cover photo preparation, local
> commands, outputs, sound controls, TouchDesigner, and troubleshooting.

This offline Python project turns explainable fingerprint geometry into a
deterministic, multi-layer sound composition for later manual use in
TouchDesigner.

```text
Fingerprint image
  → preprocessing and enhancement
  → CV or genuine checkpoint-backed ML ridge segmentation
  → skeleton/path geometry extraction
  → four independent rule-based sound layers
  → stems + safely mixed WAV
```

It is an artistic analysis system, not biometric identification, demographic
classification, authentication, or generative music.

## Run it as a website

The project ships a Streamlit front end so anyone can try it in a browser
without installing anything:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload a fingerprint or tick **Use the built-in sample**, pick a duration, and
press **Generate sound**. The page returns a playable mix, downloadable stems
(`.zip`), the geometry analysis figure, the layer waveforms, a spectrogram, and
the per-segment data table.

Deployment to Streamlit Community Cloud is covered in
[DEPLOYMENT.md](DEPLOYMENT.md).

## Current implementation status

| Component | Status |
| --- | --- |
| Traditional CV foreground/ridge analysis | Working |
| Structure-tensor orientation modulo 180° | Working |
| Skeleton graph/path tracing | Working |
| Length, position, density, curvature, spacing measurements | Working |
| Four deterministic audio layers and stems | Working |
| Streamlit web interface | Working |
| Compact PyTorch U-Net architecture | Implemented and smoke-tested |
| Manifest-backed supervised training/evaluation | Implemented |
| Synthetic images with exact ridge/orientation labels | Implemented |
| Real trained segmentation checkpoint | **Available — distilled from the CV baseline** |
| Hand-verified SOCOFing ridge masks | **Not available** |

ML mode never falls back silently to CV. It requires a real checkpoint path.
A checkpoint trained on CV-derived masks has been produced and evaluated, but
those masks are algorithm-produced references, not human annotation — see
[What the ML model actually is](#what-the-ml-model-actually-is).

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv venv
source venv/bin/activate                 # macOS/Linux
python -m pip install -r requirements.txt
```

For tests and development tooling:

```bash
python -m pip install -r requirements-dev.txt
```

For U-Net training or ML inference:

```bash
python -m pip install -r requirements-ml.txt
```

Windows PowerShell activation:

```powershell
venv\Scripts\Activate.ps1
```

## Run the working CV baseline

```bash
python main.py \
  --input images/sample_fingerprint.png \
  --output output/fingerprint_geometry.wav \
  --analysis cv \
  --duration 30 \
  --export-stems \
  --debug
```

Use `--overwrite` when intentionally replacing existing outputs.

Disable individual sound layers when comparing them:

```bash
python main.py \
  --input images/sample_fingerprint.png \
  --output output/no_texture.wav \
  --analysis cv \
  --disable-layer texture \
  --export-stems
```

Available layer names are `length`, `position`, `orientation`, and `texture`.
Layer balance can be adjusted with `--length-gain`, `--position-gain`,
`--orientation-gain`, `--texture-gain`, and `--global-gain`.

The original pixel-feature system is preserved as an explicit legacy mode:

```bash
python main.py --input images/sample_fingerprint.png --output output/legacy.wav --analysis pixel
```

## Geometry analysis

### Foreground ROI versus ridge mask

The pipeline keeps two distinct masks:

- **ROI mask:** the overall fingerprint-shaped foreground area;
- **ridge mask:** individual dark ridge structures inside the ROI.

The ROI is never presented as a ridge-segmentation result.

### Enhancement and orientation

The CV baseline uses grayscale normalization, CLAHE, mild Gaussian denoising,
adaptive thresholding, small-object filtering, and skeletonization. These
operations improve visibility but do not synthesize new ridges. Skeleton spurs
shorter than a few pixels are pruned before tracing, so hair-like binarisation
noise is not reported as a ridge path.

Orientation is estimated by the structure tensor and represented modulo π.
Smoothing uses the doubled-angle representation `cos(2θ), sin(2θ)`, so 0° and
180° remain equivalent.

### Ridge path measurements

The one-pixel skeleton is treated as an 8-connected graph. Paths are traced
between endpoints and branch pixels. Measurements are:

| Feature | Unit/rule |
| --- | --- |
| Length | Arc length along skeleton pixels; diagonal step = √2 pixels |
| Position | Segment centroid normalized to `0..1` **within the ROI bounding box** |
| Orientation | Axial mean tangent angle in radians modulo π |
| Density | Local fraction of ridge pixels, `0..1` |
| Curvature | Normalized mean local tangent-angle change, `0..1` |
| Confidence | Mean structure-tensor coherence, `0..1` |
| Spacing | Local ridge spacing in pixels, estimated from skeleton line density |
| Tightness | Ridge spacing relative to this print, `0..1` (1.0 = tightest) |

Bounding-box width or height is not used as ridge length.

Positions are normalized against the fingerprint itself, not the padded analysis
canvas. Normalizing against the canvas compressed every ridge into the middle
third of the timeline and left the beginning and end of a 30 s piece silent.

Ridge-pixel density turns out to be almost constant across a binarised print, so
it is recorded for diagnostics but does not drive the sound. **Spacing** — how
tightly the ridges are packed — does vary (roughly 3–10 px on a typical phone
photo) and is what the texture layer uses.

## Four sound layers

All layers share one timeline and are deterministic.

### A — Ridge length

- longest/confident ridge paths become warm sustained tones;
- arc length controls envelope duration/resonance;
- vertical position gently affects frequency.

### B — Ridge position

- horizontal centroid controls event time;
- vertical centroid selects a pentatonic pitch;
- ridge tightness controls note length;
- minimum event spacing and maximum event count prevent crowding;
- bell-like partials provide a separate spectral character.

### C — Ridge orientation

- the per-column mean ridge angle is unwrapped in the doubled-angle domain, so a
  0°/180° crossing counts as continued rotation rather than a jump;
- how far the ridge flow actually rotates sets the glide size: a quarter turn is
  one octave, a half turn two, so a print with uniform orientation stays near
  the base pitch instead of being stretched into a dramatic sweep;
- the pad glides in phase-continuous fashion rather than sitting on a fixed
  carrier;
- rotation rate controls second/third-harmonic balance;
- confidence controls the pad amplitude.

### D — Ridge spacing and curvature

- tight, curved segments create quiet FM-like texture events;
- curvature controls frequency and modulation character;
- tightness controls event amplitude and duration;
- the event count is limited and this layer can be disabled.

Layer gains are applied before mixing. All stems receive the same global fade.
The mix is normalized to exactly -3 dBFS (when there is signal) and every stem is
scaled by that same factor, so relative layer balance is preserved and stems stay
sample-aligned with the master.

## Output files

For `output/fingerprint_geometry.wav` the system writes:

```text
output/fingerprint_geometry.wav
output/fingerprint_geometry_ridge_length.wav
output/fingerprint_geometry_ridge_position.wav
output/fingerprint_geometry_ridge_orientation.wav
output/fingerprint_geometry_ridge_texture.wav
output/fingerprint_geometry_analysis/
├── geometry.json
├── ridge_segments.csv
├── geometry_analysis.png
├── stems_waveform.png         # --debug
└── mix_spectrogram.png        # --debug
```

WAV files are mono, 16-bit PCM and 44.1 kHz by default. Stems are sample-aligned.
`geometry.json` also records the sound events responsible for the composition.

## ML mode

The implemented ML task is **ridge-level segmentation**, not fingerprint ROI
segmentation and not identity classification.

```bash
python main.py \
  --input input/fingerprint.png \
  --output output/ml_fingerprint.wav \
  --analysis ml \
  --checkpoint checkpoints/ridge_segmentation.pt \
  --export-stems
```

Without a valid checkpoint, the command stops explicitly. The ML ridge mask and
CV ridge mask both feed exactly the same orientation, geometry and sound APIs,
which makes their output directly comparable.

The compact U-Net is CPU-capable and outputs ridge logits. It ships with no
pretrained weights; a checkpoint has to be trained (below).

The network is fully convolutional and runs at the pipeline's `--analysis-size`.
It is *not* rescaled to the size recorded in the checkpoint, because downscaling
a high-resolution photo to the training size destroys ridge detail. Use at least
the default 256 px: below roughly 128 px the ridges of a phone photo are only
2–3 px apart and both the CV and ML paths degrade.

### What the ML model actually is

The checkpoint in `checkpoints/ridge_segmentation.pt` was trained on SOCOFing
using **masks produced by the CV baseline**. The manifest records this as
`label_source=cv_baseline`. The weights are not committed (see `.gitignore`);
reproduce them with the commands in [Train a segmentation
model](#train-a-segmentation-model).

Consequences, stated plainly:

- the model **distils** the CV baseline; it does not outperform it;
- the reported Dice/IoU measure **agreement with the CV reference masks on
  unseen subjects**, not segmentation accuracy against ground truth;
- it is a functional, checkpoint-backed ML path that can be fine-tuned once
  hand-verified masks exist — not a research result.

Measured on 120 held-out subjects (subject-disjoint): Dice 0.966, IoU 0.934.
Predictions are smoother than the reference: the learned prior bridges short
gaps that adaptive thresholding leaves behind. To go beyond the baseline, correct
some masks with `training/correct_masks.py` and fine-tune.

## Building a paired ridge-mask dataset

`training/build_ml_dataset.py` turns a directory of fingerprint images into
aligned image/mask pairs at the model resolution, with a manifest:

```bash
python training/build_ml_dataset.py \
  --images-root ../datasets/SOCOFing \
  --output datasets/ml_ridge \
  --image-size 128 \
  --train 600 --validation 120 --test 120 \
  --overwrite
```

It samples evenly across SOCOFing's `Real`, `Altered-Easy`, `Altered-Medium` and
`Altered-Hard` groups and splits 70/15/15 by subject, so no subject spans two
splits. Masks come from the CV baseline; the script never invents annotation and
records the provenance in every manifest row. See `datasets/README.md`.

## Correcting the reference masks

CV masks are a starting point, not annotation. `training/correct_masks.py` opens
a local browser page that shows each print with its mask overlaid, so a
correction is a few seconds of painting:

```bash
python training/correct_masks.py --dataset datasets/ml_ridge
```

- `A` save the corrected mask and advance
- `S` accept the CV mask unchanged
- `X` skip this print
- `B` / `E` brush / erase, `[` `]` brush size, `U` undo, `N` / `P` next / previous

Then turn the review into a training manifest:

```bash
python training/correct_masks.py --dataset datasets/ml_ridge --export
```

`manifest_labeled.csv` records how each mask was obtained:

| `label_source` | Meaning |
| --- | --- |
| `manual` | edited by a human; this is the only human-verified data |
| `cv_baseline_accepted` | reviewed and confirmed correct, still CV-derived |
| `cv_baseline` | not reviewed |

Corrected masks are written to `ridge_masks_manual/` and never overwrite the
originals. The server binds to `127.0.0.1` only, and it refuses to open the
**test** split so held-out evaluation stays honest.

Training on the labeled manifest works without changes — the trainer prints the
`label_source` values it finds and stores them in the checkpoint:

```bash
python training/train_segmentation.py \
  --manifest datasets/ml_ridge/manifest_labeled.csv \
  --checkpoint checkpoints/ridge_segmentation_manual.pt \
  --epochs 25 --image-size 128
```

A model fine-tuned on a few dozen corrected masks is the first version of this
project that can legitimately claim to beat the CV baseline; report Dice
separately for the reviewed and unreviewed subsets.

## Dataset manifest preparation

For the full catalogue (path accounting, licence, hashes, duplicates), prepare
the locally inspected SOCOFing data without altering its BMP files:

```bash
python training/prepare_dataset.py \
  --images-root ../datasets/SOCOFing \
  --output datasets/manifests/socofing.csv \
  --dataset-source SOCOFing \
  --licence noncommercial-research \
  --overwrite
```

The manifest records source, subject/group, hand, finger, alteration, severity,
paired Real path, resolution, pixel hash, split, annotation paths and licence.
Splits are subject-disjoint. Exact cross-subject duplicate images are placed in
`quarantine` rather than leaking between train and test.

The current manifest contains 55,270 canonical images and **zero ridge masks**.
The preparation script leaves annotation paths empty; it does not fabricate
labels, so the full-catalogue manifest is a path/licence index and not a
segmentation training set. For that, use `build_ml_dataset.py` above, which
pairs images with CV-derived reference masks and records their provenance.

For another dataset, provide a parallel mask directory with matching relative
paths and stems. The supervised loader validates that every image has a real
mask and that mask/image dimensions align.

## Synthetic ground truth

`src/ml/synthetic.py` creates deterministic ridge patterns with:

- grayscale image;
- foreground ROI mask;
- exact ridge mask;
- exact orientation field.

These patterns support geometry unit tests and short pipeline demonstrations.
They are not a substitute for a manually checked real-image validation set.

Generate an optional labeled synthetic set:

```bash
python training/generate_synthetic.py \
  --output datasets/synthetic/demo \
  --count 100 \
  --size 128 \
  --overwrite
```

Each sample includes a PNG image, exact ridge-mask PNG and orientation `.npy`.

## Train a segmentation model

The supervised loader refuses a manifest whose records have no real
`ridge_mask_path`, so either build one with `build_ml_dataset.py` or supply your
own masks:

```bash
python training/train_segmentation.py \
  --manifest datasets/ml_ridge/manifest.csv \
  --checkpoint checkpoints/ridge_segmentation.pt \
  --epochs 25 \
  --batch-size 8 \
  --image-size 128 \
  --seed 2026
```

The training loss is BCE-with-logits plus soft Dice loss. The script reports
validation loss, Dice and IoU and saves only the best validation checkpoint.
It prints the manifest's `label_source` values on startup, and the checkpoint
records them, so a distilled model can never be mistaken for a hand-labeled one.
Training is never started by the audio command.

Evaluate a checkpoint, with an optional prediction preview and JSON report:

```bash
python training/evaluate_model.py \
  --manifest datasets/ml_ridge/manifest.csv \
  --checkpoint checkpoints/ridge_segmentation.pt \
  --split test \
  --preview-dir output/ml_eval \
  --output-json output/ml_eval/metrics.json
```

The preview grid shows `image / reference mask / prediction / disagreement` for
each sample, with missed ridge pixels in red and false positives in blue.

## Tests

```bash
python -m pytest -q
```

Tests cover image formats (including BMP), deterministic audio, PCM export,
overwrite safety, orientation direction, exact skeleton arc length, centroids,
ROI-relative positions, spur pruning, ridge spacing, curvature, synthetic
geometry, four aligned stems, layer switching, -3 dBFS normalisation, event
coverage, orientation glide scaling, manifest label refusal, and ML inference at
the analysis resolution (a trained checkpoint is not required).

## TouchDesigner

Import the final mix or individual stems manually with Audio File In CHOPs.
Possible mappings:

- length stem → long-form scale or brightness;
- position stem → timed events and spatial movement;
- orientation stem → slow harmonic deformation;
- texture stem → particle or detail activity;
- final mix spectrum → overall color/energy.

This repository does not create TouchDesigner projects, live streams, webcam
capture, scanner integration, or TouchDesigner Python scripts.

## Limitations and next steps

- The CV ridge mask is a baseline, not pixel-level ground truth.
- SOCOFing has paired alterations but no ridge masks or orientation annotations.
- The shipped checkpoint distils the CV baseline; it cannot beat its teacher.
  Beating it requires hand-verified masks or a licensed annotated dataset.
- The model is scale-sensitive. It was trained on SOCOFing, whose prints are
  ~96 px wide. Running it on a much higher-resolution photo works, but the
  ridge spacing at the analysis resolution should stay in the same rough range
  as the training data.
- Orientation ML is not yet implemented because no verified orientation labels
  are available; the structure tensor remains the baseline.
- Low-resolution fingerprints limit fine curvature accuracy: below roughly
  128 px of analysis width both the CV and ML ridge masks break down.
- Training used 600 images for 25 epochs on CPU (~13 minutes). That is enough to
  demonstrate the path, not enough to claim a production model.
