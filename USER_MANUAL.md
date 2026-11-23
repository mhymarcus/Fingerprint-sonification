# Fingerprint Image Sonification System — User Manual

This manual explains how to convert a fingerprint photo into sound, separate
audio stems, and geometry-analysis data on your own computer.

The system runs locally and offline. It converts visible fingerprint geometry
into sound; it does not perform identity recognition, authentication, or
demographic classification.

## 1. What you need

For the simplest workflow, you only need:

- a computer with Python 3.11 or newer;
- this project;
- one fingerprint photo.

When using the working `cv` analysis mode:

- you do not need SOCOFing or another dataset;
- you do not need to train a model;
- you do not need an ML checkpoint;
- you do not need an internet connection.

### Recommended photo quality

- Format: JPG, JPEG, PNG, or BMP.
- Photograph one finger at a time when possible.
- Make the fingerprint occupy most of the image.
- Keep the image sharply focused and avoid camera shake.
- Use even lighting and avoid strong reflections.
- Use a background that contrasts with the finger.
- Prefer an original photo that has not been compressed by social media.

HEIC/HEIF files are not accepted directly. On macOS, convert one to JPG first:

```bash
sips -s format jpeg "/path/to/input.heic" --out "/path/to/fingerprint.jpg"
```

> Fingerprints are sensitive biometric data. Make sure you have permission to
> use the image, and store the original photo, analysis results, and output
> files securely.

## 2. First-time installation

Open a terminal and enter the project directory:

```bash
cd /path/to/Classic-Fingerprint-Pipeline
```

Create a Python virtual environment:

```bash
python3 -m venv venv
```

Activate the environment.

macOS/Linux:

```bash
source venv/bin/activate
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Install the basic dependencies:

```bash
python -m pip install -r requirements.txt
```

Normal photo sonification does not require `requirements-ml.txt`.

## 3. Quick start

Before each session, enter the project and activate its environment:

```bash
cd /path/to/Classic-Fingerprint-Pipeline
source venv/bin/activate
```

Run the analysis:

```bash
python main.py \
  --input "/path/to/fingerprint.jpg" \
  --output "output/my_fingerprint.wav" \
  --analysis cv \
  --duration 30 \
  --export-stems \
  --debug
```

Replace `/path/to/fingerprint.jpg` with the real location of your photo. Keep
the quotation marks whenever a path contains spaces.

For example:

```bash
python main.py \
  --input "/Users/your-name/Downloads/my fingerprint.jpg" \
  --output "output/my_fingerprint.wav" \
  --analysis cv \
  --duration 30 \
  --export-stems \
  --debug
```

If an output with the same name already exists and you intentionally want to
replace it, add:

```text
--overwrite
```

Complete example:

```bash
python main.py \
  --input "/path/to/fingerprint.jpg" \
  --output "output/my_fingerprint.wav" \
  --analysis cv \
  --duration 30 \
  --export-stems \
  --debug \
  --overwrite
```

## 4. Play and inspect the result

On macOS, play the final mix with:

```bash
open output/my_fingerprint.wav
```

Open the geometry diagnostic image with:

```bash
open output/my_fingerprint_analysis/geometry_analysis.png
```

You can also open the files in Finder, Windows Explorer, a DAW, or
TouchDesigner.

## 5. Output files

If you specify:

```text
--output output/my_fingerprint.wav
```

the system creates:

```text
output/my_fingerprint.wav
output/my_fingerprint_ridge_length.wav
output/my_fingerprint_ridge_position.wav
output/my_fingerprint_ridge_orientation.wav
output/my_fingerprint_ridge_texture.wav
output/my_fingerprint_analysis/
├── geometry.json
├── ridge_segments.csv
├── geometry_analysis.png
├── stems_waveform.png
└── mix_spectrogram.png
```

### Audio files

| File | Contents |
| --- | --- |
| `my_fingerprint.wav` | Final mix of all four sound layers |
| `my_fingerprint_ridge_length.wav` | Sustained tones derived from ridge length and confidence |
| `my_fingerprint_ridge_position.wav` | Timed and pitched events derived from horizontal and vertical position |
| `my_fingerprint_ridge_orientation.wav` | Continuous timbral changes derived from the orientation field |
| `my_fingerprint_ridge_texture.wav` | Detail sounds derived from ridge spacing and curvature |

Audio is mono, 16-bit PCM, and 44.1 kHz by default. All stems use the same
timeline, so they can be layered in a DAW or TouchDesigner without manual
realignment.

### Analysis files

| File | Purpose |
| --- | --- |
| `geometry_analysis.png` | Compare the input, enhancement, ROI, ridge mask, skeleton, and orientation field |
| `geometry.json` | Complete ridge-segment and sound-event data |
| `ridge_segments.csv` | Tabular segment data for Excel, Python, or TouchDesigner |
| `stems_waveform.png` | Compare the timing of the final mix and four stems |
| `mix_spectrogram.png` | Inspect the frequency content of the final sound |

## 6. Check whether the analysis succeeded

Open `geometry_analysis.png` and check that:

1. `Raw image` shows the correct photo.
2. `Foreground ROI mask` mainly covers the finger.
3. `Ridge-level mask` follows visible ridges instead of turning the whole
   finger into one solid region.
4. `Skeleton` preserves the main fingerprint lines.
5. `Orientation field` follows the loop, arch, or whorl direction of the print.

Shadows, skin creases, reflections, and background textures can also be
interpreted as lines. They then become part of the sound. Crop the photo before
running the program if you only want to analyse the central fingerprint area.

## 7. Common command options

Display every available option:

```bash
python main.py --help
```

| Option | Purpose | Example |
| --- | --- | --- |
| `--input` | Select the input image | `--input "photo.jpg"` |
| `--output` | Select the final WAV path | `--output "output/result.wav"` |
| `--analysis` | Select the analysis mode | Use `--analysis cv` |
| `--duration` | Set audio length in seconds | `--duration 45` |
| `--export-stems` | Export four separate sound layers | No value required |
| `--debug` | Export diagnostic images | No value required |
| `--overwrite` | Replace files with the same output name | No value required |
| `--max-events` | Limit the number of sound events | `--max-events 100` |
| `--minimum-ridge-length` | Ignore very short segments | `--minimum-ridge-length 6` |
| `--sample-rate` | Set the audio sample rate | `--sample-rate 44100` |

## 8. Adjust the sound

Use these options to adjust the gain of each layer:

```text
--length-gain
--position-gain
--orientation-gain
--texture-gain
--global-gain
```

Example:

```bash
python main.py \
  --input "/path/to/fingerprint.jpg" \
  --output "output/softer_texture.wav" \
  --analysis cv \
  --duration 45 \
  --length-gain 0.8 \
  --position-gain 0.5 \
  --orientation-gain 0.7 \
  --texture-gain 0.25 \
  --global-gain 0.8 \
  --export-stems \
  --debug
```

Disable one layer:

```bash
python main.py \
  --input "/path/to/fingerprint.jpg" \
  --output "output/no_texture.wav" \
  --analysis cv \
  --disable-layer texture \
  --export-stems
```

Valid layer names are `length`, `position`, `orientation`, and `texture`.
Repeat `--disable-layer` to disable more than one layer.

## 9. Datasets and ML mode

### For normal use

Use this mode when converting a new photo into sound:

```text
--analysis cv
```

This workflow does not require a dataset.

### For research or model development

Use `--analysis ml` only when you have a genuinely trained ridge-segmentation
checkpoint:

```bash
python main.py \
  --input "/path/to/fingerprint.png" \
  --output "output/ml_result.wav" \
  --analysis ml \
  --checkpoint "checkpoints/ridge_segmentation.pt" \
  --export-stems
```

The project currently has no real checkpoint that can be claimed as validated.
SOCOFing provides original and altered fingerprints, but it does not provide
pixel-level ridge masks or orientation annotations. It therefore cannot be
used directly for supervised ridge-segmentation training.

Do not present an untrained U-Net or a synthetic-data smoke test as a real
model result.

## 10. TouchDesigner workflow

1. Create an `Audio File In CHOP` in TouchDesigner.
2. Load the final mix or each stem separately.
3. All stems share the same timeline and can be played together.
4. Possible mappings include:
   - length → large-scale form, brightness, or slow change;
   - position → event triggers and spatial movement;
   - orientation → rotation, direction, or colour deformation;
   - texture → particles, noise, or detail density.

The system does not currently create a `.toe` project automatically and does
not include live camera or scanner input.

## 11. Troubleshooting

### `python: command not found`

Try:

```bash
python3 main.py --help
```

Confirm that Python 3.11 or newer is installed.

### `ModuleNotFoundError`

The virtual environment may not be active, or the dependencies may not be
installed:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
```

### The output file already exists

Choose a new output name, or add this only when you intend to replace the
existing file:

```text
--overwrite
```

### The image cannot be opened

- Confirm that the path is correct.
- Put quotation marks around paths containing spaces.
- Confirm that the format is JPG, JPEG, PNG, or BMP.
- Convert HEIC/HEIF to JPG first.

### The processed image looks similar to the input

Inspect the ridge mask, skeleton, and orientation field in
`geometry_analysis.png`, not only the raw and enhanced grayscale panels. The
segment data and orientation field are what control the geometry sound.

### Too much background or too many creases were extracted

- Crop the image to the main fingerprint region first.
- Retake the photo with better lighting and focus.
- Increase `--minimum-ridge-length` to reject short fragments.
- Treat the diagnostic image as the final manual quality check.

### Matplotlib cache warning

Set a temporary writable cache location:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fingerprint-mpl python main.py \
  --input "/path/to/fingerprint.jpg" \
  --output "output/result.wav" \
  --analysis cv \
  --debug
```

### The sound is too quiet or too loud

Use `--global-gain` and the individual layer gains. To prevent clipping, the
mixer reduces the complete mix when necessary.

## 12. Test the installation

Run the automated tests:

```bash
python -m pytest -q
```

Or first display the command help:

```bash
python main.py --help
```

If `--help` opens normally and the test suite passes, the basic installation is
working.

## 13. Recommended workflow

```text
Keep the original photo
  → copy it before cropping or conversion
  → run CV mode with debug output
  → inspect geometry_analysis.png manually
  → compare the four stems
  → adjust layer gains or crop boundaries
  → import the result into a DAW or TouchDesigner
```

The same image and parameters produce the same output, which makes it possible
to compare different photos or sound settings consistently.
