# datasets/

This directory is generated, not committed. `.gitignore` keeps everything here
out of git except this file and `.gitkeep`, because the datasets are large and
some carry non-commercial research licences.

## `ml_ridge/` — paired ridge-segmentation data

Produced by `training/build_ml_dataset.py`:

```bash
python training/build_ml_dataset.py \
  --images-root /path/to/SOCOFing \
  --output datasets/ml_ridge \
  --image-size 128 \
  --train 600 --validation 120 --test 120
```

Layout:

```text
datasets/ml_ridge/
├── images/         # preprocessed 128 x 128 grayscale PNG
├── ridge_masks/    # 128 x 128 binary ridge masks (255 = ridge)
└── manifest.csv
```

`manifest.csv` columns: `image_path`, `ridge_mask_path`, `subject_id`, `split`,
`label_source`, `difficulty`, `source_image`.

## What the labels are — and are not

`label_source` is `cv_baseline`: every mask was produced by the CV ridge
segmentation in `src.geometry`, not by a human annotator. They are
**algorithm-produced references**, so:

- a model trained on them learns to reproduce the CV baseline (distillation);
- Dice/IoU scores measure agreement with that baseline on unseen subjects;
- they are **not** evidence that the ML model outperforms the CV baseline;
- they are not suitable for a paper claim about segmentation quality.

Changing that requires hand-verified masks. Put them in the same layout with
`label_source=manual` and the training script will report the provenance.

## Splits

Subjects are split 70/15/15 and the splits are subject-disjoint, so no subject
appears in both training and test. Images are sampled evenly across SOCOFing's
`Real`, `Altered-Easy`, `Altered-Medium` and `Altered-Hard` groups.

## Correcting masks

```bash
python training/correct_masks.py --dataset datasets/ml_ridge
python training/correct_masks.py --dataset datasets/ml_ridge --export
```

The corrector writes:

```text
datasets/ml_ridge/
├── ridge_masks/          # original CV masks (never modified)
├── ridge_masks_manual/   # human-corrected masks, created on first save
├── review_state.json     # per-sample review status
└── manifest_labeled.csv  # exported, with label_source per row
```

Review status is `edited`, `accepted` or `skipped`. `--export` maps those to
`label_source`: `manual` for edited masks, `cv_baseline_accepted` for masks
reviewed and left unchanged, `cv_baseline` for everything else.

Only the train and validation splits are served; the corrector refuses to open
the test split. Interactive sessions do not touch `manifest.csv`, so you can
always re-export from `review_state.json`.
