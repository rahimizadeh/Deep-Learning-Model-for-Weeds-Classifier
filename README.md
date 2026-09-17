# Deep Learning Model for Weed Classification

This repository contains an image dataset of 12 Iranian weed classes and deep-learning experiments for multi-class weed recognition.

Species represented in the project include *Anthemis arvensis*, *Malva sylvestris*, *Oxalis corniculata*, *Prosopis farcta*, *Tribulus terrestris*, *Cuscuta*, *Euphorbia helioscopia*, *Urtica dioica*, *Artemisia annua*, *Phragmites australis*, *Anchusa italica* Retz., and *Setaria*.

## Important methodological correction

The original notebook is retained for historical/reference purposes, but its original workflow has two methodological problems:

1. augmentation images are written back into the dataset before the random train/test split, which can put transformed versions of the same source image into both training and test sets;
2. the manual slicing uses `0:train_end_indx-1` and `train_end_indx:-1`, which drops samples at the split boundaries.

For new experiments, use **`train_leak_free.py`**. It:

- collects the original image files first;
- performs a reproducible stratified 70%/15%/15% train/validation/test split;
- applies stochastic augmentation only inside the model during training;
- leaves validation and test images unaugmented;
- uses a fixed random seed (`42`);
- reports the final score on the untouched test set.

This produces a defensible estimate of model generalization and prevents train/test leakage caused by augmentation.

## Setup

```bash
git clone https://github.com/rahimizadeh/Deep-Learning-Model-for-Weeds-Classifier.git
cd Deep-Learning-Model-for-Weeds-Classifier
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the corrected baseline

```bash
python train_leak_free.py --data "Weeds Dataset" --epochs 20
```

The default output is:

```text
weed_classifier.keras
```

You can change the image size, batch size, number of epochs, or output filename:

```bash
python train_leak_free.py \
  --data "Weeds Dataset" \
  --image-size 224 \
  --batch-size 32 \
  --epochs 30 \
  --output weed_classifier.keras
```

## What to verify before reporting results

- Training, validation and test file lists must be disjoint.
- No augmented files should be pre-generated and mixed back into the source dataset before splitting.
- Model selection/tuning should use the validation set only.
- The test set should be evaluated once at the end of the experiment.
- For publication-quality work, report per-class precision, recall, F1 score, a confusion matrix and class counts in addition to overall accuracy.

## Legacy notebook

`Weeds_Deep_Learning.ipynb` documents the earlier experimental workflow and model exploration. Use it as a reference, but use the corrected split/augmentation strategy above for any new reported performance values.
