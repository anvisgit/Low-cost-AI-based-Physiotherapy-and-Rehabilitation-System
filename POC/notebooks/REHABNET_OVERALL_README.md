# RehabNet Project: Classic + RGDA Pipelines

This project contains two separate versions of the rehab assessment pipeline.

## 1. Classic RehabNet

Folder:

```text
rehabnet_classic_github/
```

Use this if you want the original architecture:

```text
MediaPipe / dataset skeletons
→ rep segmentation
→ scalar features
→ ST-GCN
→ Transformer
→ BiLSTM
→ correctness + quality + exercise heads
→ feedback + hardware payload
```

Main files:

```text
src/rehabnet_classic/data_pipeline.py
src/rehabnet_classic/classic_model.py
src/rehabnet_classic/classic_train.py
src/rehabnet_classic/classic_inference.py
scripts/train_classic.py
scripts/infer_video_classic.py
```

## 2. RGDA RehabNet

Folder:

```text
rehabnet_rgda_github/
```

Use this if you want the improved architecture:

```text
MediaPipe / dataset skeletons
→ rep segmentation
→ scalar + rule features
→ ST-GCN
→ Transformer
→ BiLSTM
→ rule-guided fusion
→ domain-adversarial branch
→ correct / incorrect / review
```

Main files:

```text
src/rehabnet/data_pipeline.py
src/rehabnet/rgda_model.py
scripts/train_rgda.py
scripts/infer_video.py
```

RGDA is recommended for final experiments because it handles mixed datasets more honestly and gives a `review` output when model/rule signals disagree.

## Dataset Setup

Both pipelines expect the dataset CSV files in one folder:

```text
datasets/
  uiprmd.csv
  squats_tabular_timeseries_binary.csv
  keraal_final_binary.csv
```

On Colab, the expected folder can be:

```text
/content/drive/MyDrive/datasets
```

## Install

Open the folder for the pipeline you want.

For Classic:

```bash
cd rehabnet_classic_github
pip install -r requirements.txt
pip install -e .
```

For RGDA:

```bash
cd rehabnet_rgda_github
pip install -r requirements.txt
pip install -e .
```

If MediaPipe gives import issues in Colab, run:

```python
exec(open("scripts/colab_setup.py").read())
```

Then restart the Colab runtime and continue.

## Train Classic RehabNet

```bash
python scripts/train_classic.py \
  --data-dir datasets \
  --output models/rehabnet_best.pth \
  --epochs 60
```

Colab:

```python
!python scripts/train_classic.py \
  --data-dir /content/drive/MyDrive/datasets \
  --output /content/drive/MyDrive/models/rehabnet_best.pth \
  --epochs 60
```

## Test Video With Classic RehabNet

```bash
python scripts/infer_video_classic.py \
  --video path/to/video.mp4 \
  --checkpoint models/rehabnet_best.pth \
  --exercise squat \
  --output results_classic.json
```

## Train RGDA RehabNet

```bash
python scripts/train_rgda.py \
  --data-dir datasets \
  --output models/rgda_rehabnet_best.pth \
  --epochs 40
```

Colab:

```python
!python scripts/train_rgda.py \
  --data-dir /content/drive/MyDrive/datasets \
  --output /content/drive/MyDrive/models/rgda_rehabnet_best.pth \
  --epochs 40
```

## Test Video With RGDA RehabNet

```bash
python scripts/infer_video.py \
  --video path/to/video.mp4 \
  --checkpoint models/rgda_rehabnet_best.pth \
  --exercise squat \
  --output results_rgda.json
```

Colab upload example:

```python
from google.colab import files

uploaded = files.upload()
video_path = next(iter(uploaded))

!python scripts/infer_video.py \
  --video "$video_path" \
  --checkpoint /content/drive/MyDrive/models/rgda_rehabnet_best.pth \
  --exercise squat \
  --output /content/results_rgda.json
```

## Output Meaning

Classic output gives:

```text
CORRECT / INCORRECT
quality score
feedback
torque signal
hardware payload
```

RGDA output gives:

```text
model prediction
rule flags
final decision
quality
torque
```

RGDA final labels:

```text
correct    = model/rules are comfortable with the rep
incorrect  = rules found a clear biomechanical problem
review     = model and rules disagree, or confidence is not strong enough
```

Rule flag order:

```text
[insufficient_rom, asymmetric, jerky, too_fast]
```

## Which Pipeline Should You Use?

Use Classic RehabNet when:

```text
you need to reproduce the original baseline
you want binary correct/incorrect output
you want simpler architecture
```

Use RGDA RehabNet when:

```text
you want the final improved model
you are working with mixed datasets
you want rule-guided feedback
you want correct / incorrect / review instead of forced binary labels
```

## Important Limitation

Both models are trained mainly on:

```text
UI-PRMD
KIMORE
Keraal
```

Uploaded/live MediaPipe videos are a different domain. For best live-video performance, add some MediaPipe-processed videos to training. If only correct videos are available, generate synthetic incorrect keypoint samples from those correct reps.

## Push To GitHub

Recommended: push two separate repos.

Classic:

```bash
cd rehabnet_classic_github
git init
git add .
git commit -m "Initial classic RehabNet pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/rehabnet-classic.git
git push -u origin main
```

RGDA:

```bash
cd rehabnet_rgda_github
git init
git add .
git commit -m "Initial RGDA RehabNet pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/rgda-rehabnet.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

