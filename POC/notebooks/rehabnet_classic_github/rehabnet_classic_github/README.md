# Classic RehabNet

This is the original non-RGDA pipeline:

```text
MediaPipe / dataset skeletons
→ rep segmentation
→ scalar features
→ ST-GCN
→ Transformer
→ BiLSTM
→ classification + quality + exercise heads
→ feedback + hardware payload
```

It is separate from the RGDA pipeline. Use this repo when you want the original RehabNet architecture.

## Folder Layout

```text
src/rehabnet_classic/
  data_pipeline.py       # dataset loading, video processing, segmentation, scalar rules
  classic_model.py       # original ST-GCN + Transformer + BiLSTM RehabNet
  classic_train.py       # original LOSO training loop
  classic_inference.py   # original video inference and feedback
scripts/
  train_classic.py
  infer_video_classic.py
  colab_setup.py
requirements.txt
pyproject.toml
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Train

```bash
python scripts/train_classic.py \
  --data-dir datasets \
  --output models/rehabnet_best.pth \
  --epochs 60
```

Colab:

```python
!pip install -r requirements.txt -q
!pip install -e . -q
!python scripts/train_classic.py \
  --data-dir /content/drive/MyDrive/datasets \
  --output /content/drive/MyDrive/models/rehabnet_best.pth \
  --epochs 60
```

## Test A Video

```bash
python scripts/infer_video_classic.py \
  --video path/to/video.mp4 \
  --checkpoint models/rehabnet_best.pth \
  --exercise squat \
  --output results_classic.json
```

## Difference From RGDA

Classic RehabNet has a binary correctness head and does not include:

- domain-adversarial source invariance
- explicit rule-flag fusion
- 3-way correct / incorrect / review output
- RGDA multi-task loss

Use `rehabnet_rgda_github` for the upgraded architecture.

