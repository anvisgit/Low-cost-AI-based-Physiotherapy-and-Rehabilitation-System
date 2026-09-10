# RGDA RehabNet

Rule-guided domain-adaptive RehabNet for lower-limb physiotherapy assessment.

The project has two layers:

- `rehabnet.data_pipeline`: dataset loading, MediaPipe video processing, rep segmentation, scalar features, and rules.
- `rehabnet.rgda_model`: ST-GCN + Transformer + BiLSTM + scalar/rule fusion + domain-adversarial RGDA model.

## Folder Layout

```text
src/rehabnet/
  data_pipeline.py      # preprocessing, dataset loaders, video-to-reps
  rgda_model.py         # RGDA model architecture and loss
scripts/
  train_rgda.py         # train RGDA model
  infer_video.py        # test a video with a trained checkpoint
  colab_setup.py        # optional MediaPipe audio monkey-patch
requirements.txt
pyproject.toml
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

On Colab, if MediaPipe import fails, run:

```python
exec(open("scripts/colab_setup.py").read())
```

Then restart the runtime after installing dependencies.

## Expected Dataset Files

Place these files in one folder, for example `datasets/`:

```text
datasets/
  uiprmd.csv
  squats_tabular_timeseries_binary.csv
  keraal_final_binary.csv
```

## Train

```bash
python scripts/train_rgda.py --data-dir datasets --output models/rgda_rehabnet_best.pth --epochs 40
```

Colab example:

```python
!pip install -r requirements.txt -q
!pip install -e . -q
!python scripts/train_rgda.py \
  --data-dir /content/drive/MyDrive/datasets \
  --output /content/drive/MyDrive/models/rgda_rehabnet_best.pth \
  --epochs 40
```

## Test A Video

```bash
python scripts/infer_video.py \
  --video path/to/video.mp4 \
  --checkpoint models/rgda_rehabnet_best.pth \
  --exercise squat \
  --output results.json
```

Colab example:

```python
from google.colab import files
uploaded = files.upload()
video_path = next(iter(uploaded))

!python scripts/infer_video.py \
  --video "$video_path" \
  --checkpoint /content/drive/MyDrive/models/rgda_rehabnet_best.pth \
  --exercise squat \
  --output /content/results.json
```

## Output Meaning

The video script prints one line per detected rep:

```text
model=incorrect final=review p_correct=... p_incorrect=... flags=[...]
```

- `model`: neural model prediction.
- `final`: rule-gated final output.
- `flags`: `[insufficient_rom, asymmetric, jerky, too_fast]`.
- `review`: model/rule uncertainty or disagreement; do not treat it as a confident incorrect label.

## Important Notes

The model can train on UI-PRMD, KIMORE, and Keraal, but uploaded/live videos are a different domain. For best real-video behavior, add some MediaPipe-processed videos to training, or use synthetic incorrect samples from correct MediaPipe reps.

