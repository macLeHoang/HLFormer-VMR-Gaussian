# ICCV25-HLFormer-Gaussian

GaussianFormer-VMR is a video moment retrieval codebase built from HLFormer-style
Gaussian temporal attention and QD-DETR-style decoding. It supports:

- Charades-STA video moment retrieval.
- QVHighlights moment retrieval and highlight scoring.
- Multi-stream video/text features, including SlowFast, CLIP, and BLIP.
- Boundary refinement for high-IoU temporal localization.
- Optional EMA evaluation and checkpointing.

The main entry point is `src/VMR/main_vmr.py`.

## Repository Layout

```text
ICCV25-HLFormer-Gaussian/
  figures/                      # Project images
  src/
    Models/HLFormer/            # Gaussian attention blocks reused by VMR
    Utils/                       # Logging, checkpoint, table, and metric helpers
    VMR/
      Configs/                  # Dataset-specific configs
      Datasets/                 # JSONL + feature loading
      Losses/                   # Set criterion and auxiliary losses
      Models/                   # GaussianFormer-VMR model, matcher, span utils
      Validations/              # VMR/highlight evaluation
      main_vmr.py               # Train/eval entry point
  requirements.txt
  LICENSE
```

## Installation

Create an environment with PyTorch installed for your CUDA version, then install
the remaining dependencies:

```bash
cd ICCV25-HLFormer-Gaussian
pip install -r requirements.txt
```

Example PyTorch installation commands are intentionally not pinned here because
they depend on your CUDA/runtime setup.

## Data Format

Annotations are JSONL files, one sample per line. The loader expects records
with fields similar to:

```json
{
  "qid": 7803,
  "query": "Man in gray top walks from outside to inside.",
  "duration": 150.0,
  "vid": "RoripwjYFp8_360.0_510.0",
  "relevant_windows": [[26, 36]],
  "relevant_clip_ids": [13, 14, 15, 16, 17],
  "saliency_scores": [[2, 2, 3]]
}
```

Feature files are loaded as `.npz` files:

```text
video: <v_feat_dir>/<vid>.npz       key "features" -> (L_vid, D_vid)
text:  <q_feat_dir>/qid<qid>.npz    key "last_hidden_state" -> (L_txt, D_txt)
```

For Charades-STA, the current config expects:

```text
Video streams:
  /content/charades/slowfast_features      D=2304
  /content/charades/clip_features          D=512
  /content/charades/blip_video_features    D=768

Text streams:
  /content/charades/clip_text_features     D=512
  /content/charades/blip_text_features     D=768
```

Edit paths in `src/VMR/Configs/charades.py` or
`src/VMR/Configs/qvhighlights.py` before training.

## Temporal Alignment

The Charades config currently uses:

```python
cfg["v_feat_len_mode"] = "time_grid"
```

This aligns all video feature streams to a canonical duration-based grid before
concatenation:

```text
N = min(round(duration / clip_len), max_v_l)
```

If a feature `.npz` contains a `timestamps` array, those timestamps are used.
Otherwise, each stream is assumed to uniformly cover the full video duration.
Streams are interpolated onto the shared timeline before per-stream L2
normalization.

Legacy modes are still available for ablation:

```python
cfg["v_feat_len_mode"] = "max"  # resample to longest raw stream length
cfg["v_feat_len_mode"] = "min"  # resample to shortest raw stream length
```

## Configuration

Supported dataset names for the CLI:

```text
qvhighlights
charades
```

Important config files:

- `src/VMR/Configs/charades.py`
- `src/VMR/Configs/qvhighlights.py`

Useful settings:

- `model_name`: experiment folder name under `./experiments/<dataset>/`.
- `v_feat_dirs`, `q_feat_dir`: feature directories.
- `v_feat_dims`, `t_feat_dims`: per-stream dimensions.
- `use_multistream_projection`: use separate stream projections before fusion.
- `use_boundary_refinement`: enable the boundary refinement head.
- `use_refined_spans`: prefer final/refined spans at evaluation.
- `eval_span_source_metrics`: log coarse/refined/final span metrics.
- `eval_refine_diagnostics`: log refinement gate and boundary movement diagnostics.
- `val_freq`, `val_full_epoch`: validation schedule.

For clean ablations, change one major setting at a time and use a new
`model_name` to avoid overwriting checkpoints.

## Training

Run from the `src` directory so imports resolve correctly:

```bash
cd ICCV25-HLFormer-Gaussian/src
python VMR/main_vmr.py -d charades --gpu 0
```

QVHighlights:

```bash
cd ICCV25-HLFormer-Gaussian/src
python VMR/main_vmr.py -d qvhighlights --gpu 0
```

Training writes outputs to:

```text
experiments/<dataset_name>/<model_name>/
  best.ckpt
  last.pt
  log.txt
  hyperparams.yaml
```

`best.ckpt` stores the best validation model by the configured primary metric.
`last.pt` stores the latest raw model and, when enabled, EMA weights.

## Evaluation

Evaluate a checkpoint:

```bash
cd ICCV25-HLFormer-Gaussian/src
python VMR/main_vmr.py -d charades --gpu 0 --eval --resume ../experiments/charades_sta/GaussianFormer_VMR_v23/best.ckpt
```

If a test split is configured, eval mode uses it; otherwise it falls back to the
validation loader.

## Metrics

The validation code reports:

- `R1@0.3`, `R1@0.5`, `R1@0.7`
- `mAP@0.3`, `mAP@0.5`, `mAP@0.7`
- `primary`, computed as the mean of `R1@0.5` and `R1@0.7`
- `hl_mAP` and `HIT@1` when highlight labels are available

When `eval_span_source_metrics=True`, the logs also compare:

- `coarse`: decoder span output
- `refined`: `BoundaryRefinementHead` output
- `final`: gate-blended coarse/refined span

These diagnostics are useful for deciding whether boundary refinement helps and
whether `refine_gate_max` is too restrictive.

## Notes On Multi-Stream Fusion

`MultiStreamVidProjection` projects each stream independently, then fuses them
with a learned softmax gate. This is useful for heterogeneous features such as
SlowFast, CLIP, and BLIP, but the gate can become dominated by one stream.

Recommended diagnostics:

- Compare all-stream training against single-stream baselines.
- Enable stream-weight logging when evaluating fusion behavior.
- Keep temporal alignment fixed while testing fusion choices.
- Treat SlowFast dominance as expected on motion-heavy Charades queries unless
  CLIP/BLIP single-stream baselines are competitive.

## Reproducibility Tips

- Keep `seed` fixed for ablations.
- Use a new `model_name` for every run.
- Keep architecture, learning rate, and validation settings unchanged when
  testing a data-loader change such as `v_feat_len_mode`.
- Log `eval_span_source_metrics=True` and `eval_refine_diagnostics=True` during
  development runs so regression sources are visible.

## License

This project is released under the Apache License 2.0. See `LICENSE`.
