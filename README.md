# EfficientDroneDepth : Lightweight Monocular Depth for Aerial Vision

Goal: train and optimize a lightweight monocular depth model for aerial imagery (TartanAir / VisDrone), focusing on accuracy–latency trade-offs for embedded deployment.

Highlights:
- Fine-tune a compact DPT/MiDaS-style model on aerial data.
- Losses: L1 depth, Scale-Invariant (SI-Log). Metrics: AbsRel, RMSE, δ<1.25.
- Quantization and ONNX export, with CPU latency benchmark.
- Colab notebook for one-click reproducibility.

## Quickstart (Colab)
1. Open `notebooks/EfficientDroneDepth.ipynb` and run all.
2. Set `DATA_ROOT` to your TartanAir subset.
3. Train on a small subset, then scale up.
4. Export ONNX and measure latency.

## Local Setup
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Datasets
- **TartanAir**: https://theairlab.org/tartanair-dataset/
  Expected pairs under `.../left/*_left.*` and `.../depth/*_depth.png`.
  Adjust paths in `src/edd/data.py` if needed.

## Commands
```bash
# smoke test
python -m src.edd.train --data_root DATA_ROOT/tartanair --batch_size 4 --epochs 1 --limit_samples 800

# full
python -m src.edd.train --data_root DATA_ROOT/tartanair --batch_size 8 --epochs 10 --img_size 384

# eval + bench
python -m src.edd.eval --data_root DATA_ROOT/tartanair --ckpt runs/edd_midas.pt --bench

# export + quantize
python -m src.edd.export_onnx --ckpt runs/edd_midas.pt --onnx runs/edd_midas.onnx
python -m src.edd.quantize_dynamic --onnx runs/edd_midas.onnx --out runs/edd_midas_int8.onnx
```
