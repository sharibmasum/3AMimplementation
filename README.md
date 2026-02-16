# 3AM Implementation - SAM2 with MUSt3R Geometry Fusion

This project implements geometry-enhanced video segmentation by fusing SAM2's appearance features with MUSt3R's 3D geometric priors. It includes a complete training pipeline, A/B testing framework, and quantitative evaluation against hand-annotated ground truth.

---

## Quick Start

### Prerequisites
- **Hardware**: GPU recommended (12GB+ VRAM) or Apple Silicon Mac (16GB+ unified memory). CPU-only supported but slow.
- **Software**: Python 3.10+, PyTorch 2.0+, CUDA 11.8+ (if using GPU).

### Installation

**1. Clone Repository**
```bash
git clone https://github.com/sharibmasum/3AMimplementation
cd 3AMimplementation
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt

# Install SAM2
cd sam2_repo
pip install -e .
cd ..

# Optional: Install MUSt3R for geometry fusion
pip install git+https://github.com/naver/must3r.git
```

**3. Download Model Weights**

SAM2 checkpoints (~1.5GB):
```bash
cd sam2_repo/checkpoints
./download_ckpts.sh
cd ../..
```

MUSt3R checkpoint (~1.6GB, optional):  
Download `MUSt3R_512.pth` from [MUSt3R releases](https://github.com/naver/must3r) → place in `must3r_weights/MUSt3R_512.pth`.

---

## Usage (Inference & Evaluation)

### Step 1: Extract Video Frames
```bash
# Place your video in videos/ folder, then edit main.py:
# Uncomment: getFrames('your_video.mp4')
python main.py
```

This extracts frames to `videos/frames/` numbered as `0000.jpg`, `0001.jpg`, etc.

### Step 2: Create Ground Truth Masks (for evaluation)

For quantitative evaluation, you need hand-annotated masks:
1. Use [CVAT](https://www.cvat.ai/) (Computer Vision Annotation Tool) or similar.
2. Annotate your target object across all frames.
3. Export masks as PNG files (binary: 0=background, 255=object).
4. Place in `videos/gt_masks/` with matching frame numbers:
   - `videos/gt_masks/0000.png`, `0001.png`, etc.

### Step 3: Select Click Point

Open `videos/frames/0000.jpg` in an image viewer and note pixel coordinates (x, y) of your target object.

### Step 4: Run A/B Comparison
```bash
python ab_iou.py --point "180,296"  # Use your click coordinates
```

**Outputs:**
- `videos/pred_masks_baseline/` — Baseline SAM2 masks
- `videos/pred_masks_geom/` — Geometry-enhanced masks
- `videos/output_segmented_baseline.mp4` — Baseline visualization
- `videos/output_segmented_geom.mp4` — Geometry visualization
- Console prints IoU comparison

**Optional arguments:**
```bash
python ab_iou.py --point "x,y" \
    --gt /path/to/ground_truth_masks \
    --pred-a /path/to/baseline_output \
    --pred-b /path/to/geometry_output
```

---

## Training (Optional)

To train your own fusion weights from scratch:

### 1. Prepare Dataset
```bash
# Extract frames
python -c "from main import getFrames; getFrames('your_video.mp4')"

# Create ground truth masks (place in videos/gt_masks/)
# See annotation instructions above
```

### 2. Create Training File List
```bash
# Create a text file listing your videos
echo "myvideo" > sam2_repo/training/assets/my_video_train_list.txt
```

### 3. Configure Training
Edit `sam2_repo/sam2/configs/sam2.1_training/sam2.1_hiera_l_myvideo_finetune.yaml`:
- Update `dataset.img_folder` to point to your frames (absolute path)
- Update `dataset.gt_folder` to point to your masks (absolute path)
- Adjust `scratch.num_epochs` (default: 5)
- Set `trainer.accelerator` to `cuda`, `mps`, or `cpu`

### 4. Run Training
```bash
cd sam2_repo
python training/train.py \
  --config configs/sam2.1_training/sam2.1_hiera_l_myvideo_finetune.yaml \
  --use-cluster 0
```

### 5. Monitor Training
```bash
# View TensorBoard logs
tensorboard --logdir sam2_logs/myvideo_geom_must3r/tensorboard

# Check training stats
cat sam2_logs/myvideo_geom_must3r/logs/train_stats.json
```

**Training outputs**:
- Checkpoints: `sam2_logs/myvideo_geom_must3r/checkpoints/checkpoint_*.pt`
- Logs: `sam2_logs/myvideo_geom_must3r/logs/log.txt`
- Config: `sam2_logs/myvideo_geom_must3r/config.yaml`

**Expected training time**: 15-20 minutes on Apple M2 Max, ~10 minutes on RTX 3090.

---

## Project Structure

```
3AMimplementation/
├── main.py                 # Main entry point and helper functions
├── ab_iou.py              # A/B comparison script (baseline vs geometry)
├── sam2_model.py          # SAM2 wrapper with MUSt3R integration
├── metrics.py             # IoU evaluation metrics
├── must3r_loader.py       # MUSt3R model loader
├── check_checkpoint_geometry.py  # Utility to check geometry in checkpoints
├── requirements.txt       # Python dependencies
├── sam2_repo/             # SAM2 submodule (from Meta)
│   └── sam2/modeling/geometry_aware.py  # Custom fusion architecture
├── must3r_weights/        # MUSt3R model weights (download separately)
└── videos/
    ├── frames/           # Extracted video frames
    ├── gt_masks/         # Ground-truth annotation masks
    ├── pred_masks_baseline/  # Baseline predictions
    └── pred_masks_geom/      # Geometry-enhanced predictions
```

---

## Runtime Benchmarks

**Apple M1 Pro, 16GB:**
- **Frame extraction**: 1-2 min for 100 frames
- **Ground truth annotation**: 30-60 min (one-time, manual, using CVAT)
- **Training** (5 epochs): 15-20 min
- **Inference (baseline)**: ~3 min for 100 frames (SAM2 only)
- **Inference (geometry)**: ~5 min for 100 frames (SAM2 + MUSt3R + fusion)
- **Total development cycle**: Frame extraction → annotation → training → evaluation ≈ 1-2 hours

**GPU comparison** (estimated for RTX 3090):
- Training: ~10 min (1.5× faster)
- Inference: ~2 min baseline, ~3 min geometry (1.5-2× faster)

---

## Features

- **Baseline SAM2**: Standard video segmentation with appearance-only features
- **Geometry Fusion**: Enhanced with MUSt3R 3D geometry awareness (2.1M trainable params)
- **A/B Testing Framework**: Controlled comparison with environment-variable switching
- **IoU Evaluation**: Three-metric quantitative comparison (all-frames, positive-only, success rate ≥0.5)
- **Video Export**: Visual results as MP4 files with overlay masks

---

## Technical Details

### Architecture
- **Frozen Backbones**: SAM2 (224M params) + MUSt3R (48M params) remain frozen
- **Trainable Fusion**: Custom `FeatureMerger` module (2.1M params) with hierarchical cross-attention
- **Training Strategy**: Transfer learning - freeze backbones, train fusion only
- **3D Positional Encoding**: Combines point clouds + camera rays (6-channel spatial prior)

### Key Components
1. **FeatureMerger** — Multi-head cross-attention fusion with geometric reasoning
2. **MUSt3RGeometryExtractor** — Frozen 3D feature extraction with normalization pipeline
3. **A/B Testing** — Controlled experiments (baseline vs geometry on identical inputs)
4. **Evaluation** — IoU metrics with ground truth alignment

For detailed technical documentation, see the source code comments in `sam2_repo/sam2/modeling/geometry_aware.py`.

---

## Notes

- Geometry fusion is only active if `must3r_weights/MUSt3R_512.pth` exists and trained fusion weights are loaded
- Without MUSt3R weights, only baseline segmentation runs
- Frame extraction skips frames (configurable in `main.py`, default: every 6th frame)
- All paths are relative to project root for portability
- Training requires annotated ground truth masks for at least one video

---

## Citation

If you use this work, please cite:
- **SAM2**: Ravi, N., et al. "SAM 2: Segment Anything in Images and Videos." Meta AI, 2024.
- **MUSt3R**: Leroy, V., et al. "Grounding Image Matching in 3D with MASt3R." Naver Labs Europe, 2024.
- **3AM**: [Original 3AM Project](https://jayisaking.github.io/3AM-Page/) (note: original implementation not publicly available as of February 2026) 

---

## Contact

**Author**: Sharib Masum  
**GitHub**: [3AMimplementation](https://github.com/sharibmasum/3AMimplementation)

Questions? Feel free to open an issue on GitHub or reach out via the repository.
