## 3AMimplementation

This project runs SAM2 video segmentation and optionally evaluates predictions against ground-truth masks.

### Prereqs

- Python environment with the project dependencies (PyTorch, OpenCV, NumPy, Matplotlib, Pillow).
- The `sam2_repo` directory must be present with SAM2 configs/checkpoints.
- Extracted video frames in `videos/frames` (see `main.py` helpers).
- **Ground-truth masks folder**: create `videos/gt_masks`.

`gt_masks` should contain the ground-truth segmentation masks for each frame, one image per frame (e.g., `0000.png`, `0001.png`, ...). These masks are binary (foreground > 0) and must match the frame resolution and filenames used by `pred_masks`. They are used by `metrics.evaluate_masks()` to compute IoU and success metrics.

