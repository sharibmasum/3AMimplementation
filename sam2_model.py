import os
import sys
import glob
import torch

repo_root = os.path.dirname(os.path.abspath(__file__))
sam2_repo_path = os.path.join(repo_root, "sam2_repo")
if sam2_repo_path not in sys.path:
    sys.path.insert(0, sam2_repo_path)

from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.geometry_aware import MUSt3RGeometryExtractor, MUSt3RLikeGeometryExtractor
from must3r_loader import load_must3r_naver
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def showMask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([0.0, 1.0, 0.0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def showMaskCV(mask, frame, random_color=False, borders=True):
    color = np.array([0.0, 1.0, 0.0, 0.6])  # neon green

    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)

    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    mask_image = (mask_image[:, :, :3] * 255).astype(np.uint8)
    frame = cv2.addWeighted(frame, 1.0, mask_image, 0.6, 0)
    return frame


class SAM2:
    def __init__(self):
        print("starting on paths")
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint = os.path.join(
            self.project_root,
            "sam2_logs",
            "myvideo_geom_must3r",
            "checkpoints",
            "checkpoint.pt",
        )
        self.fallback_checkpoint = os.path.join(
            self.project_root,
            "sam2_repo",
            "checkpoints",
            "sam2.1_hiera_large.pt",
        )
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.video_dir = os.path.join(self.project_root, "videos", "frames")
        if not os.path.isdir(self.video_dir):
            raise FileNotFoundError(
                "Frames directory not found. Create it by extracting frames, e.g.\n"
                "  python3 main.py  # then call getFrames(...) in main.py\n"
                f"Missing path: {self.video_dir}"
            )
        self.must3r_checkpoint = os.path.join(
            self.project_root,
            "must3r_weights",
            "MUSt3R_512.pth",
        )
        self.must3r_image_size = 512
        self.frame_names = [
            p for p in os.listdir(self.video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    def viewFirstFrame(self):
        print('Viewing First Frame...')
        plt.figure(figsize=(12, 8))
        plt.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[0])))
        plt.show()

    def prepareModel(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device {device}")

        # Build model without auto-loading so we can load custom checkpoints with strict=False.
        use_must3r_like = os.getenv("USE_MUST3R_LIKE", "0") == "1"
        geom_dims = "[64,128,256]" if use_must3r_like else "[512,512,512,512]"
        hydra_overrides = [
            f"++model.feature_merger.geometry_dims={geom_dims}",
            "++model.feature_merger.appearance_dim=256",
        ]
        self.predictor = build_sam2_video_predictor(
            self.model_cfg,
            ckpt_path=None,
            device=device,
            hydra_overrides_extra=hydra_overrides,
        )
        checkpoint_path = self.checkpoint
        if not os.path.isfile(checkpoint_path):
            checkpoint_dir = os.path.dirname(self.checkpoint)
            if os.path.isdir(checkpoint_dir):
                candidates = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
                if candidates:
                    checkpoint_path = max(candidates, key=os.path.getmtime)
        if not os.path.isfile(checkpoint_path):
            checkpoint_path = self.fallback_checkpoint
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint} or {self.fallback_checkpoint}"
            )
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if use_must3r_like:
            # MUSt3R-like uses 64/128/256 geometry dims; skip 512-dim merger weights.
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("feature_merger.")
            }
        missing, unexpected = self.predictor.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                f"Loaded checkpoint with missing keys: {len(missing)}, "
                f"unexpected keys: {len(unexpected)}"
            )
        missing_feature_merger = [k for k in missing if k.startswith("feature_merger.")]
        enable_geometry = len(missing_feature_merger) == 0
        forced_disable = os.getenv("DISABLE_GEOMETRY", "0") == "1"
        if forced_disable:
            enable_geometry = False
            print("=" * 72)
            print("GEOMETRY FUSION: FORCED OFF (DISABLE_GEOMETRY=1)")
            print("=" * 72)
        if not enable_geometry and not forced_disable:
            print(
                "Feature merger weights are missing from the checkpoint; "
                "disabling geometry fusion to avoid random-weight regression."
            )
        if not enable_geometry:
            self.predictor.feature_merger = None
            self.predictor.geometry_extractor = None
        else:
            print("=" * 72)
            print("GEOMETRY FUSION: ENABLED (feature_merger weights found)")
            print(f"Checkpoint used: {checkpoint_path}")
            print("Next: will try to load MUSt3R checkpoint for geometry features.")
            print("=" * 72)
        if enable_geometry and use_must3r_like:
            self.predictor.geometry_extractor = MUSt3RLikeGeometryExtractor(
                in_channels=3, feature_dims=(64, 128, 256), return_dict=True
            )
            if self.predictor.feature_merger is None:
                print("Warning: feature_merger is None; geometry features will be unused.")
            print("MUSt3R-like geometry extractor enabled (64/128/256).")
            print("=" * 72)
            print("GEOMETRY FUSION: ACTIVE (MUSt3R-like)")
            print("=" * 72)
        else:
            must3r_model = None
            if enable_geometry and os.path.isfile(self.must3r_checkpoint):
                must3r_model = load_must3r_naver(
                    checkpoint_path=self.must3r_checkpoint,
                    image_size=self.must3r_image_size,
                    device=device,
                )
            if enable_geometry and must3r_model is not None:
                feature_indices_str = os.getenv("MUST3R_FEATURE_INDICES", "0,4,7,11")
                feature_indices = [
                    int(idx.strip())
                    for idx in feature_indices_str.split(",")
                    if idx.strip()
                ]
                self.predictor.geometry_extractor = MUSt3RGeometryExtractor(
                    must3r_model, feature_indices=feature_indices, freeze=True
                )
                if self.predictor.feature_merger is None:
                    print("Warning: feature_merger is None; geometry features will be unused.")
                print(f"MUSt3R geometry extractor enabled with layers {feature_indices}.")
                print("=" * 72)
                print("GEOMETRY FUSION: ACTIVE")
                print(f"MUSt3R checkpoint: {self.must3r_checkpoint}")
                print(f"MUSt3R image_size: {self.must3r_image_size}")
                print(f"Feature indices: {feature_indices}")
                print("=" * 72)
            elif enable_geometry:
                print("MUSt3R checkpoint not found; geometry extractor disabled.")
        require_must3r = os.getenv("REQUIRE_MUST3R", "0") == "1"
        if require_must3r and self.predictor.geometry_extractor is None:
            raise RuntimeError("REQUIRE_MUST3R=1 but MUSt3R geometry extractor is not enabled.")
        self.inference_state = self.predictor.init_state(video_path=self.video_dir)
        self.predictor.reset_state(self.inference_state)

    def maskFirstFrame (self, point, show):
        print('masking first frame')
        frame_idx = 0
        obj_id = 1
        points = np.array(point, dtype=np.float32)

        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[frame_idx])))
            showMask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
            plt.show()

    def _save_palettized_mask(self, mask: np.ndarray, output_path: str) -> None:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        # Ensure binary values (0=background, 1=foreground) for palette consistency.
        mask = (mask > 0).astype(np.uint8)
        img = Image.fromarray(mask, mode="P")
        palette = [0, 0, 0, 0, 255, 0] + [0, 0, 0] * 254
        img.putpalette(palette[:768])
        img.save(output_path)

    def segmentVideo(
        self,
        export_annotations: bool = False,
        annotations_root: str | None = None,
        annotations_video_name: str | None = None,
        pred_masks_dir: str | None = None,
        output_video_path: str | None = None,
    ):
        print('segmenting the video')
        videoSegments = {}
        for outFrameIdx, outObjIds, outMaskLogits in self.predictor.propagate_in_video(self.inference_state):
            videoSegments[outFrameIdx] = {
                outObjectId: (outMaskLogits[i] > 0.0).cpu().numpy()
                for i, outObjectId in enumerate(outObjIds)
            }
        firstFrame = cv2.imread(os.path.join(self.video_dir, self.frame_names[0]))
        height, width = firstFrame.shape[:2]

        output_path = output_video_path or os.path.join(
            self.project_root, 'videos', 'output_segmented.mp4'
        )
        if pred_masks_dir is None:
            pred_masks_dir = os.path.join(self.project_root, 'videos', 'pred_masks')
        os.makedirs(pred_masks_dir, exist_ok=True)
        annotations_dir = None
        if export_annotations:
            annotations_root = annotations_root or os.path.join(
                self.project_root, "videos", "gt_masks"
            )
            annotations_video_name = annotations_video_name or "video1"
            annotations_dir = os.path.join(annotations_root, annotations_video_name)
            os.makedirs(annotations_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # adjust to the input videos FPS
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print('Saving video...')
        for outFrameIdx in range(0, len(self.frame_names)):
            frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[outFrameIdx]))
            combined_mask = np.zeros((height, width), dtype=np.uint8)

            for out_obj_id, out_mask in videoSegments.get(outFrameIdx, {}).items():
                mask_uint8 = out_mask[0].astype(np.uint8)
                combined_mask = np.maximum(combined_mask, mask_uint8)
                frame = showMaskCV(out_mask[0], frame, borders=True)

            pred_mask_path = os.path.join(pred_masks_dir, f"{outFrameIdx:04d}.png")
            cv2.imwrite(pred_mask_path, (combined_mask * 255).astype(np.uint8))
            if annotations_dir is not None:
                annotation_path = os.path.join(annotations_dir, f"{outFrameIdx:04d}.png")
                self._save_palettized_mask(combined_mask, annotation_path)

            # Write frame to video file
            out.write(frame)

            # Optional: also display while saving
            cv2.imshow('Mask Detection', frame)
            cv2.waitKey(1)

        out.release()
        cv2.destroyAllWindows()
        print(f'video saved in: {output_path}')
