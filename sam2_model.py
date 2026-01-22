import torch
import os
from sam2.build_sam import build_sam2_video_predictor
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
            self.checkpoint = os.path.join(os.getcwd(), 'sam2_repo', 'checkpoints', 'sam2.1_hiera_large.pt')
            self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            self.video_dir = os.path.join(os.getcwd(), 'videos', 'frames')
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

        self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=device)
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

    def segmentVideo (self):
        print('segmenting the video')
        videoSegments = {}
        for outFrameIdx, outObjIds, outMaskLogits in self.predictor.propagate_in_video(self.inference_state):
            videoSegments[outFrameIdx] = {
                outObjectId: (outMaskLogits[i] > 0.0).cpu().numpy()
                for i, outObjectId in enumerate(outObjIds)
            }
        firstFrame = cv2.imread(os.path.join(self.video_dir, self.frame_names[0]))
        height, width = firstFrame.shape[:2]

        output_path = os.path.join(os.getcwd(), 'videos', 'output_segmented.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # adjust to the input videos FPS
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print('Saving video...')
        for outFrameIdx in range(0, len(self.frame_names)):
            frame = cv2.imread(os.path.join(self.video_dir, self.frame_names[outFrameIdx]))

            for out_obj_id, out_mask in videoSegments[outFrameIdx].items():
                frame = showMaskCV(out_mask[0], frame, borders=True)

            # Write frame to video file
            out.write(frame)

            # Optional: also display while saving
            cv2.imshow('Mask Detection', frame)
            cv2.waitKey(1)

        out.release()
        cv2.destroyAllWindows()
        print(f'video saved in: {output_path}')
