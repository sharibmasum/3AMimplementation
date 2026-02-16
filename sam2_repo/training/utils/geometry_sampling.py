"""
Training-only geometry-aware sampling utilities.

These are not used at inference time and do not require geometry at inference.
"""

from typing import Optional

import numpy as np
import torch


def _as_numpy(array):
    if array is None:
        return None
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _get_meta_value(meta, keys):
    if meta is None:
        return None
    for key in keys:
        if key in meta:
            return meta[key]
    return None


def _as_homogeneous(matrix):
    matrix = _as_numpy(matrix)
    if matrix is None:
        return None
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        bottom = np.array([[0, 0, 0, 1]], dtype=matrix.dtype)
        return np.concatenate([matrix, bottom], axis=0)
    raise ValueError("Extrinsics must be 3x4 or 4x4.")


def _get_cam_to_world(meta):
    cam_to_world = _get_meta_value(
        meta, ["cam_to_world", "camera_to_world", "T_cam_to_world"]
    )
    world_to_cam = _get_meta_value(
        meta, ["world_to_cam", "T_world_to_cam"]
    )
    if cam_to_world is None and world_to_cam is None:
        return None
    if cam_to_world is None:
        world_to_cam = _as_homogeneous(world_to_cam)
        return np.linalg.inv(world_to_cam)
    return _as_homogeneous(cam_to_world)


def _get_world_to_cam(meta):
    world_to_cam = _get_meta_value(
        meta, ["world_to_cam", "T_world_to_cam"]
    )
    cam_to_world = _get_meta_value(
        meta, ["cam_to_world", "camera_to_world", "T_cam_to_world"]
    )
    if world_to_cam is None and cam_to_world is None:
        return None
    if world_to_cam is None:
        cam_to_world = _as_homogeneous(cam_to_world)
        return np.linalg.inv(cam_to_world)
    return _as_homogeneous(world_to_cam)


def _get_intrinsics(meta):
    intrinsics = _get_meta_value(meta, ["intrinsics", "K", "camera_intrinsics"])
    if intrinsics is None:
        return None
    return _as_numpy(intrinsics)


def _get_depth(meta):
    return _get_meta_value(meta, ["depth", "depth_map", "depths"])


def _union_mask(segment_loader, frame_idx):
    segments = segment_loader.load(frame_idx)
    masks = []
    if hasattr(segments, "keys") and not isinstance(segments, dict):
        for obj_id in segments.keys():
            mask = segments[obj_id]
            if mask is not None:
                masks.append(mask)
    else:
        for mask in segments.values():
            if mask is not None:
                masks.append(mask)
    if not masks:
        return None
    masks = torch.stack([m.bool() for m in masks], dim=0)
    return masks.any(dim=0)


def _sample_points(xs, ys, max_points, rng):
    if max_points is None or len(xs) <= max_points:
        return xs, ys
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(len(xs), size=max_points, replace=False)
    return xs[idx], ys[idx]


def _project_points(points_cam, intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    u = (x / z) * fx + cx
    v = (y / z) * fy + cy
    return u, v, z


def _backproject(mask, depth, intrinsics, cam_to_world, max_points=None, rng=None):
    mask = _as_numpy(mask)
    depth = _as_numpy(depth)
    intrinsics = _as_numpy(intrinsics)
    if mask is None or depth is None or intrinsics is None or cam_to_world is None:
        return None

    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    xs, ys = _sample_points(xs, ys, max_points, rng)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    if zs.size == 0:
        return None

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = (xs - cx) / fx * zs
    y = (ys - cy) / fy * zs
    points_cam = np.stack([x, y, zs, np.ones_like(zs)], axis=1)
    points_world = (cam_to_world @ points_cam.T).T[:, :3]
    return points_world


def _inside_frustum(points_world, world_to_cam, intrinsics, image_size):
    world_to_cam = _as_homogeneous(world_to_cam)
    intrinsics = _as_numpy(intrinsics)
    if points_world is None or world_to_cam is None or intrinsics is None:
        return None
    num_points = points_world.shape[0]
    points_h = np.concatenate([points_world, np.ones((num_points, 1))], axis=1)
    points_cam = (world_to_cam @ points_h.T).T[:, :3]
    u, v, z = _project_points(points_cam, intrinsics)
    h, w = image_size
    inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return inside


def field_of_view_aware_sampling(
    video_frames,
    camera_metadata,
    rng,
    segment_loader=None,
    tau: float = 0.25,
    max_points: Optional[int] = 20000,
):
    """
    Training-only: select frames with overlapping field-of-view (FOV).

    Args:
        video_frames: list/sequence of frames (VOSFrame or indices).
        camera_metadata: per-frame camera/FOV metadata (if available).
        rng: random generator for reproducibility.
        segment_loader: segment loader used to fetch masks for backprojection.
        tau: overlap ratio threshold.
        max_points: max number of masked points to sample for geometry checks.

    Returns:
        A subset of frames or indices that favor shared FOV for supervision.
    """
    if (
        camera_metadata is None
        or segment_loader is None
        or len(video_frames) <= 1
    ):
        return video_frames

    ref_frame = video_frames[0]
    ref_idx = ref_frame.frame_idx if hasattr(ref_frame, "frame_idx") else ref_frame
    ref_meta = camera_metadata.get(ref_idx)
    if ref_meta is None:
        return video_frames

    ref_intrinsics = _get_intrinsics(ref_meta)
    ref_world_to_cam = _get_world_to_cam(ref_meta)
    ref_depth = _get_depth(ref_meta)
    if ref_intrinsics is None or ref_world_to_cam is None or ref_depth is None:
        return video_frames
    ref_image_size = (
        ref_depth.shape[-2],
        ref_depth.shape[-1],
    )

    kept = [ref_frame]
    for frame in video_frames[1:]:
        frame_idx = frame.frame_idx if hasattr(frame, "frame_idx") else frame
        meta = camera_metadata.get(frame_idx)
        if meta is None:
            continue
        intrinsics = _get_intrinsics(meta)
        cam_to_world = _get_cam_to_world(meta)
        depth = _get_depth(meta)
        if intrinsics is None or cam_to_world is None or depth is None:
            continue

        mask = _union_mask(segment_loader, frame_idx)
        if mask is None:
            continue

        points_world = _backproject(
            mask,
            depth,
            intrinsics,
            cam_to_world,
            max_points=max_points,
            rng=rng,
        )
        if points_world is None:
            continue

        inside = _inside_frustum(
            points_world, ref_world_to_cam, ref_intrinsics, ref_image_size
        )
        if inside is None:
            continue
        ratio = inside.mean() if inside.size > 0 else 0.0
        if ratio > tau:
            kept.append(frame)

    return kept if len(kept) > 1 else video_frames


def geometry_consistent_frame_selection(frame_pairs, geometry_matches=None, rng=None):
    """
    Training-only: pick frame pairs with consistent geometry cues.

    Args:
        frame_pairs: candidate pairs to sample from.
        geometry_matches: optional precomputed correspondence / depth consistency data.
        rng: random generator for reproducibility.

    Returns:
        Filtered or reweighted frame pairs that emphasize geometry-consistent motion.
    """
    # Placeholder for additional geometry filters (not used by default).
    return frame_pairs

