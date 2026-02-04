"""
Training-only geometry-aware sampling utilities.

These are intentionally lightweight stubs to document the expected interfaces.
They are not used at inference time and do not require geometry at inference.
"""


def field_of_view_aware_sampling(video_frames, camera_metadata, rng):
    """
    Training-only: select frames with overlapping field-of-view (FOV).

    Args:
        video_frames: list/sequence of frames or frame indices.
        camera_metadata: optional per-frame camera/FOV metadata (if available).
        rng: random generator for reproducibility.

    Returns:
        A subset of frames or indices that favor shared FOV for supervision.
    """
    # Stub: return input as-is. Implement in training to enforce FOV overlap.
    return video_frames


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
    # Stub: return input as-is. Implement in training to enforce geometric consistency.
    return frame_pairs

