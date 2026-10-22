"""Conversions between source frame conventions and submission rows."""

import numpy as np


def tracking_ids_to_video_frames(frame_ids: np.ndarray) -> np.ndarray:
    """ATD frame IDs are one-based; return zero-based tactical video frame indices."""
    ids = np.asarray(frame_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("tracking frame IDs must be a one-dimensional integer array")
    return ids - 1


def tagging_seconds_to_frames(seconds: np.ndarray | float, fps: int = 25) -> np.ndarray:
    """Convert Smart Tagging seconds from tactical-video start to nearest frame."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    values = np.asarray(seconds, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("tagging seconds must be finite and non-negative")
    return np.rint(values * fps).astype(np.int64)


def validate_frame_sequence(frame_ids: np.ndarray, expected_frames: int) -> None:
    """Require ordered ATD IDs to span 1..expected_frames exactly."""
    ids = np.asarray(frame_ids)
    expected = np.arange(1, expected_frames + 1, dtype=np.int64)
    if ids.ndim != 1 or not np.array_equal(ids, expected):
        raise ValueError(
            f"tracking frame IDs must be contiguous 1..{expected_frames} "
            "in file order (video frames are IDs minus one)"
        )
