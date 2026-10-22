"""Ball-only features extracted from the shared tracking representation."""

import numpy as np


def ball_trajectory_features(
    ball: np.ndarray,
    fps: int = 25,
    recent_window: int = 5,
    pitch_length_m: float = 105.3,
    pitch_width_m: float = 68.0,
) -> dict[str, np.ndarray]:
    """Compute causal motion from normalized coordinates and short-window travel in metres."""
    values = np.asarray(ball, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("ball coordinates must have shape (frames, 2)")
    if fps <= 0 or recent_window <= 0 or pitch_length_m <= 0 or pitch_width_m <= 0:
        raise ValueError("fps, recent_window, and pitch dimensions must be positive")
    values = values * np.asarray([pitch_length_m, pitch_width_m])
    visible = np.isfinite(values).all(axis=1)
    n_frames = len(values)
    displacement = np.full((n_frames, 2), np.nan, dtype=np.float32)
    reliable = np.zeros(n_frames, dtype=bool)
    if n_frames > 1:
        reliable[1:] = visible[1:] & visible[:-1]
        adjacent = np.flatnonzero(reliable)
        displacement[adjacent] = values[adjacent] - values[adjacent - 1]
    velocity = displacement * fps
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.full(n_frames, np.nan, dtype=np.float32)
    if n_frames > 1:
        acc_reliable = reliable[1:] & reliable[:-1]
        reliable_frames = np.flatnonzero(acc_reliable) + 1
        acceleration[reliable_frames] = (speed[reliable_frames] - speed[reliable_frames - 1]) * fps
    last_seen = np.maximum.accumulate(np.where(visible, np.arange(n_frames), -1))
    gap = np.arange(n_frames) - last_seen
    gap[~visible] = np.maximum(gap[~visible], 1)
    gap[visible] = 0
    travel = np.full(n_frames, np.nan, dtype=np.float32)
    consistent = np.zeros(n_frames, dtype=bool)
    step_lengths = np.linalg.norm(displacement, axis=1)
    for index in range(n_frames):
        start = max(1, index - recent_window + 1)
        steps = step_lengths[start : index + 1]
        step_valid = reliable[start : index + 1]
        if len(steps) and step_valid.all():
            travel[index] = float(np.sum(steps))
            consistent[index] = True
    return {
        "visible": visible,
        "displacement": displacement,
        "velocity": velocity,
        "speed": speed.astype(np.float32),
        "acceleration": acceleration,
        "recent_distance_m": travel,
        "trajectory_reliable": consistent,
        "motion_reliable": reliable,
        "gap_frames": gap.astype(np.int32),
    }
