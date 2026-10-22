"""Safe, non-pickle NumPy feature-cache persistence."""

from pathlib import Path

import numpy as np

REQUIRED_FEATURES = {
    "ball_visible",
    "home_nearest_m",
    "away_nearest_m",
    "ball_gap_frames",
    "home_player_mean_speed_m_s",
    "away_player_mean_speed_m_s",
    "home_player_track_count",
    "away_player_track_count",
}


def save_feature_cache(path: Path | str, features: dict[str, np.ndarray]) -> Path:
    """Write numeric/boolean feature arrays as a compressed NPZ archive."""
    missing = REQUIRED_FEATURES - features.keys()
    if missing:
        raise ValueError(f"cannot save feature cache; missing arrays: {sorted(missing)}")
    for name, values in features.items():
        if np.asarray(values).dtype.kind not in "biuf":
            raise ValueError(f"cannot save non-numeric feature array {name!r}")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez_compressed(handle, **features)
    return destination


def load_feature_cache(path: Path | str, expected_frames: int) -> dict[str, np.ndarray]:
    """Load an NPZ cache without pickle support and check required frame-aligned arrays."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"feature cache not found: {source}")
    with np.load(source, allow_pickle=False) as archive:
        missing = REQUIRED_FEATURES - set(archive.files)
        if missing:
            raise ValueError(f"{source}: missing required feature arrays {sorted(missing)}")
        features = {name: archive[name] for name in archive.files}
    for name, values in features.items():
        if values.ndim == 0 or len(values) != expected_frames:
            raise ValueError(f"{source}: feature {name!r} must have {expected_frames} frame rows")
    if features["ball_visible"].dtype != np.bool_:
        raise ValueError(f"{source}: ball_visible must be boolean")
    return features
