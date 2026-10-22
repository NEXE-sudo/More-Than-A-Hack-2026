"""End-to-end tracking load, feature extraction, inference, and CSV output."""

import csv
from pathlib import Path

import numpy as np

from possession.config import Config
from possession.data.loaders import load_smart_tagging, load_tracking
from possession.data.validation import validate_predictions
from possession.features.cache import load_feature_cache
from possession.features.context import tagging_context
from possession.features.tracking import extract_tracking_features
from possession.models.baseline import dead_ball_evidence, evidence_from_features
from possession.temporal.decoder import decode


def predict(
    metadata_path: Path | str | None,
    tracking_path: Path | str | None,
    output_path: Path | str,
    config: Config = Config(),
    expected_frames: int | None = None,
    feature_cache_path: Path | str | None = None,
    smart_tagging_path: Path | str | None = None,
) -> np.ndarray:
    """Generate deterministic frame labels and write a validated frame,label submission."""
    frame_count = config.data.frames if expected_frames is None else expected_frames
    if frame_count <= 0:
        raise ValueError("expected frame count must be positive")
    if feature_cache_path is not None:
        features = load_feature_cache(feature_cache_path, frame_count)
    else:
        if metadata_path is None or tracking_path is None:
            raise ValueError("metadata and tracking paths are required without a feature cache")
        tracking = load_tracking(metadata_path, tracking_path, expected_frames=frame_count)
        features = extract_tracking_features(
            tracking,
            pitch_length_m=config.data.pitch_length_m,
            pitch_width_m=config.data.pitch_width_m,
            fps=config.data.fps,
        )
    if smart_tagging_path is not None:
        events = load_smart_tagging(smart_tagging_path, config.data.fps)
        features.update(tagging_context(events, frame_count))
    dead_score = dead_ball_evidence(
        features,
        minimum_gap_frames=config.model.dead_ball_min_gap_frames,
        maximum_team_speed_m_s=config.model.dead_ball_max_team_speed_m_s,
        minimum_tracks_per_team=config.model.dead_ball_min_tracks_per_team,
        score=config.model.dead_ball_evidence_score,
    )
    evidence = evidence_from_features(
        features, config.model.distance_scale_m, dead_ball_score=dead_score
    )
    labels = decode(evidence, config.decoder)
    frames = np.arange(frame_count, dtype=np.int64)
    validate_predictions(frames, labels, frame_count)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "label"])
        writer.writerows(zip(frames, labels, strict=True))
    return labels
