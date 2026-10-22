"""Typed project configuration loaded from TOML."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    frames: int = 199319
    fps: int = 25
    pitch_length_m: float = 105.3
    pitch_width_m: float = 68.0


@dataclass(frozen=True)
class ModelConfig:
    distance_scale_m: float = 5.0
    dead_ball_min_gap_frames: int = 25
    dead_ball_max_team_speed_m_s: float = 0.5
    dead_ball_min_tracks_per_team: int = 4
    dead_ball_evidence_score: float = 0.85


@dataclass(frozen=True)
class DecoderConfig:
    switch_threshold: float = 0.55
    switch_margin: float = 0.15
    switch_frames: int = 3
    start_threshold: float = 0.45
    continue_threshold: float = 0.35
    dead_threshold: float = 0.75
    dead_margin: float = 0.15
    missing_hold_frames: int = 15


@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    decoder: DecoderConfig = DecoderConfig()


def load_config(path: Path | str) -> Config:
    """Read and validate a TOML configuration file."""
    with Path(path).expanduser().resolve().open("rb") as handle:
        raw = tomllib.load(handle)
    try:
        data = DataConfig(**raw.get("data", {}))
        model = ModelConfig(**raw.get("model", {}))
        decoder = DecoderConfig(**raw.get("decoder", {}))
    except TypeError as exc:
        raise ValueError(f"{path}: unknown or invalid configuration field: {exc}") from exc
    if data.frames <= 0 or data.fps <= 0:
        raise ValueError("data.frames and data.fps must be positive")
    if data.pitch_length_m <= 0 or data.pitch_width_m <= 0:
        raise ValueError("pitch dimensions must be positive")
    if model.distance_scale_m <= 0:
        raise ValueError("model.distance_scale_m must be positive")
    if (
        model.dead_ball_min_gap_frames < 1
        or model.dead_ball_min_tracks_per_team < 1
        or model.dead_ball_max_team_speed_m_s < 0
    ):
        raise ValueError("model dead-ball thresholds are invalid")
    if not 0 <= model.dead_ball_evidence_score <= 1:
        raise ValueError("model.dead_ball_evidence_score must be between 0 and 1")
    if decoder.switch_frames < 1 or decoder.missing_hold_frames < 0:
        raise ValueError("decoder frame limits are invalid")
    for name in (
        "switch_threshold",
        "start_threshold",
        "continue_threshold",
        "dead_threshold",
    ):
        if not 0 <= getattr(decoder, name) <= 1:
            raise ValueError(f"decoder.{name} must be between 0 and 1")
    if decoder.switch_margin < 0 or decoder.dead_margin < 0:
        raise ValueError("decoder margins cannot be negative")
    if decoder.continue_threshold > decoder.start_threshold:
        raise ValueError("decoder.continue_threshold cannot exceed start_threshold")
    if decoder.continue_threshold > decoder.switch_threshold:
        raise ValueError("decoder.continue_threshold cannot exceed switch_threshold")
    return Config(data=data, model=model, decoder=decoder)
