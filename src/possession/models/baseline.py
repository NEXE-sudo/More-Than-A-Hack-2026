"""Interpretable nearest-team baseline and frame-level evidence conversion."""

import numpy as np


def nearest_team_labels(features: dict[str, np.ndarray]) -> np.ndarray:
    """Reproduce the starter's nearest-player team decision; missing ball maps to D."""
    home = np.asarray(features["home_nearest_m"])
    away = np.asarray(features["away_nearest_m"])
    labels = np.full(len(home), "D", dtype="<U1")
    usable = np.isfinite(home) | np.isfinite(away)
    labels[usable] = np.where(
        np.nan_to_num(home[usable], nan=np.inf) <= np.nan_to_num(away[usable], nan=np.inf),
        "H",
        "A",
    )
    return labels


def evidence_from_features(
    features: dict[str, np.ndarray],
    distance_scale_m: float = 5.0,
    dead_ball_score: np.ndarray | None = None,
) -> np.ndarray:
    """Convert nearest-team distances into smooth [H, A, D] evidence scores."""
    if distance_scale_m <= 0:
        raise ValueError("distance_scale_m must be positive")
    home = np.asarray(features["home_nearest_m"], dtype=np.float64)
    away = np.asarray(features["away_nearest_m"], dtype=np.float64)
    visible = np.asarray(features["ball_visible"], dtype=bool)
    scores = np.zeros((len(home), 3), dtype=np.float32)
    scores[:, 0] = np.where(np.isfinite(home), np.exp(-home / distance_scale_m), 0.0)
    scores[:, 1] = np.where(np.isfinite(away), np.exp(-away / distance_scale_m), 0.0)
    scores[~visible, :2] = 0.0
    if dead_ball_score is not None:
        dead = np.asarray(dead_ball_score, dtype=np.float32)
        if dead.shape != (len(home),) or not np.isfinite(dead).all():
            raise ValueError("dead_ball_score must be a finite score for every frame")
        if ((dead < 0) | (dead > 1)).any():
            raise ValueError("dead_ball_score values must be in [0, 1]")
        scores[:, 2] = dead
    return scores


def dead_ball_evidence(
    features: dict[str, np.ndarray],
    minimum_gap_frames: int = 25,
    maximum_team_speed_m_s: float = 0.5,
    minimum_tracks_per_team: int = 4,
    score: float = 0.85,
) -> np.ndarray:
    """Infer a conservative dead-ball cue from sustained ball absence and team inactivity.

    A missing/high ball alone is never sufficient. This heuristic is an unvalidated prior, not
    ground truth; its thresholds must be tested against legitimate labelled data.
    """
    if minimum_gap_frames < 1 or minimum_tracks_per_team < 1 or maximum_team_speed_m_s < 0:
        raise ValueError("dead-ball inference thresholds are invalid")
    if not 0 <= score <= 1:
        raise ValueError("dead-ball score must be in [0, 1]")
    gaps = np.asarray(features["ball_gap_frames"])
    home_speed = np.asarray(features["home_player_mean_speed_m_s"], dtype=np.float64)
    away_speed = np.asarray(features["away_player_mean_speed_m_s"], dtype=np.float64)
    home_tracks = np.asarray(features["home_player_track_count"])
    away_tracks = np.asarray(features["away_player_track_count"])
    lengths = {len(value) for value in (gaps, home_speed, away_speed, home_tracks, away_tracks)}
    if len(lengths) != 1:
        raise ValueError("dead-ball features must have equal frame counts")
    quiet_teams = (
        np.isfinite(home_speed)
        & np.isfinite(away_speed)
        & (home_speed <= maximum_team_speed_m_s)
        & (away_speed <= maximum_team_speed_m_s)
        & (home_tracks >= minimum_tracks_per_team)
        & (away_tracks >= minimum_tracks_per_team)
    )
    return np.where((gaps >= minimum_gap_frames) & quiet_teams, score, 0.0).astype(np.float32)
