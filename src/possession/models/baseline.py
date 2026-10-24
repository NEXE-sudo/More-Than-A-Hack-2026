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
    if not np.isfinite(distance_scale_m) or distance_scale_m <= 0:
        raise ValueError("distance_scale_m must be positive")
    home = np.asarray(features["home_nearest_m"], dtype=np.float64)
    away = np.asarray(features["away_nearest_m"], dtype=np.float64)
    visible = np.asarray(features["ball_visible"], dtype=bool)
    if home.ndim != 1 or away.shape != home.shape or visible.shape != home.shape:
        raise ValueError("nearest distances and ball visibility must be aligned 1D arrays")
    if (
        np.isinf(home).any()
        or np.isinf(away).any()
        or (home[np.isfinite(home)] < 0).any()
        or (away[np.isfinite(away)] < 0).any()
    ):
        raise ValueError("nearest distances must be non-negative finite values or NaN")
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


def revised_evidence_from_features(
    features: dict[str, np.ndarray],
    distance_scale_m: float = 5.0,
    dead_ball_score: np.ndarray | None = None,
    tagging_team_weight: float = 0.0,
    use_proximity: bool = True,
    use_support: bool = True,
    use_approach: bool = True,
) -> np.ndarray:
    """Combine interpretable proximity, local support, and approach evidence.

    Output values are bounded evidence scores, not calibrated probabilities. Correlated
    nearest-distance and distance-margin features are deliberately not counted twice.
    """
    if not np.isfinite(distance_scale_m) or distance_scale_m <= 0:
        raise ValueError("distance_scale_m must be positive")
    if not 0 <= tagging_team_weight <= 0.2:
        raise ValueError("tagging_team_weight must be between 0 and 0.2")
    home = np.asarray(features["home_nearest_m"], dtype=np.float64)
    away = np.asarray(features["away_nearest_m"], dtype=np.float64)
    visible = np.asarray(features["ball_visible"], dtype=bool)
    if home.ndim != 1 or away.shape != home.shape or visible.shape != home.shape:
        raise ValueError("nearest distances and ball visibility must be aligned 1D arrays")
    if (
        np.isinf(home).any()
        or np.isinf(away).any()
        or (home[np.isfinite(home)] < 0).any()
        or (away[np.isfinite(away)] < 0).any()
    ):
        raise ValueError("nearest distances must be non-negative finite values or NaN")

    scores = np.zeros((len(home), 3), dtype=np.float64)
    if use_proximity:
        distances = np.column_stack((home, away))
        proximity = np.where(np.isfinite(distances), np.exp(-distances / distance_scale_m), 0.0)
        scores[:, :2] = 0.75 * proximity

    if use_support:
        for team, key in enumerate(("home_nearby_count", "away_nearby_count")):
            counts = np.asarray(features.get(key, np.zeros(len(home))), dtype=np.float64)
            if counts.shape != home.shape or not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError(f"{key} must contain aligned, finite, non-negative counts")
            scores[:, team] += 0.15 * np.clip(counts / 3.0, 0.0, 1.0)

    if use_approach:
        for team, key in enumerate(("home_ball_distance_change_m", "away_ball_distance_change_m")):
            changes = np.asarray(features.get(key, np.zeros(len(home))), dtype=np.float64)
            if changes.shape != home.shape:
                raise ValueError(f"{key} must align with nearest-distance features")
            if np.isinf(changes).any():
                raise ValueError(f"{key} cannot contain infinite values")
            approaching = np.isfinite(changes) & (changes < 0)
            scores[approaching, team] += 0.1 * np.clip(-changes[approaching] / 2.0, 0.0, 1.0)

    scores[~visible, :] = 0.0
    if tagging_team_weight:
        tagged_home = np.asarray(
            features.get("tagged_home", np.zeros(len(home), dtype=bool)), dtype=bool
        )
        tagged_away = np.asarray(
            features.get("tagged_away", np.zeros(len(home), dtype=bool)), dtype=bool
        )
        if tagged_home.shape != home.shape or tagged_away.shape != home.shape:
            raise ValueError("tagged team masks must align with evidence frames")
        unambiguous_home = tagged_home & ~tagged_away
        unambiguous_away = tagged_away & ~tagged_home
        scores[unambiguous_home, 0] += tagging_team_weight
        scores[unambiguous_away, 1] += tagging_team_weight

    scores[:, :2] = np.clip(scores[:, :2], 0.0, 1.0)
    if dead_ball_score is not None:
        dead = np.asarray(dead_ball_score, dtype=np.float32)
        if dead.shape != home.shape or not np.isfinite(dead).all():
            raise ValueError("dead_ball_score must be a finite score for every frame")
        if ((dead < 0) | (dead > 1)).any():
            raise ValueError("dead_ball_score values must be in [0, 1]")
        scores[:, 2] = dead
    return scores.astype(np.float32)


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
    if (
        minimum_gap_frames < 1
        or minimum_tracks_per_team < 1
        or not np.isfinite(maximum_team_speed_m_s)
        or maximum_team_speed_m_s < 0
    ):
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
