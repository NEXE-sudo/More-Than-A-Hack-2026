"""Vectorized, missingness-aware ball and player tracking features."""

import numpy as np

from possession.data.validation import TrackingData
from possession.features.ball import ball_trajectory_features


def extract_tracking_features(
    data: TrackingData,
    pitch_length_m: float = 105.3,
    pitch_width_m: float = 68.0,
    nearby_radius_m: float = 5.0,
    fps: int = 25,
) -> dict[str, np.ndarray]:
    """Extract distances, local counts, ball motion, and observation reliability."""
    if pitch_length_m <= 0 or pitch_width_m <= 0 or nearby_radius_m <= 0 or fps <= 0:
        raise ValueError("pitch dimensions, nearby radius, and fps must be positive")
    players = data.players.astype(np.float64, copy=False)
    ball = data.ball.astype(np.float64, copy=False)
    visible = np.isfinite(ball).all(axis=1)
    ball_m = ball * np.asarray([pitch_length_m, pitch_width_m])
    player_m = players * np.asarray([pitch_length_m, pitch_width_m])
    displacement = player_m - ball_m[:, None, :]
    distances = np.linalg.norm(displacement, axis=2)
    distances[~np.isfinite(displacement).all(axis=2)] = np.nan

    teams = data.player_teams
    team_min = np.full((data.n_frames, 2), np.nan, dtype=np.float32)
    nearby = np.zeros((data.n_frames, 2), dtype=np.int16)
    missing_tracks = np.zeros((data.n_frames, 2), dtype=np.int16)
    team_speed = np.full((data.n_frames, 2), np.nan, dtype=np.float32)
    team_track_count = np.zeros((data.n_frames, 2), dtype=np.int16)
    for team in (0, 1):
        team_mask = teams == team
        team_distances = distances[:, team_mask]
        team_track_count[:, team] = np.isfinite(players[:, team_mask]).all(axis=2).sum(axis=1)
        if team_distances.shape[1]:
            finite = np.isfinite(team_distances)
            minimum = np.min(np.where(finite, team_distances, np.inf), axis=1)
            minimum[~finite.any(axis=1)] = np.nan
            team_min[:, team] = minimum
            nearby[:, team] = np.sum(finite & (team_distances <= nearby_radius_m), axis=1)
            missing_tracks[:, team] = np.sum(
                ~np.isfinite(players[:, team_mask]).all(axis=2), axis=1
            )
            speeds = data.player_speed[:, team_mask]
            speed_count = np.isfinite(speeds).sum(axis=1)
            team_speed[:, team] = np.divide(
                np.nansum(speeds, axis=1),
                speed_count,
                out=np.full(data.n_frames, np.nan),
                where=speed_count > 0,
            )
    ordered = np.sort(team_min, axis=1)
    nearest = ordered[:, 0].copy()
    second_nearest = ordered[:, 1].copy()
    both_known = np.isfinite(team_min).all(axis=1)
    distance_margin = np.full(data.n_frames, np.nan, dtype=np.float32)
    distance_margin[both_known] = team_min[both_known, 1] - team_min[both_known, 0]
    team_distance_change = np.full((data.n_frames, 2), np.nan, dtype=np.float32)
    if data.n_frames > 1:
        adjacent = np.isfinite(team_min[1:]).all(axis=1) & np.isfinite(team_min[:-1]).all(axis=1)
        changed = np.flatnonzero(adjacent) + 1
        team_distance_change[changed] = team_min[changed] - team_min[changed - 1]
    last_visible = np.maximum.accumulate(np.where(visible, np.arange(data.n_frames), -1))
    gap = np.arange(data.n_frames) - last_visible
    gap[~visible] = np.maximum(gap[~visible], 1)
    gap[visible] = 0

    motion = np.full((data.n_frames, 2), np.nan, dtype=np.float32)
    consecutive = visible.copy()
    consecutive[0] = False
    consecutive[1:] &= visible[:-1]
    motion[consecutive] = ball_m[consecutive] - ball_m[np.flatnonzero(consecutive) - 1]
    velocity = motion * fps
    speed = np.linalg.norm(velocity, axis=1)

    player_valid = np.isfinite(players).all(axis=2)
    trajectory = ball_trajectory_features(
        data.ball,
        fps=fps,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
    )
    return {
        "ball_visible": visible,
        "ball_x_m": ball_m[:, 0].astype(np.float32),
        "ball_y_m": ball_m[:, 1].astype(np.float32),
        "ball_dx_m": motion[:, 0],
        "ball_dy_m": motion[:, 1],
        "ball_speed_m_s": speed.astype(np.float32),
        "ball_motion_reliable": consecutive,
        "ball_recent_distance_m": trajectory["recent_distance_m"],
        "ball_trajectory_reliable": trajectory["trajectory_reliable"],
        "ball_gap_frames": gap.astype(np.int32),
        "home_nearest_m": team_min[:, 0],
        "away_nearest_m": team_min[:, 1],
        "nearest_player_m": nearest,
        "second_team_nearest_m": second_nearest,
        "nearest_team_distance_margin_m": distance_margin,
        "home_nearby_count": nearby[:, 0],
        "away_nearby_count": nearby[:, 1],
        "home_missing_tracks": missing_tracks[:, 0],
        "away_missing_tracks": missing_tracks[:, 1],
        "home_player_mean_speed_m_s": team_speed[:, 0],
        "away_player_mean_speed_m_s": team_speed[:, 1],
        "home_ball_distance_change_m": team_distance_change[:, 0],
        "away_ball_distance_change_m": team_distance_change[:, 1],
        "home_player_track_count": team_track_count[:, 0],
        "away_player_track_count": team_track_count[:, 1],
        "player_track_count": player_valid.sum(axis=1).astype(np.int16),
    }
