import numpy as np

from possession.models.baseline import dead_ball_evidence


def test_dead_ball_cue_requires_long_gap_quiet_teams_and_track_coverage():
    features = {
        "ball_gap_frames": np.array([5, 30, 30, 30]),
        "home_player_mean_speed_m_s": np.array([0.1, 0.1, 1.2, 0.1]),
        "away_player_mean_speed_m_s": np.array([0.1, 0.1, 0.1, 0.1]),
        "home_player_track_count": np.array([5, 5, 5, 2]),
        "away_player_track_count": np.array([5, 5, 5, 5]),
    }
    np.testing.assert_allclose(dead_ball_evidence(features), [0.0, 0.85, 0.0, 0.0])


def test_missing_ball_alone_is_not_dead_ball_evidence():
    features = {
        "ball_gap_frames": np.array([30, 40]),
        "home_player_mean_speed_m_s": np.array([np.nan, 1.1]),
        "away_player_mean_speed_m_s": np.array([np.nan, 1.0]),
        "home_player_track_count": np.array([0, 6]),
        "away_player_track_count": np.array([0, 6]),
    }
    np.testing.assert_allclose(dead_ball_evidence(features), [0.0, 0.0])
