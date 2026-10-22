import numpy as np
import pytest

from possession.features.ball import ball_trajectory_features


def test_ball_motion_uses_only_past_and_adjacent_observations():
    ball = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [np.nan, np.nan], [0.9, 0.9]])
    features = ball_trajectory_features(ball, fps=25, recent_window=2)
    assert features["visible"].tolist() == [True, True, True, False, True]
    assert features["motion_reliable"].tolist() == [False, True, True, False, False]
    assert features["displacement"][1, 0] == pytest.approx(1.053)
    assert features["recent_distance_m"][2] == pytest.approx(2.106)
    assert not features["trajectory_reliable"][4]
    assert features["gap_frames"].tolist() == [0, 0, 0, 1, 0]
