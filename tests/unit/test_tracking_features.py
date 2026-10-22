import numpy as np
import pytest

from possession.data.validation import TrackingData
from possession.features.tracking import extract_tracking_features


def test_team_nearest_distances_and_missingness_features():
    data = TrackingData(
        frame_ids=np.arange(2),
        players=np.array(
            [
                [[0.20, 0.50], [0.80, 0.50]],
                [[np.nan, np.nan], [0.80, 0.50]],
            ],
            dtype=np.float32,
        ),
        ball=np.array([[0.21, 0.50], [np.nan, np.nan]], dtype=np.float32),
        player_teams=np.array([0, 1]),
        player_speed=np.zeros((2, 2), dtype=np.float32),
    )
    features = extract_tracking_features(data)
    assert features["home_nearest_m"][0] == pytest.approx(1.053)
    assert features["away_nearest_m"][0] > 50
    assert features["ball_visible"].tolist() == [True, False]
    assert np.isnan(features["home_nearest_m"][1])
    assert features["home_missing_tracks"].tolist() == [0, 1]
    assert features["ball_gap_frames"].tolist() == [0, 1]
