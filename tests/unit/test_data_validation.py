from dataclasses import replace

import numpy as np
import pytest

from possession.data.validation import TrackingData, validate_tracking


def fixture_data():
    return TrackingData(
        frame_ids=np.arange(2),
        players=np.array([[[0.1, 0.2], [np.nan, np.nan]], [[0.3, 0.4], [0.5, 0.6]]]),
        ball=np.array([[0.2, 0.3], [np.nan, np.nan]]),
        player_teams=np.array([0, 1]),
        player_speed=np.array([[1.0, np.nan], [2.0, 3.0]]),
    )


def test_missing_ball_and_player_observations_are_preserved():
    data = fixture_data()
    validate_tracking(data, expected_frames=2)
    assert np.isnan(data.ball[1]).all()
    assert np.isnan(data.players[0, 1]).all()


def test_invalid_coordinate_and_malformed_shapes_are_rejected():
    data = fixture_data()
    data.players[0, 0, 0] = 1.2
    with pytest.raises(ValueError, match="normalized"):
        validate_tracking(data)

    original = fixture_data()
    malformed = replace(original, ball=original.ball[:-1])
    with pytest.raises(ValueError, match="ball must have shape"):
        validate_tracking(malformed)


def test_non_finite_infinity_is_not_treated_as_missing():
    data = fixture_data()
    data.ball[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_tracking(data)


def test_partial_coordinate_pairs_are_rejected_instead_of_being_treated_as_missing():
    data = fixture_data()
    data.players[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="both coordinates or neither"):
        validate_tracking(data)
