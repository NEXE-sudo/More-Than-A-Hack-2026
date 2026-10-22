import numpy as np

from possession.data.validation import validate_submission_file
from possession.features.cache import save_feature_cache
from possession.pipeline import predict


def test_pipeline_writes_deterministic_submission_from_synthetic_tracking(tmp_path):
    metadata = tmp_path / "meta.xml"
    tracking = tmp_path / "tracking.txt"
    metadata.write_text(
        '<Root><Team id="home"/><Team id="away"/>'
        '<Player teamId="home"/><Player teamId="away"/></Root>',
        encoding="utf-8",
    )
    tracking.write_text(
        "1:0.20,0.5,0;0.80,0.5,0:0.21,0.5,0\n"
        "2:0.20,0.5,0;0.80,0.5,0:0.21,0.5,0\n"
        "3:0.20,0.5,0;0.80,0.5,0:N\n",
        encoding="utf-8",
    )
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    labels_one = predict(metadata, tracking, first, expected_frames=3)
    labels_two = predict(metadata, tracking, second, expected_frames=3)
    assert labels_one.tolist() == ["H", "H", "H"]
    assert np.array_equal(labels_one, labels_two)
    validate_submission_file(first, expected_frames=3)
    assert first.read_text(encoding="utf-8").splitlines() == [
        "frame,label",
        "0,H",
        "1,H",
        "2,H",
    ]


def test_pipeline_can_predict_from_a_validated_feature_cache(tmp_path):
    cache = save_feature_cache(
        tmp_path / "features.npz",
        {
            "ball_visible": np.array([True, True, False]),
            "home_nearest_m": np.array([1.0, 1.0, np.nan]),
            "away_nearest_m": np.array([50.0, 50.0, np.nan]),
            "ball_gap_frames": np.array([0, 0, 1]),
            "home_player_mean_speed_m_s": np.zeros(3),
            "away_player_mean_speed_m_s": np.zeros(3),
            "home_player_track_count": np.full(3, 5),
            "away_player_track_count": np.full(3, 5),
        },
    )
    labels = predict(
        None,
        None,
        tmp_path / "cached.csv",
        expected_frames=3,
        feature_cache_path=cache,
    )
    assert labels.tolist() == ["H", "H", "H"]
