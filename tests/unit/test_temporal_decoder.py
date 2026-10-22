import numpy as np

from possession.config import DecoderConfig
from possession.temporal.decoder import decode


def scores(home: float, away: float, dead: float = 0.0) -> list[float]:
    return [home, away, dead]


def test_home_possession_survives_a_brief_tracking_gap():
    values = np.array([scores(0.8, 0.1), scores(0.0, 0.0), scores(0.0, 0.0), scores(0.7, 0.2)])
    assert decode(values, DecoderConfig(missing_hold_frames=2)).tolist() == [
        "H",
        "H",
        "H",
        "H",
    ]


def test_sustained_opponent_evidence_causes_a_team_switch():
    values = np.array([scores(0.8, 0.1)] + [scores(0.1, 0.8)] * 3)
    assert decode(values, DecoderConfig(switch_frames=2)).tolist() == ["H", "H", "A", "A"]


def test_airborne_ball_motion_does_not_imply_dead_ball():
    values = np.array([scores(0.8, 0.2), scores(0.0, 0.0), scores(0.0, 0.0), scores(0.2, 0.8)])
    assert decode(values, DecoderConfig(missing_hold_frames=2)).tolist() == ["H", "H", "H", "H"]


def test_dead_ball_score_and_restart_resume_are_explicit():
    values = np.array([scores(0.8, 0.1), scores(0.1, 0.2, 0.9), scores(0.1, 0.8)])
    assert decode(values).tolist() == ["H", "D", "A"]


def test_crowded_area_ties_do_not_flip_the_existing_team():
    values = np.array([scores(0.75, 0.2)] + [scores(0.6, 0.6)] * 4)
    assert decode(values).tolist() == ["H"] * 5


def test_continue_threshold_is_lower_than_start_and_switch_thresholds():
    values = np.array([scores(0.8, 0.1), scores(0.4, 0.3), scores(0.0, 0.0)])
    config = DecoderConfig(continue_threshold=0.35, start_threshold=0.45)
    assert decode(values, config).tolist() == ["H", "H", "H"]


def test_short_possession_can_be_represented_with_one_frame_switch_rule():
    values = np.array([scores(0.8, 0.1), scores(0.1, 0.8), scores(0.8, 0.1)])
    config = DecoderConfig(switch_frames=1, switch_margin=0.0)
    assert decode(values, config).tolist() == ["H", "A", "H"]


def test_turnover_after_missing_ball_does_not_keep_old_team_forever():
    values = np.array([scores(0.8, 0.1)] + [scores(0.0, 0.0)] * 4 + [scores(0.1, 0.85)] * 2)
    decoded = decode(values, DecoderConfig(missing_hold_frames=2, switch_frames=2))
    assert decoded.tolist() == ["H", "H", "H", "X", "X", "A", "A"]


def test_sequence_boundaries_reset_previous_possession():
    values = np.array([scores(0.8, 0.1), scores(0.0, 0.0), scores(0.0, 0.0)])
    starts = [True, False, True]
    assert decode(values, sequence_starts=starts).tolist() == ["H", "H", "X"]


def test_identical_inputs_are_deterministic():
    values = np.array([scores(0.5, 0.5), scores(0.2, 0.8)] * 5)
    assert np.array_equal(decode(values), decode(values))
