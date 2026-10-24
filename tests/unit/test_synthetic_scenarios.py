import numpy as np
import pytest

from possession.models.baseline import (
    evidence_from_features,
    revised_evidence_from_features,
)
from possession.synthetic import RESTART_TYPES, SCENARIOS, generate_synthetic_sequence


@pytest.mark.parametrize("family", SCENARIOS)
def test_scenario_generator_is_reproducible_and_keeps_labels_out_of_features(family):
    first = generate_synthetic_sequence(family, 31, duration_seconds=2.0, fps=10)
    second = generate_synthetic_sequence(family, 31, duration_seconds=2.0, fps=10)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.features["ball_visible"], second.features["ball_visible"])
    assert not np.isin("labels", first.features)
    assert first.sequence_starts.shape == first.labels.shape


def test_revised_evidence_is_bounded_tag_conflicts_are_ignored_and_ablations_work():
    features = {
        "ball_visible": np.array([True, True]),
        "home_nearest_m": np.array([2.0, 2.0]),
        "away_nearest_m": np.array([4.0, 4.0]),
        "home_nearby_count": np.array([2, 2]),
        "away_nearby_count": np.array([1, 1]),
        "home_ball_distance_change_m": np.array([-0.5, -0.5]),
        "away_ball_distance_change_m": np.array([0.0, 0.0]),
        "tagged_home": np.array([True, True]),
        "tagged_away": np.array([False, True]),
    }
    revised = revised_evidence_from_features(features, tagging_team_weight=0.1)
    no_context = revised_evidence_from_features(features, tagging_team_weight=0.0)
    assert revised.shape == (2, 3)
    assert np.all((revised >= 0) & (revised <= 1))
    assert revised[0, 0] > no_context[0, 0]
    np.testing.assert_allclose(revised[1, :2], no_context[1, :2])
    no_groups = revised_evidence_from_features(
        features,
        use_support=False,
        use_approach=False,
    )
    no_proximity = revised_evidence_from_features(
        features,
        use_proximity=False,
    )
    baseline = evidence_from_features(features)
    assert not np.allclose(no_groups[:, :2], baseline[:, :2])
    assert not np.allclose(no_proximity[:, :2], revised[:, :2])


def test_unknown_scenario_and_invalid_generator_parameters_are_rejected():
    with pytest.raises(ValueError, match="unknown synthetic scenario"):
        generate_synthetic_sequence("unknown", 1)
    with pytest.raises(ValueError, match="player_availability"):
        generate_synthetic_sequence("stable_home", 1, player_availability=2)


@pytest.mark.parametrize(
    "family",
    [
        "turnovers",
        "ball_dropout",
        "long_dead_ball",
        "long_ball_occlusion",
        "boundary_start",
        "no_evidence",
    ],
)
def test_alias_scenarios_are_supported_and_reproducible(family):
    first = generate_synthetic_sequence(family, 17, duration_seconds=2.0, fps=10)
    second = generate_synthetic_sequence(family, 17, duration_seconds=2.0, fps=10)
    assert np.array_equal(first.labels, second.labels)
    assert first.sequence_starts.shape == first.labels.shape


@pytest.mark.parametrize("restart_type", RESTART_TYPES)
def test_restart_scenario_supports_known_set_piece_types(restart_type):
    sequence = generate_synthetic_sequence(
        "dead_ball_restart",
        13,
        duration_seconds=2.0,
        fps=10,
        restart_type=restart_type,
    )
    assert sequence.features["tagged_set_piece"].any()
    if restart_type == "Throw-in":
        assert not sequence.features["tagged_home"].any()
        assert not sequence.features["tagged_away"].any()
