import numpy as np
import pytest

from possession.models.baseline import (
    evidence_from_features,
    revised_evidence_from_features,
)
from possession.synthetic import (
    CANONICAL_SCENARIOS,
    DEPRECATED_SCENARIOS,
    RESTART_TYPES,
    SCENARIO_ALIASES,
    SCENARIO_DESCRIPTIONS,
    SCENARIOS,
    generate_synthetic_sequence,
)


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
    assert set(DEPRECATED_SCENARIOS) == {"interception", "goalkeeper"}
    for deprecated, explanation in DEPRECATED_SCENARIOS.items():
        with pytest.raises(ValueError, match=f"scenario '{deprecated}' is deprecated"):
            generate_synthetic_sequence(deprecated, 1)
        assert explanation
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


def test_aliases_are_declared_and_report_the_canonical_family():
    assert set(CANONICAL_SCENARIOS).isdisjoint(SCENARIO_ALIASES)
    assert set(SCENARIOS) == set(CANONICAL_SCENARIOS) | set(SCENARIO_ALIASES)
    assert set(SCENARIO_DESCRIPTIONS) == set(CANONICAL_SCENARIOS)
    for alias, canonical in SCENARIO_ALIASES.items():
        alias_sequence = generate_synthetic_sequence(alias, 23, duration_seconds=2.0, fps=10)
        canonical_sequence = generate_synthetic_sequence(
            canonical, 23, duration_seconds=2.0, fps=10
        )
        assert alias_sequence.family == canonical
        assert np.array_equal(alias_sequence.labels, canonical_sequence.labels)
        assert np.array_equal(alias_sequence.sequence_starts, canonical_sequence.sequence_starts)
        assert alias_sequence.features.keys() == canonical_sequence.features.keys()
        for name in alias_sequence.features:
            assert np.array_equal(
                alias_sequence.features[name],
                canonical_sequence.features[name],
                equal_nan=True,
            )


def test_stable_and_turnover_scenarios_have_expected_ground_truth_transitions():
    stable_home = generate_synthetic_sequence("stable_home", 11, duration_seconds=16)
    stable_away = generate_synthetic_sequence("stable_away", 11, duration_seconds=16)
    assert np.all(stable_home.labels == "H")
    assert np.all(stable_away.labels == "A")
    assert not stable_home.labels[1:].__ne__(stable_home.labels[:-1]).any()
    assert not stable_away.labels[1:].__ne__(stable_away.labels[:-1]).any()

    turnover = generate_synthetic_sequence("passes_turnovers", 11, duration_seconds=16)
    switches = np.flatnonzero(turnover.labels[1:] != turnover.labels[:-1]) + 1
    assert turnover.labels[switches - 1].tolist() == ["H", "A", "H"]
    assert turnover.labels[switches].tolist() == ["A", "H", "A"]
    np.testing.assert_array_equal(
        switches,
        [round(0.28 * 400), round(0.56 * 400), round(0.78 * 400)],
    )


@pytest.mark.parametrize(
    ("family", "before", "after"),
    [
        ("home_to_away_turnover", "H", "A"),
        ("away_to_home_turnover", "A", "H"),
    ],
)
def test_directional_turnover_scenarios_have_one_expected_transition(family, before, after):
    sequence = generate_synthetic_sequence(family, 19, duration_seconds=12, fps=20)
    switches = np.flatnonzero(sequence.labels[1:] != sequence.labels[:-1]) + 1

    np.testing.assert_array_equal(switches, [round(0.5 * 240)])
    assert sequence.labels[0] == before
    assert sequence.labels[switches[0] - 1] == before
    assert sequence.labels[switches[0]] == after
    assert sequence.labels[-1] == after


def test_ball_and_player_dropout_scenarios_mask_the_intended_channels():
    ball_dropout = generate_synthetic_sequence("ball_dropout", 11, duration_seconds=16)
    start, stop = round(0.35 * 400), round(0.55 * 400)
    assert np.all(ball_dropout.labels == "H")
    assert not ball_dropout.features["ball_visible"][start:stop].any()
    assert ball_dropout.features["ball_visible"][:start].any()
    assert ball_dropout.features["ball_visible"][stop:].any()
    assert np.all(ball_dropout.features["player_track_count"][start:stop] > 0)

    missing_players = generate_synthetic_sequence("missing_player_tracks", 11, duration_seconds=16)
    assert np.all(missing_players.labels == "H")
    assert np.all(missing_players.features["away_player_track_count"][start:stop] == 0)
    assert np.all(missing_players.features["home_player_track_count"][start:stop] > 0)
    assert missing_players.features["away_player_track_count"][:start].any()


def test_long_no_evidence_removes_ball_and_player_evidence_during_x_labels():
    sequence = generate_synthetic_sequence("long_no_evidence", 29, duration_seconds=12, fps=20)
    short_gap = generate_synthetic_sequence("half_reset", 29, duration_seconds=12, fps=20)
    start, stop = round(0.30 * 240), round(0.70 * 240)
    no_evidence = slice(start, max(start + 1, stop))

    assert np.count_nonzero(sequence.labels == "X") > np.count_nonzero(short_gap.labels == "X")
    assert np.all(sequence.labels[no_evidence] == "X")
    assert np.all(~sequence.features["ball_visible"][no_evidence])
    assert np.all(sequence.features["player_track_count"][no_evidence] == 0)
    assert np.isnan(sequence.features["home_nearest_m"][no_evidence]).all()
    assert np.isnan(sequence.features["away_nearest_m"][no_evidence]).all()
    assert sequence.features["player_track_count"][:start].any()
    assert sequence.features["ball_visible"][stop:].any()


def test_dead_ball_duration_and_consecutive_restart_intervals_are_observable():
    duration = 16
    fps = 25
    n_frames = duration * fps
    short = generate_synthetic_sequence("short_dead_ball", 11, duration_seconds=duration)
    long = generate_synthetic_sequence("long_dead_ball", 11, duration_seconds=duration)
    standard = generate_synthetic_sequence("dead_ball_restart", 11, duration_seconds=duration)

    short_count = np.count_nonzero(short.labels == "D")
    long_count = np.count_nonzero(long.labels == "D")
    standard_count = np.count_nonzero(standard.labels == "D")
    assert short_count == round((0.36 - 0.30) * n_frames)
    assert long_count == round((0.55 - 0.30) * n_frames)
    assert standard_count == round((0.48 - 0.30) * n_frames)
    assert short_count < standard_count < long_count
    for sequence in (short, long, standard):
        dead = sequence.labels == "D"
        assert np.array_equal(dead, sequence.features["tagged_set_piece"])
        assert np.all(sequence.labels[np.flatnonzero(dead)[-1] + 1 :] == "A")
        assert sequence.features["tagged_away"][dead].all()
        assert not sequence.features["tagged_home"][dead].any()

    consecutive = generate_synthetic_sequence(
        "consecutive_dead_balls", 11, duration_seconds=duration
    )
    dead = consecutive.labels == "D"
    edges = np.diff(np.r_[False, dead, False].astype(np.int8))
    intervals = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))
    assert len(intervals) == 2
    assert np.all(consecutive.labels[intervals[0][1] : intervals[1][0]] == "A")
    assert np.array_equal(dead, consecutive.features["tagged_set_piece"])
    assert np.all(consecutive.labels[intervals[1][1] :] == "H")
    assert consecutive.features["tagged_away"][intervals[0][0] : intervals[0][1]].all()
    assert consecutive.features["tagged_home"][intervals[1][0] : intervals[1][1]].all()


def test_rapid_ball_movement_and_brief_nearest_player_blip_are_measurable():
    rapid = generate_synthetic_sequence("rapid_ball_movement", 11, duration_seconds=16)
    assert np.all(rapid.labels == "H")
    rapid_window = rapid.features["ball_speed_m_s"][180:220]
    assert np.nanmax(rapid_window) > 20.0

    blip = generate_synthetic_sequence("brief_nearest_player_blip", 11, duration_seconds=16)
    start, stop = round(0.48 * 400), round(0.52 * 400)
    valid = np.isfinite(blip.features["home_nearest_m"][start:stop]) & np.isfinite(
        blip.features["away_nearest_m"][start:stop]
    )
    home = blip.features["home_nearest_m"][start:stop][valid]
    away = blip.features["away_nearest_m"][start:stop][valid]
    assert np.all(blip.labels == "H")
    assert len(home) >= 8
    assert np.mean(away < home) >= 0.75
    assert np.count_nonzero(blip.labels == "A") == 0


def test_home_goal_area_proxy_and_periodic_adversarial_tracks_are_observable():
    goal_area = generate_synthetic_sequence("home_goal_area_possession", 11, duration_seconds=16)
    visible = goal_area.features["ball_visible"]
    assert np.all(goal_area.labels == "H")
    assert visible.any()
    assert np.nanmedian(goal_area.features["ball_x_m"][visible]) < 5.0
    assert np.nanmedian(goal_area.features["home_nearest_m"][visible]) < 1.0

    adversarial = generate_synthetic_sequence("adversarial_tracks", 11, duration_seconds=16)
    event_frames = np.asarray([0, 1, 125, 126, 250, 251])
    finite = np.isfinite(adversarial.features["away_nearest_m"][event_frames])
    assert finite.sum() >= 3
    assert np.nanmax(adversarial.features["away_nearest_m"][event_frames]) < 1.0
    assert np.all(adversarial.labels[event_frames] == "H")


def test_half_reset_marks_the_unknown_gap_and_starts_a_new_sequence():
    sequence = generate_synthetic_sequence("half_reset", 11, duration_seconds=16)
    start, stop = round(0.48 * 400), round(0.52 * 400)

    assert np.all(sequence.labels[:start] == "H")
    assert np.all(sequence.labels[start:stop] == "X")
    assert np.all(sequence.labels[stop:] == "A")
    assert sequence.sequence_starts[stop]
    assert np.count_nonzero(sequence.sequence_starts) == 1


def test_dead_ball_occlusion_and_ambiguous_proximity_conditions_are_observable():
    dead_ball = generate_synthetic_sequence("dead_ball_restart", 11, duration_seconds=16)
    assert np.count_nonzero(dead_ball.labels == "D") > 0
    assert np.count_nonzero(dead_ball.labels == "X") == 0
    assert dead_ball.features["tagged_set_piece"].any()

    occlusion = generate_synthetic_sequence("occlusion_aerial", 11, duration_seconds=16)
    assert np.all(~occlusion.features["ball_visible"][200:250])
    assert np.all(occlusion.labels[:200] == "H")
    assert np.all(occlusion.labels[200:] == "A")

    contested = generate_synthetic_sequence("contested_ball", 11, duration_seconds=16)
    ungraded = contested.labels == "X"
    valid = (
        ungraded
        & np.isfinite(contested.features["home_nearest_m"])
        & np.isfinite(contested.features["away_nearest_m"])
    )
    margins = np.abs(
        contested.features["home_nearest_m"][valid] - contested.features["away_nearest_m"][valid]
    )
    assert len(margins) >= 8
    assert np.mean(margins < 2.0) >= 0.5


def test_boundary_scenarios_differ_in_transition_position_and_contradiction_is_measured():
    boundary_start = generate_synthetic_sequence("boundary_start", 11, duration_seconds=16)
    boundary_end = generate_synthetic_sequence("boundary_end", 11, duration_seconds=16)
    start_switch = np.flatnonzero(boundary_start.labels[1:] != boundary_start.labels[:-1]) + 1
    end_switch = np.flatnonzero(boundary_end.labels[1:] != boundary_end.labels[:-1]) + 1
    np.testing.assert_array_equal(start_switch, [round(0.08 * 400)])
    np.testing.assert_array_equal(end_switch, [round(0.92 * 400)])
    assert boundary_start.labels[0] == boundary_end.labels[0] == "H"
    assert boundary_start.labels[-1] == boundary_end.labels[-1] == "A"

    conflict = generate_synthetic_sequence("contradictory_evidence", 11, duration_seconds=16)
    eligible = (
        (conflict.labels == "H")
        & (np.arange(len(conflict.labels)) >= 100)
        & (np.arange(len(conflict.labels)) < 300)
        & np.isfinite(conflict.features["home_nearest_m"])
        & np.isfinite(conflict.features["away_nearest_m"])
    )
    assert eligible.sum() >= 50
    assert (
        np.mean(
            conflict.features["away_nearest_m"][eligible]
            < conflict.features["home_nearest_m"][eligible]
        )
        > 0.5
    )


def test_seed_changes_random_observations_without_changing_scenario_invariants():
    first = generate_synthetic_sequence("ball_dropout", 31, duration_seconds=4)
    second = generate_synthetic_sequence("ball_dropout", 32, duration_seconds=4)
    assert np.all(first.labels == "H")
    assert np.all(second.labels == "H")
    start, stop = round(0.35 * 100), round(0.55 * 100)
    assert not first.features["ball_visible"][start:stop].any()
    assert not second.features["ball_visible"][start:stop].any()
    assert not np.array_equal(first.features["ball_visible"], second.features["ball_visible"])
    for sequence in (first, second):
        assert np.all(sequence.labels == "H")
        assert not sequence.features["ball_visible"][start:stop].any()


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
