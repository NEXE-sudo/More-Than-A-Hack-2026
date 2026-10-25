import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

OFFICIAL_BASELINE = (
    Path(__file__).resolve().parents[3] / "more-than-a-hack-2027" / "starter" / "baseline.py"
)


@pytest.fixture(scope="module")
def official_baseline():
    if not OFFICIAL_BASELINE.is_file():
        pytest.skip("sibling official baseline checkout is unavailable")
    spec = importlib.util.spec_from_file_location(
        "_official_baseline_for_smoothing_tests",
        OFFICIAL_BASELINE,
    )
    if spec is None or spec.loader is None:
        pytest.skip("official baseline module cannot be loaded")
    original_path = sys.path[:]
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


def test_published_variants_use_the_current_half_widths(official_baseline):
    assert official_baseline.VARIANTS == (
        ("baseline_raw", 0),
        ("baseline_smooth25", 25),
        ("baseline_smooth50", 50),
        ("baseline_smooth125", 125),
    )


def test_zero_width_preserves_home_away_and_maps_missing_to_dead_ball(official_baseline):
    codes = np.array([official_baseline.HOME, official_baseline.NOBODY, official_baseline.AWAY])

    assert official_baseline.smooth(codes, 0).tolist() == ["H", "D", "A"]


def test_centered_window_uses_truncated_endpoints_and_home_ties(official_baseline):
    codes = np.array([official_baseline.AWAY, official_baseline.AWAY, official_baseline.HOME])

    assert official_baseline.smooth(codes, 1).tolist() == ["A", "A", "H"]


def test_long_window_on_short_sequence_counts_only_observed_teams(official_baseline):
    codes = np.array([official_baseline.AWAY, official_baseline.NOBODY, official_baseline.HOME])

    assert official_baseline.smooth(codes, 125).tolist() == ["H", "H", "H"]
    assert official_baseline.smooth(
        np.array([official_baseline.NOBODY, official_baseline.NOBODY]),
        125,
    ).tolist() == ["D", "D"]


def test_centered_vote_delays_abrupt_change_and_removes_short_spell(official_baseline):
    abrupt = np.array(
        [
            official_baseline.HOME,
            official_baseline.HOME,
            official_baseline.AWAY,
            official_baseline.AWAY,
        ]
    )
    short_spell = np.array(
        [
            official_baseline.HOME,
            official_baseline.HOME,
            official_baseline.AWAY,
            official_baseline.HOME,
            official_baseline.HOME,
        ]
    )

    assert official_baseline.smooth(abrupt, 1).tolist() == ["H", "H", "A", "A"]
    assert official_baseline.smooth(short_spell, 1).tolist() == ["H"] * 5


def test_smoothing_is_deterministic_and_submission_keeps_frame_order(official_baseline, tmp_path):
    codes = np.array(
        [
            official_baseline.HOME,
            official_baseline.NOBODY,
            official_baseline.AWAY,
            official_baseline.AWAY,
            official_baseline.HOME,
        ]
    )
    first = official_baseline.smooth(codes, 2)
    second = official_baseline.smooth(codes.copy(), 2)
    output = tmp_path / "submission.csv"
    official_baseline.write_submission(output, first)

    assert np.array_equal(first, second)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["frame", "label"]
    assert [int(row[0]) for row in rows[1:]] == list(range(len(codes)))
    assert [row[1] for row in rows[1:]] == first.tolist()
