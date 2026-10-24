import numpy as np

from possession.features.context import tagging_context


def test_smart_tagging_produces_context_masks_not_a_direct_possession_label():
    masks = tagging_context(
        [
            {
                "code": "SET PIECES",
                "start_frame": 2,
                "end_frame": 5,
                "tags": {"Team": "N/A"},
            },
            {
                "code": "BUILD UP",
                "start_frame": 6,
                "end_frame": 8,
                "tags": {"Team": "Barcelona"},
            },
        ],
        n_frames=10,
    )
    assert np.flatnonzero(masks["tagged_set_piece"]).tolist() == [2, 3, 4]
    assert np.flatnonzero(masks["tagged_home"]).tolist() == [6, 7]
    assert not masks["tagged_away"].any()


def test_unknown_conflicting_and_out_of_range_tags_are_explicitly_bounded():
    masks = tagging_context(
        [
            {
                "code": "UNKNOWN EVENT",
                "start_frame": -2,
                "end_frame": 2,
                "tags": {"Team": "Unknown Club"},
            },
            {
                "code": "BUILD UP",
                "start_frame": 1,
                "end_frame": 4,
                "tags": {"Team": "Barcelona"},
            },
            {
                "code": "PROGRESSION",
                "start_frame": 2,
                "end_frame": 5,
                "tags": {"Team": "Atletic Club"},
            },
            {
                "code": "SET PIECES",
                "start_frame": 4,
                "end_frame": 8,
                "tags": {"Team": None},
            },
        ],
        n_frames=6,
    )
    assert np.flatnonzero(masks["tagged_phase"]).tolist() == [0, 1, 2, 3, 4, 5]
    assert np.flatnonzero(masks["tagged_team_conflict"]).tolist() == [2, 3]
    assert np.flatnonzero(masks["tagged_set_piece"]).tolist() == [4, 5]
    assert np.flatnonzero(masks["tagged_home"]).tolist() == [1, 2, 3]
    assert np.flatnonzero(masks["tagged_away"]).tolist() == [2, 3, 4]
