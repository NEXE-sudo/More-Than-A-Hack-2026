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
