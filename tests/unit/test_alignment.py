import numpy as np
import pytest

from possession.data.alignment import (
    tagging_seconds_to_frames,
    tracking_ids_to_video_frames,
    validate_frame_sequence,
)


def test_tracking_frame_ids_map_to_zero_based_video_frames():
    assert tracking_ids_to_video_frames(np.array([1, 2, 199319])).tolist() == [
        0,
        1,
        199318,
    ]


def test_smart_tagging_uses_video_zero_timebase_and_nearest_frame():
    assert tagging_seconds_to_frames([0, 1.02, 25], fps=25).tolist() == [0, 26, 625]


def test_alignment_rejects_missing_or_reordered_tracking_ids():
    with pytest.raises(ValueError, match="contiguous"):
        validate_frame_sequence(np.array([1, 3]), expected_frames=2)
