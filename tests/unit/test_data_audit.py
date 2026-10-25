import json
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from possession.data.audit import (
    audit_tracking_homography,
    inspect_video,
    summarize_video_timestamps,
)
from possession.data.validation import TrackingData


def test_tracking_homography_audit_reports_aggregates_without_frame_exports():
    tracking = TrackingData(
        frame_ids=np.arange(3),
        players=np.array(
            [
                [[0.1, 0.2], [np.nan, np.nan]],
                [[0.2, 0.3], [0.8, 0.7]],
                [[0.3, 0.4], [0.7, 0.6]],
            ]
        ),
        ball=np.array([[0.2, 0.3], [np.nan, np.nan], [1.0, 1.0]]),
        player_teams=np.array([0, 1]),
        player_speed=np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]),
        player_coordinate_out_of_range_observations=1,
        ball_coordinate_out_of_range_observations=2,
    )
    matrices = np.repeat(np.diag([100.0, 50.0, 1.0])[None, :, :], 3, axis=0)
    matrices[1] = np.nan
    errors = np.array([0.1, np.nan, 0.3])
    valid = np.array([True, False, True])

    report = audit_tracking_homography(
        tracking,
        matrices,
        errors,
        valid,
        pitch_length_m=105,
        pitch_width_m=68,
    )

    assert report["frame_index_range"] == [0, 2]
    assert report["player_observations_usable"] == 5
    assert report["ball_observations_usable"] == 2
    assert report["ball_visibility_usable_fraction"] == pytest.approx(2 / 3)
    assert report["ball_observations_rejected_out_of_range"] == 2
    assert report["homography"]["valid_rows"] == 2
    assert report["homography"]["ball_points_round_trip_mapped"] == 2
    assert report["homography"]["round_trip_normalized_error"]["max"] == pytest.approx(0)
    assert report["homography"]["projected_ball_points_outside_video"] is None
    assert "frames" not in report


def test_tracking_homography_audit_rejects_misaligned_inputs():
    tracking = TrackingData(
        frame_ids=np.arange(2),
        players=np.empty((2, 0, 2)),
        ball=np.full((2, 2), np.nan),
        player_teams=np.array([], dtype=int),
        player_speed=np.empty((2, 0)),
    )
    with pytest.raises(ValueError, match="homography matrices must align"):
        audit_tracking_homography(
            tracking,
            np.empty((1, 3, 3)),
            np.zeros(2),
            np.ones(2, dtype=bool),
            pitch_length_m=105,
            pitch_width_m=68,
        )


def test_tracking_homography_audit_requires_both_video_dimensions():
    tracking = TrackingData(
        frame_ids=np.arange(1),
        players=np.empty((1, 0, 2)),
        ball=np.full((1, 2), np.nan),
        player_teams=np.array([], dtype=int),
        player_speed=np.empty((1, 0)),
    )
    with pytest.raises(ValueError, match="width and height must be supplied together"):
        audit_tracking_homography(
            tracking,
            np.eye(3)[None, :, :],
            np.zeros(1),
            np.ones(1, dtype=bool),
            pitch_length_m=105,
            pitch_width_m=68,
            video_width=1920,
        )


@pytest.mark.parametrize(
    ("matrix", "error", "message"),
    [
        (np.zeros((1, 3, 3)), np.zeros(1), "non-singular"),
        (np.full((1, 3, 3), np.nan), np.zeros(1), "finite values"),
        (np.eye(3)[None, :, :], np.array([np.inf]), "reprojection errors"),
    ],
)
def test_tracking_homography_audit_rejects_invalid_valid_rows(matrix, error, message):
    tracking = TrackingData(
        frame_ids=np.arange(1),
        players=np.empty((1, 0, 2)),
        ball=np.full((1, 2), np.nan),
        player_teams=np.array([], dtype=int),
        player_speed=np.empty((1, 0)),
    )
    with pytest.raises(ValueError, match=message):
        audit_tracking_homography(
            tracking,
            matrix,
            error,
            np.ones(1, dtype=bool),
            pitch_length_m=105,
            pitch_width_m=68,
        )


def test_video_timestamp_summary_detects_missing_duplicate_and_drift():
    summary = summarize_video_timestamps(
        [0.0, 0.04, 0.04, None, 0.20],
        fps=Fraction(25, 1),
    )

    assert summary["timestamp_samples"] == 5
    assert summary["valid_timestamp_count"] == 4
    assert summary["missing_timestamp_count"] == 1
    assert summary["duplicate_timestamps"] == 1
    assert summary["timestamp_discontinuities"] == 2
    assert summary["max_relative_timeline_drift_seconds"] is None
    assert not summary["cadence_within_tolerance"]


def test_video_timestamp_summary_rejects_nonpositive_frame_rate():
    with pytest.raises(ValueError, match="frame rate must be positive"):
        summarize_video_timestamps([], fps=Fraction(0, 1))


def test_inspect_video_reports_timestamps_and_runs_optional_full_decode(tmp_path, monkeypatch):
    video = tmp_path / "sample.mp4"
    video.touch()
    monkeypatch.setattr(
        "possession.data.audit.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0].endswith("ffmpeg"):
            return subprocess.CompletedProcess(command, 0, "", "")
        if "frame=best_effort_timestamp_time" in command:
            return subprocess.CompletedProcess(command, 0, "0\n0.04\n0.08\n0.16\n", "")
        metadata = {
            "streams": [
                {
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "25/1",
                    "r_frame_rate": "25/1",
                    "duration": "0.16",
                    "nb_frames": "4",
                    "nb_read_frames": "4",
                }
            ],
            "format": {"duration": "0.16"},
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(metadata), "")

    monkeypatch.setattr("possession.data.audit.subprocess.run", run)

    report = inspect_video(video, decode=True)

    assert report["codec"] == "h264"
    assert report["width"] == 1920
    assert report["decoded_frame_count"] == 4
    assert report["timestamp_samples"] == 4
    assert report["valid_timestamp_count"] == 4
    assert report["timestamp_discontinuities"] == 1
    assert report["max_relative_timeline_drift_seconds"] == pytest.approx(0.04)
    assert not report["cadence_within_tolerance"]
    assert report["full_decode"] == "passed"
    assert calls[-1][0].endswith("ffmpeg")
