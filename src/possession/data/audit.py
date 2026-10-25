"""Aggregate local-only audit helpers for tracking, homography, and tactical video."""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from possession.data.validation import TrackingData


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"p50": None, "p95": None, "max": None}
    return {
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def summarize_video_timestamps(timestamps: list[float | None], fps: Fraction) -> dict[str, Any]:
    """Summarize timestamp cadence while keeping missing frames visible."""
    if fps <= 0:
        raise ValueError("video frame rate must be positive")
    valid = [
        (index, timestamp) for index, timestamp in enumerate(timestamps) if timestamp is not None
    ]
    values = np.asarray([timestamp for _, timestamp in valid], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("video timestamps contain non-finite values")
    deltas = np.diff(values)
    expected_delta = float(1 / fps)
    tolerance = expected_delta * 0.01
    discontinuities = int((np.abs(deltas - expected_delta) > tolerance).sum())
    duplicate_timestamps = int((deltas == 0).sum())
    non_monotonic_timestamps = int((deltas < 0).sum())
    max_drift = None
    if len(valid) == len(timestamps) and valid:
        indices = np.arange(len(timestamps), dtype=np.float64)
        relative_times = values - values[0]
        max_drift = float(np.max(np.abs(relative_times - indices * expected_delta)))
    return {
        "timestamp_samples": len(timestamps),
        "valid_timestamp_count": len(valid),
        "missing_timestamp_count": len(timestamps) - len(valid),
        "first_timestamp_seconds": None if not valid else float(valid[0][1]),
        "last_timestamp_seconds": None if not valid else float(valid[-1][1]),
        "timestamp_offset_seconds": None if not valid else float(valid[0][1]),
        "timestamp_discontinuities": discontinuities,
        "duplicate_timestamps": duplicate_timestamps,
        "non_monotonic_timestamps": non_monotonic_timestamps,
        "max_relative_timeline_drift_seconds": max_drift,
        "cadence_tolerance_seconds": tolerance,
        "cadence_within_tolerance": (
            len(valid) == len(timestamps)
            and len(timestamps) > 0
            and discontinuities == 0
            and max_drift is not None
            and max_drift <= tolerance
        ),
    }


def audit_tracking_homography(
    tracking: TrackingData,
    matrices: np.ndarray,
    reprojection_errors: np.ndarray,
    homography_valid: np.ndarray,
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    video_width: int | None = None,
    video_height: int | None = None,
) -> dict[str, Any]:
    """Summarize observations and homography round trips without exporting frame data."""
    if matrices.shape != (tracking.n_frames, 3, 3):
        raise ValueError("homography matrices must align with tracking frames")
    if reprojection_errors.shape != (tracking.n_frames,):
        raise ValueError("homography errors must align with tracking frames")
    if homography_valid.shape != (tracking.n_frames,):
        raise ValueError("homography validity must align with tracking frames")
    if (
        not np.isfinite([pitch_length_m, pitch_width_m]).all()
        or pitch_length_m <= 0
        or pitch_width_m <= 0
    ):
        raise ValueError("pitch dimensions must be positive")
    if (video_width is None) != (video_height is None):
        raise ValueError("video width and height must be supplied together")
    if video_width is not None and (video_width <= 0 or video_height <= 0):
        raise ValueError("video dimensions must be positive")
    if homography_valid.dtype != np.bool_:
        raise ValueError("homography validity mask must be boolean")
    valid_matrices = matrices[homography_valid]
    valid_errors = reprojection_errors[homography_valid]
    if not np.isfinite(valid_matrices).all():
        raise ValueError("valid homography matrices must contain only finite values")
    if not np.isfinite(valid_errors).all() or (valid_errors < 0).any():
        raise ValueError("valid homography reprojection errors must be finite and non-negative")
    if len(valid_matrices) and (np.abs(np.linalg.det(valid_matrices)) < 1e-12).any():
        raise ValueError("valid homography matrices must be non-singular")

    conditions = np.linalg.cond(valid_matrices) if len(valid_matrices) else np.array([])
    inverse_failures = 0
    round_trip_errors: list[np.ndarray] = []
    projected_outside_image = 0
    projected_count = 0
    pitch_mapped_outside = 0
    pitch_mapped_count = 0

    for start in range(0, tracking.n_frames, 4096):
        stop = min(start + 4096, tracking.n_frames)
        observed = homography_valid[start:stop] & np.isfinite(tracking.ball[start:stop]).all(axis=1)
        indices = np.flatnonzero(observed) + start
        if not len(indices):
            continue
        transforms = matrices[indices]
        try:
            inverses = np.linalg.inv(transforms)
        except np.linalg.LinAlgError:
            inverse_failures += int(len(indices))
            continue
        pitch_points = tracking.ball[indices].astype(np.float64)
        homogeneous = np.column_stack((pitch_points, np.ones(len(indices))))
        image_points = np.einsum("nij,nj->ni", transforms, homogeneous)
        usable = np.abs(image_points[:, 2]) > 1e-12
        inverse_failures += int((~usable).sum())
        if not usable.any():
            continue
        pixels = image_points[usable, :2] / image_points[usable, 2, None]
        pixel_homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
        recovered = np.einsum("nij,nj->ni", inverses[usable], pixel_homogeneous)
        stable = np.abs(recovered[:, 2]) > 1e-12
        inverse_failures += int((~stable).sum())
        if stable.any():
            normalized = recovered[stable, :2] / recovered[stable, 2, None]
            expected = pitch_points[usable][stable]
            round_trip_errors.append(np.linalg.norm(normalized - expected, axis=1))
            pitch_mapped_count += len(normalized)
            pitch_mapped_outside += int(((normalized < 0) | (normalized > 1)).any(axis=1).sum())
        accepted_pixels = pixels[stable]
        projected_count += len(accepted_pixels)
        if video_width is not None and video_height is not None:
            projected_outside_image += int(
                (
                    (accepted_pixels[:, 0] < 0)
                    | (accepted_pixels[:, 0] >= video_width)
                    | (accepted_pixels[:, 1] < 0)
                    | (accepted_pixels[:, 1] >= video_height)
                ).sum()
            )

    round_trip = np.concatenate(round_trip_errors) if round_trip_errors else np.array([])
    finite_ball = np.isfinite(tracking.ball).all(axis=1)
    finite_players = np.isfinite(tracking.players).all(axis=2)
    usable_speeds = np.isfinite(tracking.player_speed)
    return {
        "frame_count": tracking.n_frames,
        "frame_index_range": (
            [int(tracking.frame_ids[0]), int(tracking.frame_ids[-1])] if tracking.n_frames else None
        ),
        "track_slots": int(tracking.players.shape[1]),
        "player_observations_usable": int(finite_players.sum()),
        "player_observations_rejected_out_of_range": (
            tracking.player_coordinate_out_of_range_observations
        ),
        "player_speed_observations_usable": int(usable_speeds.sum()),
        "ball_observations_usable": int(finite_ball.sum()),
        "ball_observations_rejected_out_of_range": (
            tracking.ball_coordinate_out_of_range_observations
        ),
        "ball_visibility_usable_fraction": (
            float(finite_ball.mean()) if tracking.n_frames else None
        ),
        "speed_provenance": {
            "source_unit": tracking.source_speed_unit,
            "canonical_unit": tracking.speed_unit,
            "conversion_factor": tracking.speed_conversion_factor,
        },
        "coordinate_policy": {
            "normalized_range": [0, 1],
            "out_of_range_observations_rejected_in_memory": True,
            "frames_dropped_or_shifted": False,
        },
        "homography": {
            "row_count": int(len(homography_valid)),
            "valid_rows": int(homography_valid.sum()),
            "invalid_rows": int((~homography_valid).sum()),
            "reprojection_error": _percentiles(reprojection_errors[homography_valid]),
            "condition_number": _percentiles(conditions),
            "direction": "normalized pitch coordinates to image pixels",
            "inverse_mapping": "image pixels to normalized pitch coordinates",
            "ball_observations_projected": projected_count,
            "projected_ball_points_outside_video": (
                projected_outside_image if video_width is not None else None
            ),
            "ball_points_round_trip_mapped": pitch_mapped_count,
            "round_trip_normalized_error": _percentiles(round_trip),
            "round_trip_points_outside_normalized_pitch": pitch_mapped_outside,
            "inverse_or_projection_failures": inverse_failures,
            "pitch_dimensions_m": [pitch_length_m, pitch_width_m],
        },
    }


def inspect_video(video_path: Path | str, *, decode: bool = False) -> dict[str, Any]:
    """Read video metadata and optionally fully decode to a null sink using local ffmpeg tools."""
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("ffprobe is required to inspect video metadata")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,duration,nb_frames,nb_read_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"ffprobe failed for video: {result.stderr.strip()}")
    metadata = json.loads(result.stdout)
    streams = metadata.get("streams", [])
    if len(streams) != 1:
        raise ValueError("ffprobe did not return exactly one selected video stream")
    stream = streams[0]
    video_info = {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream.get("avg_frame_rate"),
        "reported_frame_rate": stream.get("r_frame_rate"),
        "duration_seconds": float(stream.get("duration", metadata["format"]["duration"])),
        "reported_frame_count": int(stream.get("nb_frames", 0)),
        "decoded_frame_count": int(stream.get("nb_read_frames", 0)),
        "full_decode": "not_run",
        "timestamp_samples": None,
    }
    try:
        fps = Fraction(stream["avg_frame_rate"])
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"video reports invalid frame rate {stream['avg_frame_rate']!r}") from exc
    if fps <= 0:
        raise ValueError(f"video reports invalid frame rate {stream['avg_frame_rate']!r}")
    timestamps = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if timestamps.returncode:
        raise ValueError(f"ffprobe timestamp scan failed: {timestamps.stderr.strip()}")
    frame_times: list[float | None] = []
    for line in timestamps.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        if value == "N/A":
            frame_times.append(None)
            continue
        try:
            frame_times.append(float(value))
        except ValueError as exc:
            raise ValueError(f"ffprobe returned a malformed timestamp: {value!r}") from exc
    video_info.update(summarize_video_timestamps(frame_times, fps))
    if decode:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg is required when full video decoding is requested")
        decoded = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostats",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if decoded.returncode:
            raise ValueError(f"ffmpeg decode failed: {decoded.stderr.strip()}")
        video_info["full_decode"] = "passed"

    return video_info
