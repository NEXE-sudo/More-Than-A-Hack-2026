"""Validation of in-memory tracking arrays and submission predictions."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LABELS = frozenset({"H", "A", "D", "X"})


@dataclass(frozen=True)
class TrackingData:
    frame_ids: np.ndarray
    players: np.ndarray
    ball: np.ndarray
    player_teams: np.ndarray
    player_speed: np.ndarray

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)


def validate_tracking(data: TrackingData, expected_frames: int | None = None) -> None:
    """Validate frame alignment, team assignments, shapes, and coordinate missingness."""
    n_frames = len(data.frame_ids)
    if expected_frames is not None and n_frames != expected_frames:
        raise ValueError(f"expected {expected_frames} tracking frames, found {n_frames}")
    if (
        data.frame_ids.ndim != 1
        or not np.issubdtype(data.frame_ids.dtype, np.integer)
        or not np.array_equal(data.frame_ids, np.arange(n_frames, dtype=np.int64))
    ):
        raise ValueError("video frame indices must be ordered and contiguous from zero")
    if data.players.ndim != 3 or data.players.shape[0] != n_frames or data.players.shape[2] != 2:
        raise ValueError("players must have shape (frames, tracks, 2)")
    n_tracks = data.players.shape[1]
    if data.ball.shape != (n_frames, 2):
        raise ValueError("ball must have shape (frames, 2)")
    if data.player_teams.shape != (n_tracks,) or not np.isin(data.player_teams, [0, 1]).all():
        raise ValueError("player_teams must assign every track to home=0 or away=1")
    if data.player_speed.shape != (n_frames, n_tracks):
        raise ValueError("player_speed must have shape (frames, tracks)")
    for name, values in (("players", data.players), ("ball", data.ball)):
        if np.isinf(values).any():
            raise ValueError(f"{name} coordinates contain infinite values")
        observed_coordinates = np.isfinite(values)
        partial_observation = observed_coordinates[..., 0] != observed_coordinates[..., 1]
        if partial_observation.any():
            raise ValueError(f"{name} observations must provide both coordinates or neither")
        finite = np.isfinite(values)
        if finite.any() and ((values[finite] < 0) | (values[finite] > 1)).any():
            raise ValueError(f"{name} coordinates must be normalized to [0, 1] or NaN")
    if np.isinf(data.player_speed).any():
        raise ValueError("player speeds contain infinite values")
    if (data.player_speed[np.isfinite(data.player_speed)] < 0).any():
        raise ValueError("player speeds cannot be negative")


def validate_predictions(frames: np.ndarray, labels: np.ndarray, expected_frames: int) -> None:
    """Check exact submission frame IDs, label alphabet, count, and order."""
    frame_values = np.asarray(frames)
    label_values = np.asarray(labels)
    if frame_values.ndim != 1 or len(frame_values) != expected_frames:
        raise ValueError(f"submission must contain exactly {expected_frames} frames")
    if not np.issubdtype(frame_values.dtype, np.integer) or not np.array_equal(
        frame_values, np.arange(expected_frames, dtype=np.int64)
    ):
        raise ValueError("submission frames must be unique, ordered, and equal to 0..N-1")
    if label_values.shape != (expected_frames,):
        raise ValueError(f"submission must contain exactly {expected_frames} labels")
    invalid = sorted(set(label_values.tolist()) - LABELS)
    if invalid:
        raise ValueError(f"invalid labels {invalid}; allowed labels are H, A, D, X")


def validate_submission_file(path: Path | str, expected_frames: int = 199319) -> None:
    """Validate a CSV file against the competition's exact frame,label contract."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["frame", "label"]:
            raise ValueError("submission header must be exactly frame,label")
        frames: list[int] = []
        labels: list[str] = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"line {line_number}: unexpected extra CSV columns")
            try:
                frames.append(int(row["frame"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: frame must be an integer") from exc
            label = (row["label"] or "").strip().upper()
            if label not in LABELS:
                raise ValueError(f"line {line_number}: invalid label {row['label']!r}")
            labels.append(label)
    validate_predictions(
        np.asarray(frames, dtype=np.int64),
        np.asarray(labels, dtype="<U1"),
        expected_frames,
    )
