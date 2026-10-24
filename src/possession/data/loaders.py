"""Standard-library loaders for the official Metrica tracking sidecar formats."""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from possession.data.alignment import tagging_seconds_to_frames, tracking_ids_to_video_frames
from possession.data.validation import TrackingData, validate_tracking

MISSING_TOKENS = {"N", "NAN", ""}
HOMOGRAPHY_COLUMNS = [f"h{row}{col}" for row in range(3) for col in range(3)]


def _parse_group(text: str) -> tuple[float, float, float]:
    if text.strip().upper() in MISSING_TOKENS:
        return np.nan, np.nan, np.nan
    try:
        values = [float(item) for item in text.split(",")]
    except ValueError as exc:
        raise ValueError(f"invalid tracking coordinate group {text!r}") from exc
    if len(values) != 3:
        raise ValueError(f"expected x,y,speed tracking group, found {text!r}")
    return values[0], values[1], values[2]


def load_tracking(
    metadata_path: Path | str,
    raw_path: Path | str,
    expected_frames: int = 199319,
) -> TrackingData:
    """Read ATD XML metadata and raw frame data into compact NumPy arrays."""
    if expected_frames <= 0:
        raise ValueError("expected_frames must be positive")
    meta = Path(metadata_path).expanduser().resolve()
    raw = Path(raw_path).expanduser().resolve()
    if not meta.is_file():
        raise FileNotFoundError(f"tracking metadata not found: {meta}")
    if not raw.is_file():
        raise FileNotFoundError(f"tracking data not found: {raw}")
    try:
        root = ET.parse(meta).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{meta}: malformed tracking metadata XML: {exc}") from exc
    team_ids = [node.get("id") for node in root.iter("Team")]
    players = list(root.iter("Player"))
    if len(team_ids) != 2 or not players or len(set(team_ids)) != 2 or None in team_ids:
        raise ValueError(f"{meta}: expected two teams and at least one Player entry")
    team_lookup = {team_ids[0]: 0, team_ids[1]: 1}
    try:
        team_assignments = np.asarray(
            [team_lookup[player.get("teamId")] for player in players], dtype=np.int8
        )
    except KeyError as exc:
        raise ValueError(f"{meta}: player refers to an unknown team ID {exc.args[0]!r}") from exc

    n_tracks = len(players)
    frame_ids = np.empty(expected_frames, dtype=np.int64)
    track_positions = np.full((expected_frames, n_tracks, 2), np.nan, dtype=np.float32)
    track_speeds = np.full((expected_frames, n_tracks), np.nan, dtype=np.float32)
    ball_positions = np.full((expected_frames, 2), np.nan, dtype=np.float32)
    row_count = 0
    with raw.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                raise ValueError(f"{raw}:{line_number}: expected frame:players:ball")
            if row_count >= expected_frames:
                raise ValueError(f"{raw}: contains more than {expected_frames} frames")
            try:
                frame_id = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"{raw}:{line_number}: invalid frame ID {parts[0]!r}") from exc
            frame_ids[row_count] = frame_id
            groups = parts[1].split(";") if parts[1] else []
            if len(groups) != n_tracks:
                raise ValueError(
                    f"{raw}:{line_number}: expected {n_tracks} player tracks, found {len(groups)}"
                )
            for track, group in enumerate(groups):
                x, y, speed = _parse_group(group)
                track_positions[row_count, track] = (x, y)
                track_speeds[row_count, track] = speed
            ball_x, ball_y, _ = _parse_group(parts[2])
            ball_positions[row_count] = (ball_x, ball_y)
            row_count += 1
    if row_count != expected_frames:
        raise ValueError(f"{raw}: expected {expected_frames} frames, found {row_count}")
    video_frames = tracking_ids_to_video_frames(frame_ids)
    expected = np.arange(expected_frames, dtype=np.int64)
    if not np.array_equal(video_frames, expected):
        raise ValueError(
            f"{raw}: ATD frame IDs must be contiguous 1..{expected_frames} in file order"
        )
    data = TrackingData(
        frame_ids=video_frames,
        players=track_positions,
        ball=ball_positions,
        player_teams=team_assignments,
        player_speed=track_speeds,
    )
    validate_tracking(data, expected_frames)
    return data


def load_homography(path: Path | str, expected_frames: int = 199319) -> tuple[np.ndarray, ...]:
    """Load zero-indexed homography rows, preserving invalid matrices as NaN."""
    if expected_frames <= 0:
        raise ValueError("expected_frames must be positive")
    source = Path(path).expanduser().resolve()
    required = ["frame", *HOMOGRAPHY_COLUMNS, "reprojection_error", "valid"]
    matrices = np.full((expected_frames, 3, 3), np.nan, dtype=np.float64)
    errors = np.full(expected_frames, np.nan, dtype=np.float64)
    valid = np.zeros(expected_frames, dtype=bool)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise ValueError(f"{source}: missing required homography columns")
        count = 0
        for line, row in enumerate(reader, start=2):
            if count >= expected_frames:
                raise ValueError(f"{source}: contains more than {expected_frames} rows")
            try:
                frame = int(row["frame"])
                valid_value = int(row["valid"])
                if valid_value not in (0, 1):
                    raise ValueError
                is_valid = valid_value == 1
                error = float(row["reprojection_error"])
                matrix = np.asarray([float(row[name]) for name in HOMOGRAPHY_COLUMNS])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line}: malformed homography row") from exc
            if frame != count:
                raise ValueError(f"{source}:{line}: expected frame {count}, found {frame}")
            if is_valid and (not np.isfinite(matrix).all() or not np.isfinite(error)):
                raise ValueError(f"{source}:{line}: valid homography has non-finite values")
            if is_valid and error < 0:
                raise ValueError(f"{source}:{line}: reprojection error cannot be negative")
            if is_valid and abs(np.linalg.det(matrix.reshape(3, 3))) < 1e-12:
                raise ValueError(f"{source}:{line}: valid homography matrix is singular")
            if is_valid:
                matrices[frame] = matrix.reshape(3, 3)
                errors[frame] = error
                valid[frame] = True
            count += 1
    if count != expected_frames:
        raise ValueError(f"{source}: expected {expected_frames} rows, found {count}")
    return matrices, errors, valid


def load_smart_tagging(path: Path | str, fps: int = 25) -> list[dict[str, object]]:
    """Load Smart Tagging intervals; seconds are relative to tactical video frame zero."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    source = Path(path).expanduser().resolve()
    events: list[dict[str, object]] = []
    try:
        instances = ET.parse(source).iter("instance")
    except ET.ParseError as exc:
        raise ValueError(f"{source}: malformed Smart Tagging XML: {exc}") from exc
    for instance in instances:
        try:
            start = float(instance.findtext("start", ""))
            end = float(instance.findtext("end", ""))
        except ValueError as exc:
            raise ValueError(f"{source}: instance has invalid start/end seconds") from exc
        if not np.isfinite([start, end]).all():
            raise ValueError(f"{source}: instance start/end seconds must be finite")
        if end < start:
            raise ValueError(f"{source}: instance ends before it starts")
        tags = {
            label.findtext("group", ""): label.findtext("text", "")
            for label in instance.findall("label")
        }
        events.append(
            {
                "code": instance.findtext("code", ""),
                "start_s": start,
                "end_s": end,
                "start_frame": int(tagging_seconds_to_frames(start, fps)),
                "end_frame": int(tagging_seconds_to_frames(end, fps)),
                "tags": tags,
            }
        )
    return events
