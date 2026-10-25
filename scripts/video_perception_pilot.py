"""Run a local, frame-aligned YOLOX-Nano tactical-video feasibility sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess

from possession.data.loaders import load_homography, load_tracking
from possession.features.geometry import project_to_image

FPS = 25
PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0
BALL_CLASS = 32
PERSON_CLASS = 0
INPUT_SIZE = (416, 416)


def parse_anchor(value: str) -> tuple[str, int]:
    try:
        name, raw_frame = value.split("=", maxsplit=1)
        frame = int(raw_frame)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("anchors must be NAME=FRAME") from exc
    if not name or frame < 0:
        raise argparse.ArgumentTypeError("anchor name must be non-empty and frame non-negative")
    return name, frame


def frame_samples(
    anchors: list[tuple[str, int]], radius: int, step: int
) -> tuple[dict[int, list[str]], dict[str, int]]:
    selected: dict[int, list[str]] = {}
    centers = dict(anchors)
    for name, center in anchors:
        start = max(center - radius, 0)
        stop = center + radius
        sample_frames = set(range(start, stop + 1, step))
        sample_frames.add(center)
        for frame in sample_frames:
            selected.setdefault(frame, []).append(name)
    return selected, centers


def detections_for_tiles(
    image: np.ndarray,
    model: torch.nn.Module,
    preproc: ValTransform,
    num_classes: int,
    confidence: float,
    nms_threshold: float,
) -> list[dict[str, float | int]]:
    height, width = image.shape[:2]
    x_mid, y_mid = width // 2, height // 2
    overlap_x, overlap_y = round(width * 0.05), round(height * 0.05)
    regions = (
        (0, 0, x_mid + overlap_x, y_mid + overlap_y),
        (x_mid - overlap_x, 0, width, y_mid + overlap_y),
        (0, y_mid - overlap_y, x_mid + overlap_x, height),
        (x_mid - overlap_x, y_mid - overlap_y, width, height),
    )
    tensors: list[np.ndarray] = []
    ratios: list[float] = []
    for x1, y1, x2, y2 in regions:
        crop = image[y1:y2, x1:x2]
        ratio = min(INPUT_SIZE[0] / crop.shape[0], INPUT_SIZE[1] / crop.shape[1])
        tensor, _ = preproc(crop, None, INPUT_SIZE)
        tensors.append(tensor)
        ratios.append(ratio)

    with torch.inference_mode():
        outputs = model(torch.from_numpy(np.stack(tensors)).float())
        batch_outputs = postprocess(
            outputs,
            num_classes,
            confidence,
            nms_threshold,
            class_agnostic=False,
        )

    detections: list[dict[str, float | int]] = []
    for region, ratio, output in zip(regions, ratios, batch_outputs, strict=True):
        if output is None:
            continue
        offset_x, offset_y = region[:2]
        for row in output.cpu().numpy():
            x1, y1, x2, y2 = row[:4] / ratio
            detections.append(
                {
                    "x1": float(x1 + offset_x),
                    "y1": float(y1 + offset_y),
                    "x2": float(x2 + offset_x),
                    "y2": float(y2 + offset_y),
                    "score": float(row[4] * row[5]),
                    "class_id": int(row[6]),
                }
            )

    if not detections:
        return detections
    boxes = torch.tensor(
        [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=torch.float32
    )
    scores = torch.tensor([d["score"] for d in detections], dtype=torch.float32)
    classes = torch.tensor([d["class_id"] for d in detections], dtype=torch.int64)
    keep = batched_nms(boxes, scores, classes, nms_threshold).tolist()
    return [detections[index] for index in keep]


def pitch_position(
    matrix: np.ndarray, image_x: float, image_y: float
) -> tuple[float, float] | None:
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    projected = inverse @ np.asarray([image_x, image_y, 1.0])
    if not np.isfinite(projected).all() or abs(projected[2]) <= 1e-12:
        return None
    point = projected[:2] / projected[2]
    return float(point[0]), float(point[1])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotate(
    image: np.ndarray, detections: list[dict[str, float | int]], frame: int, timestamp: float
) -> np.ndarray:
    output = image.copy()
    for detection in detections:
        class_id = int(detection["class_id"])
        if class_id not in (BALL_CLASS, PERSON_CLASS):
            continue
        x1, y1, x2, y2 = (int(detection[key]) for key in ("x1", "y1", "x2", "y2"))
        color = (0, 255, 0) if class_id == BALL_CLASS else (0, 165, 255)
        score = float(detection["score"])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{'ball' if class_id == BALL_CLASS else 'person'} {score:.2f}",
            (x1, max(y1 - 5, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    cv2.putText(
        output,
        f"frame={frame} t={timestamp:.2f}s",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--homography", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor", action="append", type=parse_anchor, required=True)
    parser.add_argument("--radius-frames", type=int, default=25)
    parser.add_argument("--step-frames", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    if args.radius_frames < 0 or args.step_frames < 1 or args.threads < 1:
        parser.error("radius must be non-negative and step/threads must be positive")
    if not 0 < args.confidence < 1 or not 0 < args.nms_threshold < 1:
        parser.error("confidence and NMS threshold must be in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tracking = load_tracking(args.metadata, args.tracking)
    matrices, _, homography_valid = load_homography(args.homography, tracking.n_frames)
    data_loaded = time.perf_counter()

    torch.set_num_threads(args.threads)
    exp = get_exp(None, "yolox-nano")
    exp.test_size = INPUT_SIZE
    model = exp.get_model()
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    preproc = ValTransform(legacy=False)
    model_loaded = time.perf_counter()

    selected, centers = frame_samples(args.anchor, args.radius_frames, args.step_frames)
    max_frame = max(selected)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise OSError(f"could not open tactical video: {args.video}")
    video_info = {
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "reported_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    if video_info["fps"] != FPS:
        raise ValueError(f"expected {FPS} fps, video reports {video_info['fps']}")

    rows: list[dict[str, object]] = []
    inspected: set[str] = set()
    frame_start = time.perf_counter()
    inference_seconds = 0.0
    for frame_index in range(max_frame + 1):
        ok, image = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"video ended before requested frame {frame_index}")
        anchor_names = selected.get(frame_index)
        if anchor_names is None:
            continue
        inference_started = time.perf_counter()
        detections = detections_for_tiles(
            image,
            model,
            preproc,
            exp.num_classes,
            args.confidence,
            args.nms_threshold,
        )
        inference_seconds += time.perf_counter() - inference_started
        ball_detections = [d for d in detections if d["class_id"] == BALL_CLASS]
        ball = max(ball_detections, key=lambda detection: detection["score"], default=None)
        person_detections = [d for d in detections if d["class_id"] == PERSON_CLASS]
        visible_track = bool(np.isfinite(tracking.ball[frame_index]).all())
        h_valid = bool(homography_valid[frame_index])
        detected_pitch = None
        track_pixel = None
        disagreement_px = None
        disagreement_m = None
        nearest_player_m = None
        nearest_player_team = None
        ball_x = ball_y = ball_score = None
        if ball is not None:
            ball_x = (float(ball["x1"]) + float(ball["x2"])) / 2
            ball_y = (float(ball["y1"]) + float(ball["y2"])) / 2
            ball_score = float(ball["score"])
            if h_valid:
                detected_pitch = pitch_position(matrices[frame_index], ball_x, ball_y)
        if visible_track and h_valid:
            projection, projection_valid = project_to_image(
                matrices[frame_index : frame_index + 1],
                tracking.ball[frame_index : frame_index + 1],
                np.asarray([h_valid]),
            )
            if projection_valid[0]:
                track_pixel = projection[0].tolist()
                if ball is not None:
                    disagreement_px = float(
                        np.linalg.norm(np.asarray([ball_x, ball_y]) - np.asarray(track_pixel))
                    )
                    if detected_pitch is not None:
                        normalized_delta = np.asarray(detected_pitch) - tracking.ball[frame_index]
                        disagreement_m = float(
                            np.linalg.norm(
                                normalized_delta * np.asarray([PITCH_LENGTH_M, PITCH_WIDTH_M])
                            )
                        )
        if ball is not None and detected_pitch is not None:
            positions = tracking.players[frame_index]
            finite_players = np.isfinite(positions).all(axis=1)
            if finite_players.any():
                distances = np.linalg.norm(
                    (positions[finite_players] - np.asarray(detected_pitch))
                    * np.asarray([PITCH_LENGTH_M, PITCH_WIDTH_M]),
                    axis=1,
                )
                nearest_index = int(np.flatnonzero(finite_players)[np.argmin(distances)])
                nearest_player_m = float(np.min(distances))
                nearest_player_team = int(tracking.player_teams[nearest_index])

        row: dict[str, object] = {
            "frame_index": frame_index,
            "timestamp_s": frame_index / FPS,
            "anchor_names": "|".join(anchor_names),
            "homography_valid": h_valid,
            "ball_track_visible": visible_track,
            "video_ball_detected": ball is not None,
            "ball_x_px": ball_x,
            "ball_y_px": ball_y,
            "ball_score": ball_score,
            "ball_x_pitch_norm": None if detected_pitch is None else detected_pitch[0],
            "ball_y_pitch_norm": None if detected_pitch is None else detected_pitch[1],
            "track_ball_x_px": None if track_pixel is None else track_pixel[0],
            "track_ball_y_px": None if track_pixel is None else track_pixel[1],
            "track_disagreement_px": disagreement_px,
            "track_disagreement_m": disagreement_m,
            "nearest_tracked_player_m": nearest_player_m,
            "nearest_tracked_player_team": nearest_player_team,
            "person_detection_count": len(person_detections),
            "person_detections_json": json.dumps(person_detections, separators=(",", ":")),
        }
        rows.append(row)

        for name in anchor_names:
            if frame_index == centers[name] and name not in inspected:
                output = annotate(image, detections, frame_index, frame_index / FPS)
                cv2.imwrite(str(args.output_dir / f"inspection_{name}_{frame_index}.jpg"), output)
                inspected.add(name)
    capture.release()
    frame_finished = time.perf_counter()

    by_anchor = {name: [] for name in centers}
    for row in rows:
        for name in str(row["anchor_names"]).split("|"):
            by_anchor[name].append(row)
    for anchor_rows in by_anchor.values():
        anchor_rows.sort(key=lambda row: int(row["frame_index"]))
        previous_detection: dict[str, object] | None = None
        for row in anchor_rows:
            row["frames_since_video_detection"] = (
                None
                if previous_detection is None
                else int(row["frame_index"]) - int(previous_detection["frame_index"])
            )
            row["ball_movement_speed_m_s"] = None
            if row["video_ball_detected"] and row["ball_x_pitch_norm"] is not None:
                if (
                    previous_detection is not None
                    and previous_detection["ball_x_pitch_norm"] is not None
                ):
                    elapsed = (
                        int(row["frame_index"]) - int(previous_detection["frame_index"])
                    ) / FPS
                    delta = np.asarray(
                        [
                            float(row["ball_x_pitch_norm"])
                            - float(previous_detection["ball_x_pitch_norm"]),
                            float(row["ball_y_pitch_norm"])
                            - float(previous_detection["ball_y_pitch_norm"]),
                        ]
                    )
                    row["ball_movement_speed_m_s"] = float(
                        np.linalg.norm(delta * [PITCH_LENGTH_M, PITCH_WIDTH_M]) / elapsed
                    )
                previous_detection = row
            if row["video_ball_detected"] and row["frames_since_video_detection"] is None:
                row["frames_since_video_detection"] = 0

    csv_path = args.output_dir / "video_features.csv"
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    comparisons = [row for row in rows if row["track_disagreement_px"] is not None]
    track_gap_detections = [
        row for row in rows if not row["ball_track_visible"] and row["video_ball_detected"]
    ]
    speeds = [
        float(row["ball_movement_speed_m_s"])
        for row in rows
        if row["ball_movement_speed_m_s"] is not None
    ]
    summary = {
        "model": "YOLOX-Nano COCO checkpoint",
        "model_source_revision": "419778480ab6ec0590e5d3831b3afb3b46ab2aa3",
        "checkpoint_sha256": sha256_file(args.weights),
        "confidence_threshold": args.confidence,
        "nms_threshold": args.nms_threshold,
        "tile_layout": "four overlapping image quadrants, 416x416 each",
        "sampled_frames": len(rows),
        "anchor_count": len(centers),
        "ball_detections": sum(bool(row["video_ball_detected"]) for row in rows),
        "ball_detection_availability": (
            sum(bool(row["video_ball_detected"]) for row in rows) / len(rows) if rows else None
        ),
        "tracker_visible_frames": sum(bool(row["ball_track_visible"]) for row in rows),
        "detections_during_tracker_gaps": len(track_gap_detections),
        "tracker_comparison_frames": len(comparisons),
        "tracker_disagreement_px_median": (
            float(np.median([row["track_disagreement_px"] for row in comparisons]))
            if comparisons
            else None
        ),
        "tracker_disagreement_px_p90": (
            float(np.percentile([row["track_disagreement_px"] for row in comparisons], 90))
            if comparisons
            else None
        ),
        "tracker_disagreement_m_median": (
            float(np.median([row["track_disagreement_m"] for row in comparisons]))
            if comparisons
            else None
        ),
        "movement_speed_m_s_p90": float(np.percentile(speeds, 90)) if speeds else None,
        "video": video_info,
        "timings_s": {
            "tracking_and_homography_load": round(data_loaded - started, 3),
            "model_load": round(model_loaded - data_loaded, 3),
            "frame_decode_and_inference": round(frame_finished - frame_start, 3),
            "model_inference": round(inference_seconds, 3),
            "video_decode_and_bookkeeping": round(
                frame_finished - frame_start - inference_seconds, 3
            ),
            "model_inference_per_sampled_frame": (
                round(inference_seconds / len(rows), 4) if rows else None
            ),
            "model_tile_images_per_second": (
                round(4 * len(rows) / inference_seconds, 2) if inference_seconds else None
            ),
            "total": round(frame_finished - started, 3),
        },
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1),
        "torch_threads": torch.get_num_threads(),
        "outputs_are_local_data_derived_artifacts": True,
    }
    with (args.output_dir / "pilot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
