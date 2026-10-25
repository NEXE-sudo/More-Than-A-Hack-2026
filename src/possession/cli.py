"""Command-line entry points for validation, prediction, and official integration."""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

from possession import __version__
from possession.config import Config, load_config
from possession.data.audit import audit_tracking_homography, inspect_video
from possession.data.loaders import load_homography, load_smart_tagging, load_tracking
from possession.data.validation import validate_submission_file
from possession.evaluation.error_analysis import analyze_baseline_audit
from possession.evaluation.official_adapter import run_official_baseline, run_official_evaluator
from possession.evaluation.reproducibility import (
    compare_experiment_reports,
    ensure_local_output_path,
    record_prediction_experiment,
    run_baseline_audit,
    run_smoothing_sensitivity_audit,
)
from possession.experiments import run_experiments
from possession.features.cache import save_feature_cache
from possession.features.context import tagging_context
from possession.features.tracking import extract_tracking_features
from possession.pipeline import predict
from possession.synthetic import CANONICAL_SCENARIOS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="possession")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    environment = commands.add_parser("check-environment")
    environment.set_defaults(action=_check_environment)

    validate = commands.add_parser("validate-data")
    validate.add_argument("--metadata", required=True, type=Path)
    validate.add_argument("--tracking", required=True, type=Path)
    validate.add_argument("--homography", type=Path)
    validate.add_argument("--smart-tagging", type=Path)
    validate.add_argument("--frames", type=int, default=199319)
    validate.set_defaults(action=_validate_data)

    audit = commands.add_parser("audit-data")
    audit.add_argument("--metadata", required=True, type=Path)
    audit.add_argument("--tracking", required=True, type=Path)
    audit.add_argument("--homography", required=True, type=Path)
    audit.add_argument("--video", type=Path)
    audit.add_argument("--decode-video", action="store_true")
    audit.add_argument("--frames", type=int, default=199319)
    audit.add_argument("--config", type=Path)
    audit.add_argument("--output", required=True, type=Path)
    audit.set_defaults(action=_audit_data)

    baseline = commands.add_parser("run-baseline")
    baseline.add_argument("--repo-root", required=True, type=Path)
    baseline.add_argument("--data-root", required=True, type=Path)
    baseline.add_argument("--output-dir", required=True, type=Path)
    baseline.set_defaults(action=_run_baseline)

    baseline_audit = commands.add_parser("run-baseline-audit")
    baseline_audit.add_argument("--repo-root", required=True, type=Path)
    baseline_audit.add_argument("--data-root", required=True, type=Path)
    baseline_audit.add_argument("--output-dir", required=True, type=Path)
    baseline_audit.add_argument("--bootstrap", type=int, default=1000)
    baseline_audit.set_defaults(action=_run_baseline_audit)

    sensitivity = commands.add_parser("run-smoothing-sensitivity")
    sensitivity.add_argument("--repo-root", required=True, type=Path)
    sensitivity.add_argument("--data-root", required=True, type=Path)
    sensitivity.add_argument("--output-dir", required=True, type=Path)
    sensitivity.add_argument("--bootstrap", type=int, default=1000)
    sensitivity.set_defaults(action=_run_smoothing_sensitivity)

    error_analysis = commands.add_parser("analyze-baseline-errors")
    error_analysis.add_argument("--repo-root", required=True, type=Path)
    error_analysis.add_argument("--audit-dir", required=True, type=Path)
    error_analysis.add_argument("--output-dir", required=True, type=Path)
    error_analysis.add_argument("--run-id", choices=("run-1", "run-2"), default="run-1")
    error_analysis.set_defaults(action=_analyze_baseline_errors)

    record = commands.add_parser("record-experiment")
    record.add_argument("--repo-root", required=True, type=Path)
    record.add_argument("--prediction", required=True, type=Path)
    record.add_argument("--output-dir", required=True, type=Path)
    record.add_argument("--experiment-id", required=True)
    record.add_argument("--description", required=True)
    record.add_argument("--generation-command", required=True)
    record.add_argument("--config", type=Path)
    record.add_argument(
        "--source-input",
        action="append",
        default=[],
        type=Path,
        help="input file to checksum (repeat for each input used to generate the prediction)",
    )
    record.add_argument("--ground-truth", default="public")
    record.add_argument("--bootstrap", type=int, default=1000)
    record.set_defaults(action=_record_experiment)

    compare = commands.add_parser("compare-experiments")
    compare.add_argument("reports", nargs="+", type=Path)
    compare.set_defaults(action=_compare_experiments)

    build = commands.add_parser("build-features")
    build.add_argument("--metadata", required=True, type=Path)
    build.add_argument("--tracking", required=True, type=Path)
    build.add_argument("--smart-tagging", type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--frames", type=int, default=199319)
    build.add_argument("--config", type=Path)
    build.set_defaults(action=_build_features)

    prediction = commands.add_parser("predict")
    prediction.add_argument("--metadata", type=Path)
    prediction.add_argument("--tracking", type=Path)
    prediction.add_argument("--feature-cache", type=Path)
    prediction.add_argument("--smart-tagging", type=Path)
    prediction.add_argument("--output", required=True, type=Path)
    prediction.add_argument("--config", type=Path)
    prediction.add_argument("--frames", type=int)
    prediction.add_argument("--repository-root", type=Path)
    prediction.add_argument("--evaluate", action="store_true")
    prediction.set_defaults(action=_predict)

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--repo-root", required=True, type=Path)
    evaluation.add_argument("--results-dir", required=True, type=Path)
    evaluation.add_argument("--output-dir", required=True, type=Path)
    evaluation.add_argument("--ground-truth", default="public")
    evaluation.add_argument("--bootstrap", type=int, default=0)
    evaluation.set_defaults(action=_evaluate)

    synthetic = commands.add_parser("run-synthetic-experiments")
    synthetic.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic-experiments"))
    synthetic.add_argument("--seeds", type=int, nargs="+", default=(11, 29, 47))
    synthetic.add_argument(
        "--scenarios",
        nargs="+",
        default=CANONICAL_SCENARIOS,
        help="canonical scenario names or declared aliases",
    )
    synthetic.add_argument("--duration-seconds", type=float, default=16.0)
    synthetic.set_defaults(action=_run_synthetic_experiments)
    return parser


def _check_environment(_: argparse.Namespace) -> int:
    print(f"Python {sys.version.split()[0]}")
    print(f"NumPy {np.__version__}")
    return 0


def _validate_data(args: argparse.Namespace) -> int:
    data = load_tracking(args.metadata, args.tracking, args.frames)
    print(f"tracking: {data.n_frames} aligned frames, {data.players.shape[1]} track slots")
    print(
        f"player speed: {data.source_speed_unit} -> {data.speed_unit} "
        f"(factor {data.speed_conversion_factor:g})"
    )
    print(
        "out-of-range coordinate observations rejected: "
        f"{data.player_coordinate_out_of_range_observations} player "
        f"({data.player_coordinate_out_of_range_components} components), "
        f"{data.ball_coordinate_out_of_range_observations} ball "
        f"({data.ball_coordinate_out_of_range_components} components)"
    )
    usable_ball_frames = int(np.isfinite(data.ball).all(axis=1).sum())
    raw_ball_frames = usable_ball_frames + data.ball_coordinate_out_of_range_observations
    print(
        f"ball observations: {raw_ball_frames / data.n_frames:.1%} raw complete; "
        f"{usable_ball_frames / data.n_frames:.1%} usable after coordinate validation"
    )
    if args.homography:
        matrices, _, valid = load_homography(args.homography, args.frames)
        print(f"homography: {len(matrices)} frames, {valid.mean():.1%} valid")
    if args.smart_tagging:
        events = load_smart_tagging(args.smart_tagging)
        masks = tagging_context(events, args.frames)
        set_piece_frames = int(masks["tagged_set_piece"].sum())
        print(
            f"Smart Tagging context only (not possession labels): {len(events)} intervals, "
            f"{set_piece_frames} set-piece frames"
        )
    return 0


def _run_baseline(args: argparse.Namespace) -> int:
    run_official_baseline(args.repo_root, args.data_root, args.output_dir)
    return 0


def _audit_data(args: argparse.Namespace) -> int:
    if args.decode_video and args.video is None:
        raise ValueError("--video is required when --decode-video is used")
    destination = args.output.expanduser().resolve()
    ensure_local_output_path(destination)
    if destination.exists():
        raise ValueError(f"audit output already exists; refusing to overwrite: {destination}")
    config = load_config(args.config) if args.config else Config()
    tracking = load_tracking(args.metadata, args.tracking, args.frames)
    matrices, errors, valid = load_homography(args.homography, args.frames)
    video_info = inspect_video(args.video, decode=args.decode_video) if args.video else None
    audit = audit_tracking_homography(
        tracking,
        matrices,
        errors,
        valid,
        pitch_length_m=config.data.pitch_length_m,
        pitch_width_m=config.data.pitch_width_m,
        video_width=None if video_info is None else video_info["width"],
        video_height=None if video_info is None else video_info["height"],
    )
    if video_info is not None:
        fps = config.data.fps
        audit["video"] = video_info
        count_matches = video_info["decoded_frame_count"] == tracking.n_frames
        timestamps_match = video_info["timestamp_samples"] == video_info["decoded_frame_count"]
        fps_matches = Fraction(video_info["frame_rate"]) == fps
        cadence_ok = video_info["cadence_within_tolerance"]
        audit["alignment"] = {
            "video_tracking_frame_count_matches": count_matches,
            "video_frame_rate_matches_config": fps_matches,
            "video_timestamp_count_matches_decoded_count": timestamps_match,
            "video_timestamp_cadence_matches_frame_indices": cadence_ok,
            "tracking_indices_zero_based": audit["frame_index_range"] == [0, args.frames - 1],
            "homography_rows_align_with_tracking": len(valid) == tracking.n_frames,
            "temporal_origin_independently_verified": False,
        }
        audit["video_sync_status"] = (
            "index_and_cadence_consistent_but_temporal_origin_not_independently_verified"
            if all((count_matches, timestamps_match, fps_matches, cadence_ok))
            else "alignment_checks_failed"
        )
    else:
        audit["video_status"] = {
            "status": "blocked",
            "reason": "tactical video was not supplied",
            "next_step": "set TACTICAL_VIDEO to the authorized local file and pass --video "
            '"$TACTICAL_VIDEO"',
        }
    audit["calibration_validation"] = {
        "status": "blocked",
        "method": "independent image-to-pitch landmark reprojection",
        "independent_reference_count": 0,
        "reason": "no independently established pitch landmarks and corresponding image "
        "coordinates were supplied",
        "required_evidence": (
            "authorized, independently measured pitch landmarks with image coordinates for "
            "matching tactical frames, documented units, and measurement uncertainty"
        ),
    }
    report = {
        "report_type": "aggregate_local_data_audit",
        "sensitive_inputs_not_recorded": True,
        "video_decode_requested": args.decode_video,
        "summary": audit,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(destination)
    except (OSError, TypeError, ValueError):
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    print(f"wrote aggregate local audit to {destination}")
    return 0


def _run_baseline_audit(args: argparse.Namespace) -> int:
    report = run_baseline_audit(
        args.repo_root,
        args.data_root,
        args.output_dir,
        bootstrap=args.bootstrap,
    )
    print(f"baseline audit status: {report['status']}")
    print(
        f"prediction files byte-identical: "
        f"{report.get('determinism', {}).get('prediction_files_byte_identical')}"
    )
    return 0 if report["status"] == "complete" else 1


def _run_smoothing_sensitivity(args: argparse.Namespace) -> int:
    report = run_smoothing_sensitivity_audit(
        args.repo_root,
        args.data_root,
        args.output_dir,
        bootstrap=args.bootstrap,
    )
    print(f"smoothing sensitivity status: {report['status']}")
    print(
        "prediction files byte-identical: "
        f"{report.get('determinism', {}).get('prediction_files_byte_identical')}"
    )
    return 0 if report["status"] == "complete" else 1


def _analyze_baseline_errors(args: argparse.Namespace) -> int:
    report = analyze_baseline_audit(
        args.repo_root,
        args.audit_dir,
        args.output_dir,
        run_id=args.run_id,
    )
    print(f"wrote aggregate official error analysis to {args.output_dir.expanduser().resolve()}")
    print(
        f"variants: {len(report['variants'])}; "
        f"graded frames: {report['evaluation_scope']['graded_frames']}"
    )
    return 0


def _record_experiment(args: argparse.Namespace) -> int:
    report = record_prediction_experiment(
        args.repo_root,
        args.prediction,
        args.output_dir,
        experiment_id=args.experiment_id,
        description=args.description,
        generation_command=args.generation_command,
        config_path=args.config,
        source_inputs=tuple(args.source_input),
        ground_truth=args.ground_truth,
        bootstrap=args.bootstrap,
    )
    print(f"experiment status: {report['status']}")
    print(f"report: {args.output_dir.expanduser().resolve() / 'experiment-report.json'}")
    return 0


def _compare_experiments(args: argparse.Namespace) -> int:
    experiments, scope_mismatch = compare_experiment_reports(args.reports)
    if scope_mismatch:
        print(
            "WARNING: report provenance, configuration, or evaluation scope differs or is "
            "incomplete; review each row's comparison warnings before comparing scores.",
            file=sys.stderr,
        )
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        (
            "experiment_id",
            "scope",
            "spell_f1_mean",
            "macro_f1",
            "status",
            "project_revision",
            "configuration_fingerprint",
            "comparison_warnings",
        )
    )
    for experiment in experiments:
        metrics = experiment.get("metrics", {})
        writer.writerow(
            (
                experiment.get("experiment_id"),
                experiment.get("scope"),
                metrics.get("spell_f1_mean", "n/a"),
                metrics.get("macro_f1", "n/a"),
                experiment.get("status", "unknown"),
                experiment.get("project_revision", "unknown"),
                experiment.get("configuration_fingerprint", "unknown"),
                "; ".join(experiment.get("comparison_warnings", [])),
            )
        )
    return 0


def _build_features(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else Config()
    data = load_tracking(args.metadata, args.tracking, args.frames)
    features = extract_tracking_features(
        data,
        pitch_length_m=config.data.pitch_length_m,
        pitch_width_m=config.data.pitch_width_m,
        fps=config.data.fps,
    )
    if args.smart_tagging:
        features.update(tagging_context(load_smart_tagging(args.smart_tagging), args.frames))
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved_to = save_feature_cache(
        destination,
        features,
        source_paths=(args.metadata, args.tracking),
        context_source_paths=(() if args.smart_tagging is None else (args.smart_tagging,)),
        feature_config={
            "fps": config.data.fps,
            "pitch_length_m": config.data.pitch_length_m,
            "pitch_width_m": config.data.pitch_width_m,
        },
    )
    print(f"saved {len(features)} feature arrays to {saved_to}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else Config()
    expected_frames = config.data.frames if args.frames is None else args.frames
    if args.metadata is None or args.tracking is None:
        raise ValueError(
            "--metadata and --tracking are required, including when validating a feature cache"
        )
    labels = predict(
        args.metadata,
        args.tracking,
        args.output,
        config,
        expected_frames=expected_frames,
        feature_cache_path=args.feature_cache,
        smart_tagging_path=args.smart_tagging,
    )
    validate_submission_file(args.output, expected_frames)
    print(f"wrote {len(labels)} validated frame labels to {args.output.expanduser().resolve()}")
    if args.evaluate:
        if not args.repository_root:
            raise ValueError("--repository-root is required when --evaluate is used")
        run_official_evaluator(
            args.repository_root,
            args.output.expanduser().resolve().parent,
            args.output.expanduser().resolve().parent / "evaluation",
        )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    run_official_evaluator(
        args.repo_root,
        args.results_dir,
        args.output_dir,
        args.ground_truth,
        bootstrap=args.bootstrap,
    )
    return 0


def _run_synthetic_experiments(args: argparse.Namespace) -> int:
    paths = run_experiments(
        args.output_dir,
        seeds=tuple(args.seeds),
        scenarios=tuple(args.scenarios),
        duration_seconds=args.duration_seconds,
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


def main() -> None:
    args = _parser().parse_args()
    try:
        status = args.action(args)
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
