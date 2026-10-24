"""Command-line entry points for validation, prediction, and official integration."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from possession import __version__
from possession.config import Config, load_config
from possession.data.loaders import load_homography, load_smart_tagging, load_tracking
from possession.data.validation import validate_submission_file
from possession.evaluation.official_adapter import run_official_baseline, run_official_evaluator
from possession.experiments import run_experiments
from possession.features.cache import save_feature_cache
from possession.features.context import tagging_context
from possession.features.tracking import extract_tracking_features
from possession.pipeline import predict
from possession.synthetic import SCENARIOS


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

    baseline = commands.add_parser("run-baseline")
    baseline.add_argument("--repo-root", required=True, type=Path)
    baseline.add_argument("--data-root", required=True, type=Path)
    baseline.add_argument("--output-dir", required=True, type=Path)
    baseline.set_defaults(action=_run_baseline)

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
    synthetic.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
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
    print(f"ball visibility: {np.isfinite(data.ball).all(axis=1).mean():.1%}")
    if args.homography:
        matrices, _, valid = load_homography(args.homography, args.frames)
        print(f"homography: {len(matrices)} frames, {valid.mean():.1%} valid")
    if args.smart_tagging:
        events = load_smart_tagging(args.smart_tagging)
        masks = tagging_context(events, args.frames)
        set_piece_frames = int(masks["tagged_set_piece"].sum())
        print(f"Smart Tagging: {len(events)} intervals, {set_piece_frames} set-piece frames")
    return 0


def _run_baseline(args: argparse.Namespace) -> int:
    run_official_baseline(args.repo_root, args.data_root, args.output_dir)
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
