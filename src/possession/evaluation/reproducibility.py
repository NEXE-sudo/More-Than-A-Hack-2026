"""Reproducible wrappers and metadata for official baseline/evaluation runs."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from possession.config import Config, load_config
from possession.evaluation.official_adapter import run_official_baseline, run_official_evaluator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SMOOTHING_SENSITIVITY_WIDTHS = (75, 100)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SCORE_METRICS = ("macro_f1", "spell_f1_mean", "spell_f1_at_headline")
_COUNT_METRICS = (
    "graded_frames",
    "ground_truth_spells",
    "predicted_spells",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path | str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(Path(path).resolve()), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_worktree_dirty(path: Path | str) -> bool | None:
    result = subprocess.run(
        ["git", "-C", str(Path(path).resolve()), "status", "--porcelain", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _official_variants(
    script: Path,
    smoothing_half_widths: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    if smoothing_half_widths is not None:
        if not smoothing_half_widths or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 0
            for width in smoothing_half_widths
        ):
            raise ValueError("smoothing half-widths must be non-negative integers")
        if len(set(smoothing_half_widths)) != len(smoothing_half_widths):
            raise ValueError("smoothing half-widths must be unique")
        return [
            {
                "name": f"baseline_smooth{width}" if width else "baseline_raw",
                "smoothing_half_width_frames": width,
            }
            for width in smoothing_half_widths
        ]

    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "VARIANTS" for target in node.targets
        ):
            values = ast.literal_eval(node.value)
            variants = []
            for value in values:
                if (
                    not isinstance(value, list | tuple)
                    or len(value) != 2
                    or not isinstance(value[0], str)
                    or not value[0]
                    or isinstance(value[1], bool)
                    or not isinstance(value[1], int)
                    or value[1] < 0
                ):
                    raise ValueError(f"invalid official baseline variant: {value!r}")
                variants.append({"name": value[0], "smoothing_half_width_frames": value[1]})
            names = [variant["name"] for variant in variants]
            if not variants or len(names) != len(set(names)):
                raise ValueError("official baseline variants must have unique names")
            return variants
    raise ValueError(f"official baseline variant configuration not found in {script}")


def _scored_metrics(evaluation_dir: Path) -> dict[str, dict[str, Any]]:
    reports_dir = evaluation_dir / "reports"
    if not reports_dir.is_dir():
        raise FileNotFoundError(f"official evaluator reports missing: {reports_dir}")
    metrics: dict[str, dict[str, Any]] = {}
    for path in sorted(reports_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError(f"official evaluator report must be a JSON object: {path}")
        name = report.get("submission")
        if not isinstance(name, str) or not name:
            raise ValueError(f"official evaluator report has invalid submission name: {path}")
        if name in metrics:
            raise ValueError(f"official evaluator produced duplicate reports for {name}")
        transitions = report.get("transitions", {})
        if not isinstance(transitions, dict):
            raise ValueError(f"official evaluator report has invalid transitions: {path}")
        try:
            metrics[name] = {
                "macro_f1": report["macro_f1"],
                "spell_f1_mean": report["spell_f1_mean"],
                "spell_f1_at_headline": report["spell_f1_at_headline"],
                "graded_frames": report["n_graded"],
                "ground_truth_spells": report["n_gt_spells"],
                "predicted_spells": report["n_pred_spells"],
                "transitions": {
                    str(tau): {
                        "f1": values["f1"],
                        "matched": values["n_matched"],
                        "missed": values["n_missed"],
                        "spurious": values["n_spurious"],
                        "mean_latency": values.get("mean_latency"),
                        "mean_abs_latency": values.get("mean_abs_latency"),
                    }
                    for tau, values in transitions.items()
                },
                "spell_bootstrap_low": report.get("spell_bootstrap_low"),
                "spell_bootstrap_high": report.get("spell_bootstrap_high"),
                "bootstrap_n": report.get("bootstrap_n"),
                "n_gt_segments": report.get("n_gt_segments"),
                "n_pred_segments": report.get("n_pred_segments"),
                "fragmentation": report.get("fragmentation"),
            }
        except (KeyError, TypeError) as exc:
            raise ValueError(f"official evaluator report has an incomplete schema: {path}") from exc
        _validate_metrics(metrics[name], f"official evaluator report {path}")
    if not metrics:
        raise ValueError(f"official evaluator produced no per-submission reports in {reports_dir}")
    return metrics


def _stable_id(*parts: Any) -> str:
    serialized = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()[:16]


def _scoreboard_metrics(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "scored":
            continue
        name = row.get("submission")
        if not isinstance(name, str) or not name:
            raise ValueError(f"official scoreboard has invalid submission name: {path}")
        if name in metrics:
            raise ValueError(f"official scoreboard contains duplicate submission {name}")
        try:
            metrics[name] = {
                "macro_f1": float(row["macro_f1"]),
                "spell_f1_mean": float(row["spell_f1_mean"]),
                "spell_f1_at_headline": float(row["spell_f1_at_headline"]),
                "graded_frames": int(row["n_graded"]),
                "ground_truth_spells": int(row["n_gt_spells"]),
                "predicted_spells": int(row["n_pred_spells"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"official scoreboard has incomplete metrics: {path}") from exc
        _validate_metrics(metrics[row["submission"]], f"official scoreboard {path}")
    if not metrics:
        raise ValueError(f"official scoreboard contains no scored submissions: {path}")
    return metrics


def _validate_metrics(metrics: dict[str, Any], source: str) -> None:
    missing = sorted(set((*_SCORE_METRICS, *_COUNT_METRICS)) - metrics.keys())
    if missing:
        raise ValueError(f"{source} is missing metrics: {', '.join(missing)}")
    for name in _SCORE_METRICS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{source} has invalid {name}: expected a finite number")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{source} has invalid {name}: expected a finite value from 0 to 1")
    for name in _COUNT_METRICS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{source} has invalid {name}: expected a non-negative integer")
    if metrics["graded_frames"] == 0:
        raise ValueError(f"{source} has no graded frames")


def _public_ground_truth(repository_root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted((repository_root / "evaluation" / "gt").glob("*_possession_gt_public.csv"))
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one authorized public ground-truth CSV under "
            f"{repository_root / 'evaluation' / 'gt'}, found {len(candidates)}"
        )
    ground_truth = candidates[0]
    sidecar = ground_truth.with_name(f"{ground_truth.stem}_meta.json")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        scope = {
            "ground_truth": "public",
            "ground_truth_sha256": sha256_file(ground_truth),
            "ground_truth_metadata_sha256": sha256_file(sidecar),
            "graded_frames": metadata["n_graded_frames"],
            "ground_truth_spells": metadata["n_possessions"],
            "video_frames": metadata["n_video_frames"],
            "public_window": metadata["public_window"],
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"public ground-truth metadata is missing or malformed: {sidecar}"
        ) from exc
    if any(
        isinstance(scope[key], bool) or not isinstance(scope[key], int) or scope[key] <= 0
        for key in ("graded_frames", "ground_truth_spells", "video_frames")
    ) or not isinstance(scope["public_window"], dict):
        raise ValueError(f"public ground-truth metadata has invalid counts: {sidecar}")
    return ground_truth, scope


def _provenance(
    root: Path,
    baseline_script: Path,
    evaluator_script: Path,
    metrics_script: Path,
    baseline_dependencies: tuple[Path, ...],
    inputs: tuple[Path, ...],
    scope: dict[str, Any],
    configuration: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "project_revision": git_revision(PROJECT_ROOT),
        "project_worktree_dirty": git_worktree_dirty(PROJECT_ROOT),
        "official_repository_revision": git_revision(root),
        "official_worktree_dirty": git_worktree_dirty(root),
        "official_baseline_sha256": sha256_file(baseline_script),
        "official_evaluator_sha256": sha256_file(evaluator_script),
        "official_metrics_sha256": sha256_file(metrics_script),
        "official_baseline_dependencies_sha256": {
            path.name: sha256_file(path) for path in baseline_dependencies
        },
        "input_sha256": {path.name: sha256_file(path) for path in inputs},
        "evaluation_scope": scope,
        "configuration": configuration,
    }


def ensure_local_output_path(destination: Path) -> None:
    resolved = destination.expanduser().resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return
    if not relative.parts or relative.parts[0] not in {"outputs", "cache", "artifacts"}:
        raise ValueError(
            "output paths inside the repository must be under ignored outputs/, cache/, "
            f"or artifacts/: {resolved}"
        )
    ignored = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "check-ignore", "-q", str(resolved)],
        capture_output=True,
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError(f"output path is not ignored by Git: {resolved}")


def _ensure_empty_output_directory(destination: Path) -> None:
    if destination.is_symlink():
        raise ValueError(f"output path must not be a symlink: {destination}")
    if not destination.exists():
        return
    if not destination.is_dir():
        raise ValueError(f"output path must be a real directory: {destination}")
    if any(destination.iterdir()):
        raise ValueError(f"output directory is not empty; refusing to overwrite: {destination}")


def _write_report(
    destination: Path,
    report: dict[str, Any],
    *,
    stem: str = "baseline-audit",
    title: str = "Official baseline audit",
) -> None:
    json_path = f"{stem}.json"
    markdown_path = f"{stem}.md"
    rows = []
    experiments = report.get("experiments", [])
    if experiments:
        for experiment in experiments:
            values = experiment.get("metrics", {})
            rows.append(
                f"| {experiment['experiment_id']} | {experiment['name']} "
                f"| {values.get('spell_f1_mean', float('nan')):.4f} "
                f"| {values.get('macro_f1', float('nan')):.4f} "
                f"| {values.get('graded_frames', '—')} "
                f"| {values.get('ground_truth_spells', '—')} |"
            )
    else:
        for run in report["runs"]:
            for name, values in sorted(run.get("metrics", {}).items()):
                rows.append(
                    f"| {run['run_id']} | {name} | {values['spell_f1_mean']:.4f} "
                    f"| {values['macro_f1']:.4f} | {values['graded_frames']} "
                    f"| {values['ground_truth_spells']} |"
                )
    table = "\n".join(rows) if rows else "| — | — | — | — | — | — |"
    status = report["status"]
    if "runs" in report:
        determinism = report.get("determinism", {}).get("prediction_files_byte_identical")
        introduction = (
            "This audit runs the official baseline generator twice and scores each run with the "
            "unchanged official evaluator on the public ground truth. The public window contains "
            "7,500 graded frames in one contiguous five-minute interval and 17 possession spells. "
            "These metrics are for that window only; they do not validate full-match accuracy or "
            "generalization."
        )
        table_header = (
            "| Run | Submission | spellF1 mean | macro-F1 | Graded frames | GT spells |\n"
            "|---|---|---:|---:|---:|---:|"
        )
        reproducibility = f"\n\nPrediction files byte-identical across runs: **{determinism}**"
    else:
        introduction = (
            "This record scores one candidate prediction with the unchanged official evaluator "
            "using the authorized public ground truth. Public-window metrics do not validate "
            "full-match accuracy or generalization."
        )
        table_header = (
            "| Experiment ID | Name | spellF1 mean | macro-F1 | Graded frames | GT spells |\n"
            "|---|---|---:|---:|---:|---:|"
        )
        reproducibility = ""
    body = f"""# {title}

Status: **{status}**

{introduction}{reproducibility}

{table_header}
{table}

Machine-readable metadata and checksums are in `{json_path}`. Prediction files and
evaluator reports remain local under this output directory and must not be committed or shared.
"""
    json_content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination.mkdir(parents=True, exist_ok=True)
    temporary_files: list[tuple[Path, Path]] = []
    try:
        for filename, content in (
            (markdown_path, body),
            (json_path, json_content),
        ):
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination,
                prefix=f".{filename}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(content)
                temporary_files.append((Path(handle.name), destination / filename))
        for temporary, final in temporary_files:
            temporary.replace(final)
    except OSError:
        for temporary, _ in temporary_files:
            temporary.unlink(missing_ok=True)
        raise


def run_baseline_audit(
    repository_root: Path | str,
    data_root: Path | str,
    output_dir: Path | str,
    *,
    bootstrap: int = 1000,
    smoothing_half_widths: tuple[int, ...] | None = None,
    sensitivity_question: str | None = None,
) -> dict[str, Any]:
    """Run official baseline generation and public evaluation twice, recording local provenance."""
    if bootstrap < 0:
        raise ValueError("bootstrap draw count cannot be negative")
    root = Path(repository_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    destination = Path(output_dir).expanduser()
    ensure_local_output_path(destination)
    _ensure_empty_output_directory(destination)
    destination = destination.resolve()
    if not (data / "tactical").is_dir():
        raise FileNotFoundError(f"local data root must contain tactical/: {data}")
    input_paths = (
        data / "tactical" / "BAR-ATH_tactical_atd.xml",
        data / "tactical" / "BAR-ATH_tactical_atd.txt",
    )
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"official baseline input not found: {path}")
    baseline_script = root / "starter" / "baseline.py"
    evaluator_script = root / "evaluation" / "run_evaluation.py"
    metrics_script = root / "evaluation" / "possession_metrics.py"
    for script in (baseline_script, evaluator_script, metrics_script):
        if not script.is_file():
            raise FileNotFoundError(f"required official source not found: {script}")
    baseline_dependencies = (root / "games.py",)
    for dependency in baseline_dependencies:
        if not dependency.is_file():
            raise FileNotFoundError(
                f"required official baseline dependency not found: {dependency}"
            )
    variants = _official_variants(baseline_script, smoothing_half_widths)
    _, scope = _public_ground_truth(root)
    baseline_provenance = _provenance(
        root,
        baseline_script,
        evaluator_script,
        metrics_script,
        baseline_dependencies,
        input_paths,
        scope,
        variants,
    )
    baseline_command = [sys.executable, str(baseline_script), "--out", "<RUN_PREDICTIONS>"]
    if smoothing_half_widths is not None:
        baseline_command.extend(["--window", *(str(width) for width in smoothing_half_widths)])
    report: dict[str, Any] = {
        "status": "incomplete",
        "scope": "public",
        "evaluation_scope": scope,
        "metadata": {
            **baseline_provenance,
            "configuration": variants,
            "bootstrap_draws": bootstrap,
            "commands": [
                baseline_command,
                [
                    sys.executable,
                    str(evaluator_script),
                    "--gt",
                    "public",
                    "--results",
                    "<RUN_PREDICTIONS>",
                    "--out",
                    "<RUN_EVALUATION>",
                    "--bootstrap",
                    str(bootstrap),
                ],
            ],
        },
        "runs": [],
        "experiments": [],
    }
    if sensitivity_question is not None:
        if smoothing_half_widths is None or not sensitivity_question.strip():
            raise ValueError("sensitivity questions require explicit smoothing half-widths")
        report["metadata"]["smoothing_sensitivity"] = {
            "question": sensitivity_question,
            "candidate_half_widths": [
                variant["smoothing_half_width_frames"] for variant in variants
            ],
            "changes_production_default": False,
        }
    try:
        for index in (1, 2):
            run_id = f"run-{index}"
            run_dir = destination / run_id
            predictions = run_dir / "predictions"
            evaluation = run_dir / "evaluation"
            started = time.perf_counter()
            if smoothing_half_widths is None:
                run_official_baseline(root, data, predictions)
            else:
                run_official_baseline(
                    root,
                    data,
                    predictions,
                    smoothing_half_widths=smoothing_half_widths,
                )
            after_generation = _provenance(
                root,
                baseline_script,
                evaluator_script,
                metrics_script,
                baseline_dependencies,
                input_paths,
                scope,
                variants,
            )
            if after_generation != baseline_provenance or _public_ground_truth(root)[1] != scope:
                raise ValueError("baseline inputs or source provenance changed during generation")
            files = sorted(predictions.glob("*.csv"))
            if {path.stem for path in files} != {variant["name"] for variant in variants}:
                raise ValueError(
                    f"official baseline output variants do not match declared variants: "
                    f"{[path.name for path in files]}"
                )
            checksums = {path.stem: sha256_file(path) for path in files}
            run_official_evaluator(root, predictions, evaluation, "public", bootstrap)
            after_evaluation = _provenance(
                root,
                baseline_script,
                evaluator_script,
                metrics_script,
                baseline_dependencies,
                input_paths,
                scope,
                variants,
            )
            if after_evaluation != baseline_provenance or _public_ground_truth(root)[1] != scope:
                raise ValueError("evaluation inputs or source provenance changed during scoring")
            metrics = _scored_metrics(evaluation)
            if set(metrics) != set(checksums):
                raise ValueError("official evaluator reports do not match generated predictions")
            if any(
                values["graded_frames"] != scope["graded_frames"]
                or values["ground_truth_spells"] != scope["ground_truth_spells"]
                for values in metrics.values()
            ):
                raise ValueError(
                    "official evaluator metrics do not match the recorded public scope"
                )
            report["runs"].append(
                {
                    "run_id": run_id,
                    "status": "complete",
                    "provenance": baseline_provenance,
                    "metrics": metrics,
                    "prediction_sha256": checksums,
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
        first, second = report["runs"]
        report["determinism"] = {
            "prediction_files_byte_identical": (
                first["prediction_sha256"] == second["prediction_sha256"]
            ),
            "metrics_identical": first["metrics"] == second["metrics"],
        }
        if not report["determinism"]["prediction_files_byte_identical"]:
            report["status"] = "non_reproducible"
        elif not report["determinism"]["metrics_identical"]:
            report["status"] = "non_reproducible"
        for variant in variants:
            name = variant["name"]
            configuration = {
                "prediction_source": "unchanged official starter/baseline.py",
                **variant,
            }
            configuration_fingerprint = hashlib.sha256(
                json.dumps(
                    configuration,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            report["experiments"].append(
                {
                    "experiment_id": _stable_id(
                        "official-baseline",
                        report["metadata"]["official_baseline_sha256"],
                        "public",
                        configuration,
                    ),
                    "name": name,
                    "description": "Official nearest-player baseline with its published smoothing.",
                    "status": "complete",
                    "scope": "public",
                    "configuration": configuration,
                    "configuration_fingerprint": configuration_fingerprint,
                    "metrics": first["metrics"][name],
                    "prediction_sha256": first["prediction_sha256"][name],
                    "repeated_prediction_sha256": second["prediction_sha256"][name],
                    "prediction_reproducible": (
                        first["prediction_sha256"][name] == second["prediction_sha256"][name]
                    ),
                }
            )
        if report["status"] != "non_reproducible":
            report["status"] = "complete"
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        KeyError,
    ) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        _write_report(destination, report)
        raise
    _write_report(destination, report)
    return report


def run_smoothing_sensitivity_audit(
    repository_root: Path | str,
    data_root: Path | str,
    output_dir: Path | str,
    *,
    bootstrap: int = 1000,
) -> dict[str, Any]:
    """Run the prespecified intermediate-width sensitivity check without changing defaults."""
    return run_baseline_audit(
        repository_root,
        data_root,
        output_dir,
        bootstrap=bootstrap,
        smoothing_half_widths=SMOOTHING_SENSITIVITY_WIDTHS,
        sensitivity_question=(
            "Check whether the observed public-window spellF1 increase from half-width 50 to 125 "
            "also appears at the two prespecified intermediate widths 75 and 100; descriptive "
            "only, not parameter selection."
        ),
    )


def record_prediction_experiment(
    repository_root: Path | str,
    prediction: Path | str,
    output_dir: Path | str,
    *,
    experiment_id: str,
    description: str,
    generation_command: str,
    config_path: Path | str | None = None,
    source_inputs: tuple[Path | str, ...] = (),
    ground_truth: str = "public",
    bootstrap: int = 1000,
) -> dict[str, Any]:
    """Score one locally generated prediction with the official evaluator and record provenance."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", experiment_id):
        raise ValueError(
            "experiment ID must be 1-80 safe alphanumeric, dot, underscore, or dash characters"
        )
    if not description.strip() or not generation_command.strip():
        raise ValueError("experiment description and generation command are required")
    if ground_truth.strip().casefold() != "public":
        raise ValueError("only the authorized public ground truth may be recorded")
    if bootstrap < 0:
        raise ValueError("bootstrap draw count cannot be negative")
    source_prediction = Path(prediction).expanduser().resolve()
    if not source_prediction.is_file():
        raise FileNotFoundError(f"prediction file not found: {source_prediction}")
    root = Path(repository_root).expanduser().resolve()
    destination = Path(output_dir).expanduser()
    ensure_local_output_path(destination)
    _ensure_empty_output_directory(destination)
    destination = destination.resolve()
    evaluator = root / "evaluation" / "run_evaluation.py"
    metrics_source = root / "evaluation" / "possession_metrics.py"
    if not evaluator.is_file() or not metrics_source.is_file():
        raise FileNotFoundError(f"official evaluator sources not found under {root}")
    _, scope = _public_ground_truth(root)
    inputs = tuple(Path(path).expanduser().resolve() for path in source_inputs)
    if len({path.name for path in inputs}) != len(inputs):
        raise ValueError("source input filenames must be unique for checksum recording")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"source input not found: {path}")
    resolved_config_path = (
        Path(config_path).expanduser().resolve() if config_path is not None else None
    )
    config = load_config(resolved_config_path) if resolved_config_path else Config()
    prediction_hash = sha256_file(source_prediction)
    config_hash = sha256_file(resolved_config_path) if resolved_config_path else None
    config_fingerprint = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    initial_source_hashes = {path.name: sha256_file(path) for path in inputs}
    initial_evaluator_hash = sha256_file(evaluator)
    initial_metrics_hash = sha256_file(metrics_source)
    initial_ground_truth_hash = scope["ground_truth_sha256"]
    report: dict[str, Any] = {
        "status": "incomplete",
        "scope": "public",
        "evaluation_scope": scope,
        "metadata": {
            "project_revision": git_revision(PROJECT_ROOT),
            "project_worktree_dirty": git_worktree_dirty(PROJECT_ROOT),
            "official_repository_revision": git_revision(root),
            "official_worktree_dirty": git_worktree_dirty(root),
            "official_evaluator_sha256": initial_evaluator_hash,
            "official_metrics_sha256": initial_metrics_hash,
            "input_sha256": initial_source_hashes,
            "generation_command": generation_command,
            "scoring_command": [
                sys.executable,
                str(evaluator),
                "--gt",
                "public",
                "--results",
                "<EXPERIMENT_PREDICTION_DIR>",
                "--out",
                "<EXPERIMENT_EVALUATION_DIR>",
                "--bootstrap",
                str(bootstrap),
            ],
            "configuration": asdict(config),
            "configuration_sha256": config_hash,
            "configuration_fingerprint": config_fingerprint,
            "prediction_sha256": prediction_hash,
            "bootstrap_draws": bootstrap,
        },
        "experiments": [],
    }
    destination.mkdir(parents=True, exist_ok=True)
    try:
        predictions_dir = destination / "predictions"
        predictions_dir.mkdir()
        staged_prediction = predictions_dir / f"{experiment_id}.csv"
        shutil.copyfile(source_prediction, staged_prediction)
        evaluation_dir = destination / "evaluation"
        run_official_evaluator(root, predictions_dir, evaluation_dir, "public", bootstrap)
        if (
            sha256_file(source_prediction) != prediction_hash
            or {path.name: sha256_file(path) for path in inputs} != initial_source_hashes
            or sha256_file(evaluator) != initial_evaluator_hash
            or sha256_file(metrics_source) != initial_metrics_hash
            or _public_ground_truth(root)[1]["ground_truth_sha256"] != initial_ground_truth_hash
            or (sha256_file(resolved_config_path) if resolved_config_path is not None else None)
            != config_hash
        ):
            raise ValueError("experiment inputs or scoring provenance changed during evaluation")
        metrics = _scoreboard_metrics(evaluation_dir / "scoreboard.csv")
        if set(metrics) != {experiment_id}:
            raise ValueError(
                "official evaluator did not produce exactly one scored prediction; "
                f"scored={sorted(metrics)}"
            )
        measured = metrics[experiment_id]
        if (
            measured["graded_frames"] != scope["graded_frames"]
            or measured["ground_truth_spells"] != scope["ground_truth_spells"]
        ):
            raise ValueError("official evaluator metrics do not match the recorded public scope")
        report["evaluation_scope"].update(
            {
                "graded_frames": measured["graded_frames"],
                "ground_truth_spells": measured["ground_truth_spells"],
                "description": "official public ground truth; contiguous published window only",
            }
        )
        report["experiments"].append(
            {
                "experiment_id": experiment_id,
                "name": experiment_id,
                "description": description,
                "status": "complete",
                "scope": "public",
                "configuration": asdict(config),
                "configuration_fingerprint": config_fingerprint,
                "metrics": measured,
                "prediction_sha256": prediction_hash,
                "source_sha256": initial_source_hashes,
            }
        )
        report["status"] = "complete"
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        KeyError,
    ) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        _write_report(destination, report, stem="experiment-report", title="Experiment record")
        raise
    _write_report(destination, report, stem="experiment-report", title="Experiment record")
    return report


def compare_experiment_reports(report_paths: list[Path | str]) -> tuple[list[dict[str, Any]], bool]:
    """Collect valid experiment records and report missing or incompatible provenance."""
    if len(report_paths) < 2:
        raise ValueError("at least two experiment reports are required for comparison")
    loaded: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for source in report_paths:
        path = Path(source).expanduser().resolve()
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read experiment report {path}: {exc}") from exc
        if not isinstance(report, dict) or report.get("status") != "complete":
            raise ValueError(f"experiment report is not complete: {path}")
        metadata = report.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"experiment report has no metadata object: {path}")
        scope = report.get("evaluation_scope")
        if not isinstance(scope, dict):
            scope = metadata.get("evaluation_scope")
        if not isinstance(scope, dict):
            raise ValueError(f"experiment report has no evaluation-scope object: {path}")
        rows = report.get("experiments")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"no experiment records found in {path}")
        loaded.append((report, {"scope": scope, **metadata}, path))

    report_ids: set[str] = set()
    experiments: list[dict[str, Any]] = []
    warning = False
    protocol_signatures: list[tuple[Any, ...]] = []
    for report, provenance, path in loaded:
        report_warnings = []
        scope = provenance["scope"]
        evaluator_hash = provenance.get("official_evaluator_sha256")
        metrics_hash = provenance.get("official_metrics_sha256")
        ground_truth_hash = scope.get("ground_truth_sha256")
        if not isinstance(evaluator_hash, str) or not _SHA256_PATTERN.fullmatch(evaluator_hash):
            report_warnings.append("missing or invalid official evaluator SHA-256")
        if not isinstance(metrics_hash, str) or not _SHA256_PATTERN.fullmatch(metrics_hash):
            report_warnings.append("missing or invalid official metric SHA-256")
        if not isinstance(ground_truth_hash, str) or not _SHA256_PATTERN.fullmatch(
            ground_truth_hash
        ):
            report_warnings.append("missing or invalid public ground-truth SHA-256")
        inputs = provenance.get("input_sha256")
        if not isinstance(inputs, dict) or not inputs:
            report_warnings.append("source-input checksums are missing")
            inputs = {}
        elif any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not _SHA256_PATTERN.fullmatch(value)
            for name, value in inputs.items()
        ):
            report_warnings.append("source-input checksums are invalid")
            inputs = {}
        if (
            isinstance(scope.get("graded_frames"), bool)
            or not isinstance(scope.get("graded_frames"), int)
            or isinstance(scope.get("ground_truth_spells"), bool)
            or not isinstance(scope.get("ground_truth_spells"), int)
        ):
            report_warnings.append("evaluation-window counts are missing or invalid")
        public_window = scope.get("public_window")
        if not isinstance(public_window, dict):
            report_warnings.append("evaluation-window definition is missing or invalid")
            public_window = {}
        project_revision = provenance.get("project_revision")
        if not isinstance(project_revision, str) or not project_revision:
            report_warnings.append("project source revision is invalid")
            project_revision = None
        official_revision = provenance.get("official_repository_revision")
        if not isinstance(official_revision, str) or not official_revision:
            report_warnings.append("official repository revision is missing or invalid")
            official_revision = None
        baseline_dependencies = provenance.get("official_baseline_dependencies_sha256")
        if baseline_dependencies is not None and (
            not isinstance(baseline_dependencies, dict)
            or any(
                not isinstance(name, str)
                or not isinstance(value, str)
                or not _SHA256_PATTERN.fullmatch(value)
                for name, value in baseline_dependencies.items()
            )
        ):
            report_warnings.append("official baseline dependency checksums are invalid")
        protocol_signatures.append(
            (
                report.get("scope"),
                official_revision,
                ground_truth_hash,
                scope.get("ground_truth_metadata_sha256"),
                scope.get("graded_frames"),
                scope.get("ground_truth_spells"),
                json.dumps(public_window, sort_keys=True, allow_nan=False),
                evaluator_hash,
                metrics_hash,
                tuple(sorted(inputs.items())),
            )
        )
        rows = report["experiments"]
        for experiment in rows:
            if not isinstance(experiment, dict):
                raise ValueError(f"invalid experiment record in {path}: expected an object")
            experiment_id = experiment.get("experiment_id")
            if not isinstance(experiment_id, str) or not experiment_id:
                raise ValueError(f"experiment record has an invalid identifier in {path}")
            if experiment_id in report_ids:
                raise ValueError(f"duplicate experiment identifier across reports: {experiment_id}")
            report_ids.add(experiment_id)
            if experiment.get("status") != "complete":
                raise ValueError(f"experiment record is not complete: {experiment_id}")
            metrics = experiment.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"experiment record has no metrics object: {experiment_id}")
            _validate_metrics(metrics, f"experiment record {experiment_id}")
            if metrics["graded_frames"] != scope.get("graded_frames") or metrics[
                "ground_truth_spells"
            ] != scope.get("ground_truth_spells"):
                report_warnings.append(
                    f"{experiment_id}: metrics do not match the report evaluation window"
                )
            prediction_hash = experiment.get("prediction_sha256")
            if not isinstance(prediction_hash, str) or not _SHA256_PATTERN.fullmatch(
                prediction_hash
            ):
                raise ValueError(
                    f"experiment record has an invalid prediction checksum: {experiment_id}"
                )
            if experiment.get("scope", report.get("scope")) != report.get("scope"):
                report_warnings.append(f"{experiment_id}: record scope differs from report scope")
            configuration = experiment.get("configuration")
            if not isinstance(configuration, dict):
                report_warnings.append(f"{experiment_id}: configuration is missing")
                configuration = {}
            try:
                canonical_configuration = json.dumps(
                    configuration,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"experiment record has invalid configuration: {experiment_id}"
                ) from exc
            experiment_inputs = experiment.get("source_sha256")
            if experiment_inputs is not None and experiment_inputs != inputs:
                report_warnings.append(f"{experiment_id}: record input checksums differ")
            configuration_fingerprint = experiment.get("configuration_fingerprint")
            expected_fingerprint = hashlib.sha256(canonical_configuration.encode()).hexdigest()
            if configuration_fingerprint is not None and (
                not isinstance(configuration_fingerprint, str)
                or not _SHA256_PATTERN.fullmatch(configuration_fingerprint)
                or configuration_fingerprint != expected_fingerprint
            ):
                report_warnings.append(
                    f"{experiment_id}: configuration fingerprint does not match configuration"
                )
                configuration_fingerprint = None
            elif configuration_fingerprint is None:
                report_warnings.append(f"{experiment_id}: configuration fingerprint is missing")
            experiments.append(
                {
                    **experiment,
                    "scope": report.get("scope"),
                    "project_revision": project_revision,
                    "configuration_fingerprint": configuration_fingerprint,
                    "configuration": configuration,
                    "canonical_configuration": canonical_configuration,
                    "comparison_warnings": report_warnings,
                }
            )
        if report.get("scope") is None:
            report_warnings.append("evaluation scope is missing")
        if project_revision is None:
            report_warnings.append("project source revision is missing")
        warning = warning or bool(report_warnings)

    if len(set(protocol_signatures)) != 1:
        warning = True
        for experiment in experiments:
            experiment["comparison_warnings"].append(
                "reports differ in evaluation protocol, evaluator, ground truth, or source inputs"
            )
    revisions = {provenance.get("project_revision") for _, provenance, _ in loaded}
    official_revisions = {
        provenance.get("official_repository_revision") for _, provenance, _ in loaded
    }
    if len(revisions) > 1:
        warning = True
        for experiment in experiments:
            experiment["comparison_warnings"].append("project source revisions differ")
    if len(official_revisions) > 1:
        warning = True
        for experiment in experiments:
            experiment["comparison_warnings"].append("official repository revisions differ")
    configurations = {
        experiment.get("configuration_fingerprint") or experiment["canonical_configuration"]
        for experiment in experiments
    }
    if len(configurations) > 1:
        warning = True
        for experiment in experiments:
            experiment["comparison_warnings"].append(
                "recorded experiment configurations differ; this is an intended factor only "
                "when declared by the experiment protocol"
            )
    experiments.sort(key=lambda experiment: experiment["experiment_id"])
    return experiments, warning
