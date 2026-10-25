"""Aggregate-only diagnostics over official public-window baseline reports."""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

from possession.evaluation.reproducibility import (
    _ensure_empty_output_directory,
    _public_ground_truth,
    _scored_metrics,
    ensure_local_output_path,
    sha256_file,
)

_LABELS = {"H", "A", "D", "X"}
_PROVENANCE_FIELDS = (
    "project_revision",
    "official_repository_revision",
    "official_baseline_sha256",
    "official_evaluator_sha256",
    "official_metrics_sha256",
    "official_baseline_dependencies_sha256",
    "input_sha256",
    "evaluation_scope",
)


def _read_labels(path: Path, expected_frames: int) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {str(field).strip().lower() for field in (reader.fieldnames or [])}
        if fields != {"frame", "label"}:
            raise ValueError(f"expected frame,label columns in {path}")
        labels: list[str] = []
        for index, row in enumerate(reader):
            try:
                frame = int(row["frame"])
                label = row["label"].strip().upper()
            except (KeyError, AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid prediction row {index + 2} in {path}") from exc
            if frame != index:
                raise ValueError(f"frame indices are not contiguous in {path}")
            if label not in _LABELS:
                raise ValueError(f"invalid label in {path} at row {index + 2}")
            labels.append(label)
    if len(labels) != expected_frames:
        raise ValueError(f"frame count mismatch in {path}: {len(labels)} != {expected_frames}")
    return labels


def _load_official_metrics(repository_root: Path, expected_sha256: str):
    metrics_path = repository_root / "evaluation" / "possession_metrics.py"
    if not metrics_path.is_file() or sha256_file(metrics_path) != expected_sha256:
        raise ValueError("official metric source does not match the recorded baseline audit")
    spec = importlib.util.spec_from_file_location(
        "_possession_official_metrics_for_error_analysis",
        metrics_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official metric source: {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _saved_metrics_match(saved: dict[str, Any], measured: dict[str, Any]) -> bool:
    for key, value in saved.items():
        if key != "transitions":
            if measured.get(key) != value:
                return False
            continue
        if not isinstance(value, dict):
            return False
        for tau, transition in value.items():
            official_transition = measured.get("transitions", {}).get(tau)
            if not isinstance(transition, dict) or not isinstance(official_transition, dict):
                return False
            if any(official_transition.get(name) != metric for name, metric in transition.items()):
                return False
    return True


def analyze_baseline_audit(
    repository_root: Path | str,
    audit_directory: Path | str,
    output_directory: Path | str,
    *,
    run_id: str = "run-1",
) -> dict[str, Any]:
    """Write local aggregate error diagnostics using the unchanged official matching helpers."""
    root = Path(repository_root).expanduser().resolve()
    source = Path(audit_directory).expanduser().resolve()
    destination = Path(output_directory).expanduser()
    ensure_local_output_path(destination)
    _ensure_empty_output_directory(destination)
    destination = destination.resolve()

    audit_path = source / "baseline-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or audit.get("status") != "complete":
        raise ValueError(f"baseline audit is not complete: {audit_path}")
    if audit.get("scope") != "public":
        raise ValueError("error analysis accepts only the authorized public evaluation scope")
    metadata = audit.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"baseline audit has no provenance metadata: {audit_path}")
    evaluator_path = root / "evaluation" / "run_evaluation.py"
    baseline_path = root / "starter" / "baseline.py"
    if (
        not evaluator_path.is_file()
        or not baseline_path.is_file()
        or sha256_file(evaluator_path) != metadata.get("official_evaluator_sha256")
        or sha256_file(baseline_path) != metadata.get("official_baseline_sha256")
    ):
        raise ValueError("official scorer or baseline source differs from the recorded audit")

    ground_truth_path, current_scope = _public_ground_truth(root)
    recorded_scope = audit.get("evaluation_scope")
    if current_scope != recorded_scope:
        raise ValueError("public ground-truth scope or checksum differs from the recorded audit")
    if run_id not in {"run-1", "run-2"}:
        raise ValueError("run ID must be run-1 or run-2")
    run_records = audit.get("runs")
    if not isinstance(run_records, list):
        raise ValueError("baseline audit has no run records")
    runs: dict[str, dict[str, Any]] = {}
    for run in run_records:
        if not isinstance(run, dict) or not isinstance(run.get("run_id"), str):
            raise ValueError("baseline audit contains an invalid run record")
        if run["run_id"] in runs:
            raise ValueError(f"baseline audit contains duplicate run ID {run['run_id']}")
        runs[run["run_id"]] = run
    selected_run = runs.get(run_id)
    if not isinstance(selected_run, dict) or selected_run.get("status") != "complete":
        raise ValueError(f"baseline audit does not contain a complete {run_id}")
    for run in runs.values():
        provenance = run.get("provenance", {})
        if not isinstance(provenance, dict) or any(
            provenance.get(field) != metadata.get(field) for field in _PROVENANCE_FIELDS
        ):
            raise ValueError("baseline runs have inconsistent recorded provenance")

    metrics_module = _load_official_metrics(
        root,
        str(metadata.get("official_metrics_sha256", "")),
    )
    ground_truth_frame = metrics_module.read_label_csv(ground_truth_path)
    ground_truth_codes = metrics_module.label_codes(ground_truth_frame)
    if len(ground_truth_codes) != current_scope["video_frames"]:
        raise ValueError("public ground-truth frame count differs from its metadata")
    valid_runs = metrics_module.valid_runs(ground_truth_codes)
    gt_spells = metrics_module.possession_spells(ground_truth_codes, valid_runs)
    if len(gt_spells) != current_scope["ground_truth_spells"]:
        raise ValueError("official spell extraction does not match the recorded scope")

    configured_variants = metadata.get("configuration")
    if not isinstance(configured_variants, list) or not configured_variants:
        raise ValueError("baseline audit has no variant configuration")
    expected_names = set()
    for variant in configured_variants:
        if (
            not isinstance(variant, dict)
            or not isinstance(variant.get("name"), str)
            or not variant["name"]
            or isinstance(variant.get("smoothing_half_width_frames"), bool)
            or not isinstance(variant.get("smoothing_half_width_frames"), int)
            or variant["smoothing_half_width_frames"] < 0
        ):
            raise ValueError("baseline audit contains an invalid variant configuration")
        if variant["name"] in expected_names:
            raise ValueError(f"baseline audit contains duplicate variant {variant['name']}")
        expected_names.add(variant["name"])
    predictions_directory = source / run_id / "predictions"
    prediction_paths = sorted(predictions_directory.glob("*.csv"))
    if {path.stem for path in prediction_paths} != expected_names:
        raise ValueError("baseline prediction files do not match the recorded variants")
    prediction_hashes = selected_run.get("prediction_sha256")
    if not isinstance(prediction_hashes, dict):
        raise ValueError("selected baseline run has no prediction checksums")

    recorded_metrics = selected_run.get("metrics")
    measured_metrics = _scored_metrics(source / run_id / "evaluation")
    if (
        not isinstance(recorded_metrics, dict)
        or set(recorded_metrics) != expected_names
        or set(measured_metrics) != expected_names
    ):
        raise ValueError("baseline metrics do not match the recorded variants")

    gt_labels = _read_labels(ground_truth_path, current_scope["video_frames"])
    graded_indexes = [index for index, label in enumerate(gt_labels) if label != "X"]
    if len(graded_indexes) != current_scope["graded_frames"]:
        raise ValueError("graded public frame count differs from the recorded scope")

    durations = [stop - start for start, stop, _ in gt_spells]
    median_spell_frames = statistics.median(durations)
    short_spell_indexes = {
        index for index, duration in enumerate(durations) if duration <= median_spell_frames
    }
    long_spell_indexes = set(range(len(durations))) - short_spell_indexes
    threshold = metrics_module.SPELL_IOU_HEADLINE

    variant_results: dict[str, dict[str, Any]] = {}
    prediction_labels: dict[str, list[str]] = {}
    for variant in configured_variants:
        name = variant["name"]
        prediction_path = predictions_directory / f"{name}.csv"
        if sha256_file(prediction_path) != prediction_hashes.get(name):
            raise ValueError(f"prediction checksum differs from baseline audit: {name}")
        labels = _read_labels(prediction_path, current_scope["video_frames"])
        prediction_labels[name] = labels
        official_metrics = measured_metrics[name]
        saved_metrics = recorded_metrics[name]
        required_metrics = {
            "macro_f1",
            "spell_f1_mean",
            "graded_frames",
            "ground_truth_spells",
            "predicted_spells",
            "transitions",
        }
        if not required_metrics.issubset(saved_metrics) or not _saved_metrics_match(
            saved_metrics,
            official_metrics,
        ):
            raise ValueError(f"stored and official evaluator metrics differ: {name}")

        prediction_codes = metrics_module.label_codes(labels)
        filled_codes = metrics_module.fill_abstentions(prediction_codes, valid_runs)
        predicted_spells = metrics_module.possession_spells(filled_codes, valid_runs)
        overlaps = metrics_module.spell_overlaps(gt_spells, predicted_spells)
        matched_flags = metrics_module.matched_gt_flags(
            overlaps,
            len(gt_spells),
            threshold,
        )
        calculated_spells = metrics_module.spell_scores(
            ground_truth_codes,
            filled_codes,
            valid_runs,
        )
        if calculated_spells["mean_f1"] != official_metrics["spell_f1_mean"]:
            raise ValueError(f"official spell metric does not reconcile with report: {name}")
        short_missed = sum(not matched_flags[index] for index in short_spell_indexes)
        long_missed = sum(not matched_flags[index] for index in long_spell_indexes)
        spell_metrics = {
            str(iou): {
                "false_positive_spells": values["n_spurious"],
                "missed_ground_truth_spells": values["n_missed"],
                "matched_spells": values["n_matched"],
                "f1": values["f1"],
            }
            for iou, values in sorted(calculated_spells["per_iou"].items())
        }
        transition_metrics = {
            str(tau): {
                "matched_boundaries": values["matched"],
                "missed_boundaries": values["missed"],
                "spurious_boundaries": values["spurious"],
                "f1": values["f1"],
                "mean_latency_frames": values["mean_latency"],
                "mean_absolute_latency_frames": values["mean_abs_latency"],
            }
            for tau, values in sorted(
                official_metrics["transitions"].items(), key=lambda item: int(item[0])
            )
        }
        variant_results[name] = {
            "smoothing_half_width_frames": variant["smoothing_half_width_frames"],
            "prediction_sha256": prediction_hashes[name],
            "spell_f1_mean": official_metrics["spell_f1_mean"],
            "macro_f1": official_metrics["macro_f1"],
            "spell_errors_by_iou": spell_metrics,
            "transition_errors": transition_metrics,
            "spell_counts": {
                "ground_truth": official_metrics["ground_truth_spells"],
                "predicted": official_metrics["predicted_spells"],
                "count_difference": (
                    official_metrics["predicted_spells"] - official_metrics["ground_truth_spells"]
                ),
            },
            "segment_fragmentation": {
                "ground_truth_segments": official_metrics.get("n_gt_segments"),
                "predicted_segments": official_metrics.get("n_pred_segments"),
                "official_ratio": official_metrics.get("fragmentation"),
            },
            "headline_iou_missed_spells_by_duration": {
                "definition": "short <= median ground-truth spell duration; long > median",
                "median_ground_truth_spell_frames": median_spell_frames,
                "short": {
                    "sample_size": len(short_spell_indexes),
                    "missed": int(short_missed),
                },
                "long": {
                    "sample_size": len(long_spell_indexes),
                    "missed": int(long_missed),
                },
                "iou_threshold": threshold,
            },
            "bootstrap": {
                "draws": official_metrics.get("bootstrap_n"),
                "spell_f1_low": official_metrics.get("spell_bootstrap_low"),
                "spell_f1_high": official_metrics.get("spell_bootstrap_high"),
                "method": (
                    "official evaluator resamples ground-truth possession spells with replacement; "
                    "it holds predicted-spell count fixed"
                ),
            },
        }

    pairwise_differences = []
    names = sorted(prediction_labels)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            differing = sum(
                prediction_labels[left][index] != prediction_labels[right][index]
                for index in graded_indexes
            )
            pairwise_differences.append(
                {
                    "left": left,
                    "right": right,
                    "different_predictions_on_graded_frames": differing,
                    "graded_frames": len(graded_indexes),
                    "fraction_different": differing / len(graded_indexes),
                }
            )

    result = {
        "report_type": "official_public_baseline_error_analysis",
        "status": "complete",
        "run_id": run_id,
        "reproducibility": audit.get("determinism"),
        "evaluation_scope": current_scope,
        "ranking_rule": {
            "primary": "spell_f1_mean, descending",
            "tie_breaks": [
                "macro_f1, descending",
                "transition F1 at the loosest configured tolerance, descending",
                "mean absolute transition latency, ascending",
                "submission name, ascending",
            ],
            "source": "unchanged official evaluation/run_evaluation.py rank_key",
        },
        "official_source_revision": metadata["official_repository_revision"],
        "official_source_checksums": {
            key: metadata[key]
            for key in (
                "official_baseline_sha256",
                "official_evaluator_sha256",
                "official_metrics_sha256",
            )
        },
        "variants": variant_results,
        "pairwise_prediction_differences": pairwise_differences,
        "limitations": [
            "All metrics describe only this public window and its 17 possession spells.",
            "Repeatability on one contiguous interval is not statistical robustness "
            "or generalization.",
            "Missed-spell duration groups use a median split of this small sample "
            "and are descriptive.",
            "Aggregate differences do not identify why any individual error occurred.",
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    json_content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown_content = _markdown_report(result)
    temporary_files: list[tuple[Path, Path]] = []
    try:
        for filename, content in (
            ("error-analysis.json", json_content),
            ("error-analysis.md", markdown_content),
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
    return result


def _markdown_report(report: dict[str, Any]) -> str:
    rows = []
    for name, result in sorted(report["variants"].items()):
        spell = result["spell_errors_by_iou"]["0.5"]
        transitions = result["transition_errors"].get("25", {})
        rows.append(
            f"| {name} | {result['smoothing_half_width_frames']} | "
            f"{result['spell_f1_mean']:.4f} | {result['macro_f1']:.4f} | "
            f"{spell['false_positive_spells']} | {spell['missed_ground_truth_spells']} | "
            f"{transitions.get('missed_boundaries', 'n/a')} | "
            f"{transitions.get('spurious_boundaries', 'n/a')} |"
        )
    header = (
        "| Variant | Half-width (frames) | spellF1 | macro-F1 | "
        "FP spells (IoU 0.5) | Missed GT spells (IoU 0.5) | "
        "Missed boundaries (±25) | Spurious boundaries (±25) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    return f"""# Official public-window error analysis

Run: `{report["run_id"]}`

Scope: {report["evaluation_scope"]["graded_frames"]} graded frames, \
{report["evaluation_scope"]["ground_truth_spells"]} possession spells.

{header}
{chr(10).join(rows)}

See the JSON for official per-IoU and transition aggregates, median-split missed-spell counts,
pairwise prediction disagreements on graded frames, and bootstrap details. No frame-level
predictions or labels are included. These diagnostics describe this public window only and do not
establish full-match accuracy or generalization.
"""
