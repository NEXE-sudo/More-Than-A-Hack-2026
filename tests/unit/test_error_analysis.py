import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import possession.evaluation.error_analysis as error_analysis
from possession.evaluation.reproducibility import _public_ground_truth, _scored_metrics

OFFICIAL_METRICS = (
    Path(__file__).resolve().parents[3]
    / "more-than-a-hack-2027"
    / "evaluation"
    / "possession_metrics.py"
)


def _load_metrics_module():
    if not OFFICIAL_METRICS.is_file():
        pytest.skip("sibling official evaluator checkout is unavailable")
    spec = importlib.util.spec_from_file_location(
        "_official_metrics_for_error_analysis_test",
        OFFICIAL_METRICS,
    )
    if spec is None or spec.loader is None:
        pytest.skip("official metric module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_labels(path, labels):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame", "label"))
        writer.writerows(enumerate(labels))


def test_error_analysis_reuses_official_matching_and_exports_aggregates_only(
    tmp_path,
    monkeypatch,
):
    metrics_module = _load_metrics_module()
    official_root = tmp_path / "official"
    evaluation_dir = official_root / "evaluation"
    gt_dir = evaluation_dir / "gt"
    starter_dir = official_root / "starter"
    gt_dir.mkdir(parents=True)
    starter_dir.mkdir()
    evaluator_path = evaluation_dir / "run_evaluation.py"
    baseline_path = starter_dir / "baseline.py"
    evaluator_path.write_text("# synthetic test evaluator\n", encoding="utf-8")
    baseline_path.write_text("# synthetic test baseline\n", encoding="utf-8")

    ground_truth = ["H"] * 2 + ["D"] * 2 + ["A"] * 4 + ["X"] * 4
    gt_path = gt_dir / "fixture_possession_gt_public.csv"
    _write_labels(gt_path, ground_truth)
    sidecar = gt_dir / "fixture_possession_gt_public_meta.json"
    sidecar.write_text(
        json.dumps(
            {
                "n_graded_frames": 8,
                "n_possessions": 2,
                "n_video_frames": 12,
                "public_window": {
                    "first_video_frame": 0,
                    "last_video_frame": 11,
                    "n_frames": 12,
                },
            }
        ),
        encoding="utf-8",
    )
    _, scope = _public_ground_truth(official_root)

    names = ("baseline_raw", "baseline_smooth50")
    predictions = {
        "baseline_raw": ground_truth,
        "baseline_smooth50": ["H", "D", "D", "D", "D", "D", "A", "A", "A", "A", "X", "X"],
    }
    audit_dir = tmp_path / "audit"
    run = audit_dir / "run-1"
    predictions_dir = run / "predictions"
    reports_dir = run / "evaluation" / "reports"
    predictions_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    hashes = {}
    for name, labels in predictions.items():
        prediction_path = predictions_dir / f"{name}.csv"
        _write_labels(prediction_path, labels)
        hashes[name] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        result = metrics_module.evaluate(
            metrics_module.label_codes(ground_truth),
            metrics_module.label_codes(labels),
            bootstrap_n=0,
        )
        result["submission"] = name
        (reports_dir / f"{name}.json").write_text(
            json.dumps(result, default=float),
            encoding="utf-8",
        )
    run_metrics = _scored_metrics(run / "evaluation")
    metadata = {
        "project_revision": "local-revision",
        "official_repository_revision": "official-revision",
        "official_baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "official_evaluator_sha256": hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
        "official_metrics_sha256": hashlib.sha256(OFFICIAL_METRICS.read_bytes()).hexdigest(),
        "official_baseline_dependencies_sha256": {"games.py": "a" * 64},
        "input_sha256": {"tracking.txt": "b" * 64},
        "evaluation_scope": scope,
        "configuration": [
            {"name": names[0], "smoothing_half_width_frames": 0},
            {"name": names[1], "smoothing_half_width_frames": 50},
        ],
    }
    provenance_fields = (
        "project_revision",
        "official_repository_revision",
        "official_baseline_sha256",
        "official_evaluator_sha256",
        "official_metrics_sha256",
        "official_baseline_dependencies_sha256",
        "input_sha256",
        "evaluation_scope",
    )
    run_provenance = {field: metadata[field] for field in provenance_fields}
    (audit_dir / "baseline-audit.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "scope": "public",
                "evaluation_scope": scope,
                "metadata": metadata,
                "determinism": {
                    "prediction_files_byte_identical": True,
                    "metrics_identical": True,
                },
                "runs": [
                    {
                        "run_id": "run-1",
                        "status": "complete",
                        "provenance": run_provenance,
                        "metrics": run_metrics,
                        "prediction_sha256": hashes,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        error_analysis,
        "_load_official_metrics",
        lambda repository_root, expected_sha256: metrics_module,
    )

    output_dir = tmp_path / "analysis"
    report = error_analysis.analyze_baseline_audit(
        official_root,
        audit_dir,
        output_dir,
    )

    raw = report["variants"]["baseline_raw"]
    smoothed = report["variants"]["baseline_smooth50"]
    assert raw["spell_errors_by_iou"]["0.5"]["false_positive_spells"] == 0
    assert raw["spell_errors_by_iou"]["0.5"]["missed_ground_truth_spells"] == 0
    assert smoothed["spell_errors_by_iou"]["0.6"]["missed_ground_truth_spells"] == 2
    assert (
        report["pairwise_prediction_differences"][0]["different_predictions_on_graded_frames"] == 3
    )
    assert raw["headline_iou_missed_spells_by_duration"]["short"]["sample_size"] == 1
    assert (output_dir / "error-analysis.json").is_file()
    markdown = (output_dir / "error-analysis.md").read_text(encoding="utf-8")
    assert "No frame-level" in markdown
    assert "predictions or labels are included." in markdown


def test_error_analysis_rejects_changed_prediction_checksum(tmp_path, monkeypatch):
    metrics_module = _load_metrics_module()
    root = tmp_path / "official"
    (root / "starter").mkdir(parents=True)
    (root / "evaluation" / "gt").mkdir(parents=True)
    (root / "starter" / "baseline.py").write_text("baseline", encoding="utf-8")
    (root / "evaluation" / "run_evaluation.py").write_text("evaluator", encoding="utf-8")
    labels = ["H", "H", "X"]
    ground_truth = root / "evaluation" / "gt" / "fixture_possession_gt_public.csv"
    _write_labels(ground_truth, labels)
    (ground_truth.with_name("fixture_possession_gt_public_meta.json")).write_text(
        json.dumps(
            {
                "n_graded_frames": 2,
                "n_possessions": 1,
                "n_video_frames": 3,
                "public_window": {"first_video_frame": 0, "last_video_frame": 2},
            }
        ),
        encoding="utf-8",
    )
    _, scope = _public_ground_truth(root)
    audit_dir = tmp_path / "audit"
    run_dir = audit_dir / "run-1"
    predictions_dir = run_dir / "predictions"
    evaluation_dir = run_dir / "evaluation"
    (evaluation_dir / "reports").mkdir(parents=True)
    predictions_dir.mkdir()
    prediction = predictions_dir / "baseline_raw.csv"
    _write_labels(prediction, labels)
    result = metrics_module.evaluate(
        metrics_module.label_codes(labels),
        metrics_module.label_codes(labels),
        bootstrap_n=0,
    )
    result["submission"] = "baseline_raw"
    (evaluation_dir / "reports" / "baseline_raw.json").write_text(
        json.dumps(result, default=float),
        encoding="utf-8",
    )
    metadata = {
        "project_revision": "local",
        "official_repository_revision": "official",
        "official_baseline_sha256": hashlib.sha256(
            (root / "starter" / "baseline.py").read_bytes()
        ).hexdigest(),
        "official_evaluator_sha256": hashlib.sha256(
            (root / "evaluation" / "run_evaluation.py").read_bytes()
        ).hexdigest(),
        "official_metrics_sha256": hashlib.sha256(OFFICIAL_METRICS.read_bytes()).hexdigest(),
        "official_baseline_dependencies_sha256": {},
        "input_sha256": {"tracking.txt": "c" * 64},
        "evaluation_scope": scope,
        "configuration": [{"name": "baseline_raw", "smoothing_half_width_frames": 0}],
    }
    provenance_fields = (
        "project_revision",
        "official_repository_revision",
        "official_baseline_sha256",
        "official_evaluator_sha256",
        "official_metrics_sha256",
        "official_baseline_dependencies_sha256",
        "input_sha256",
        "evaluation_scope",
    )
    (audit_dir / "baseline-audit.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "scope": "public",
                "evaluation_scope": scope,
                "metadata": metadata,
                "runs": [
                    {
                        "run_id": "run-1",
                        "status": "complete",
                        "provenance": {key: metadata[key] for key in provenance_fields},
                        "metrics": _scored_metrics(evaluation_dir),
                        "prediction_sha256": {"baseline_raw": "0" * 64},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        error_analysis,
        "_load_official_metrics",
        lambda repository_root, expected_sha256: metrics_module,
    )

    with pytest.raises(ValueError, match="prediction checksum differs"):
        error_analysis.analyze_baseline_audit(root, audit_dir, tmp_path / "analysis")
