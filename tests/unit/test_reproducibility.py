import csv
import hashlib
import json

import pytest

import possession.evaluation.reproducibility as reproducibility


def test_scored_metrics_rejects_non_object_report(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "malformed.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        reproducibility._scored_metrics(tmp_path)


def _make_official_repository(root):
    (root / "starter").mkdir(parents=True)
    (root / "evaluation").mkdir()
    ground_truth_dir = root / "evaluation" / "gt"
    ground_truth_dir.mkdir()
    (root / "starter" / "baseline.py").write_text(
        'VARIANTS = [("smooth-25", 25), ("smooth-125", 125)]\n',
        encoding="utf-8",
    )
    (root / "games.py").write_text("GAME = 'fixture'\n", encoding="utf-8")
    (root / "evaluation" / "run_evaluation.py").touch()
    (root / "evaluation" / "possession_metrics.py").touch()
    (ground_truth_dir / "test_possession_gt_public.csv").write_text(
        "frame,label\n0,H\n", encoding="utf-8"
    )
    (ground_truth_dir / "test_possession_gt_public_meta.json").write_text(
        json.dumps(
            {
                "n_graded_frames": 12,
                "n_possessions": 2,
                "n_video_frames": 14,
                "public_window": {"first_video_frame": 0, "last_video_frame": 13},
            }
        ),
        encoding="utf-8",
    )


def _write_metrics_report(directory, submission):
    reports = directory / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{submission}.json").write_text(
        json.dumps(
            {
                "submission": submission,
                "macro_f1": 0.75,
                "spell_f1_mean": 0.5,
                "spell_f1_at_headline": 0.6,
                "n_graded": 12,
                "n_gt_spells": 2,
                "n_pred_spells": 3,
                "transitions": {
                    "25": {
                        "f1": 0.4,
                        "n_matched": 2,
                        "n_missed": 1,
                        "n_spurious": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_repeated_baseline_audit_records_hashes_metrics_and_stable_ids(tmp_path, monkeypatch):
    root = tmp_path / "official"
    _make_official_repository(root)
    data = tmp_path / "data"
    tactical = data / "tactical"
    tactical.mkdir(parents=True)
    (tactical / "BAR-ATH_tactical_atd.xml").write_text("<root />", encoding="utf-8")
    (tactical / "BAR-ATH_tactical_atd.txt").write_text("synthetic test fixture\n", encoding="utf-8")

    def generate_baseline(repository_root, data_root, output_dir):
        output_dir.mkdir(parents=True)
        for name in ("smooth-25", "smooth-125"):
            (output_dir / f"{name}.csv").write_text("frame,label\n0,H\n", encoding="utf-8")

    def evaluate(repository_root, prediction_dir, output_dir, ground_truth, bootstrap):
        output_dir.mkdir(parents=True)
        for path in prediction_dir.glob("*.csv"):
            _write_metrics_report(output_dir, path.stem)

    monkeypatch.setattr(reproducibility, "run_official_baseline", generate_baseline)
    monkeypatch.setattr(reproducibility, "run_official_evaluator", evaluate)

    report = reproducibility.run_baseline_audit(
        root,
        data,
        tmp_path / "baseline-audit",
        bootstrap=0,
    )
    repeated = reproducibility.run_baseline_audit(
        root,
        data,
        tmp_path / "baseline-audit-repeat",
        bootstrap=0,
    )

    assert report["status"] == "complete"
    assert set(report["metadata"]["input_sha256"]) == {
        "BAR-ATH_tactical_atd.xml",
        "BAR-ATH_tactical_atd.txt",
    }
    assert report["determinism"] == {
        "prediction_files_byte_identical": True,
        "metrics_identical": True,
    }
    assert all(experiment["prediction_reproducible"] for experiment in report["experiments"])
    assert (tmp_path / "baseline-audit" / "baseline-audit.json").is_file()
    assert (tmp_path / "baseline-audit" / "baseline-audit.md").is_file()
    assert [row["experiment_id"] for row in report["experiments"]] == [
        row["experiment_id"] for row in repeated["experiments"]
    ]


def test_smoothing_sensitivity_uses_only_predeclared_intermediate_widths(
    tmp_path,
    monkeypatch,
):
    calls = []

    def run_audit(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(reproducibility, "run_baseline_audit", run_audit)

    result = reproducibility.run_smoothing_sensitivity_audit(
        tmp_path / "official",
        tmp_path / "data",
        tmp_path / "sensitivity",
    )

    assert result["status"] == "complete"
    assert calls[0][1]["smoothing_half_widths"] == (75, 100)
    assert "descriptive only, not parameter selection" in calls[0][1]["sensitivity_question"]


def test_sensitivity_report_records_custom_windows_and_reproduction_command(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "official"
    _make_official_repository(root)
    data = tmp_path / "data"
    tactical = data / "tactical"
    tactical.mkdir(parents=True)
    (tactical / "BAR-ATH_tactical_atd.xml").write_text("<root />", encoding="utf-8")
    (tactical / "BAR-ATH_tactical_atd.txt").write_text("synthetic fixture\n", encoding="utf-8")
    requested_widths = []

    def generate_baseline(repository_root, data_root, output_dir, *, smoothing_half_widths):
        requested_widths.append(smoothing_half_widths)
        output_dir.mkdir(parents=True)
        for width in smoothing_half_widths:
            (output_dir / f"baseline_smooth{width}.csv").write_text(
                "frame,label\n0,H\n",
                encoding="utf-8",
            )

    def evaluate(repository_root, prediction_dir, output_dir, ground_truth, bootstrap):
        output_dir.mkdir(parents=True)
        for path in prediction_dir.glob("*.csv"):
            _write_metrics_report(output_dir, path.stem)

    monkeypatch.setattr(reproducibility, "run_official_baseline", generate_baseline)
    monkeypatch.setattr(reproducibility, "run_official_evaluator", evaluate)

    report = reproducibility.run_baseline_audit(
        root,
        data,
        tmp_path / "sensitivity",
        bootstrap=1000,
        smoothing_half_widths=(75, 100),
        sensitivity_question="A bounded synthetic sensitivity test.",
    )

    assert requested_widths == [(75, 100), (75, 100)]
    assert [
        variant["smoothing_half_width_frames"] for variant in report["metadata"]["configuration"]
    ] == [
        75,
        100,
    ]
    assert report["metadata"]["commands"][0][-3:] == ["--window", "75", "100"]
    assert report["metadata"]["smoothing_sensitivity"]["changes_production_default"] is False


def test_record_prediction_experiment_writes_separate_experiment_report(tmp_path, monkeypatch):
    root = tmp_path / "official"
    _make_official_repository(root)
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("frame,label\n0,H\n", encoding="utf-8")
    source = tmp_path / "tracking.txt"
    source.write_text("synthetic tracking fixture", encoding="utf-8")

    def evaluate(repository_root, prediction_dir, output_dir, ground_truth, bootstrap):
        output_dir.mkdir(parents=True)
        with (output_dir / "scoreboard.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "submission",
                    "status",
                    "macro_f1",
                    "spell_f1_mean",
                    "spell_f1_at_headline",
                    "n_graded",
                    "n_gt_spells",
                    "n_pred_spells",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "submission": "pilot-1",
                    "status": "scored",
                    "macro_f1": "0.75",
                    "spell_f1_mean": "0.5",
                    "spell_f1_at_headline": "0.6",
                    "n_graded": "12",
                    "n_gt_spells": "2",
                    "n_pred_spells": "3",
                }
            )

    monkeypatch.setattr(reproducibility, "run_official_evaluator", evaluate)

    report = reproducibility.record_prediction_experiment(
        root,
        prediction,
        tmp_path / "experiment",
        experiment_id="pilot-1",
        description="A local synthetic test record",
        generation_command="possession predict --synthetic",
        source_inputs=(source,),
        bootstrap=0,
    )

    assert report["status"] == "complete"
    assert report["experiments"][0]["metrics"]["spell_f1_mean"] == 0.5
    assert (
        report["metadata"]["input_sha256"]["tracking.txt"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert (tmp_path / "experiment" / "experiment-report.json").is_file()
    assert (tmp_path / "experiment" / "experiment-report.md").is_file()
    assert (tmp_path / "experiment" / "predictions" / "pilot-1.csv").read_bytes() == (
        prediction.read_bytes()
    )


def test_compare_experiment_reports_flags_different_scopes(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(_valid_report("public", "one", "a" * 64)),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(_valid_report("synthetic", "two", "b" * 64)),
        encoding="utf-8",
    )

    experiments, mismatch = reproducibility.compare_experiment_reports([first, second])

    assert mismatch
    assert [experiment["scope"] for experiment in experiments] == ["public", "synthetic"]
    assert all(experiment["comparison_warnings"] for experiment in experiments)


def test_compare_experiment_reports_requires_two_reports(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        reproducibility.compare_experiment_reports([tmp_path / "one.json"])


def _valid_report(scope, experiment_id, ground_truth_hash):
    configuration = {"mode": experiment_id}
    return {
        "status": "complete",
        "scope": scope,
        "evaluation_scope": {
            "ground_truth_sha256": ground_truth_hash,
            "graded_frames": 12,
            "ground_truth_spells": 2,
            "public_window": {"first_video_frame": 0, "last_video_frame": 13},
        },
        "metadata": {
            "official_evaluator_sha256": "c" * 64,
            "official_metrics_sha256": "d" * 64,
            "input_sha256": {"tracking.txt": "e" * 64},
            "project_revision": "revision",
            "official_repository_revision": "official-revision",
        },
        "experiments": [
            {
                "experiment_id": experiment_id,
                "status": "complete",
                "scope": scope,
                "configuration": configuration,
                "metrics": {
                    "macro_f1": 0.5,
                    "spell_f1_mean": 0.4,
                    "spell_f1_at_headline": 0.3,
                    "graded_frames": 12,
                    "ground_truth_spells": 2,
                    "predicted_spells": 3,
                },
                "prediction_sha256": "f" * 64,
                "configuration_fingerprint": hashlib.sha256(
                    json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        ],
    }


def test_compare_experiment_reports_rejects_missing_metrics_and_duplicate_ids(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_valid_report("public", "one", "a" * 64)), encoding="utf-8")
    missing_metrics = _valid_report("public", "two", "a" * 64)
    missing_metrics["experiments"][0]["metrics"].pop("spell_f1_mean")
    second.write_text(json.dumps(missing_metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="missing metrics"):
        reproducibility.compare_experiment_reports([first, second])

    duplicate = _valid_report("public", "one", "a" * 64)
    second.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate experiment identifier"):
        reproducibility.compare_experiment_reports([first, second])


def test_compare_experiment_reports_flags_missing_checksums_and_scope_mismatch(tmp_path):
    first = _valid_report("public", "one", "a" * 64)
    second = _valid_report("public", "two", "a" * 64)
    second["metadata"]["input_sha256"] = {}
    second["evaluation_scope"]["ground_truth_sha256"] = "b" * 64
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    experiments, warning = reproducibility.compare_experiment_reports([first_path, second_path])

    assert warning
    assert any(
        "source-input checksums are missing" in item
        for item in experiments[1]["comparison_warnings"]
    )
    assert any(
        "reports differ in evaluation protocol" in item
        for item in experiments[0]["comparison_warnings"]
    )


def test_compare_experiment_reports_flags_revision_and_metric_window_mismatches(tmp_path):
    first = _valid_report("public", "one", "a" * 64)
    second = _valid_report("public", "two", "a" * 64)
    second["metadata"]["project_revision"] = "different-revision"
    second["metadata"]["official_repository_revision"] = "different-official-revision"
    second["experiments"][0]["metrics"]["graded_frames"] = 11
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    experiments, warning = reproducibility.compare_experiment_reports([first_path, second_path])

    assert warning
    second_warnings = experiments[1]["comparison_warnings"]
    assert any(
        "metrics do not match the report evaluation window" in item for item in second_warnings
    )
    assert any("project source revisions differ" in item for item in second_warnings)
    assert any("official repository revisions differ" in item for item in second_warnings)


def test_compare_experiment_reports_warns_on_malformed_input_checksum_values(tmp_path):
    first = _valid_report("public", "one", "a" * 64)
    second = _valid_report("public", "two", "a" * 64)
    second["metadata"]["input_sha256"] = {"tracking.txt": {"not": "a checksum"}}
    second["metadata"].pop("official_repository_revision")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    experiments, warning = reproducibility.compare_experiment_reports([first_path, second_path])

    assert warning
    assert any(
        "source-input checksums are invalid" in item
        for item in experiments[1]["comparison_warnings"]
    )
    assert any(
        "official repository revision is missing or invalid" in item
        for item in experiments[1]["comparison_warnings"]
    )


def test_compare_experiment_reports_does_not_trust_a_stale_configuration_fingerprint(tmp_path):
    first = _valid_report("public", "one", "a" * 64)
    second = _valid_report("public", "two", "a" * 64)
    second["experiments"][0]["configuration_fingerprint"] = first["experiments"][0][
        "configuration_fingerprint"
    ]
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    experiments, warning = reproducibility.compare_experiment_reports([first_path, second_path])

    assert warning
    assert any(
        "configuration fingerprint does not match configuration" in item
        for item in experiments[1]["comparison_warnings"]
    )


def test_compare_experiment_reports_rejects_nonfinite_metrics_and_empty_histories(tmp_path):
    first = _valid_report("public", "one", "a" * 64)
    second = _valid_report("public", "two", "a" * 64)
    second["experiments"][0]["metrics"]["macro_f1"] = float("nan")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    with pytest.raises(ValueError, match="finite value"):
        reproducibility.compare_experiment_reports([first_path, second_path])

    second["experiments"] = []
    second_path.write_text(json.dumps(second), encoding="utf-8")
    with pytest.raises(ValueError, match="no experiment records"):
        reproducibility.compare_experiment_reports([first_path, second_path])


def test_baseline_audit_rejects_symlink_output_directory(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    destination = tmp_path / "audit-link"
    destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        reproducibility.run_baseline_audit(
            tmp_path / "official",
            tmp_path / "data",
            destination,
        )


def test_output_paths_inside_repository_must_be_ignored_data_output_locations(tmp_path):
    with pytest.raises(ValueError, match="must be under ignored"):
        reproducibility.ensure_local_output_path(
            reproducibility.PROJECT_ROOT / "src" / "output.json"
        )
    reproducibility.ensure_local_output_path(
        reproducibility.PROJECT_ROOT / "outputs" / "safe-output.json"
    )


def test_repeated_baseline_audit_rejects_input_change_during_generation(tmp_path, monkeypatch):
    root = tmp_path / "official"
    _make_official_repository(root)
    data = tmp_path / "data"
    tactical = data / "tactical"
    tactical.mkdir(parents=True)
    metadata = tactical / "BAR-ATH_tactical_atd.xml"
    metadata.write_text("<root />", encoding="utf-8")
    (tactical / "BAR-ATH_tactical_atd.txt").write_text("tracking", encoding="utf-8")
    calls = 0

    def generate_baseline(repository_root, data_root, output_dir):
        nonlocal calls
        output_dir.mkdir(parents=True)
        (output_dir / "smooth-25.csv").write_text("frame,label\n0,H\n", encoding="utf-8")
        (output_dir / "smooth-125.csv").write_text("frame,label\n0,H\n", encoding="utf-8")
        calls += 1
        if calls == 1:
            metadata.write_text("<root changed='true' />", encoding="utf-8")

    monkeypatch.setattr(reproducibility, "run_official_baseline", generate_baseline)

    with pytest.raises(ValueError, match="source provenance changed"):
        reproducibility.run_baseline_audit(root, data, tmp_path / "audit", bootstrap=0)
    report = json.loads((tmp_path / "audit" / "baseline-audit.json").read_text())
    assert report["status"] == "failed"
    assert not report["runs"]
