import csv
import json
import statistics

import pytest

from possession.experiments import run_experiments


def _csv_value(raw, expected):
    if expected is None:
        assert raw == ""
        return None
    if isinstance(expected, bool):
        return raw.lower() == "true"
    if isinstance(expected, int):
        return int(raw)
    if isinstance(expected, float):
        return float(raw)
    return raw


def test_experiment_runner_writes_machine_readable_configured_results(tmp_path):
    paths = run_experiments(
        tmp_path,
        seeds=(7,),
        scenarios=("passes_turnovers",),
        duration_seconds=2.0,
    )
    assert set(paths) == {"csv", "json", "report"}
    with paths["csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert rows[0]["seed"] == "7"
    assert json.loads(rows[0]["configuration_json"])["synthetic"]["fps"] == 25
    result = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert result["evaluation_kind"] == "synthetic_only_not_competition_performance"
    assert len(result["rows"]) == 9
    assert "not official" in paths["report"].read_text(encoding="utf-8")


def test_experiment_runner_counts_aliases_once(tmp_path):
    paths = run_experiments(
        tmp_path,
        seeds=(7,),
        scenarios=("passes_turnovers", "turnovers"),
        duration_seconds=2.0,
    )
    result = json.loads(paths["json"].read_text(encoding="utf-8"))
    report = paths["report"].read_text(encoding="utf-8")

    assert result["requested_scenarios"] == ["passes_turnovers", "turnovers"]
    assert result["scenarios"] == ["passes_turnovers"]
    assert result["scenario_aliases_collapsed"] == {"turnovers": "passes_turnovers"}
    assert len(result["rows"]) == 9
    assert {row["scenario"] for row in result["rows"]} == {"passes_turnovers"}
    assert "- Effective scenario families (1): passes_turnovers" in report
    assert "- Runs per configuration: 1" in report


def test_experiment_report_keeps_distinct_requested_conditions(tmp_path):
    requested = (
        "home_to_away_turnover",
        "away_to_home_turnover",
        "long_dead_ball",
        "consecutive_dead_balls",
    )
    paths = run_experiments(
        tmp_path,
        seeds=(7,),
        scenarios=requested,
        duration_seconds=2.0,
    )
    result = json.loads(paths["json"].read_text(encoding="utf-8"))
    report = paths["report"].read_text(encoding="utf-8")

    assert result["requested_scenarios"] == list(requested)
    assert result["scenarios"] == list(requested)
    assert {row["scenario"] for row in result["rows"]} == set(requested)
    assert len(result["rows"]) == len(requested) * 9
    for scenario in requested:
        assert scenario in report


def test_experiment_runner_rejects_deprecated_scenario_with_replacement(tmp_path):
    with pytest.raises(ValueError, match="scenario 'interception' is deprecated"):
        run_experiments(
            tmp_path,
            seeds=(7,),
            scenarios=("interception",),
            duration_seconds=2.0,
        )


def test_experiment_outputs_are_consistent_and_reproducible(tmp_path):
    options = {
        "seeds": (7, 13),
        "scenarios": ("stable_home",),
        "duration_seconds": 2.0,
    }
    first_paths = run_experiments(tmp_path / "first", **options)
    second_paths = run_experiments(tmp_path / "second", **options)

    first_json = json.loads(first_paths["json"].read_text(encoding="utf-8"))
    second_json = json.loads(second_paths["json"].read_text(encoding="utf-8"))
    assert len(first_json["rows"]) == len(second_json["rows"]) == 18
    assert {key: value for key, value in first_json.items() if key != "rows"} == {
        key: value for key, value in second_json.items() if key != "rows"
    }

    first_rows = {
        (row["scenario"], row["seed"], row["experiment"]): row for row in first_json["rows"]
    }
    second_rows = {
        (row["scenario"], row["seed"], row["experiment"]): row for row in second_json["rows"]
    }
    assert first_rows.keys() == second_rows.keys()
    for key, first in first_rows.items():
        second = second_rows[key]
        assert {name: value for name, value in first.items() if name != "runtime_seconds"} == {
            name: value for name, value in second.items() if name != "runtime_seconds"
        }

    for output_paths, json_rows in (
        (first_paths, first_rows),
        (second_paths, second_rows),
    ):
        with output_paths["csv"].open(encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        assert len(csv_rows) == len(json_rows)
        for csv_row in csv_rows:
            key = (csv_row["scenario"], int(csv_row["seed"]), csv_row["experiment"])
            json_row = json_rows[key]
            for field, raw in csv_row.items():
                if field == "configuration_json":
                    assert json.loads(raw) == json_row["configuration"]
                else:
                    assert _csv_value(raw, json_row[field]) == json_row[field]

    first_report = first_paths["report"].read_text(encoding="utf-8")
    second_report = second_paths["report"].read_text(encoding="utf-8")
    assert first_report == second_report
    aggregate = first_report.split("## Aggregate results\n", maxsplit=1)[1].split(
        "## Scenario-level results", maxsplit=1
    )[0]
    baseline_line = next(
        line
        for line in aggregate.splitlines()
        if line.startswith("| baseline_model_baseline_decoder |")
    )
    fields = baseline_line.strip("| ").split("|")
    metrics = [[float(value.strip()) for value in summary.split("±")] for summary in fields[1:3]]
    baseline_rows = [
        row for row in first_json["rows"] if row["experiment"] == "baseline_model_baseline_decoder"
    ]
    expected_metrics = [
        [
            statistics.mean(row[name] for row in baseline_rows),
            statistics.stdev(row[name] for row in baseline_rows),
        ]
        for name in ("macro_f1", "spell_f1_mean")
    ]
    for actual, expected in zip(metrics, expected_metrics, strict=True):
        assert actual == pytest.approx(expected, abs=0.00051)
