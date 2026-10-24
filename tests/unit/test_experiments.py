import csv
import json

import pytest

from possession.experiments import run_experiments


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
