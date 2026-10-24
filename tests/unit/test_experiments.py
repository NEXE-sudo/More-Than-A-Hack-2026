import csv
import json

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
