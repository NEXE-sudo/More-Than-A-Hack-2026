import csv
import json
import re
import statistics

import pytest

from possession.experiments import run_experiments


def _csv_value(raw, expected):
    if expected is None:
        assert raw == "", f"expected blank CSV value for null, got {raw!r}"
        return None
    if isinstance(expected, bool):
        normalized = raw.casefold()
        assert normalized in {"true", "false"}, f"invalid CSV boolean: {raw!r}"
        return normalized == "true"
    if isinstance(expected, int):
        return int(raw)
    if isinstance(expected, float):
        return float(raw)
    return raw


def _index_rows(rows, key_fields, description):
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        assert key not in indexed, f"duplicate {description} key: {key}"
        indexed[key] = row
    return indexed


def _markdown_table(section, expected_header):
    lines = section.splitlines()
    for index, line in enumerate(lines):
        header = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if header != expected_header:
            continue
        rows = []
        for row_line in lines[index + 2 :]:
            if not row_line.strip().startswith("|"):
                break
            cells = [cell.strip() for cell in row_line.strip().strip("|").split("|")]
            assert len(cells) == len(expected_header), f"malformed Markdown table row: {row_line}"
            rows.append(dict(zip(expected_header, cells, strict=True)))
        return rows
    raise AssertionError(f"Markdown table not found with columns {expected_header!r}")


def _displayed_mean_sd(value, context):
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?) ± ([+-]?\d+(?:\.\d+)?)", value)
    assert match is not None, f"invalid mean ± SD for {context}: {value!r}"
    return float(match.group(1)), float(match.group(2))


def _assert_displayed(actual, expected, precision, context):
    tolerance = 0.5 * 10**-precision + 1e-12
    assert actual == pytest.approx(expected, abs=tolerance), (
        f"{context}: displayed {actual}, expected {expected} rounded to {precision} decimal places"
    )


def _mean_sd(rows, metric):
    values = [row[metric] for row in rows if row[metric] is not None]
    if not values:
        return 0.0, 0.0
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("FALSE", False)])
def test_csv_boolean_values_are_parsed_case_insensitively(raw, expected):
    assert _csv_value(raw, False) is expected


@pytest.mark.parametrize("raw", ["", "yes", "0", "truth"])
def test_csv_boolean_values_reject_invalid_text(raw):
    with pytest.raises(AssertionError, match="invalid CSV boolean"):
        _csv_value(raw, False)


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
        "scenarios": ("stable_home", "passes_turnovers"),
        "duration_seconds": 2.0,
    }
    first_paths = run_experiments(tmp_path / "first", **options)
    second_paths = run_experiments(tmp_path / "second", **options)

    first_json = json.loads(first_paths["json"].read_text(encoding="utf-8"))
    second_json = json.loads(second_paths["json"].read_text(encoding="utf-8"))
    assert len(first_json["rows"]) == len(second_json["rows"]) == 36
    assert {key: value for key, value in first_json.items() if key != "rows"} == {
        key: value for key, value in second_json.items() if key != "rows"
    }

    json_fields = set(first_json["rows"][0])
    for report in (first_json, second_json):
        assert all(set(row) == json_fields for row in report["rows"])
        assert report["scenarios"] == list(options["scenarios"])
        assert report["seeds"] == list(options["seeds"])

    key_fields = ("scenario", "seed", "experiment")
    first_rows = _index_rows(first_json["rows"], key_fields, "first JSON")
    second_rows = _index_rows(second_json["rows"], key_fields, "second JSON")
    experiments = [experiment["name"] for experiment in first_json["experiments"]]
    expected_keys = {
        (scenario, seed, experiment)
        for scenario in options["scenarios"]
        for seed in options["seeds"]
        for experiment in experiments
    }
    assert first_rows.keys() == second_rows.keys() == expected_keys
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
            reader = csv.DictReader(handle)
            csv_rows = list(reader)
            csv_fields = reader.fieldnames
        expected_csv_fields = {
            "configuration_json" if field == "configuration" else field
            for field in first_json["rows"][0]
        }
        assert len(csv_fields) == len(expected_csv_fields)
        assert set(csv_fields) == expected_csv_fields
        assert len(csv_rows) == len(json_rows)
        indexed_csv = _index_rows(
            [{**row, "seed": int(row["seed"])} for row in csv_rows],
            key_fields,
            "CSV",
        )
        assert indexed_csv.keys() == json_rows.keys()
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
    assert first_report == second_report, (
        "Markdown report reproducibility failure: identical runs produced different reports"
    )

    aggregate = first_report.split("## Aggregate results", maxsplit=1)[1].split(
        "## Scenario-level results", maxsplit=1
    )
    aggregate_rows = _markdown_table(
        aggregate[0],
        [
            "Configuration",
            "macro-F1 mean ± SD",
            "spellF1 mean ± SD",
            "switches/min",
            "boundary error (frames)",
        ],
    )
    aggregate_by_name = _index_rows(aggregate_rows, ("Configuration",), "Markdown aggregate")
    assert aggregate_by_name.keys() == {(name,) for name in experiments}
    for name in experiments:
        group = [row for row in first_json["rows"] if row["experiment"] == name]
        markdown = aggregate_by_name[(name,)]
        for column, metric in (
            ("macro-F1 mean ± SD", "macro_f1"),
            ("spellF1 mean ± SD", "spell_f1_mean"),
        ):
            actual = _displayed_mean_sd(markdown[column], f"{name} {metric}")
            expected = _mean_sd(group, metric)
            for value, target in zip(actual, expected, strict=True):
                _assert_displayed(value, target, 3, f"{name} {metric}")
        _assert_displayed(
            float(markdown["switches/min"]),
            _mean_sd(group, "false_switches_per_minute")[0],
            2,
            f"{name} false_switches_per_minute",
        )
        boundaries = [
            row["boundary_timing_error_frames"]
            for row in group
            if row["boundary_timing_error_frames"] is not None
        ]
        if boundaries:
            assert markdown["boundary error (frames)"] != "n/a"
            _assert_displayed(
                float(markdown["boundary error (frames)"]),
                statistics.mean(boundaries),
                2,
                f"{name} boundary_timing_error_frames",
            )
        else:
            assert markdown["boundary error (frames)"] == "n/a"

    scenario_section = first_report.split("## Scenario-level results", maxsplit=1)[1].split(
        "## Paired descriptive deltas", maxsplit=1
    )[0]
    scenario_blocks = re.split(r"(?m)^### ", scenario_section)
    scenario_tables = {}
    for block in scenario_blocks[1:]:
        name, _, body = block.partition("\n")
        table_rows = _markdown_table(
            body,
            [
                "Scenario",
                "macro-F1 mean ± SD",
                "spellF1 mean ± SD",
                "true/predicted spells (mean)",
                "alive/dead F1",
                "team accuracy",
            ],
        )
        indexed = _index_rows(table_rows, ("Scenario",), f"{name} scenario")
        assert name not in scenario_tables, f"duplicate scenario table for {name}"
        scenario_tables[name] = indexed
    assert scenario_tables.keys() == set(experiments)
    for name in experiments:
        indexed = scenario_tables[name]
        assert indexed.keys() == {(scenario,) for scenario in options["scenarios"]}
        for scenario in options["scenarios"]:
            context = f"{name}/{scenario}"
            group = [
                row
                for row in first_json["rows"]
                if row["experiment"] == name and row["scenario"] == scenario
            ]
            markdown = indexed[(scenario,)]
            for column, metric in (
                ("macro-F1 mean ± SD", "macro_f1"),
                ("spellF1 mean ± SD", "spell_f1_mean"),
            ):
                actual = _displayed_mean_sd(markdown[column], f"{context} {metric}")
                expected = _mean_sd(group, metric)
                for value, target in zip(actual, expected, strict=True):
                    _assert_displayed(value, target, 3, f"{context} {metric}")
            spell_counts = markdown["true/predicted spells (mean)"].split("/")
            assert len(spell_counts) == 2, f"invalid spell count summary for {context}"
            for value, metric in zip(
                spell_counts, ("ground_truth_spells", "predicted_spells"), strict=True
            ):
                _assert_displayed(
                    float(value), _mean_sd(group, metric)[0], 1, f"{context} {metric}"
                )
            for column, metric in (
                ("alive/dead F1", "alive_dead_f1"),
                ("team accuracy", "team_accuracy"),
            ):
                _assert_displayed(
                    float(markdown[column]),
                    _mean_sd(group, metric)[0],
                    3,
                    f"{context} {metric}",
                )

    paired_section = first_report.split("## Paired descriptive deltas", maxsplit=1)[1]
    paired_rows = _markdown_table(
        paired_section,
        [
            "Configuration",
            "Δ macro-F1 mean ± SD",
            "Δ spellF1 mean ± SD",
        ],
    )
    paired_by_name = _index_rows(paired_rows, ("Configuration",), "Markdown paired-delta")
    non_baseline = set(experiments) - {"baseline_model_baseline_decoder"}
    assert paired_by_name.keys() == {(name,) for name in non_baseline}
    baseline_rows = [
        row for row in first_json["rows"] if row["experiment"] == "baseline_model_baseline_decoder"
    ]
    baseline = _index_rows(baseline_rows, ("scenario", "seed"), "baseline paired-delta")
    expected_baseline_keys = {
        (scenario, seed) for scenario in options["scenarios"] for seed in options["seeds"]
    }
    assert baseline.keys() == expected_baseline_keys
    for name in non_baseline:
        group = [row for row in first_json["rows"] if row["experiment"] == name]
        markdown = paired_by_name[(name,)]
        for column, metric in (
            ("Δ macro-F1 mean ± SD", "macro_f1"),
            ("Δ spellF1 mean ± SD", "spell_f1_mean"),
        ):
            differences = [
                row[metric] - baseline[(row["scenario"], row["seed"])][metric] for row in group
            ]
            actual = _displayed_mean_sd(markdown[column], f"{name} paired {metric}")
            expected = _mean_sd(
                [{metric: value} for value in differences],
                metric,
            )
            for value, target in zip(actual, expected, strict=True):
                _assert_displayed(value, target, 3, f"{name} paired {metric}")
