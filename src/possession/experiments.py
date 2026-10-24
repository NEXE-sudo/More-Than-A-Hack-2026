"""Reproducible synthetic model/decoder comparison runner."""

import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from possession.config import DecoderConfig, ModelConfig
from possession.evaluation.synthetic import evaluate_synthetic
from possession.models.baseline import (
    dead_ball_evidence,
    evidence_from_features,
    revised_evidence_from_features,
)
from possession.synthetic import (
    CANONICAL_SCENARIOS,
    SCENARIO_ALIASES,
    SCENARIOS,
    generate_synthetic_sequence,
)
from possession.temporal.decoder import decode, decode_revised

EXPERIMENTS = (
    {"name": "baseline_model_baseline_decoder", "model": "baseline", "decoder": "baseline"},
    {"name": "revised_model_baseline_decoder", "model": "revised", "decoder": "baseline"},
    {"name": "baseline_model_revised_decoder", "model": "baseline", "decoder": "revised"},
    {"name": "revised_model_revised_decoder", "model": "revised", "decoder": "revised"},
    {
        "name": "revised_without_proximity",
        "model": "revised",
        "decoder": "revised",
        "use_proximity": False,
    },
    {
        "name": "revised_without_support",
        "model": "revised",
        "decoder": "revised",
        "use_support": False,
    },
    {
        "name": "revised_without_approach",
        "model": "revised",
        "decoder": "revised",
        "use_approach": False,
    },
    {
        "name": "revised_smart_tagging_off",
        "model": "revised",
        "decoder": "revised",
        "tagging_team_weight": 0.0,
    },
    {
        "name": "revised_smart_tagging_on",
        "model": "revised",
        "decoder": "revised",
        "tagging_team_weight": 0.05,
    },
)


def _run_one(
    sequence,
    experiment: dict[str, object],
    synthetic_config: dict[str, int | float | str],
) -> dict[str, object]:
    tagging_team_weight = float(experiment.get("tagging_team_weight", 0.0))
    model_config = ModelConfig(
        evidence_model=str(experiment["model"]),
        tagging_team_weight=tagging_team_weight,
    )
    dead_score = dead_ball_evidence(
        sequence.features,
        minimum_gap_frames=model_config.dead_ball_min_gap_frames,
        maximum_team_speed_m_s=model_config.dead_ball_max_team_speed_m_s,
        minimum_tracks_per_team=model_config.dead_ball_min_tracks_per_team,
        score=model_config.dead_ball_evidence_score,
    )
    if experiment["model"] == "baseline":
        evidence = evidence_from_features(
            sequence.features, model_config.distance_scale_m, dead_score
        )
    else:
        evidence = revised_evidence_from_features(
            sequence.features,
            model_config.distance_scale_m,
            dead_score,
            tagging_team_weight=model_config.tagging_team_weight,
            use_proximity=bool(experiment.get("use_proximity", True)),
            use_support=bool(experiment.get("use_support", True)),
            use_approach=bool(experiment.get("use_approach", True)),
        )
    decoder_config = DecoderConfig(
        strategy=str(experiment["decoder"]),
        contested_tolerance_frames=1,
    )
    decoder = decode_revised if experiment["decoder"] == "revised" else decode
    prediction = decoder(evidence, decoder_config, sequence.sequence_starts)
    metrics = evaluate_synthetic(
        sequence.labels,
        prediction,
        ball_visible=sequence.features["ball_visible"],
    )
    return {
        "scenario": sequence.family,
        "seed": sequence.seed,
        "experiment": experiment["name"],
        "model": experiment["model"],
        "decoder": experiment["decoder"],
        "tagging_team_weight": model_config.tagging_team_weight,
        "use_proximity": bool(experiment.get("use_proximity", True)),
        "use_support": bool(experiment.get("use_support", True)),
        "use_approach": bool(experiment.get("use_approach", True)),
        "samples": 1,
        "warnings": "",
        "configuration": {
            "model_version": f"{experiment['model']}-v1",
            "decoder_version": f"{experiment['decoder']}-v1",
            "model": {
                **asdict(model_config),
                "use_proximity": bool(experiment.get("use_proximity", True)),
                "use_support": bool(experiment.get("use_support", True)),
                "use_approach": bool(experiment.get("use_approach", True)),
                "evidence_weights": {
                    "proximity": 0.75,
                    "support": 0.15,
                    "approach": 0.10,
                },
            },
            "decoder": asdict(decoder_config),
            "synthetic": {
                **synthetic_config,
                "scenario": sequence.family,
                "seed": sequence.seed,
            },
        },
        **metrics,
    }


def run_experiments(
    output_dir: Path | str,
    *,
    seeds: tuple[int, ...] = (11, 29, 47),
    scenarios: tuple[str, ...] = CANONICAL_SCENARIOS,
    duration_seconds: float = 16.0,
) -> dict[str, Path]:
    """Run the fixed experiment matrix and write detailed JSON/CSV/Markdown outputs."""
    if not seeds:
        raise ValueError("at least one deterministic seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique to avoid duplicate runs")
    if not scenarios or any(scenario not in SCENARIOS for scenario in scenarios):
        raise ValueError("scenarios must be selected from the predefined synthetic families")
    selected_scenarios = tuple(
        dict.fromkeys(SCENARIO_ALIASES.get(scenario, scenario) for scenario in scenarios)
    )
    aliases_collapsed = {
        scenario: SCENARIO_ALIASES[scenario]
        for scenario in scenarios
        if scenario in SCENARIO_ALIASES
    }
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    rows = []
    synthetic_config: dict[str, int | float | str] = {
        "duration_seconds": duration_seconds,
        "fps": 25,
        "pitch_length_m": 105.3,
        "pitch_width_m": 68.0,
        "players_per_team": 8,
        "player_availability": 0.94,
        "ball_visibility": 0.70,
        "coordinate_noise_m": 0.45,
        "restart_type": "Corner Kick",
    }
    for scenario in selected_scenarios:
        for seed in seeds:
            sequence = generate_synthetic_sequence(
                scenario,
                seed,
                **synthetic_config,
            )
            for experiment in EXPERIMENTS:
                started = time.perf_counter()
                result = _run_one(sequence, experiment, synthetic_config)
                result["runtime_seconds"] = time.perf_counter() - started
                rows.append(result)

    csv_path = destination / "synthetic-results.csv"
    json_path = destination / "synthetic-results.json"
    report_path = destination / "synthetic-report.md"
    metric_names = (
        "accuracy",
        "macro_f1",
        "spell_f1_mean",
        "spell_f1_iou_0_5",
        "spell_matched_at_iou_0_5",
        "spell_spurious_at_iou_0_5",
        "spell_missed_at_iou_0_5",
        "false_switches_per_minute",
        "missed_switches",
        "incorrect_switches",
        "boundary_timing_error_frames",
        "fragmentation",
        "missing_ball_accuracy",
        "dead_ball_f1",
        "alive_dead_f1",
        "team_accuracy",
        "ground_truth_segments",
        "predicted_segments",
        "ground_truth_spells",
        "predicted_spells",
        "graded_frames",
        "runtime_seconds",
    )
    fieldnames = [
        "scenario",
        "seed",
        "experiment",
        "model",
        "decoder",
        "tagging_team_weight",
        "use_proximity",
        "use_support",
        "use_approach",
        "samples",
        "warnings",
        "configuration_json",
        *metric_names,
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {
                **{key: value for key, value in row.items() if key != "configuration"},
                "configuration_json": json.dumps(row["configuration"], sort_keys=True),
            }
            for row in rows
        )
    json_path.write_text(
        json.dumps(
            {
                "evaluation_kind": "synthetic_only_not_competition_performance",
                "seeds": list(seeds),
                "requested_scenarios": list(scenarios),
                "scenarios": list(selected_scenarios),
                "scenario_aliases_collapsed": aliases_collapsed,
                "synthetic_config": synthetic_config,
                "duration_seconds": duration_seconds,
                "experiments": list(EXPERIMENTS),
                "rows": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment"])].append(row)
    report_lines = [
        "# Synthetic possession-system experiment",
        "",
        "Controlled synthetic tracking only. These scores are not official public-window or "
        "competition results and do not establish real-match generalization.",
        "",
        f"- Seeds: {', '.join(map(str, seeds))}",
        f"- Effective scenario families ({len(selected_scenarios)}): "
        f"{', '.join(selected_scenarios)}",
        f"- Sequence duration: {duration_seconds:g} seconds",
        f"- Runs per configuration: {len(seeds) * len(selected_scenarios)}",
        "",
        "Alias inputs are normalized to canonical generators and counted only once. These "
        "descriptive means and standard deviations are pooled over scenario/seed cases; they are "
        "not confidence intervals and do not establish independent scenario sampling.",
        "",
        "## Aggregate results",
        "",
        "Mean ± sample SD across canonical scenario/seed runs. Frame rows are not independent "
        "observations.",
        "",
        "| Configuration | macro-F1 mean ± SD | spellF1 mean ± SD | switches/min | "
        "boundary error (frames) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, group in grouped.items():
        macro_mean, macro_sd = _mean_sd(group, "macro_f1")
        spell_mean, spell_sd = _mean_sd(group, "spell_f1_mean")
        switch_mean, _ = _mean_sd(group, "false_switches_per_minute")
        boundary_values = [
            float(row["boundary_timing_error_frames"])
            for row in group
            if row["boundary_timing_error_frames"] is not None
        ]
        boundary_mean = float(np.mean(boundary_values)) if boundary_values else None
        boundary_text = f"{boundary_mean:.2f}" if boundary_mean is not None else "n/a"
        report_lines.append(
            f"| {name} | {macro_mean:.3f} ± {macro_sd:.3f} | "
            f"{spell_mean:.3f} ± {spell_sd:.3f} | {switch_mean:.2f} | "
            f"{boundary_text} |"
        )
    report_lines.extend(("", "## Scenario-level results", ""))
    for name, group in grouped.items():
        report_lines.append(f"### {name}")
        report_lines.append("")
        report_lines.append(
            "| Scenario | macro-F1 mean ± SD | spellF1 mean ± SD | "
            "true/predicted spells (mean) | alive/dead F1 | team accuracy |"
        )
        report_lines.append("|---|---:|---:|---:|---:|---:|")
        for scenario in selected_scenarios:
            subset = [row for row in group if row["scenario"] == scenario]
            macro_mean, macro_sd = _mean_sd(subset, "macro_f1")
            spell_mean, spell_sd = _mean_sd(subset, "spell_f1_mean")
            true_spells, _ = _mean_sd(subset, "ground_truth_spells")
            pred_spells, _ = _mean_sd(subset, "predicted_spells")
            alive_dead, _ = _mean_sd(subset, "alive_dead_f1")
            team_accuracy, _ = _mean_sd(subset, "team_accuracy")
            report_lines.append(
                f"| {scenario} | {macro_mean:.3f} ± {macro_sd:.3f} | "
                f"{spell_mean:.3f} ± {spell_sd:.3f} | "
                f"{true_spells:.1f}/{pred_spells:.1f} | {alive_dead:.3f} | "
                f"{team_accuracy:.3f} |"
            )
        report_lines.append("")
    report_lines.extend(("", "## Paired descriptive deltas", ""))
    report_lines.append(
        "Per-case differences from `baseline_model_baseline_decoder`, matched by canonical "
        "scenario and seed. These are descriptive paired summaries only, not significance tests."
    )
    report_lines.extend(
        (
            "",
            "| Configuration | Δ macro-F1 mean ± SD | Δ spellF1 mean ± SD |",
            "|---|---:|---:|",
        )
    )
    baseline_rows = {
        (str(row["scenario"]), int(row["seed"])): row
        for row in rows
        if row["experiment"] == "baseline_model_baseline_decoder"
    }
    for experiment in EXPERIMENTS:
        name = str(experiment["name"])
        if name == "baseline_model_baseline_decoder":
            continue
        comparisons = [
            row
            for row in rows
            if row["experiment"] == name
            and (str(row["scenario"]), int(row["seed"])) in baseline_rows
        ]
        macro_deltas = [
            float(row["macro_f1"])
            - float(baseline_rows[(str(row["scenario"]), int(row["seed"]))]["macro_f1"])
            for row in comparisons
        ]
        spell_deltas = [
            float(row["spell_f1_mean"])
            - float(baseline_rows[(str(row["scenario"]), int(row["seed"]))]["spell_f1_mean"])
            for row in comparisons
        ]
        macro_mean, macro_sd = _values_mean_sd(macro_deltas)
        spell_mean, spell_sd = _values_mean_sd(spell_deltas)
        report_lines.append(
            f"| {name} | {macro_mean:+.3f} ± {macro_sd:.3f} | {spell_mean:+.3f} ± {spell_sd:.3f} |"
        )
    report_lines.extend(
        (
            "## Interpretation limits",
            "",
            "- The generator approximates selected tracking failure modes, not football dynamics.",
            "- Tagging-on runs use synthetic phase annotations with known timing and deliberately "
            "unreliable team context; the context is only a small evidence bonus.",
            "- No parameter is selected as a competition winner from these scores.",
            "- Re-run with gated match data and the unchanged official evaluator when authorized "
            "tracking and labels are available.",
        )
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "report": report_path}


def _values_mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return 0.0, 0.0
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def _mean_sd(rows: list[dict[str, object]], metric: str) -> tuple[float, float]:
    return _values_mean_sd([float(row[metric]) for row in rows if row[metric] is not None])
