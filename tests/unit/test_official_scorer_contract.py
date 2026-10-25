import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

OFFICIAL_REPOSITORY = Path(__file__).resolve().parents[3] / "more-than-a-hack-2027"


def _score(tmp_path, ground_truth, submissions):
    script = OFFICIAL_REPOSITORY / "evaluation" / "run_evaluation.py"
    if not script.is_file():
        pytest.skip("sibling official evaluator checkout is unavailable")

    ground_truth_path = tmp_path / "ground-truth.csv"
    results_dir = tmp_path / "submissions"
    output_dir = tmp_path / "evaluation"
    results_dir.mkdir()
    for path, labels in [
        (ground_truth_path, ground_truth),
        *[(results_dir / f"{name}.csv", labels) for name, labels in submissions.items()],
    ]:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("frame", "label"))
            writer.writerows(enumerate(labels))

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--gt",
            str(ground_truth_path),
            "--results",
            str(results_dir),
            "--out",
            str(output_dir),
            "--bootstrap",
            "0",
            "--no-detail",
        ],
        cwd=OFFICIAL_REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    reports = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (output_dir / "reports").glob("*.json")
    }
    with (output_dir / "scoreboard.csv").open(encoding="utf-8", newline="") as handle:
        scoreboard = list(csv.DictReader(handle))
    return reports, scoreboard


def test_official_scorer_exact_labels_and_deterministic_tie_break(tmp_path):
    labels = ["H"] * 5 + ["D"] * 2 + ["A"] * 5 + ["X"] * 2

    reports, scoreboard = _score(
        tmp_path,
        labels,
        {"beta": labels, "alpha": labels},
    )

    assert reports["alpha"]["macro_f1"] == 1.0
    assert reports["alpha"]["spell_f1_mean"] == 1.0
    assert reports["alpha"]["n_graded"] == 12
    assert [row["submission"] for row in scoreboard] == ["alpha", "beta"]


def test_official_scorer_handles_spell_iou_threshold_boundary(tmp_path):
    ground_truth = ["H"] * 10 + ["D"] * 2 + ["X"] * 2
    prediction = ["H"] * 6 + ["D"] * 6 + ["X"] * 2

    reports, _ = _score(tmp_path, ground_truth, {"boundary": prediction})

    report = reports["boundary"]
    assert report["n_gt_spells"] == 1
    assert report["n_pred_spells"] == 1
    assert report["spell_f1"]["0.6"]["n_matched"] == 1
    assert report["spell_f1"]["0.7"]["n_matched"] == 0
    assert report["spell_f1_mean"] == pytest.approx(0.8)


def test_official_scorer_matches_overlapping_ground_truth_spells_one_to_one(tmp_path):
    ground_truth = ["H"] * 4 + ["D"] + ["H"] * 4 + ["X"] * 2
    prediction = ["H"] * 9 + ["X"] * 2

    reports, _ = _score(tmp_path, ground_truth, {"overlap": prediction})

    report = reports["overlap"]
    assert report["n_gt_spells"] == 2
    assert report["n_pred_spells"] == 1
    assert report["spell_f1"]["0.3"]["n_matched"] == 1
    assert report["spell_f1"]["0.3"]["n_missed"] == 1
    assert report["spell_f1"]["0.3"]["n_spurious"] == 0
    assert report["spell_f1"]["0.5"]["n_matched"] == 0


def test_official_scorer_does_not_match_two_tied_predictions_to_one_spell(tmp_path):
    ground_truth = ["H"] * 10 + ["X"] * 2
    prediction = ["H"] * 4 + ["D"] * 2 + ["H"] * 4 + ["X"] * 2

    reports, _ = _score(tmp_path, ground_truth, {"duplicate": prediction})

    spell = reports["duplicate"]["spell_f1"]["0.4"]
    assert spell["n_matched"] == 1
    assert spell["n_spurious"] == 1
    assert spell["n_missed"] == 0
    assert spell["f1"] == pytest.approx(2 / 3)
