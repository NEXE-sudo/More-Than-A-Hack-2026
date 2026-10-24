import numpy as np
import pytest

from possession.evaluation.synthetic import evaluate_synthetic


def test_exact_prediction_scores_perfect_frame_and_spell_metrics():
    labels = np.array(["H"] * 5 + ["A"] * 5)
    result = evaluate_synthetic(labels, labels)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["spell_f1_mean"] == 1.0
    assert result["spell_f1_iou_0_5"] == 1.0
    assert result["missed_switches"] == 0
    assert result["incorrect_switches"] == 0
    assert result["boundary_timing_error_frames"] == 0


def test_perfect_single_team_run_keeps_absent_classes_in_macro_f1():
    labels = np.array(["H"] * 12)
    result = evaluate_synthetic(labels, labels)

    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["spell_f1_mean"] == 1.0
    assert result["ground_truth_spells"] == 1
    assert result["predicted_spells"] == 1


def test_abstentions_are_frame_misses_but_filled_for_spell_evaluation():
    truth = np.array(["H"] * 10 + ["A"] * 10)
    prediction = np.array(["H"] * 5 + ["X"] * 5 + ["A"] * 10)
    result = evaluate_synthetic(truth, prediction)
    assert result["accuracy"] == 0.75
    assert result["spell_f1_mean"] == 1.0


def test_ground_truth_x_splits_runs_and_empty_sequences_are_defined():
    truth = np.array(["H"] * 5 + ["X"] * 3 + ["H"] * 5)
    result = evaluate_synthetic(truth, truth)
    assert result["spell_f1_mean"] == 1.0
    assert result["ground_truth_segments"] == 2

    empty = evaluate_synthetic(np.array([], dtype="<U1"), np.array([], dtype="<U1"))
    assert empty["accuracy"] == 0.0
    assert empty["spell_f1_mean"] == 0.0
    assert empty["false_switches_per_minute"] == 0.0


def test_spell_matching_counts_fragmentation_and_one_frame_spells():
    truth = np.array(["H"])
    assert evaluate_synthetic(truth, truth)["spell_f1_mean"] == 1.0

    truth = np.array(["H"] * 10)
    prediction = np.array(["H"] * 4 + ["A"] * 2 + ["H"] * 4)
    result = evaluate_synthetic(truth, prediction)
    assert result["predicted_segments"] == 3
    assert result["fragmentation"] == 3.0
    assert result["spell_f1_iou_0_5"] == 0.0
    assert result["spell_spurious_at_iou_0_5"] == 3


def test_malformed_metric_inputs_are_rejected():
    with pytest.raises(ValueError, match="same-length"):
        evaluate_synthetic(np.array(["H"]), np.array(["H", "A"]))
    with pytest.raises(ValueError, match="one of"):
        evaluate_synthetic(np.array(["Z"]), np.array(["H"]))
