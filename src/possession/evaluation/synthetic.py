"""Small spell/frame evaluator verified against the official evaluator contract."""

import numpy as np

LABELS = ("H", "A", "D", "X")
POSSESSION_LABELS = {"H", "A"}
SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def _runs(truth: np.ndarray) -> list[tuple[int, int]]:
    graded = truth != "X"
    edges = np.diff(np.concatenate(([False], graded, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _fill_abstentions(prediction: np.ndarray, runs: list[tuple[int, int]]) -> np.ndarray:
    filled = prediction.copy()
    for start, stop in runs:
        values = filled[start:stop]
        known = np.flatnonzero(values != "X")
        if len(known):
            carry = np.maximum.accumulate(np.where(values != "X", np.arange(len(values)), -1))
            carry[carry < 0] = known[0]
            filled[start:stop] = values[carry]
    return filled


def _segments(labels: np.ndarray, runs: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    result = []
    for start, stop in runs:
        values = labels[start:stop]
        if not len(values):
            continue
        changes = np.flatnonzero(values[1:] != values[:-1]) + 1
        bounds = np.concatenate(([0], changes, [len(values)]))
        result.extend(
            (start + int(bounds[i]), start + int(bounds[i + 1]), str(values[bounds[i]]))
            for i in range(len(bounds) - 1)
        )
    return result


def _spell_f1(
    truth: np.ndarray, prediction: np.ndarray, runs: list[tuple[int, int]], threshold: float
) -> tuple[float, int, int, int]:
    actual = [segment for segment in _segments(truth, runs) if segment[2] in POSSESSION_LABELS]
    predicted = [
        segment for segment in _segments(prediction, runs) if segment[2] in POSSESSION_LABELS
    ]
    overlaps = []
    for actual_index, (a_start, a_stop, a_label) in enumerate(actual):
        for pred_index, (p_start, p_stop, p_label) in enumerate(predicted):
            if p_label != a_label or p_stop <= a_start or a_stop <= p_start:
                continue
            intersection = min(a_stop, p_stop) - max(a_start, p_start)
            union = max(a_stop, p_stop) - min(a_start, p_start)
            overlaps.append((intersection / union, actual_index, pred_index))
    overlaps.sort(key=lambda match: (-match[0], match[1], match[2]))
    matched_actual: set[int] = set()
    matched_predicted: set[int] = set()
    for iou, actual_index, pred_index in overlaps:
        if iou < threshold:
            break
        if actual_index not in matched_actual and pred_index not in matched_predicted:
            matched_actual.add(actual_index)
            matched_predicted.add(pred_index)
    true_positive = len(matched_actual)
    false_positive = len(predicted) - true_positive
    false_negative = len(actual) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else 0.0
    return f1, true_positive, false_positive, false_negative


def _transitions(labels: np.ndarray, runs: list[tuple[int, int]]) -> list[tuple[int, str, str]]:
    result = []
    for start, stop in runs:
        window = labels[start:stop]
        changes = np.flatnonzero(window[1:] != window[:-1]) + 1
        result.extend(
            (start + int(index), str(window[index - 1]), str(window[index])) for index in changes
        )
    return result


def _matched_switch_latencies(
    truth_transitions: list[tuple[int, str, str]],
    predicted_transitions: list[tuple[int, str, str]],
    tolerance: int,
) -> tuple[list[int], int, int]:
    actual = [
        value
        for value in truth_transitions
        if value[1] in POSSESSION_LABELS and value[2] in POSSESSION_LABELS and value[1] != value[2]
    ]
    predicted = [
        value
        for value in predicted_transitions
        if value[1] in POSSESSION_LABELS and value[2] in POSSESSION_LABELS and value[1] != value[2]
    ]
    by_destination = {
        label: [value for value in actual if value[2] == label] for label in POSSESSION_LABELS
    }
    predicted_by_destination = {
        label: [value for value in predicted if value[2] == label] for label in POSSESSION_LABELS
    }
    latencies: list[int] = []
    missed = spurious = 0
    for label in sorted(POSSESSION_LABELS):
        ground_truth = by_destination[label]
        guesses = predicted_by_destination[label]
        i = j = 0
        while i < len(ground_truth) and j < len(guesses):
            delta = guesses[j][0] - ground_truth[i][0]
            if delta < -tolerance:
                spurious += 1
                j += 1
            elif delta > tolerance:
                missed += 1
                i += 1
            else:
                latencies.append(delta)
                i += 1
                j += 1
        missed += len(ground_truth) - i
        spurious += len(guesses) - j
    return latencies, missed, spurious


def evaluate_synthetic(
    truth_labels: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    fps: int = 25,
    ball_visible: np.ndarray | None = None,
    switch_tolerance_frames: int = 12,
) -> dict[str, float | int | None]:
    """Return official-style frame/spell metrics and interpretable switch diagnostics."""
    truth = np.asarray(truth_labels, dtype="<U1")
    prediction = np.asarray(predicted_labels, dtype="<U1")
    if truth.ndim != 1 or prediction.shape != truth.shape:
        raise ValueError("truth and predictions must be same-length 1D label arrays")
    if fps <= 0 or switch_tolerance_frames < 0:
        raise ValueError("fps must be positive and switch tolerance cannot be negative")
    if set(truth.tolist()) - set(LABELS) or set(prediction.tolist()) - set(LABELS):
        raise ValueError("labels must be one of H, A, D, X")

    runs = _runs(truth)
    graded = truth != "X"
    n_graded = int(graded.sum())
    class_f1 = []
    for label in ("H", "A", "D"):
        true_positive = int(np.sum(graded & (truth == label) & (prediction == label)))
        false_positive = int(np.sum(graded & (truth != label) & (prediction == label)))
        false_negative = int(np.sum(graded & (truth == label) & (prediction != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1.append(2 * true_positive / denominator if denominator else 0.0)
    accuracy = float(np.mean(truth[graded] == prediction[graded])) if n_graded else 0.0

    filled = _fill_abstentions(prediction, runs)
    spell_results = [_spell_f1(truth, filled, runs, threshold) for threshold in SPELL_IOUS]
    gt_segments = len(_segments(truth, runs))
    pred_segments = len(_segments(filled, runs))
    gt_transitions = _transitions(truth, runs)
    pred_transitions = _transitions(filled, runs)
    latencies, missed_switches, false_switches = _matched_switch_latencies(
        gt_transitions, pred_transitions, switch_tolerance_frames
    )
    duration_minutes = n_graded / (fps * 60.0)

    missing_ball_accuracy: float | None = None
    if ball_visible is not None:
        visible = np.asarray(ball_visible, dtype=bool)
        if visible.shape != truth.shape:
            raise ValueError("ball_visible must align with label arrays")
        missing = graded & ~visible
        if missing.any():
            missing_ball_accuracy = float(np.mean(truth[missing] == prediction[missing]))

    dead_tp = int(np.sum(graded & (truth == "D") & (prediction == "D")))
    dead_fp = int(np.sum(graded & (truth != "D") & (prediction == "D")))
    dead_fn = int(np.sum(graded & (truth == "D") & (prediction != "D")))
    dead_denominator = 2 * dead_tp + dead_fp + dead_fn

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(class_f1)),
        "spell_f1_mean": float(np.mean([result[0] for result in spell_results])),
        "spell_f1_iou_0_5": spell_results[2][0],
        "spell_matched_at_iou_0_5": spell_results[2][1],
        "spell_spurious_at_iou_0_5": spell_results[2][2],
        "spell_missed_at_iou_0_5": spell_results[2][3],
        "false_switches_per_minute": (
            false_switches / duration_minutes if duration_minutes else 0.0
        ),
        "missed_switches": missed_switches,
        "incorrect_switches": false_switches,
        "boundary_timing_error_frames": (float(np.mean(np.abs(latencies))) if latencies else None),
        "fragmentation": pred_segments / gt_segments if gt_segments else 0.0,
        "ground_truth_segments": gt_segments,
        "predicted_segments": pred_segments,
        "missing_ball_accuracy": missing_ball_accuracy,
        "dead_ball_f1": (2 * dead_tp / dead_denominator if dead_denominator else 0.0),
        "graded_frames": n_graded,
    }
