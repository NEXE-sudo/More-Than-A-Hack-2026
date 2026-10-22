"""Deterministic hysteresis decoder for possession evidence sequences."""

from collections.abc import Sequence

import numpy as np

from possession.config import DecoderConfig

TEAM_LABELS = ("H", "A")


def decode(
    scores: np.ndarray,
    config: DecoderConfig = DecoderConfig(),
    sequence_starts: Sequence[bool] | None = None,
) -> np.ndarray:
    """Decode [home, away, dead] scores with explicit gaps and switch persistence.

    Scores are evidence in [0, 1], not probabilities. A short gap holds the previous state;
    longer gaps produce X instead of retaining possession indefinitely. Sequence starts reset
    all state. Ties between teams are resolved in favor of home for reproducibility.
    """
    evidence = np.asarray(scores, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[1] != 3:
        raise ValueError("scores must have shape (frames, 3) in H,A,D order")
    if not np.isfinite(evidence).all() or ((evidence < 0) | (evidence > 1)).any():
        raise ValueError("scores must be finite values in [0, 1]")
    n_frames = len(evidence)
    starts = np.zeros(n_frames, dtype=bool)
    if n_frames:
        starts[0] = True
    if sequence_starts is not None:
        supplied = np.asarray(sequence_starts, dtype=bool)
        if supplied.shape != (n_frames,):
            raise ValueError("sequence_starts must have one value per frame")
        starts |= supplied
    output = np.full(n_frames, "X", dtype="<U1")
    current: str | None = None
    missing_run = 0
    pending: str | None = None
    pending_run = 0

    for frame in range(n_frames):
        if starts[frame]:
            current = None
            missing_run = 0
            pending = None
            pending_run = 0
        home, away, dead = evidence[frame]
        best_team = "H" if home >= away else "A"
        team_score = max(home, away)
        dead_wins = dead >= config.dead_threshold and dead >= team_score + config.dead_margin
        if dead_wins:
            current = "D"
            missing_run = 0
            pending = None
            pending_run = 0
            output[frame] = "D"
            continue

        if current not in TEAM_LABELS and team_score < config.start_threshold:
            missing_run += 1
            output[frame] = "X"
            continue

        if current not in TEAM_LABELS:
            current = best_team
            missing_run = 0
            output[frame] = current
            continue
        if team_score < config.continue_threshold:
            missing_run += 1
            pending = None
            pending_run = 0
            if missing_run <= config.missing_hold_frames:
                output[frame] = current
            else:
                current = None
                output[frame] = "X"
            continue

        missing_run = 0
        if best_team == current:
            pending = None
            pending_run = 0
            output[frame] = current
            continue

        current_score = home if current == "H" else away
        if (
            team_score >= config.switch_threshold
            and team_score - current_score >= config.switch_margin
        ):
            if pending == best_team:
                pending_run += 1
            else:
                pending = best_team
                pending_run = 1
            if pending_run >= config.switch_frames:
                current = best_team
                pending = None
                pending_run = 0
        else:
            pending = None
            pending_run = 0
        output[frame] = current
    return output
