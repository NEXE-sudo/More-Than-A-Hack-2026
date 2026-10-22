"""Small helpers for adding Smart Tagging interval context without treating it as truth."""

import numpy as np


def tagging_context(events: list[dict[str, object]], n_frames: int) -> dict[str, np.ndarray]:
    """Return phase, set-piece, and tagged-team masks indexed by tactical frame."""
    phase = np.zeros(n_frames, dtype=bool)
    set_piece = np.zeros(n_frames, dtype=bool)
    tagged_home = np.zeros(n_frames, dtype=bool)
    tagged_away = np.zeros(n_frames, dtype=bool)
    for event in events:
        start = max(0, int(event["start_frame"]))
        end = min(n_frames, int(event["end_frame"]))
        if end <= start:
            continue
        phase[start:end] = True
        if str(event["code"]).strip().casefold() == "set pieces":
            set_piece[start:end] = True
        tags = event["tags"]
        if not isinstance(tags, dict):
            continue
        team = str(tags.get("Team", "")).strip().casefold()
        if team == "barcelona":
            tagged_home[start:end] = True
        elif team in {"athletic club", "atletic club"}:
            tagged_away[start:end] = True
    return {
        "tagged_phase": phase,
        "tagged_set_piece": set_piece,
        "tagged_home": tagged_home,
        "tagged_away": tagged_away,
    }
