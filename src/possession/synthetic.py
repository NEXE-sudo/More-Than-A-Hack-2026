"""Seeded, controlled tracking simulations for possession-system development."""

from dataclasses import dataclass

import numpy as np

from possession.data.validation import TrackingData
from possession.features.context import tagging_context
from possession.features.tracking import extract_tracking_features

CANONICAL_SCENARIOS = (
    "stable_home",
    "stable_away",
    "passes_turnovers",
    "interception",
    "contested_ball",
    "occlusion_aerial",
    "goalkeeper",
    "dead_ball_restart",
    "adversarial_tracks",
    "half_reset",
    "ball_dropout",
    "missing_player_tracks",
    "boundary_start",
    "boundary_end",
    "contradictory_evidence",
)
SCENARIO_ALIASES = {
    "stable_home_possession": "stable_home",
    "stable_away_possession": "stable_away",
    "home_to_away_turnover": "passes_turnovers",
    "away_to_home_turnover": "passes_turnovers",
    "turnovers": "passes_turnovers",
    "ball_tracking_dropout": "contested_ball",
    "long_dead_ball": "dead_ball_restart",
    "short_dead_ball": "dead_ball_restart",
    "dead_ball": "dead_ball_restart",
    "ball_occlusion": "occlusion_aerial",
    "long_ball_occlusion": "occlusion_aerial",
    "rapid_ball_movement": "passes_turnovers",
    "crowded_ball_area": "contested_ball",
    "ambiguous_proximity": "contested_ball",
    "brief_nearest_player_blip": "interception",
    "nearest_player_blip": "interception",
    "consecutive_dead_balls": "dead_ball_restart",
    "repeated_possessions": "passes_turnovers",
    "long_no_evidence": "half_reset",
    "no_evidence": "half_reset",
}
SCENARIOS = (*CANONICAL_SCENARIOS, *SCENARIO_ALIASES)
RESTART_TYPES = (
    "Throw-in",
    "Corner Kick",
    "Free Kick",
    "Goal Kick",
    "Kick Off (After Goal)",
    "Other",
)


@dataclass(frozen=True)
class SyntheticSequence:
    family: str
    seed: int
    features: dict[str, np.ndarray]
    labels: np.ndarray
    sequence_starts: np.ndarray


def _possession_schedule(family: str, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(n_frames, "H", dtype="<U1")
    starts = np.zeros(n_frames, dtype=bool)

    def fill(start_fraction: float, end_fraction: float, label: str) -> None:
        start = round(start_fraction * n_frames)
        stop = round(end_fraction * n_frames)
        labels[start:stop] = label

    if family == "stable_away":
        labels[:] = "A"
    elif family == "passes_turnovers":
        fill(0.28, 0.56, "A")
        fill(0.56, 0.78, "H")
        fill(0.78, 1.0, "A")
    elif family == "interception":
        fill(0.45, 1.0, "A")
    elif family == "contested_ball":
        fill(0.40, 0.43, "X")
        fill(0.43, 0.72, "A")
        fill(0.72, 0.75, "X")
        fill(0.75, 1.0, "H")
    elif family == "occlusion_aerial":
        fill(0.50, 1.0, "A")
    elif family == "dead_ball_restart":
        fill(0.30, 0.48, "D")
        fill(0.48, 1.0, "A")
    elif family == "adversarial_tracks":
        fill(0.72, 1.0, "A")
    elif family == "half_reset":
        fill(0.48, 0.52, "X")
        fill(0.52, 1.0, "A")
        starts[min(n_frames - 1, round(0.52 * n_frames))] = True
    elif family == "boundary_start":
        labels[round(0.08 * n_frames) :] = "A"
    elif family == "boundary_end":
        labels[round(0.92 * n_frames) :] = "A"
    return labels, starts


def generate_synthetic_sequence(
    family: str,
    seed: int,
    *,
    duration_seconds: float = 20.0,
    fps: int = 25,
    pitch_length_m: float = 105.3,
    pitch_width_m: float = 68.0,
    players_per_team: int = 8,
    player_availability: float = 0.94,
    ball_visibility: float = 0.70,
    coordinate_noise_m: float = 0.45,
    restart_type: str = "Corner Kick",
) -> SyntheticSequence:
    """Generate a reproducible tracking sequence without exposing labels as model features."""
    normalized_family = SCENARIO_ALIASES.get(family, family)
    if normalized_family not in CANONICAL_SCENARIOS:
        raise ValueError(
            f"unknown synthetic scenario {family!r}; choose from {CANONICAL_SCENARIOS} "
            f"and aliases {sorted(SCENARIO_ALIASES)}"
        )
    if restart_type not in RESTART_TYPES:
        raise ValueError(f"unknown restart type {restart_type!r}; choose from {RESTART_TYPES}")
    if (
        not np.isfinite(duration_seconds)
        or duration_seconds <= 0
        or fps <= 0
        or players_per_team < 1
    ):
        raise ValueError("duration, fps, and players_per_team must be positive")
    if (
        not np.isfinite([pitch_length_m, pitch_width_m, coordinate_noise_m]).all()
        or pitch_length_m <= 0
        or pitch_width_m <= 0
        or coordinate_noise_m < 0
    ):
        raise ValueError("pitch dimensions must be positive and noise cannot be negative")
    for name, value in (
        ("player_availability", player_availability),
        ("ball_visibility", ball_visibility),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")

    rng = np.random.default_rng(seed)
    n_frames = max(1, round(duration_seconds * fps))
    n_tracks = 2 * players_per_team
    family_name = normalized_family
    labels, sequence_starts = _possession_schedule(family_name, n_frames)
    players = np.full((n_frames, n_tracks, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    speeds = np.full((n_frames, n_tracks), np.nan, dtype=np.float32)
    teams = np.repeat(np.arange(2, dtype=np.int8), players_per_team)
    scales = np.asarray([pitch_length_m, pitch_width_m])
    center = np.asarray([pitch_length_m * 0.5, pitch_width_m * 0.5])
    state = center.copy()
    formation_offsets = rng.normal(0.0, 9.0, (n_tracks, 2))
    player_world = center + formation_offsets
    defending_slots = rng.integers(0, players_per_team, size=2)
    active_slots = np.full(2, -1, dtype=np.int64)
    previous_label = ""
    aerial_begin, aerial_end = int(0.38 * n_frames), int(0.55 * n_frames)
    restart_owner = int(rng.integers(0, 2))
    pass_windows: list[tuple[int, int, np.ndarray]] = []
    if family_name == "passes_turnovers":
        switch_frames = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        for switch_frame in switch_frames:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            distance = rng.uniform(6.0, 25.0)
            pass_speed_m_s = rng.uniform(12.0, 24.0)
            pass_duration = max(2, round(distance / pass_speed_m_s * fps))
            pass_start = max(0, int(switch_frame) - pass_duration)
            pass_stop = min(n_frames, int(switch_frame) + 1)
            delta = distance * np.asarray([np.cos(angle), np.sin(angle)])
            pass_windows.append((pass_start, pass_stop, delta))
    active_pass: tuple[int, int, np.ndarray, np.ndarray] | None = None

    for frame, label in enumerate(labels):
        if sequence_starts[frame]:
            state = center.copy()
            formation_offsets = rng.normal(0.0, 9.0, (n_tracks, 2))
            player_world = center + formation_offsets
            active_slots[:] = -1
            previous_label = ""
            active_pass = None
        if active_pass is not None and frame >= active_pass[1]:
            active_pass = None
        for pass_start, pass_stop, delta in pass_windows:
            if frame == pass_start:
                active_pass = (pass_start, pass_stop, delta, state.copy())
                break
        if active_pass is not None:
            pass_start, pass_stop, delta, origin = active_pass
            progress = (frame - pass_start) / max(1, pass_stop - pass_start - 1)
            state = origin + progress * delta
        else:
            state += rng.normal(0.0, [0.22, 0.16])
        state = np.clip(
            state,
            [pitch_length_m * 0.12, pitch_width_m * 0.12],
            [pitch_length_m * 0.88, pitch_width_m * 0.88],
        )
        is_dead = label == "D"
        active_team = 0 if label == "H" else 1 if label == "A" else -1
        if active_team >= 0 and label != previous_label:
            active_slots[active_team] = int(rng.integers(0, players_per_team))
        previous_label = str(label)
        formation_offsets += rng.normal(0.0, 0.025, formation_offsets.shape)
        targets = state + formation_offsets
        for team in (0, 1):
            team_offset = team * players_per_team
            target_slot = int(defending_slots[team])
            angle = 0.8 + team * 1.9
            radius = rng.uniform(4.0, 8.0)
            if is_dead:
                radius = rng.uniform(1.0, 5.0)
            elif label == "X" and family_name == "contested_ball":
                radius = rng.uniform(0.5, 2.0)
            targets[team_offset + target_slot] = state + radius * np.asarray(
                [np.cos(angle), np.sin(angle)]
            )
            if team == active_team and active_pass is None and active_slots[team] >= 0:
                controller_offset = rng.normal(0.0, max(0.25, coordinate_noise_m), 2)
                targets[team_offset + active_slots[team]] = state + controller_offset
            elif active_pass is not None:
                targets[team_offset + target_slot] = state + radius * np.asarray(
                    [np.cos(angle), np.sin(angle)]
                )
        movement = targets - player_world
        movement_distance = np.linalg.norm(movement, axis=1)
        player_speeds = (
            rng.uniform(0.05, 0.35, n_tracks) if is_dead else rng.uniform(0.5, 4.2, n_tracks)
        )
        maximum_step = player_speeds / fps
        step_scale = np.divide(
            np.minimum(movement_distance, maximum_step),
            movement_distance,
            out=np.zeros_like(movement_distance),
            where=movement_distance > 0,
        )
        player_world += movement * step_scale[:, None]
        if movement_distance.max(initial=0.0) > 0 and frame == 0:
            player_world = targets.copy()
        visibility = rng.random() < (0.08 if is_dead else ball_visibility)
        if family_name == "occlusion_aerial" and n_frames // 2 <= frame < min(
            n_frames, n_frames // 2 + fps * 2
        ):
            visibility = False
        if visibility:
            observed_ball = state + rng.normal(0.0, coordinate_noise_m, size=2)
            if family_name == "occlusion_aerial" and aerial_begin <= frame < aerial_end:
                observed_ball += rng.normal(0.0, [10.0, 7.0])
            ball[frame] = np.clip(observed_ball / scales, 0.01, 0.99)

        for team in (0, 1):
            team_slice = slice(team * players_per_team, (team + 1) * players_per_team)
            locations = player_world[team_slice] + rng.normal(
                0.0, coordinate_noise_m, (players_per_team, 2)
            )
            locations = np.clip(locations / scales, 0.01, 0.99)
            present = rng.random(players_per_team) < player_availability
            team_speed = player_speeds[team_slice]

            if (
                family_name == "adversarial_tracks"
                and label == "H"
                and frame % max(1, fps * 5) < 2
                and team == 1
            ):
                locations[0] = np.clip(state / scales, 0.01, 0.99)
                present[0] = True
            if family_name == "contradictory_evidence" and label == "H" and team == 1:
                locations[0] = np.clip(state / scales, 0.01, 0.99)
                present[0] = True

            players[frame, team_slice][present] = locations[present]
            speeds[frame, team_slice][present] = team_speed[present]

        if family_name == "goalkeeper" and visibility:
            ball[frame] = np.clip(
                np.asarray([0.035, 0.50]) + rng.normal(0, coordinate_noise_m, 2) / scales,
                0.01,
                0.99,
            )
            players[frame, 0] = np.clip(ball[frame] + rng.normal(0, 0.004, 2), 0.01, 0.99)

    if family_name == "ball_dropout":
        start = round(0.35 * n_frames)
        stop = max(start + 1, round(0.55 * n_frames))
        ball[start:stop] = np.nan
    if family_name == "missing_player_tracks":
        start = round(0.35 * n_frames)
        stop = max(start + 1, round(0.55 * n_frames))
        away_tracks = teams == 1
        players[start:stop, away_tracks] = np.nan
        speeds[start:stop, away_tracks] = np.nan

    tracking = TrackingData(
        frame_ids=np.arange(n_frames, dtype=np.int64),
        players=players,
        ball=ball,
        player_teams=teams,
        player_speed=speeds,
    )
    features = extract_tracking_features(
        tracking,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        fps=fps,
    )

    events: list[dict[str, object]] = []
    if family_name == "dead_ball_restart":
        start = round(0.30 * n_frames)
        end = round(0.55 * n_frames)
        team_name = (
            "N/A"
            if restart_type == "Throw-in"
            else "Barcelona"
            if restart_owner == 0
            else "Atletic Club"
        )
        events.append(
            {
                "code": "SET PIECES",
                "start_frame": start,
                "end_frame": end,
                "tags": {"Team": team_name, "Type": restart_type},
            }
        )
    features.update(tagging_context(events, n_frames))
    return SyntheticSequence(family_name, seed, features, labels, sequence_starts)
