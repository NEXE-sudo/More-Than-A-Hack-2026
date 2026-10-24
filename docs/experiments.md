# Experiments and evaluation

## Official metric contract

The metric implementation was checked in the unchanged sibling repository at
`../more-than-a-hack-2027/evaluation/possession_metrics.py` and its `EVALUATION.md`.

- Ground-truth `X` frames are outside scored runs. Prediction `X` on a graded frame is a
  frame-level miss.
- For temporal metrics, predicted X is forward-filled with the previous non-X label within each
  ground-truth non-X run, then back-filled from the first prediction at the run head. An entirely-X
  run stays X. Ground-truth X gaps split temporal runs, including halftime.
- A possession spell is a maximal H or A run. D is excluded from possession spells.
- Same-team spells are greedily matched one-to-one in descending temporal-IoU order at 0.3, 0.4,
  0.5, 0.6 and 0.7. `spellF1` is the mean F1 over these thresholds.
- The synthetic helper implements these rules. Its hand-built tests cover empty sequences,
  one-frame spells, abstention fills, ungraded gaps and fragmented/overlapping spells.

## Synthetic harness and reproducible command

The local generator creates frame-aligned labels independently of the feature dictionary and
deterministically generates tracking positions from explicit seeds. The canonical registry is
listed below. These are controlled tracking fixtures, not validated representations of real match
frequency or football dynamics.

| Canonical scenario | Generated condition |
|---|---|
| `stable_home` | All frames are graded home possession with ordinary tracking noise. |
| `stable_away` | All frames are graded away possession with ordinary tracking noise. |
| `passes_turnovers` | H-A-H-A possession spells with a moving-ball window before each change. |
| `home_to_away_turnover` | One H-to-A possession change at sequence midpoint. |
| `away_to_home_turnover` | One A-to-H possession change at sequence midpoint. |
| `contested_ball` | Two short X intervals with both teams close to the ball. |
| `occlusion_aerial` | An H-to-A change plus a fixed ball-occlusion interval and noisy aerial track. |
| `home_goal_area_possession` | Home possession near its goal; home track slot 0 is a proximity proxy, not a labeled role. |
| `dead_ball_restart` | One D interval followed by an away restart. |
| `short_dead_ball` | One short D interval followed by an away restart. |
| `long_dead_ball` | One longer D interval followed by an away restart. |
| `consecutive_dead_balls` | Two D intervals separated by an away live-possession restart. |
| `adversarial_tracks` | Periodic away-track proximity conflicts during home possession. |
| `half_reset` | An X boundary interval and decoder reset before away possession. |
| `long_no_evidence` | An X interval with ball and all player observations unavailable. |
| `ball_dropout` | A missing-ball interval while player observations and H labels continue. |
| `missing_player_tracks` | Away player tracks are absent in one interval; ball and home tracks remain. |
| `boundary_start` | One H-to-A possession change near sequence start. |
| `boundary_end` | One H-to-A possession change near sequence end. |
| `contradictory_evidence` | Persistent away proximity conflicts with H ground-truth labels. |
| `rapid_ball_movement` | A pitch-safe ball movement targeting 12 m/s for 0.5 seconds during uninterrupted H possession. |
| `brief_nearest_player_blip` | A brief away-nearest-player observation during H possession. |

The following accepted names are true aliases; generation is identical to the stated canonical
target under identical seed and configuration:

| Alias | Canonical target |
|---|---|
| `stable_home_possession` | `stable_home` |
| `stable_away_possession` | `stable_away` |
| `turnovers` | `passes_turnovers` |
| `ball_tracking_dropout` | `ball_dropout` |
| `dead_ball` | `dead_ball_restart` |
| `ball_occlusion` | `occlusion_aerial` |
| `long_ball_occlusion` | `occlusion_aerial` |
| `crowded_ball_area` | `contested_ball` |
| `ambiguous_proximity` | `contested_ball` |
| `nearest_player_blip` | `brief_nearest_player_blip` |
| `repeated_possessions` | `passes_turnovers` |
| `no_evidence` | `long_no_evidence` |

Directional turnover, dead-ball duration/count, long no-evidence, rapid-ball-movement, and nearest
player-blip names are distinct canonical scenarios, not aliases. Names that previously redirected
to an unrelated family have been corrected to generate their promised condition. `interception` is
deprecated because this generator does not encode an event subtype; use `home_to_away_turnover` for
the supported H-to-A label transition. `goalkeeper` is deprecated because no player-role identity
is represented; use `home_goal_area_possession` for the available proximity proxy. Both deprecated
names fail explicitly rather than silently changing condition. Parameters include duration, frame
rate, pitch dimensions, player count, track availability, ball visibility and coordinate noise. The
restart scenario supports throw-ins, corners, free kicks, goal kicks, kick-offs and other stoppages;
throw-ins deliberately carry `Team=N/A`. For other restart types, set-piece team context matches the
team holding the restart in ground truth.

The rapid-movement fixture starts at 45% of the sequence and constructs a straight-line trajectory
over `round(0.5 * fps)` frame intervals. At the default 25 fps this is 12 intervals (0.48 seconds)
and 5.76 metres of travel, for 12 m/s.
The direction is chosen along the pitch axis with the most room, and travel is capped to retain a
margin inside the generator's 12%-88% pitch bounds. On unusually small pitches or near-boundary
starting states, that cap can reduce the target speed. `SyntheticSequence.ball_state_m` exposes the
resulting ground-truth trajectory for simulator tests only; it is separate from `features` and is
never passed to the model. Model input still uses intermittently visible, coordinate-noisy ball
observations. Consequently, observed frame-to-frame speed can have noise spikes and is not itself
a ground-truth speed oracle. This controlled impulse validates the feature path, not the frequency
or realism of such passes in matches.

Run the complete fixed matrix:

```bash
possession run-synthetic-experiments --output-dir outputs/synthetic-experiments
```

Optional `--seeds`, `--scenarios`, and `--duration-seconds` arguments narrow or extend the run.
The default matrix uses seeds 11, 29 and 47; 22 canonical scenario families; and nine
configurations:

1. baseline model + baseline decoder,
2. revised model + baseline decoder,
3. baseline model + revised decoder,
4. revised model + revised decoder,
5. revised-model ablation without proximity evidence,
6. revised-model ablation without local-support evidence,
7. revised-model ablation without approach evidence,
8. revised evidence with synthetic Smart Tagging context disabled, and
9. revised evidence with the same context enabled.

Each row records its seed, canonical scenario, full configuration, model/decoder version, sample
count, runtime, warnings, frame and spell metrics, transition diagnostics, and abstention/missing-
ball behavior. CSV and JSON are machine-readable; the Markdown report states requested names,
effective canonical families, aliases, seeds, duration, and all configurations. Means and standard
deviations are pooled across scenario/seed runs, not frames. Alias inputs normalize only to their
declared true-equivalent canonical family; distinct requested conditions are never substituted.
Outputs are local and ignored by Git.

## Synthetic results (not competition performance)

Results from the canonical run per configuration (22 families × 3 seeds):

| Configuration | macro-F1 mean ± SD | spellF1 mean ± SD | switches/min |
|---|---:|---:|---:|
| Baseline model + baseline decoder | 0.358 ± 0.149 | 0.640 ± 0.320 | 2.40 |
| Revised model + baseline decoder | 0.328 ± 0.154 | 0.614 ± 0.352 | 1.83 |
| Baseline model + revised decoder | 0.362 ± 0.150 | 0.645 ± 0.318 | 2.50 |
| Revised model + revised decoder | 0.335 ± 0.155 | 0.630 ± 0.350 | 1.89 |
| Revised without proximity | 0.010 ± 0.034 | 0.000 ± 0.000 | 0.00 |
| Revised without local support | 0.355 ± 0.143 | 0.641 ± 0.329 | 2.29 |
| Revised without approach | 0.339 ± 0.151 | 0.644 ± 0.345 | 1.77 |
| Revised, Smart Tagging off | 0.335 ± 0.155 | 0.630 ± 0.350 | 1.89 |
| Revised, Smart Tagging on | 0.335 ± 0.155 | 0.630 ± 0.350 | 1.89 |

This does not demonstrate a robust improvement: the revised evidence model regressed on the
aggregate synthetic spell score, and the revised decoder's switch-contest handling did not improve
aggregate spellF1. Smart Tagging's aggregate scores were unchanged. Means and sample standard
deviations are pooled descriptively across scenario/seed cases, which are not independent samples
from real matches or confidence intervals. Per-family variability is substantial (stable
single-team cases score perfectly on spellF1, while restart, transition and half-reset cases are
harder). Keep the baseline defaults. Do not tune from these few stylized sequences or interpret
them as predictions for the match.

## Public-window evaluation

The unchanged official evaluator was run on the four existing public baseline CSVs using
`--ground-truth public --bootstrap 100`. It reproduced these scoreboard values over the public
5-minute window (7,500 graded frames; 17 annotated possession spells):

| Public baseline | macroF1 | spellF1 |
|---|---:|---:|
| smooth-125 | 0.5812 | 0.4242 |
| smooth-50 | 0.6207 | 0.3256 |
| smooth-25 | 0.6200 | 0.2868 |
| raw | 0.5972 | 0.0378 |

These are scores of existing official baseline predictions, not this project's candidate model.
The small public sample is too small to establish robust generalization. The configured tracking
bundle is absent, so candidate feature extraction, candidate prediction, full-match evaluation, and
real-data runtime/memory measurements remain unverified.

Reproduce the public-only check without writing to the official checkout:

```bash
possession evaluate \
  --repo-root /path/to/more-than-a-hack-2027 \
  --results-dir /path/to/more-than-a-hack-2027/evaluation/results \
  --output-dir outputs/public-evaluation --ground-truth public --bootstrap 100
```

## Next experiment when gated data is available

First regenerate the official baseline from the authorized ATD bundle and compare frame alignment
and labels against the repository's reference output. Then evaluate the baseline and each isolated
candidate/ablation with the unchanged official `spellF1` scorer, segment-level uncertainty, and
half-aware temporal diagnostics on held-out contiguous match segments. Do not choose thresholds
from the 17-spell public slice. Include Smart Tagging only as an ablation, measuring whether its
small context prior helps beyond tracking-only evidence.
