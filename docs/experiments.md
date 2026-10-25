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
Outputs are local and ignored by Git. Because reports may contain data-derived information, do not
publish or redistribute reports generated from gated match inputs.

## Synthetic results (not competition performance)

Results from the canonical run per configuration (22 families × 3 seeds):

| Configuration | macro-F1 mean ± SD | spellF1 mean ± SD | switches/min |
|---|---:|---:|---:|
| Baseline model + baseline decoder | 0.362 ± 0.147 | 0.661 ± 0.320 | 2.29 |
| Revised model + baseline decoder | 0.332 ± 0.153 | 0.634 ± 0.354 | 1.72 |
| Baseline model + revised decoder | 0.366 ± 0.148 | 0.665 ± 0.318 | 2.38 |
| Revised model + revised decoder | 0.339 ± 0.154 | 0.650 ± 0.351 | 1.77 |
| Revised without proximity | 0.010 ± 0.034 | 0.000 ± 0.000 | 0.00 |
| Revised without local support | 0.359 ± 0.141 | 0.661 ± 0.330 | 2.18 |
| Revised without approach | 0.343 ± 0.150 | 0.664 ± 0.345 | 1.66 |
| Revised, Smart Tagging off | 0.339 ± 0.154 | 0.650 ± 0.351 | 1.77 |
| Revised, Smart Tagging on | 0.339 ± 0.154 | 0.650 ± 0.351 | 1.77 |

This does not demonstrate a robust improvement: the revised evidence model regressed on the
aggregate synthetic spell score, and the revised decoder's switch-contest handling did not improve
aggregate spellF1. Smart Tagging's aggregate scores were unchanged. Means and sample standard
deviations are pooled descriptively across scenario/seed cases, which are not independent samples
from real matches or confidence intervals. Per-family variability is substantial (stable
single-team cases score perfectly on spellF1, while restart, transition and half-reset cases are
harder). Keep the baseline defaults. Do not tune from these few stylized sequences or interpret
them as predictions for the match.

## Public-window evaluation

The unchanged official evaluator was run on the four generated official baseline CSVs and this
project's generated candidate using `--ground-truth public`. The public ground truth covers only the
opening 5-minute window (7,500 graded frames; 17 annotated possession spells):

| Submission | macroF1 | spellF1 |
|---|---:|---:|
| Official baseline smooth-125 | 0.5812 | 0.4242 |
| Official baseline smooth-50 | 0.6207 | 0.3256 |
| Official baseline smooth-25 | 0.6200 | 0.2868 |
| Candidate | 0.6417 | 0.2349 |
| Official baseline raw | 0.5972 | 0.0378 |

### Decoder-only public-window diagnosis

The unchanged official scorer was run with `--bootstrap 1000` over the same public slice for every
row below. It contains 7,500 graded frames and only 17 H/A possession spells. Decoder variants
used one source-bound feature cache, so feature extraction was held fixed. The candidate source
snapshot was `16b6d90e20229dbff60eb924a07fba207f4db732` plus the worktree diff fingerprint
`2880e08b4cf0f4c940c8b326dcfa394de7ad6985f7ce4d9d3d92f077d8f95510`. The unchanged official
baseline was at `3de6b60aeabd2475d95879a651d38b72bc4fba00`. The default cached prediction was
byte-identical to the retained full-pipeline candidate.

Transition F1 is shown at both official tolerances (12 and 25 frames); transition TP/FN/FP counts
are at the 25-frame tolerance. Latency is the official mean signed `predicted transition frame -
ground-truth transition frame`; a negative value means the matched prediction was earlier. Latency
uses only matched transitions, which are few for these runs. Spell duration/count are measured
after the official within-run X fill. Candidate runtimes are cached prediction wall times and
exclude feature-cache generation; the four baselines were generated together in 3.67 seconds.

| Run / single change from default | macroF1 | spellF1 (5–95% bootstrap) | transF1 @12 / @25 | TP/FN/FP @25 | mean latency @25 (frames) | predicted spells / mean duration (s) | runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official baseline raw | 0.5972 | 0.0378 (0.0189–0.0595) | 0.0641 / 0.1410 | 11/9/125 | -3.1 | 131 / 2.02 | 3.67 shared |
| Official baseline smooth-25 | 0.6200 | 0.2868 (0.1962–0.3849) | 0.1695 / 0.2034 | 6/14/33 | +8.2 | 36 / 7.58 | 3.67 shared |
| Official baseline smooth-50 | 0.6207 | 0.3256 (0.2140–0.4465) | 0.0816 / 0.0816 | 2/18/27 | +4.0 | 26 / 10.80 | 3.67 shared |
| Official baseline smooth-125 | 0.5812 | 0.4242 (0.2667–0.5939) | 0.1111 / 0.1667 | 3/17/13 | -7.7 | 16 / 18.46 | 3.67 shared |
| Candidate default, `switch_frames=3` | 0.6417 | 0.2349 (0.1651–0.3051) | 0.1159 / 0.2029 | 7/13/42 | -5.3 | 46 / 6.03 | 1.64 cached |
| `switch_frames=1` | 0.6403 | 0.2349 (0.1651–0.3051) | 0.1159 / 0.2029 | 7/13/42 | -6.4 | 46 / 6.03 | 1.60 |
| `switch_frames=5` | 0.6518 | 0.2623 (0.1836–0.3410) | 0.1194 / 0.2090 | 7/13/40 | +0.4 | 44 / 6.31 | 1.52 |
| `switch_frames=8` | 0.6673 | 0.3750 (0.2750–0.4667) | 0.1481 / 0.2963 | 8/12/26 | -7.0 | 31 / 8.95 | 1.48 |
| `switch_margin=0.25` | 0.6518 | 0.2915 (0.2034–0.3797) | 0.1538 / 0.1846 | 6/14/39 | -7.2 | 42 / 6.61 | 1.66 |
| `missing_hold_frames=0` | 0.6044 | 0.2277 (0.1600–0.2957) | 0.1127 / 0.1972 | 7/13/44 | -6.0 | 48 / 5.78 | 1.52 |
| `dead_threshold=0.90` | 0.4928 | 0.2164 (0.1508–0.2885) | 0.0635 / 0.1587 | 5/15/38 | -3.0 | 44 / 6.82 | 1.60 |
| `strategy=revised` only | 0.6417 | 0.2349 (0.1651–0.3051) | 0.1159 / 0.2029 | 7/13/42 | -5.3 | 46 / 6.03 | 1.60 |

All scored runs completed without errors. The current candidate has 7 matched, 13 missed and 42
spurious transitions at the 25-frame tolerance. At spell IoU 0.5 it matches 8 of 17 ground-truth
spells, misses 9 and produces 38 spurious spells. Its 46 scored H/A spells average 6.03 seconds
(median 3.54 seconds), versus 17 ground-truth spells averaging 14.19 seconds. The smooth-125
control matches 7 spells at IoU 0.5, with 10 missed and 9 spurious; it predicts 16 spells (18.46
seconds mean, 12.30 seconds median). The `switch_frames=8` variant matches 10, misses 7 and has 21
spurious spells at IoU 0.5; its 31 spells average 8.95 seconds (median 4.64 seconds). This supports
excess possession fragmentation in the current output and a public-window sensitivity to switch
persistence; it does not establish the correct full-match persistence value.

The official temporal scorer fills X forward and then backfills at the start of each graded
ground-truth run. In the current candidate this reduces raw H/A spell fragments from 68 to 46;
the remaining over-count therefore is not explained by X boundaries alone. The 1,063 predicted X
frames on graded frames still count as frame-level errors. D is a dead-ball label, not a draw or
tie class. On the graded frames the candidate predicts D on 506 frames, of which 441 are true D:
precision 0.872 and recall 0.300 against 1,469 ground-truth D frames. Raising `dead_threshold` to
0.90 makes D impossible because the configured dead-ball evidence score is 0.85, and the observed
D F1 falls to zero. Exact home/away evidence ties on the graded frames all have both team scores
zero; none reaches the start threshold, so the decoder's deterministic home tie-break is not
evidence of a cause for this public-window result. Tied frames instead follow the no-evidence,
dead-ball, or previous-state logic.

The `switch_frames=8` result raises spellF1 from 0.2349 to 0.3750 and macroF1 from 0.6417 to
0.6673 on this slice, while remaining below smooth-125 on spellF1 (0.4242). Its spell bootstrap
interval overlaps both the current candidate and smooth-125 intervals. Consequently this
experiment is diagnostic, not a basis to change the default. Keep the current configuration and
full-match output as the reproducible reference until independent full-match labels or additional
held-out matches support a choice. No full-match accuracy is known.

### Interval diagnosis, dead-ball audit, and persistence hypothesis

The following is a public-window-only audit against the unchanged scorer's official X-filled
spell matching. It uses the same source-bound feature cache as the decoder sweep. These are
verified prediction/label relationships, not verified explanations of the underlying football
events:

| Default candidate finding | Evidence |
|---|---|
| Unmatched predicted spells at IoU 0.5 | 38 total: 17 have same-team overlap but IoU below 0.5; 19 overlap most strongly with a different-team ground-truth spell; 2 lie wholly inside ground-truth D intervals and have no H/A spell overlap. |
| Missed ground-truth spells at IoU 0.5 | 9 total: 5 have a same-team overlapping prediction below IoU 0.5; 4 are predominantly opposite-team, very short, or mixed-label overlaps. |
| Matched-spell boundaries | Eight matched spells still show boundary error; among them, predicted starts range from 12 frames early to 427 frames late, and stops range from 235 frames early to 17 frames late. |
| Transition localization | 49 predicted versus 20 ground-truth transitions; 42 are spurious at the 25-frame tolerance, and 13 ground-truth transitions are missed. Of the 42 spurious transitions, 39 are more than 25 frames from any ground-truth transition. The ball is visible and the current ball-gap feature is zero at 39 of those 42 transitions. |
| Abstentions and spell matching | Official X filling reduces raw candidate H/A spells from 68 to 46. At IoU 0.5, raw predictions score 4 matched / 64 spurious / 13 missed; filled predictions score 8 / 38 / 9. X filling removes many temporal fragments, but the remaining excess is not explained by X gaps alone. Graded X predictions remain frame-level errors. |

The interval evidence strongly supports excess switching/fragmentation as a scored failure mode:
most false predicted spells overlap a different-team or same-team ground-truth interval, and most
spurious transitions occur well inside annotated label runs while the ball is visible. This does
not prove those transitions are physically false; the labels identify scored boundaries, not
frame-level possession evidence. The matched spells also show substantial late starts and early
stops, so transition persistence cannot by itself repair every segmentation error. The two
predicted spells wholly inside GT D are verified dead-ball-period errors. Their local feature
summaries show no dead cue; one occurs with the ball visible, and the other has a maximum ball gap
of 24 frames against a 25-frame minimum. No Smart Tagging interpretation was used.

The dead-ball cue is exactly 0.85 when all of these conditions hold: ball gap at least 25 frames,
finite mean speed for both teams no greater than 0.5 m/s, and at least four tracked players for
each team. Missing ball alone is not sufficient. Among the 1,469 public GT D frames, 548 meet the
gap condition, 1,158 meet the two-team quiet-speed condition, all 1,469 meet the track-count
condition, and 441 meet the full conjunction. The 441 qualifying GT D frames also pass the
decoder's 0.75 threshold and 0.15 team-margin check and are predicted D. The other 1,028 GT D
frames are predicted H (400), A (190), or X (438). There are also 65 false-positive D frames
outside GT D. Thus D precision is 0.872 and recall is 0.300; the dead threshold is reachable at
0.75 but a value above the maximum 0.85 evidence score makes D unreachable. The principal
observed gates on GT D are ball-gap duration and the quiet-speed condition, not track coverage or
the decoder margin. Missed D may contribute to some spell-boundary errors, but cannot explain
most spurious transitions, which occur with a visible ball and zero gap.

**Falsifiable hypothesis:** increasing only `switch_frames` from 3 to 8 requires a longer
continuous opponent-favored evidence run, reducing short-lived false switches while retaining
genuine changes. The hypothesis would be weakened if false spells and spurious transitions did
not decline, or if the reduction came with a material increase in missed GT spells/transitions.
With the feature cache and all other configuration values fixed, the public-window result was:

| Variant | macroF1 | spellF1 (5–95% bootstrap) | transF1 @12 / @25 | TP/FN/FP @25 | IoU .5 spells: matched/missed/spurious | predicted spells; mean / median duration | Cached prediction runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default, `switch_frames=3` | 0.6417 | 0.2349 (0.1651–0.3051) | 0.1159 / 0.2029 | 7/13/42 | 8/9/38 | 46; 6.03 / 3.54 s | 0.999 / 0.922 s |
| Experiment, `switch_frames=8` | 0.6673 | 0.3750 (0.2750–0.4667) | 0.1481 / 0.2963 | 8/12/26 | 10/7/21 | 31; 8.95 / 4.64 s | 0.912 / 0.902 s |

Both configurations were generated twice; each pair was byte-identical. The default output SHA-256
was `ffb3149e1f8d810bab3578050f45019e5c84e39e3b5fcea98045f46d84ec54bf`, matching the retained
candidate. Both switch-8 outputs had SHA-256
`db10a5fc455d545534d32aee099c3f1519be0ce8c324630223da596608991680`. Peak process RSS across
these cached runs was 62,160,896 bytes as reported by macOS `ru_maxrss` (bytes on macOS). This
excludes feature-cache generation. The changed persistence reduced spurious spells by 17 and
spurious transitions by 16, while the matched spell count increased by two and missed spells fell
by two; at the stricter 12-frame transition tolerance, matched/missed counts were unchanged and
spurious transitions fell from 45 to 30. D precision/recall was unchanged. These measurements
support the hypothesis on this slice, but the spell bootstrap intervals overlap, the slice has
only 17 possessions, and switch-8 still trails smooth-125 on spellF1. It remains an experiment;
do not promote it or change the default from this evidence.

For this repeated run, the repository HEAD was
`16b6d90e20229dbff60eb924a07fba207f4db732`; the source/test/config worktree diff fingerprint was
`c673118dd670b811a59e4ecd6d6d007ae0a2b4240ffd665ebe13573abb1ed97e`. The only experiment
configuration change was `decoder.switch_frames=8`. Reproduce the cached prediction pair and score
it locally with:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from dataclasses import replace
from pathlib import Path

from possession.config import load_config
from possession.pipeline import predict

repo = Path.cwd()
metadata = repo / "Data/More Than A Hack 2027 - Initial Challenge - Dataset/tactical/BAR-ATH_tactical_atd.xml"
tracking = repo / "Data/More Than A Hack 2027 - Initial Challenge - Dataset 8/tactical/BAR-ATH_tactical_atd.txt"
cache = Path("/tmp/nexe-spell-investigation/features.npz")
output_dir = Path("/tmp/nexe-spell-diagnosis/repro/results")
default = load_config(repo / "configs/default.toml")
variants = (
    ("default", default),
    ("switch8", replace(default, decoder=replace(default.decoder, switch_frames=8))),
)
for name, config in variants:
    for replicate in (1, 2):
        predict(
            metadata,
            tracking,
            output_dir / f"{name}-{replicate}.csv",
            config=config,
            expected_frames=199319,
            feature_cache_path=cache,
        )
PY
PYTHONPATH=src .venv/bin/python ../more-than-a-hack-2027/evaluation/run_evaluation.py \
  --gt public \
  --results /tmp/nexe-spell-diagnosis/repro/results \
  --out /tmp/nexe-spell-diagnosis/repro/evaluation \
  --bootstrap 1000
```

The data-derived cache, predictions, and evaluator reports stay local and must not be redistributed.

The candidate output passed the official submission validator with exactly 199,319 rows and
continuous frame indices `0..199318`. Two default-configuration runs on the local tactical bundle
were byte-identical. On this host, wall times were 68.03 and 66.80 seconds and maximum resident set
sizes were 597,553,152 and 597,504,000 bytes, respectively. These are implementation/runtime
measurements, not accuracy evidence. The candidate's public-window spellF1 is lower than the
smooth-125 baseline. The public slice was used for the explicitly documented decoder diagnostics
above, not to select a production setting; 17 spells cannot establish robust generalization. No
full-match ground truth was used or inferred, so real full-match accuracy remains unverified.
Candidate files and evaluator reports are local-only and must not be redistributed.

Reproduce candidate generation and validate it using the local gated bundle, without writing
outputs into the official checkout:

```bash
possession predict \
  --metadata "$ATD_META" \
  --tracking "$ATD_RAW" \
  --output "$OUTPUT_DIR/candidate.csv" \
  --config configs/default.toml --frames 199319
python "$REPO_ROOT/evaluation/validate_submission.py" "$OUTPUT_DIR/candidate.csv"
```

For the public-window comparison, place the candidate and generated baseline CSVs in the same local
results directory, then run:

```bash
possession evaluate \
  --repo-root /path/to/more-than-a-hack-2027 \
  --results-dir "$OUTPUT_DIR/results" \
  --output-dir "$OUTPUT_DIR/public-evaluation" --ground-truth public
```

## YOLOX-Nano video feasibility pilot

The pilot checked alignment before running a bounded detector sample. The tactical video is 1920×1080
H.264 at 25 fps, with 199,319 frames over 7,972.76 seconds. Full FFmpeg decoding succeeded, and
sparse OpenCV seek/decode checks at 11 points spanning both endpoints returned the requested
frames. Tracking IDs map to video/submission frames `0..199318`, and homography rows cover the same
range. The homography maps pitch coordinates to image coordinates; converting a video detection
back to pitch coordinates requires the inverse transform.

YOLOX-Nano was sampled on 66 frames across six intervals between frames 13,050 and 17,666. This was
not a full-match detector evaluation:

| Same-frame observations | Count |
|---|---:|
| Tracking and detector both reported a ball | 39 |
| Tracking only | 5 |
| Detector only | 11 |
| Neither reported a ball | 11 |

On the 39 paired observations, median image-space disagreement was 796.5 pixels and median
homography-mapped disagreement was 37.2 metres. The p90 mapped movement estimate was 119.6 m/s,
indicating substantial instability or false detections in this sample. Manual inspection of the 11
detector-only hits found sideline detections in a tracking-gap interval. This was a small manual
review, not a labelled precision estimate. The pilot took 134.378 seconds, including 12.475 seconds
of inference (about 0.189 seconds per sampled frame); peak RSS was 652.8 MB. Full-match runtime was
not measured.

**Decision: no-go for integrating YOLOX-Nano into the production possession pipeline based on this
evidence.** The pilot does not establish that the detector adds reliable ball information. No
classifier was trained, and no production decoder, default, baseline, evaluator, or ground truth was
changed. Existing public-window scores cover only the published five-minute/17-spell window and
were not rerun for this feasibility assessment. Full-match possession accuracy remains unverified.
If pursued, first verify checkpoint licensing and assess a tactical-view-appropriate detector
against authorized, independently annotated, held-out ball visibility and bounding boxes. A
possession classifier would separately require more independent permitted possession labels across
intervals or matches. See [data handling](data_handling.md) for local-only asset restrictions.

The local pilot can be rerun with the same six sample centers and 25-frame radius, using the
separate Python 3.11 environment described in [data handling](data_handling.md). Set the variables
to local inputs; keep the output directory outside the repository:

```bash
PYTHONPATH="src:$YOLOX_SOURCE" \
  uv run --project video-pilot --only-group video-pilot \
  python scripts/video_perception_pilot.py \
  --video "$TACTICAL_VIDEO" \
  --metadata "$ATD_META" \
  --tracking "$ATD_RAW" \
  --homography "$HOMOGRAPHY" \
  --weights "$YOLOX_WEIGHTS" \
  --output-dir "$VIDEO_PILOT_OUTPUT" \
  --anchor early=13050 \
  --anchor transition=13704 \
  --anchor tracked_mid=15000 \
  --anchor stable=15750 \
  --anchor tracking_gap=17163 \
  --anchor difficult_gap=17641 \
  --radius-frames 25 \
  --step-frames 5 \
  --confidence 0.1 \
  --nms-threshold 0.45 \
  --threads 8
```

## Next evaluation

The full-match candidate is now reproducible from the local tactical inputs, but no permitted
full-match possession ground truth is available for accuracy validation. Preserve the official
baseline and scorer as controls. Do not choose thresholds from the 17-spell public slice. Include
Smart Tagging only as an ablation, never as possession labels.
