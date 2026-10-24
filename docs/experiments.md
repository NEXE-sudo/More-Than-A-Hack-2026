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
deterministically generates tracking positions from explicit seeds. It includes stable home/away,
possession turnovers with short/long pass flight intervals, an abrupt interception, contested/
ungraded intervals, ball occlusion and aerial displacement, goalkeeper possession, a dead-ball
restart, adversarial close observations, ball/player-track dropout, boundary-start/end transitions,
contradictory proximity, and a half-boundary reset. The restart scenario supports throw-ins, corners,
free kicks, goal kicks, kick-offs and other stoppages; throw-ins deliberately carry `Team=N/A`.
Parameters include duration, frame rate, pitch dimensions, player count, track availability, ball
visibility and coordinate noise. Players move coherently toward changing targets, with observation
dropout/noise and configured scenario interruptions. These simulations are test fixtures, not
realistic match replicas.

Run the complete fixed matrix:

```bash
possession run-synthetic-experiments --output-dir outputs/synthetic-experiments
```

Optional `--seeds`, `--scenarios`, and `--duration-seconds` arguments narrow or extend the run.
The default matrix uses seeds 11, 29 and 47; 15 canonical scenario families; and nine
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

Each row records its seed, scenario, full configuration, model/decoder version, sample count,
runtime, warnings, frame and spell metrics, transition diagnostics, and abstention/missing-ball
behavior. CSV and JSON are machine-readable; the Markdown report summarizes mean and standard
deviation across scenario/seed runs, not frames. The 20 additional accepted scenario names are
explicit aliases for canonical generators. Aliases normalize to their canonical family before
generation and aggregation, so requesting both a canonical name and an alias does not duplicate or
overweight a family. Outputs are local and ignored by Git.

## Synthetic results (not competition performance)

Results from the canonical 45-sequence run per configuration (15 families × 3 seeds):

| Configuration | macro-F1 mean ± SD | spellF1 mean ± SD | switches/min |
|---|---:|---:|---:|
| Baseline model + baseline decoder | 0.352 ± 0.140 | 0.662 ± 0.337 | 2.27 |
| Revised model + baseline decoder | 0.332 ± 0.127 | 0.653 ± 0.341 | 1.68 |
| Baseline model + revised decoder | 0.354 ± 0.141 | 0.665 ± 0.335 | 2.27 |
| Revised model + revised decoder | 0.333 ± 0.130 | 0.637 ± 0.345 | 1.85 |
| Revised without proximity | 0.006 ± 0.026 | 0.000 ± 0.000 | 0.00 |
| Revised without local support | 0.340 ± 0.129 | 0.650 ± 0.347 | 1.85 |
| Revised without approach | 0.340 ± 0.131 | 0.668 ± 0.335 | 1.68 |
| Revised, Smart Tagging off | 0.333 ± 0.130 | 0.637 ± 0.345 | 1.85 |
| Revised, Smart Tagging on | 0.333 ± 0.130 | 0.637 ± 0.345 | 1.85 |

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
