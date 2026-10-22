# Architecture

## Implementation plan

1. Confirm the official output rules, source frame conventions, baseline, scorer, and Git boundary.
2. Implement an independent Python package with strict loaders, explicit missingness, and synthetic
   tests.
3. Add vectorized tracking features and a deterministic, configurable temporal decoder.
4. Keep baseline reproduction and scoring as subprocess calls into the official repository.
5. Verify with data-independent tests and the public baseline evaluation; report gated-data
   blockers instead of implying full-match experiments ran.

## Data flow

`data.loaders` reads the ATD XML team/track metadata and line-oriented raw tracking into arrays;
`data.validation` checks coordinate ranges and array shapes while preserving NaN observations.
`features.tracking` computes normalized-pitch distances in metres, per-team nearby counts, ball
motion and player-track reliability. `features.ball` provides causal trajectory features;
`features.context` converts Smart Tagging intervals into context masks, not ground truth. The
optional `features.geometry` projection validates and uses homographies only on declared-valid
frames.

`models.baseline.evidence_from_features` creates independent H/A evidence from nearest distances.
`temporal.decoder.decode` adds switch persistence, dead-ball evidence from a conservative sustained
occlusion/low-team-speed heuristic, bounded short-gap holds, sequence resets, and abstention after
longer evidence gaps. `pipeline.predict` validates and writes a plain `frame,label` CSV. The
dead-ball heuristic is an unvalidated prior, not ground truth.

The official adapter launches the original baseline and scorer with explicit working directories.
It neither imports evaluator internals nor copies or edits their source. The scorer requires pandas,
which is pinned as a project dependency so the subprocess works with the isolated interpreter.

## Decoder semantics

Evidence columns are H, A, D scores in `[0,1]`, not calibrated probabilities. Ties between teams
prefer H. A switch requires the configured evidence margin over consecutive frames. Missing team
evidence temporarily holds an existing team, then emits X; it does not imply D. D requires an
explicitly supplied dead-ball score above a separate threshold and margin. A new sequence resets
all state. These conservative defaults are configurable and not tuned against the 17-possession
public slice.

## Explicitly deferred

There is no trained classifier, video model, identity tracking, or synchronized broadcast fusion.
The dead-ball prior has not been evaluated against this match. Smart Tagging is context only and
is not copied into possession labels. These are deliberate restrictions until permitted full-match
assets and sound evaluation are available.
