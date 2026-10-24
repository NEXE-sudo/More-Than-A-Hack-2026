# Architecture

## Inference path

`data.loaders` reads the ATD metadata XML and frame records into aligned arrays. Validation checks
contiguous frame IDs, team assignments, pairwise coordinate missingness, finite values, and
normalized coordinate limits. `features.tracking` computes pitch-metre distances, local counts,
team speeds, ball motion, and missingness indicators using vectorized NumPy operations.

`models.baseline.evidence_from_features` is the default, unchanged nearest-distance evidence model.
`models.baseline.revised_evidence_from_features` is opt-in and combines:

- 75% nearest-player proximity evidence (`exp(-distance / scale)`),
- up to 15% local supporting-player count evidence, and
- up to 10% causal nearest-distance approach evidence.

Those bounded values are evidence scores, not calibrated probabilities. The model deliberately
does not add the correlated nearest-team distance margin as another vote. Missing ball observations
produce no team-distance evidence. Dead-ball evidence remains a separately defined score requiring
a sustained ball gap, quiet teams, and enough observed tracks; it is a heuristic, not a trained
dead-ball classifier.

`temporal.decoder.decode` remains the default hysteresis implementation. The opt-in
`decode_revised` preserves a pending switch candidate through a configured, bounded number of
low-evidence contest frames. Both decoders assign a newly observed possession immediately, confirm
normal switches persistently, explicitly handle D and missing-evidence holds, and reset on sequence
boundaries. The revised behavior does not backdate a confirmed switch.

## Smart Tagging

The loader maps `<start>` and `<end>` seconds to half-open tactical-frame intervals at the
configured frame rate. The official schema describes automatic phase/restart annotations, not
possession ground truth; `Team` identifies the team performing the event and may be `N/A`. The
default model ignores these masks. For the revised model only, a valid unambiguous team tag may add
a small bounded evidence bonus (`tagging_team_weight`, default 0.05); overlapping conflicting team
tags are ignored. The code never maps `SET PIECES` directly to D or copies a tag into an output
label. This is a testable prior, not verified against gated match data.

## Causal and sequence semantics

Ball and team distance changes use only current and earlier frame observations. No look-ahead
smoothing is used. A sequence boundary clears the current team, pending switch, and missing-run
state. Inference emits `H`, `A`, `D`, or `X`; `X` is a model abstention in the submission contract.
The official scorer's spell metric treats predicted `X` differently from frame-level scoring:
within each non-X ground-truth run it fills a prediction's X with its previous non-X label and
back-fills at the beginning of the run. A wholly-X run remains X. The synthetic evaluator follows
that source-verified rule.

## Cache and output path

Feature caches are compressed NPZ files with `allow_pickle=False`, explicit schema version, array
shape/dtype metadata, source-file SHA-256 hashes, and feature-extraction settings. Cache reuse
requires the original source paths and matching feature settings; model/decoder settings do not
invalidate features because they do not affect extraction. Output writing validates exact frame
order, count, label alphabet, and `frame,label` CSV formatting.

## Gated-data limitation

The tactical match tracking bundle is not present in this workspace. The official scorer and
public baseline outputs are accessible through the sibling official checkout, but this project's
candidate pipeline cannot be validated against full-match tracking or private labels yet. The
synthetic generator and experiment matrix are for reproducible engineering tests only, not a
substitute for real-match evidence.
