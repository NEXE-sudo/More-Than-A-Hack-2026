# nexe-possession

An independent, deterministic possession-estimation project for the Metrica Sports / FC Barcelona
More Than A Hack 2027 challenge. The implementation uses the official tactical tracking data and
stores all project source in a sibling directory. It is a transparent baseline and research
foundation, not a claim of improved full-match performance.

## Architecture

ATD XML/text inputs are loaded into compact NumPy arrays, checked for frame alignment and missing
data, transformed into ball/player features, converted to frame-level evidence, then decoded with
switch persistence and bounded missing-data holds. The original evidence model and decoder remain
the defaults; a separately selectable evidence/decoder pair is available for controlled
experiments. The official baseline and scorer source are invoked as subprocesses without copying or
editing them. See [docs/architecture.md](docs/architecture.md) and
[docs/data_handling.md](docs/data_handling.md).

## Requirements and installation (macOS)

Supported: macOS on Intel or Apple Silicon, Python 3.11–3.13. NumPy is used for feature work;
pandas is required by the unchanged official scorer launched by the adapter. pytest and Ruff are
development dependencies. An isolated environment and pinned installs:

```bash
cd /path/to/nexe-possession
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

Set `REPO_ROOT` and `DATA_ROOT` for the official checkout and your locally supplied, permitted
assets. Neither path is hard-coded:

```bash
export REPO_ROOT=/path/to/more-than-a-hack-2027
export DATA_ROOT=/path/to/match-data
```

The expected bundle is described in the official [`DATA.md`](../more-than-a-hack-2027/DATA.md):
`tactical/BAR-ATH_tactical_atd.xml`, `tactical/BAR-ATH_tactical_atd.txt`, and optional homography
and Smart Tagging files. Competition data and prediction outputs must not be committed.

## Commands

Run from any working directory after installation:

```bash
possession check-environment
possession validate-data \
  --metadata "$DATA_ROOT/tactical/BAR-ATH_tactical_atd.xml" \
  --tracking "$DATA_ROOT/tactical/BAR-ATH_tactical_atd.txt" \
  --homography "$DATA_ROOT/tactical/BAR-ATH_tactical_homography.csv" \
  --smart-tagging "$DATA_ROOT/tactical/BAR-ATH_tactical_smart_tagging.xml"

# Reproduce the original baseline with its own implementation and gated tracking bundle
possession run-baseline --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
  --output-dir outputs/official-baselines

# Build a source-verified feature cache or generate a candidate
possession build-features --metadata "$DATA_ROOT/tactical/BAR-ATH_tactical_atd.xml" \
  --tracking "$DATA_ROOT/tactical/BAR-ATH_tactical_atd.txt" --config configs/default.toml \
  --output cache/features.npz
possession predict --metadata "$DATA_ROOT/tactical/BAR-ATH_tactical_atd.xml" \
  --tracking "$DATA_ROOT/tactical/BAR-ATH_tactical_atd.txt" --output outputs/nexe.csv \
  --config configs/default.toml
# Or reuse the archive without reparsing tracking (source files are fingerprinted):
possession predict --metadata "$DATA_ROOT/tactical/BAR-ATH_tactical_atd.xml" \
  --tracking "$DATA_ROOT/tactical/BAR-ATH_tactical_atd.txt" \
  --feature-cache cache/features.npz --output outputs/nexe.csv \
  --config configs/default.toml

# Invoke the unchanged official evaluator on a folder of candidate CSV files
possession evaluate --repo-root "$REPO_ROOT" --results-dir outputs \
  --output-dir outputs/evaluation --ground-truth public --bootstrap 1000

# Compare the fixed model/decoder matrix without gated match data
possession run-synthetic-experiments --output-dir outputs/synthetic-experiments
```

`configs/default.toml` contains the frame rate, pitch dimensions and decoder thresholds. CLI paths
are explicit, resolved via `pathlib`, and do not depend on the current directory. The optional
official evaluator is the repository's `evaluation/run_evaluation.py`; it reports its own exit
status and output without a local metric reimplementation. Do not supply or seek private ground
truth. Cache reuse requires both source paths so content fingerprints and feature-extraction
settings can be verified without reparsing the tracking rows. `model.evidence_model` and
`decoder.strategy` accept `baseline` or `revised` and default to `baseline`. Smart Tagging is
optional weak team context for the revised evidence model; unknown/conflicting team tags are not
copied into output labels.

## Tests and quality checks

```bash
python -m pytest
ruff check .
ruff format --check .
```

Tests use synthetic data and require no competition files. Run the original evaluator on the
provided baseline CSVs without gated match assets:

```bash
possession evaluate --repo-root "$REPO_ROOT" --results-dir "$REPO_ROOT/evaluation/results" \
  --output-dir outputs/public-evaluation --ground-truth public
```

Validate the final CSV with the unchanged official validator as well:

```bash
python "$REPO_ROOT/evaluation/validate_submission.py" outputs/nexe.csv
```

The synthetic command writes CSV/JSON run records and a Markdown report beneath the selected output
directory. These controlled simulations are not official evaluation results. See
[docs/experiments.md](docs/experiments.md) for their scope and limitations.

## Output contract

The CSV has exactly `frame,label`, 199,319 rows indexed `0..199318`, in tactical-video order, and
labels from `H`, `A`, `D`, `X`. The default prediction uses normalized ATD locations and emits `X`
for long periods without team evidence rather than retaining a prior team indefinitely. The baseline's
conservative dead-ball prior requires sustained ball absence plus low speeds for both teams and
adequate track coverage. It is not validated against this match because the bundle is unavailable;
an occlusion or Smart Tagging interval alone never becomes D.

## Limitations and reproducibility

The gated tracking bundle is not present in this workspace. Consequently ATD parsing, full-match
candidate generation and local baseline regeneration are implemented but unexecuted on match
data. Synthetic tests and evaluation of the committed public baseline outputs are verifiable here;
no candidate was measured on tracking data. No trained classifier is included: the public labels
cover only 17 possessions and are insufficient evidence for generalization. Predictions are
deterministic; all decoding settings are explicit. See [docs/experiments.md](docs/experiments.md)
for measured results and blockers.
# More-Than-A-Hack-2026
