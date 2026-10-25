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
assets. `DATA_ROOT` must contain a `tactical/` directory. The bundle is local-only; the root
`Data/` and `data/` spellings are ignored by Git, as are generated outputs and feature caches.
Neither path is hard-coded:

```bash
export REPO_ROOT=/path/to/more-than-a-hack-2027
export DATA_ROOT=/path/to/match-data
```

For the project CLI, set each input path to the actual local file; the files need not be copied or
moved into one directory:

```bash
export ATD_META=/path/to/local/BAR-ATH_tactical_atd.xml
export ATD_RAW=/path/to/local/BAR-ATH_tactical_atd.txt
export HOMOGRAPHY=/path/to/local/BAR-ATH_tactical_homography.csv
export SMART_TAGGING=/path/to/local/BAR-ATH_tactical_smart_tagging.xml
```

The bundle layout is described in the official [`DATA.md`](../more-than-a-hack-2027/DATA.md):
`tactical/BAR-ATH_tactical_atd.xml`, `tactical/BAR-ATH_tactical_atd.txt`, and optional homography
and Smart Tagging files. Use only the permitted local copy and keep the video, tracking, labels,
homographies, caches, predictions, and any data-derived exports out of Git and external services.
Do not redistribute supplied data or derived artifacts; follow the official
[`RULES.md`](../more-than-a-hack-2027/RULES.md).
The official baseline command expects the standard bundle layout. If the supplied archive was
extracted into separate dataset folders, use a private local directory containing symlinks to the
original files rather than copying or modifying gated assets.

## Commands

Run from any working directory after installation:

```bash
possession check-environment
possession validate-data \
  --metadata "$ATD_META" --tracking "$ATD_RAW" \
  --homography "$HOMOGRAPHY" --smart-tagging "$SMART_TAGGING"

# Reproduce the original baseline with its own implementation and gated tracking bundle
possession run-baseline --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
  --output-dir outputs/official-baselines

# Build a source-verified feature cache or generate a candidate
possession build-features --metadata "$ATD_META" \
  --tracking "$ATD_RAW" --config configs/default.toml \
  --output cache/features.npz
possession predict --metadata "$ATD_META" \
  --tracking "$ATD_RAW" --output outputs/nexe.csv \
  --config configs/default.toml
# Or reuse the archive without reparsing tracking (source files are fingerprinted):
possession predict --metadata "$ATD_META" \
  --tracking "$ATD_RAW" \
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
adequate track coverage. The local tactical bundle was available for full-match candidate generation
and public-window evaluation. Public scoring covers only 7,500 frames and 17 possession spells in
one contiguous five-minute interval. Running the candidate over all frames does not validate
full-match accuracy; an occlusion or Smart Tagging interval alone never becomes D.

## Limitations and reproducibility

The gated tracking bundle was supplied locally for ingestion and candidate generation but is not
versioned. Its use requires explicit input paths, and generated predictions and derived diagnostics
must remain local. Public-window evaluation was performed on the opening five minutes (7,500 frames,
17 possession spells); it does not validate full-match possession accuracy or generalization.
Independent permitted full-match possession ground truth is unavailable. The candidate can be run
over the full match and generates deterministic predictions, but its full-match accuracy remains
unverified. No trained classifier is included. See
[docs/experiments.md](docs/experiments.md) for measured results and blockers.
# More-Than-A-Hack-2026
