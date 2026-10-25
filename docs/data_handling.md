# Data handling and alignment

## Boundary and restrictions

The official checkout's `DATA.md` and `RULES.md` are the references for formats and terms. This
project reads the permitted local tactical bundle only when a user supplies paths. The root
`Data/` and `data/` paths are ignored by Git on case-sensitive and case-insensitive filesystems.
Never commit or redistribute videos, tracking, labels, homographies, caches, predictions, or
data-derived exports, and do not upload them to hosted services. Keep generated files in the
ignored `outputs/`, `cache/`, or `artifacts/` directories. Broadcast tracking has a separate,
unsynchronized timeline and is not accepted as a tactical frame source.

The `video-pilot` dependency group in `video-pilot/pyproject.toml` is for local detector feasibility
experiments only; it is not a project runtime extra and is not used by the production possession
pipeline. It has a separate `uv.lock` because its pinned NumPy 1.x CPU stack is incompatible with
the production project's NumPy 2.x requirement. Create its isolated Python 3.11 environment with
`uv sync --project video-pilot --only-group video-pilot --python 3.11`. YOLOX source and weights are
not installed by this group; acquire them separately for a local pilot, and keep downloaded
weights, extracted frames, detections, and derived feature tables outside the repository. Verify
the code and weight licenses before use or redistribution.

The standalone `scripts/video_perception_pilot.py` accepts explicit local video, tracking, and
homography paths and writes its sampled feature CSV, summary, and inspection frames only to the
requested output directory. Use an ignored or external directory; this feasibility tool is not
part of the possession prediction CLI or production path.

## Frame conventions

Verified against the official `DATA.md` and baseline:

| Source | Index | Tactical video frame |
|---|---|---|
| Submission | `frame = 0..199318` | direct |
| ATD raw text | frame ID `1..199319` | `frame_id - 1` |
| Smart Tagging | seconds from tactical video start | `round(seconds * 25)` |
| Tactical homography | `frame = 0..199318` | direct |

The submission length is 199,319. ATD records must be in exact contiguous order; the loader fails
clearly on missing, reordered or duplicated frames. The separate broadcast video has 164,488 frames
and must not be used directly as tactical output.

## Tracking representation and missingness

The EPTS line format is `frame_id:player;player;...:ball`, with each object represented as
`x,y,speed`. The official schema specifies normalized coordinates in `[0,1]`; team membership is
read from metadata in the same track-slot order. Track slots are fragmented tracklets, not stable
player identities. Finite, complete coordinate pairs within the inclusive range (including 0 and 1)
are accepted. Fully missing `N`/`NaN` pairs remain missing. A finite pair outside the schema range
is counted and warned about, then rejected in memory as a missing position; for a player its
associated speed is also marked missing. The original tracking file and every frame record remain
unchanged, so no frame is dropped or shifted and no coordinate is clipped. Partial coordinate
pairs, infinite coordinate values, malformed groups, and malformed or non-contiguous frame records
are fatal input errors.

The metadata declares the player speed channel as `speed_kph` in `kph`. Internal speeds are
canonicalized to metres per second at load time (`km/h / 3.6`), exactly once, before feature
extraction and the dead-ball speed threshold. The loader checks the declared channel name and unit,
records both the source unit and conversion factor in `TrackingData`, preserves missing speeds, and
rejects unsupported or conflicting metadata. Already-metric `speed_m_s`/`speed_mps` inputs use a
factor of 1.0. Infinite and negative speeds are rejected.

Homography rows contain nine row-major matrix values, reprojection error and a `valid` flag. Invalid
rows are retained as NaN and never projected. Smart Tagging is automatic phase/restart context, not
label truth; its `Team` tags may be `N/A`, and its intervals may span live play. XML parse failures
are surfaced with the offending path. Context masks use half-open intervals, clip events at video
boundaries, ignore unknown team values, and record conflicting home/away annotations separately.

## Feature-cache integrity

NPZ feature caches use `allow_pickle=False` and include:

- a cache-schema version,
- exact names, 1D frame shapes, and NumPy dtypes for every feature array,
- SHA-256 hashes of the metadata and tracking source files, and
- the frame rate and pitch dimensions used during feature extraction.

`build-features` records these values and includes Smart Tagging among the hashed sources when it is
used to create cached context masks. `predict --feature-cache` requires `--metadata` and
`--tracking` (and the same `--smart-tagging` path if tags were included in the cache) so it can hash
the inputs without reparsing tracking, and rejects a missing/old schema, changed source content,
changed extraction settings, malformed arrays, or invalid numeric values.
This makes cache reuse cheaper than feature reconstruction without silently accepting stale source
data. Rebuild caches after changing source files, extraction settings, or feature definitions.

Tracking feature extraction uses float32 intermediates for coordinate scaling and distance matrices;
precision is ample for pitch-metre distances and lowers peak memory for the approximately 199k-frame
sequence. No memory benchmark is implied by this implementation detail.

## Ground truth

The official public CSV covers only the opening five minutes and 17 possessions. It is used only
for allowed evaluation, never as model input. The private full-match labels are unavailable and
must not be inferred or sought; public-window scores do not establish full-match accuracy.
