# Data handling and alignment

## Boundary and restrictions

The official checkout is the reference for terms and formats. This project reads the permitted
tactical tracking bundle only when a user supplies paths. Never upload videos, tracking, labels, or
extracts to hosted services; keep match data, private labels, caches, and generated submissions out
of Git. Broadcast tracking has a separate, unsynchronized timeline and is not accepted as a
tactical frame source.

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
`x,y,speed`. Coordinates are normalized to `[0,1]`; team membership is read from metadata in the
same track-slot order. Track slots are fragmented tracklets, not stable player identities. `N` and
`NaN` observations stay NaN in memory. Infinite/out-of-range coordinates and malformed records are
errors rather than fabricated positions. A coordinate pair must be fully observed or fully missing;
partial x/y observations are rejected. Missing speed is allowed, but infinite or negative speeds
are not.

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
sequence. No real-match memory benchmark is claimed because the gated source bundle is unavailable.

## Ground truth

The official public CSV covers only the opening five minutes and 17 possessions. It is used only
for allowed evaluation, never as model input. The private full-match labels are unavailable and
must not be inferred or sought.
