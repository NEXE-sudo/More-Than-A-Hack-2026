"""Versioned, source-bound, non-pickle NumPy feature-cache persistence."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA_VERSION = 2
METADATA_KEY = "__cache_metadata__"
TAGGING_FEATURES = {
    "tagged_phase",
    "tagged_set_piece",
    "tagged_home",
    "tagged_away",
    "tagged_team_conflict",
}
REQUIRED_FEATURES = {
    "ball_visible",
    "home_nearest_m",
    "away_nearest_m",
    "ball_gap_frames",
    "home_player_mean_speed_m_s",
    "away_player_mean_speed_m_s",
    "home_player_track_count",
    "away_player_track_count",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_metadata(source_paths: tuple[Path | str, ...]) -> list[dict[str, str]]:
    if not source_paths:
        raise ValueError("source_paths must include the source files used to build the cache")
    sources = []
    for value in source_paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"feature-cache source file not found: {path}")
        sources.append({"name": path.name, "sha256": _file_sha256(path)})
    return sources


def save_feature_cache(
    path: Path | str,
    features: dict[str, np.ndarray],
    *,
    source_paths: tuple[Path | str, ...],
    context_source_paths: tuple[Path | str, ...] = (),
    feature_config: dict[str, Any],
) -> Path:
    """Write feature arrays and source/config fingerprints to a compressed NPZ archive."""
    missing = REQUIRED_FEATURES - features.keys()
    if missing:
        raise ValueError(f"cannot save feature cache; missing arrays: {sorted(missing)}")
    arrays: dict[str, np.ndarray] = {}
    for name, values in features.items():
        array = np.asarray(values)
        if name == METADATA_KEY or array.dtype.kind not in "biuf":
            raise ValueError(f"cannot save non-numeric feature array {name!r}")
        if array.ndim != 1:
            raise ValueError(f"feature {name!r} must be a one-dimensional frame array")
        if np.isinf(array).any():
            raise ValueError(f"feature {name!r} contains infinite values")
        arrays[name] = array
    frame_counts = {len(values) for values in arrays.values()}
    if len(frame_counts) != 1:
        raise ValueError("all feature arrays must have the same frame count")
    if arrays["ball_visible"].dtype != np.bool_:
        raise ValueError("ball_visible must be boolean")
    has_tagging_features = bool(TAGGING_FEATURES & arrays.keys())
    if has_tagging_features and not TAGGING_FEATURES <= arrays.keys():
        raise ValueError("cached Smart Tagging context must include all interval masks")
    if has_tagging_features and not context_source_paths:
        raise ValueError("context_source_paths are required when caching Smart Tagging masks")
    if not has_tagging_features and context_source_paths:
        raise ValueError("Smart Tagging sources were supplied without cached context masks")

    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "sources": _source_metadata(source_paths),
        "context_sources": (_source_metadata(context_source_paths) if has_tagging_features else []),
        "feature_config": feature_config,
        "arrays": {
            name: {"dtype": values.dtype.str, "shape": list(values.shape)}
            for name, values in sorted(arrays.items())
        },
    }
    arrays[METADATA_KEY] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return destination


def load_feature_cache(
    path: Path | str,
    expected_frames: int,
    *,
    source_paths: tuple[Path | str, ...],
    context_source_paths: tuple[Path | str, ...] = (),
    feature_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Load and validate source-bound frame arrays without enabling pickle support."""
    if expected_frames <= 0:
        raise ValueError("expected_frames must be positive")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"feature cache not found: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if METADATA_KEY not in archive.files:
            raise ValueError(f"{source}: cache metadata is missing; rebuild the feature cache")
        try:
            metadata = json.loads(str(archive[METADATA_KEY].item()))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{source}: cache metadata is malformed") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"{source}: cache metadata must be a JSON object")
        if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(f"{source}: unsupported feature-cache schema version")
        actual_sources = _source_metadata(source_paths)
        if metadata.get("sources") != actual_sources:
            raise ValueError(f"{source}: source files changed since the cache was built")
        if metadata.get("feature_config") != feature_config:
            raise ValueError(f"{source}: feature configuration changed since the cache was built")
        arrays_metadata = metadata.get("arrays")
        if not isinstance(arrays_metadata, dict):
            raise ValueError(f"{source}: cache array metadata is malformed")
        expected_names = set(arrays_metadata)
        has_tagging_features = bool(TAGGING_FEATURES & expected_names)
        cached_context_sources = metadata.get("context_sources")
        if not isinstance(cached_context_sources, list):
            raise ValueError(f"{source}: context source metadata is malformed")
        if has_tagging_features:
            if not TAGGING_FEATURES <= expected_names:
                raise ValueError(f"{source}: cached Smart Tagging context is incomplete")
            if not context_source_paths:
                raise ValueError(f"{source}: --smart-tagging is required to verify cached context")
            actual_context_sources = _source_metadata(context_source_paths)
            if cached_context_sources != actual_context_sources:
                raise ValueError(
                    f"{source}: Smart Tagging source files changed since the cache was built"
                )
        elif cached_context_sources:
            raise ValueError(f"{source}: unexpected Smart Tagging source metadata")
        if set(archive.files) != expected_names | {METADATA_KEY}:
            raise ValueError(f"{source}: cached feature set does not match its metadata")
        missing = REQUIRED_FEATURES - expected_names
        if missing:
            raise ValueError(f"{source}: missing required feature arrays {sorted(missing)}")
        features = {name: archive[name] for name in expected_names}

    for name, values in features.items():
        description = arrays_metadata.get(name, {})
        if values.ndim != 1 or len(values) != expected_frames:
            raise ValueError(f"{source}: feature {name!r} must have {expected_frames} frame rows")
        if values.dtype.kind not in "biuf" or np.isinf(values).any():
            raise ValueError(f"{source}: feature {name!r} has invalid dtype or values")
        if description != {"dtype": values.dtype.str, "shape": list(values.shape)}:
            raise ValueError(f"{source}: feature {name!r} does not match its metadata")
    if features["ball_visible"].dtype != np.bool_:
        raise ValueError(f"{source}: ball_visible must be boolean")
    for name in TAGGING_FEATURES & features.keys():
        if features[name].dtype != np.bool_:
            raise ValueError(f"{source}: {name} must be boolean")
    return features
