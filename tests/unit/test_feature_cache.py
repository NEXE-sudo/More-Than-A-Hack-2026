import numpy as np
import pytest

from possession.features.cache import load_feature_cache, save_feature_cache

FEATURE_CONFIG = {"fps": 25, "pitch_length_m": 105.3, "pitch_width_m": 68.0}


def cache_features():
    return {
        "ball_visible": np.array([True, False]),
        "home_nearest_m": np.array([1.0, np.nan]),
        "away_nearest_m": np.array([2.0, np.nan]),
        "ball_gap_frames": np.array([0, 1]),
        "home_player_mean_speed_m_s": np.array([1.0, 1.0]),
        "away_player_mean_speed_m_s": np.array([1.0, 1.0]),
        "home_player_track_count": np.array([4, 4]),
        "away_player_track_count": np.array([4, 4]),
    }


def test_cache_roundtrip_validates_schema_sources_and_feature_configuration(tmp_path):
    metadata = tmp_path / "metadata.xml"
    tracking = tmp_path / "tracking.txt"
    metadata.write_text("<metadata />", encoding="utf-8")
    tracking.write_text("tracking v1", encoding="utf-8")
    path = save_feature_cache(
        tmp_path / "features.npz",
        cache_features(),
        source_paths=(metadata, tracking),
        feature_config=FEATURE_CONFIG,
    )

    loaded = load_feature_cache(
        path,
        2,
        source_paths=(metadata, tracking),
        feature_config=FEATURE_CONFIG,
    )
    assert loaded["ball_visible"].tolist() == [True, False]
    assert np.isnan(loaded["home_nearest_m"][1])

    with pytest.raises(ValueError, match="feature configuration changed"):
        load_feature_cache(
            path,
            2,
            source_paths=(metadata, tracking),
            feature_config={**FEATURE_CONFIG, "fps": 30},
        )
    tracking.write_text("tracking v2", encoding="utf-8")
    with pytest.raises(ValueError, match="source files changed"):
        load_feature_cache(
            path,
            2,
            source_paths=(metadata, tracking),
            feature_config=FEATURE_CONFIG,
        )


def test_cache_rejects_unversioned_archives_and_invalid_arrays(tmp_path):
    old_cache = tmp_path / "old.npz"
    np.savez(old_cache, **cache_features())
    with pytest.raises(ValueError, match="metadata is missing"):
        load_feature_cache(
            old_cache,
            2,
            source_paths=(tmp_path / "source.txt",),
            feature_config=FEATURE_CONFIG,
        )

    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    with pytest.raises(ValueError, match="one-dimensional"):
        save_feature_cache(
            tmp_path / "bad.npz",
            {**cache_features(), "extra": np.zeros((2, 2))},
            source_paths=(source,),
            feature_config=FEATURE_CONFIG,
        )


def test_cache_rejects_smart_tagging_source_changes(tmp_path):
    tracking = tmp_path / "tracking.txt"
    tags = tmp_path / "tags.xml"
    tracking.write_text("tracking", encoding="utf-8")
    tags.write_text("<tags />", encoding="utf-8")
    features = cache_features()
    features["tagged_phase"] = np.array([True, True])
    features["tagged_set_piece"] = np.array([False, False])
    features["tagged_home"] = np.array([True, False])
    features["tagged_away"] = np.array([False, False])
    features["tagged_team_conflict"] = np.array([False, False])
    path = save_feature_cache(
        tmp_path / "tagged.npz",
        features,
        source_paths=(tracking,),
        context_source_paths=(tags,),
        feature_config=FEATURE_CONFIG,
    )
    with pytest.raises(ValueError, match="--smart-tagging is required"):
        load_feature_cache(
            path,
            2,
            source_paths=(tracking,),
            feature_config=FEATURE_CONFIG,
        )
    tags.write_text("<tags changed='yes' />", encoding="utf-8")
    with pytest.raises(ValueError, match="source files changed"):
        load_feature_cache(
            path,
            2,
            source_paths=(tracking,),
            context_source_paths=(tags,),
            feature_config=FEATURE_CONFIG,
        )
