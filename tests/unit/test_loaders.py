from pathlib import Path

import numpy as np
import pytest

from possession.data.loaders import load_homography, load_smart_tagging, load_tracking


def write_tracking_files(folder: Path, frames=("1", "2")) -> tuple[Path, Path]:
    metadata = folder / "meta.xml"
    raw = folder / "raw.txt"
    metadata.write_text(
        '<Root><Team id="home"/><Team id="away"/>'
        '<Player teamId="home"/><Player teamId="away"/></Root>',
        encoding="utf-8",
    )
    raw.write_text(
        "".join(f"{frame}:0.1,0.2,1;N:0.3,0.4,0\n" for frame in frames),
        encoding="utf-8",
    )
    return metadata, raw


def test_tracking_loader_aligns_frames_and_keeps_missing_observations(tmp_path):
    metadata, raw = write_tracking_files(tmp_path)
    data = load_tracking(metadata, raw, expected_frames=2)
    assert data.frame_ids.tolist() == [0, 1]
    assert data.player_teams.tolist() == [0, 1]
    assert np.isnan(data.players[:, 1]).all()
    np.testing.assert_allclose(data.ball, [[0.3, 0.4], [0.3, 0.4]])


def test_tracking_loader_rejects_frame_gaps(tmp_path):
    metadata, raw = write_tracking_files(tmp_path, frames=("1", "3"))
    with pytest.raises(ValueError, match="contiguous"):
        load_tracking(metadata, raw, expected_frames=2)


def test_homography_loader_preserves_invalid_rows(tmp_path):
    path = tmp_path / "homography.csv"
    path.write_text(
        "frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n"
        "0,1,0,0,0,1,0,0,0,1,0.1,1\n"
        "1,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0\n",
        encoding="utf-8",
    )
    matrices, errors, valid = load_homography(path, expected_frames=2)
    assert valid.tolist() == [True, False]
    assert np.isfinite(matrices[0]).all()
    assert np.isnan(matrices[1]).all()
    assert errors[0] == pytest.approx(0.1)


def test_smart_tagging_seconds_convert_to_half_open_frames(tmp_path):
    path = tmp_path / "tags.xml"
    path.write_text(
        "<root><instance><start>1.0</start><end>2.0</end><code>SET PIECES</code>"
        "<label><group>Team</group><text>Barcelona</text></label></instance></root>",
        encoding="utf-8",
    )
    event = load_smart_tagging(path)[0]
    assert (event["start_frame"], event["end_frame"]) == (25, 50)
    assert event["tags"] == {"Team": "Barcelona"}


def test_loaders_report_malformed_xml_with_the_input_path(tmp_path):
    metadata, raw = write_tracking_files(tmp_path)
    metadata.write_text("<Root>", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed tracking metadata XML"):
        load_tracking(metadata, raw, expected_frames=2)

    tagging = tmp_path / "broken-tags.xml"
    tagging.write_text("<root>", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed Smart Tagging XML"):
        load_smart_tagging(tagging)


def test_smart_tagging_rejects_non_finite_interval_boundaries(tmp_path):
    tagging = tmp_path / "invalid-tags.xml"
    tagging.write_text(
        "<root><instance><start>nan</start><end>2.0</end></instance></root>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be finite"):
        load_smart_tagging(tagging)
