import json
import subprocess
import sys
from argparse import Namespace

import numpy as np

import possession.cli as possession_cli
from possession.data.validation import TrackingData


def test_cli_build_cache_predict_and_validate_submission_smoke(tmp_path):
    metadata = tmp_path / "meta.xml"
    tracking = tmp_path / "tracking.txt"
    cache = tmp_path / "features.npz"
    output = tmp_path / "submission.csv"
    metadata.write_text(
        '<Root><Team id="home"/><Team id="away"/>'
        '<Player teamId="home"/><Player teamId="away"/>'
        '<Sensor id="speed"><Channels><Channel id="s">'
        "<Name>speed_m_s</Name><Unit>m/s</Unit>"
        "</Channel></Channels></Sensor></Root>",
        encoding="utf-8",
    )
    tracking.write_text(
        "1:0.20,0.5,0;0.80,0.5,0:0.21,0.5,0\n"
        "2:0.20,0.5,0;0.80,0.5,0:0.21,0.5,0\n"
        "3:0.20,0.5,0;0.80,0.5,0:N\n",
        encoding="utf-8",
    )

    validation = subprocess.run(
        [
            sys.executable,
            "-m",
            "possession.cli",
            "validate-data",
            "--metadata",
            str(metadata),
            "--tracking",
            str(tracking),
            "--frames",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validation.returncode == 0, validation.stderr

    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "possession.cli",
            "build-features",
            "--metadata",
            str(metadata),
            "--tracking",
            str(tracking),
            "--frames",
            "3",
            "--output",
            str(cache),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    predict = subprocess.run(
        [
            sys.executable,
            "-m",
            "possession.cli",
            "predict",
            "--metadata",
            str(metadata),
            "--tracking",
            str(tracking),
            "--feature-cache",
            str(cache),
            "--frames",
            "3",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert predict.returncode == 0, predict.stderr
    assert output.read_text(encoding="utf-8").splitlines() == [
        "frame,label",
        "0,H",
        "1,H",
        "2,H",
    ]


def test_cli_reports_deprecated_synthetic_scenario_and_replacement(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "possession.cli",
            "run-synthetic-experiments",
            "--output-dir",
            str(tmp_path),
            "--scenarios",
            "interception",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "scenario 'interception' is deprecated" in result.stderr
    assert "home_to_away_turnover" in result.stderr


def test_cli_requires_video_when_full_decode_is_requested(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "possession.cli",
            "audit-data",
            "--metadata",
            str(tmp_path / "metadata.xml"),
            "--tracking",
            str(tmp_path / "tracking.txt"),
            "--homography",
            str(tmp_path / "homography.csv"),
            "--decode-video",
            "--output",
            str(tmp_path / "audit.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "--video is required when --decode-video is used" in result.stderr


def test_cli_does_not_overwrite_existing_audit_output(tmp_path):
    output = tmp_path / "audit.json"
    output.write_text("keep this file", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "possession.cli",
            "audit-data",
            "--metadata",
            str(tmp_path / "metadata.xml"),
            "--tracking",
            str(tmp_path / "tracking.txt"),
            "--homography",
            str(tmp_path / "homography.csv"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "audit output already exists" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep this file"


def test_data_audit_report_marks_video_and_calibration_as_blocked(tmp_path, monkeypatch):
    tracking = TrackingData(
        frame_ids=np.arange(2),
        players=np.empty((2, 0, 2)),
        ball=np.full((2, 2), np.nan),
        player_teams=np.array([], dtype=int),
        player_speed=np.empty((2, 0)),
    )
    monkeypatch.setattr(possession_cli, "load_tracking", lambda *args: tracking)
    monkeypatch.setattr(
        possession_cli,
        "load_homography",
        lambda *args: (
            np.repeat(np.eye(3)[None, :, :], 2, axis=0),
            np.zeros(2),
            np.ones(2, dtype=bool),
        ),
    )

    result = possession_cli._audit_data(
        Namespace(
            decode_video=False,
            video=None,
            output=tmp_path / "local-audit.json",
            config=None,
            metadata=tmp_path / "metadata.xml",
            tracking=tmp_path / "tracking.txt",
            homography=tmp_path / "homography.csv",
            frames=2,
        )
    )

    assert result == 0
    report = json.loads((tmp_path / "local-audit.json").read_text())
    assert report["summary"]["video_status"]["status"] == "blocked"
    assert report["summary"]["calibration_validation"]["status"] == "blocked"
    assert "TACTICAL_VIDEO" in report["summary"]["video_status"]["next_step"]


def test_baseline_audit_cli_fails_when_runs_are_not_reproducible(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        possession_cli,
        "run_baseline_audit",
        lambda *args, **kwargs: {
            "status": "non_reproducible",
            "determinism": {
                "prediction_files_byte_identical": False,
                "metrics_identical": True,
            },
        },
    )

    result = possession_cli._run_baseline_audit(
        Namespace(
            repo_root=tmp_path / "official",
            data_root=tmp_path / "data",
            output_dir=tmp_path / "audit",
            bootstrap=1000,
        )
    )

    assert result == 1
    assert "baseline audit status: non_reproducible" in capsys.readouterr().out
