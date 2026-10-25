import subprocess
import sys


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
