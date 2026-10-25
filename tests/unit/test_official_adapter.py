import subprocess

import possession.evaluation.official_adapter as official_adapter


def test_official_baseline_default_command_remains_unmodified(tmp_path, monkeypatch):
    root = tmp_path / "official"
    (root / "starter").mkdir(parents=True)
    (root / "starter" / "baseline.py").touch()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(official_adapter.subprocess, "run", run)

    official_adapter.run_official_baseline(root, tmp_path / "data", tmp_path / "output")

    assert commands[0][-2:] == ["--out", str((tmp_path / "output").resolve())]
    assert "--window" not in commands[0]


def test_official_baseline_adapter_passes_only_requested_sensitivity_widths(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "official"
    (root / "starter").mkdir(parents=True)
    (root / "starter" / "baseline.py").touch()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(official_adapter.subprocess, "run", run)

    official_adapter.run_official_baseline(
        root,
        tmp_path / "data",
        tmp_path / "output",
        smoothing_half_widths=(75, 100),
    )

    assert commands[0][-5:] == [
        "--out",
        str((tmp_path / "output").resolve()),
        "--window",
        "75",
        "100",
    ]
