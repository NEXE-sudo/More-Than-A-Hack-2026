"""Subprocess integration with the unchanged official starter and evaluator."""

import os
import subprocess
import sys
from pathlib import Path


def run_official_baseline(
    repository_root: Path | str,
    data_root: Path | str,
    output_dir: Path | str,
    *,
    smoothing_half_widths: tuple[int, ...] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run starter/baseline.py in its own repository, surfacing output and exit code."""
    root = Path(repository_root).expanduser().resolve()
    script = root / "starter" / "baseline.py"
    if not script.is_file():
        raise FileNotFoundError(f"official baseline not found: {script}")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MTAH_DATA_DIR"] = str(Path(data_root).expanduser().resolve())
    command = [sys.executable, str(script), "--out", str(destination)]
    if smoothing_half_widths is not None:
        if not smoothing_half_widths or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 0
            for width in smoothing_half_widths
        ):
            raise ValueError("smoothing half-widths must be non-negative integers")
        if len(set(smoothing_half_widths)) != len(smoothing_half_widths):
            raise ValueError("smoothing half-widths must be unique")
        command.extend(["--window", *(str(width) for width in smoothing_half_widths)])
    result = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr
        )
    return result


def run_official_evaluator(
    repository_root: Path | str,
    results_dir: Path | str,
    output_dir: Path | str,
    ground_truth: str = "public",
    bootstrap: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Run the repository's evaluator without copying or importing its implementation."""
    if ground_truth.strip().casefold() == "private":
        raise ValueError("private ground truth is not available or permitted for this project")
    if bootstrap < 0:
        raise ValueError("bootstrap draw count cannot be negative")
    root = Path(repository_root).expanduser().resolve()
    script = root / "evaluation" / "run_evaluation.py"
    if not script.is_file():
        raise FileNotFoundError(f"official evaluator not found: {script}")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--gt",
            ground_truth,
            "--results",
            str(Path(results_dir).expanduser().resolve()),
            "--out",
            str(Path(output_dir).expanduser().resolve()),
            "--bootstrap",
            str(bootstrap),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr
        )
    return result
