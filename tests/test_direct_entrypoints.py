import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_help(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_direct_ablation_runner_help():
    result = _run_help("run_puc_rstattn_v2_ablation.py")
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert "--variant" in result.stdout
    assert "--grid" in result.stdout


def test_direct_ablation_summary_help():
    result = _run_help("summarize_puc_rstattn_v2_ablation.py")
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_direct_centralized_summary_help():
    result = _run_help("summarize_centralized_results.py")
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
