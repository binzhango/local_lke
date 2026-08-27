import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "demo_chapter.sh"


def test_demo_launcher_lists_all_chapters() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "list"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "1  Naive cited RAG baseline" in result.stdout
    assert "7  Bearer authentication" in result.stdout


def test_demo_launcher_rejects_an_unknown_chapter() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "8"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Usage:" in result.stderr
