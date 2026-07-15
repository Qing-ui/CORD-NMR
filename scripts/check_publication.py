from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 20 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".bib", ".csv", ".db", ".doc", ".docx", ".enw", ".pdf", ".ppt",
    ".pptx", ".rar", ".ris", ".sd", ".sdf", ".sqlite", ".sqlite3",
    ".tar", ".tex", ".xls", ".xlsx", ".zip", ".7z",
}
FORBIDDEN_NAMES = {"PACKAGE_MANIFEST.txt", "chem_data.db-shm", "chem_data.db-wal"}
FORBIDDEN_REPOSITORY_PATHS = {
    Path("nmr_trendtrack/models/enumerated_v5.py"),
    Path("single_spectrum/clustering.py"),
    Path("single_spectrum/config.py"),
}
FORBIDDEN_PARTS = {
    ("external", "NMR-Predictor-Portable", "envs"),
    ("external", "NMR-Predictor-Portable", "models"),
    ("nmr_trendtrack", "align"),
    ("nmr_trendtrack", "cluster"),
    ("nmr_trendtrack", "component"),
    ("nmr_trendtrack", "io"),
    ("nmr_trendtrack", "optimize"),
    ("nmr_trendtrack", "postprocess"),
    ("nmr_trendtrack", "preprocess"),
    ("nmr_trendtrack", "trend"),
}
FORBIDDEN_FILENAME_WORDS = ("draft", "manuscript", "supplement")
LOCAL_PATH_PATTERNS = (
    re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:\\"),
    re.compile(r"/mnt/data/"),
)


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8"
    )
    return [ROOT / line for line in output.splitlines() if line]


def main() -> int:
    failures: list[str] = []
    checker = Path(__file__).resolve()
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        parts = relative.parts
        lowered_name = path.name.lower()
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden publication file: {relative}")
        if relative in FORBIDDEN_REPOSITORY_PATHS:
            failures.append(f"non-GUI clustering compatibility file: {relative}")
        if any(tuple(parts[: len(prefix)]) == prefix for prefix in FORBIDDEN_PARTS):
            failures.append(f"excluded path tracked in Git: {relative}")
        if any(word in lowered_name for word in FORBIDDEN_FILENAME_WORDS):
            failures.append(f"manuscript or draft filename: {relative}")
        if path.stat().st_size > MAX_TRACKED_BYTES:
            failures.append(f"tracked file exceeds 20 MiB: {relative}")
        if path == checker or path.suffix.lower() not in {".py", ".md", ".txt", ".bat"}:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if any(pattern.search(text) for pattern in LOCAL_PATH_PATTERNS):
            failures.append(f"local absolute path found: {relative}")

    if failures:
        print("Publication check failed:")
        for failure in sorted(set(failures)):
            print(f"- {failure}")
        return 1
    print("Publication check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
