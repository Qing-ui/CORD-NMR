from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


RELEASE_VERSION = "1.0.0"
ARCHIVE_PREFIX = f"CORD-NMR-v{RELEASE_VERSION}"
SKIP_DIR_NAMES = {
    ".git",
    ".github",
    ".idea",
    ".pytest_cache",
    ".ruff_cache",
    ".vscode",
    "__pycache__",
    "benchmarks",
    "cache",
    "contrib",
    "docs",
    "examples",
    "include",
    "sample_data",
    "test",
    "testdata",
    "testing",
    "tests",
}
SKIP_SUFFIXES = {
    ".a",
    ".bib",
    ".c",
    ".cc",
    ".cpp",
    ".csv",
    ".cu",
    ".doc",
    ".docx",
    ".enw",
    ".h",
    ".hpp",
    ".ipynb",
    ".lib",
    ".log",
    ".pdb",
    ".pdf",
    ".ppt",
    ".pptx",
    ".pxd",
    ".pyc",
    ".pyo",
    ".pyx",
    ".rar",
    ".ris",
    ".sd",
    ".sdf",
    ".tar",
    ".tex",
    ".xls",
    ".xlsx",
    ".zip",
    ".7z",
}


def long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def tracked_source_files(repo_root: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=repo_root, text=True, encoding="utf-8"
    )
    return [repo_root / item for item in output.splitlines() if item]


def iter_runtime_files(root: Path):
    root_text = long_path(root)
    for current, directories, filenames in os.walk(root_text, topdown=True):
        directories[:] = [
            name for name in directories if name.lower() not in SKIP_DIR_NAMES
        ]
        current_path = Path(current)
        for filename in filenames:
            source = current_path / filename
            if source.suffix.lower() in SKIP_SUFFIXES:
                continue
            yield source


def archive_name(relative: Path) -> str:
    return str(PurePosixPath(ARCHIVE_PREFIX, *relative.parts))


def add_file(
    archive: zipfile.ZipFile,
    source: Path,
    relative: Path,
    counters: dict[str, int],
) -> None:
    archive.write(str(source), archive_name(relative))
    counters["files"] += 1
    counters["bytes"] += source.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runtime_root = args.runtime_root.resolve()
    predictor_root = runtime_root / "external" / "NMR-Predictor-Portable"
    required = [
        predictor_root / "envs" / "nmrnet" / "python.exe",
        predictor_root / "envs" / "cascade2" / "python.exe",
        predictor_root / "models" / "nmrnet",
        predictor_root / "models" / "cascade2",
        args.launcher,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required release inputs:\n" + "\n".join(missing))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    counters = {"files": 0, "bytes": 0}
    with zipfile.ZipFile(
        args.output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for source in tracked_source_files(repo_root):
            relative = source.relative_to(repo_root)
            if relative.parts[:2] == (".github", "workflows"):
                continue
            add_file(archive, source, relative, counters)

        add_file(archive, args.launcher, Path("CORD-NMR.exe"), counters)

        for folder in ("envs", "models"):
            source_root = predictor_root / folder
            for source in iter_runtime_files(source_root):
                relative = Path(
                    "external",
                    "NMR-Predictor-Portable",
                    folder,
                    *source.relative_to(Path(long_path(source_root))).parts,
                )
                add_file(archive, source, relative, counters)

        manifest = (
            "CORD-NMR curated Windows portable release\n"
            f"Version: v{RELEASE_VERSION}\n"
            f"Files: {counters['files']}\n"
            f"Uncompressed bytes: {counters['bytes']}\n\n"
            "Included: GUI source, runtime launcher, required Python environments, "
            "NMRNet and CASCADE-2.0 weights.\n"
            "Excluded: manuscript files, research datasets, generated databases, "
            "results, caches, tests, examples, headers, and development artifacts.\n"
        )
        archive.writestr(archive_name(Path("RELEASE_MANIFEST.txt")), manifest)

    digest = hashlib.sha256()
    with args.output.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    print(f"archive={args.output}")
    print(f"files={counters['files']}")
    print(f"uncompressed_bytes={counters['bytes']}")
    print(f"archive_bytes={args.output.stat().st_size}")
    print(f"sha256={digest.hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
