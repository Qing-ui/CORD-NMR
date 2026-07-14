from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


RELEASE_VERSION = "1.0.0"
ARCHIVE_PREFIX = f"CORD-NMR-v{RELEASE_VERSION}"
ROOT_RUNTIME_FILES = {
    "Install-CORD-NMR.bat",
    "LICENSE",
    "README.md",
    "requirements.txt",
    "run_gui.bat",
}
RUNTIME_PREFIXES = {
    "nmr_trendtrack",
    "services",
    "single_spectrum",
}
EXTERNAL_RUNTIME_FILES = {
    Path("external/NMR-Predictor-Portable/MODEL_ASSETS.md"),
    Path("external/NMR-Predictor-Portable/README.md"),
    Path("external/NMR-Predictor-Portable/requirements-cascade2.txt"),
    Path("external/NMR-Predictor-Portable/requirements-nmrnet.txt"),
}


def tracked_source_files(repo_root: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=repo_root, text=True, encoding="utf-8"
    )
    return [repo_root / item for item in output.splitlines() if item]


def is_runtime_file(relative: Path) -> bool:
    if len(relative.parts) == 1:
        return relative.name in ROOT_RUNTIME_FILES or relative.suffix == ".py"
    if relative.parts[0] in RUNTIME_PREFIXES:
        return relative.suffix == ".py"
    if relative.parts[:3] == ("external", "NMR-Predictor-Portable", "app"):
        return relative.suffix in {".bat", ".py", ".txt"}
    if relative in EXTERNAL_RUNTIME_FILES:
        return True
    return relative in {
        Path("docs/THIRD_PARTY_NOTICES.md"),
        Path("scripts/install_prediction_runtime.ps1"),
    }


def archive_name(relative: Path) -> str:
    return str(PurePosixPath(ARCHIVE_PREFIX, *relative.parts))


def add_file(
    archive: zipfile.ZipFile,
    source: Path,
    relative: Path,
    counters: dict[str, int],
) -> None:
    archive.write(source, archive_name(relative))
    counters["files"] += 1
    counters["bytes"] += source.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runtime_files = [
        source
        for source in tracked_source_files(repo_root)
        if is_runtime_file(source.relative_to(repo_root))
    ]
    required = [repo_root / name for name in ROOT_RUNTIME_FILES]
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
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for source in runtime_files:
            add_file(archive, source, source.relative_to(repo_root), counters)

        manifest = (
            "CORD-NMR Windows installer release\n"
            f"Version: v{RELEASE_VERSION}\n"
            f"Files: {counters['files']}\n"
            f"Uncompressed bytes: {counters['bytes']}\n\n"
            "Included: GUI runtime source, one-click installer, pinned dependency "
            "definitions, prediction bridge source, and third-party notices.\n"
            "Downloaded during installation: isolated Python environments and the "
            "four verified inference assets published with release v1.0.0.\n"
            "Excluded: training data, manuscripts, research datasets, generated "
            "results, caches, tests, notebooks, and development artifacts.\n"
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
