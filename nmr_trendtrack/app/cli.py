from __future__ import annotations

import argparse

from nmr_trendtrack.app.service import run_joint_from_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nmr-trendtrack")
    sub = parser.add_subparsers(dest="command", required=True)

    run_joint = sub.add_parser("run-joint", help="Run the switchable cross-spectrum clustering pipeline")
    run_joint.add_argument("--config", required=True, help="YAML config path")
    run_joint.add_argument("--manifest", required=True, help="YAML sample manifest path")
    run_joint.add_argument("--output-dir", required=True, help="Output directory")
    run_joint.add_argument(
        "--model",
        default=None,
        choices=[
            "v5_enum",
            "v5_enumerated",
            "enumerated_v5",
            "v5_pmtc",
            "v5_pmtc_r85",
            "v5",
            "v4_baseline",
            "v4",
            "original_t",
            "t_mixture",
            "t",
        ],
        help="Model backend. Defaults to config model.name, which defaults to v5_pmtc.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-joint":
        run_joint_from_files(args.config, args.manifest, args.output_dir, model=args.model)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
