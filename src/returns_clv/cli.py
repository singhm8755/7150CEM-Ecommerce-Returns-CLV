"""Command-line entry point.

    returns-clv <stage> [--config PATH] [stage options]

Every stage is independently runnable and writes its artefacts to
``outputs/``; ``all`` chains them in dependency order. Stages read each other's
outputs from disk rather than passing objects in memory, so a long pipeline can
be resumed at any point after a failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from returns_clv import __version__
from returns_clv.config import DEFAULT_CONFIG_PATH, Config, load_config
from returns_clv.logging_utils import configure_logging, get_logger, write_json

logger = get_logger("returns_clv.cli")

STAGES = ("generate", "validate", "eda", "train", "clv", "explain", "experiments", "all")
PIPELINE = ("validate", "eda", "train", "clv", "explain")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="returns-clv",
        description="Return-risk prediction and customer lifetime value pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  returns-clv all                      run the full pipeline\n"
            "  returns-clv train --split random     compare against a random split\n"
            "  returns-clv generate --out data/new.csv --transactions 50000\n"
        ),
    )
    parser.add_argument("stage", choices=STAGES, help="pipeline stage to run")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML configuration file"
    )
    parser.add_argument("--data", type=Path, help="override the input dataset path")
    parser.add_argument(
        "--split",
        choices=("temporal", "random", "grouped"),
        help="override the train/test split strategy",
    )
    parser.add_argument(
        "--resampler", choices=("smote", "adasyn", "none"), help="override the resampling method"
    )
    parser.add_argument("--seed", type=int, help="override the random seed")
    parser.add_argument(
        "--search-iterations", type=int, help="hyperparameter search budget per model"
    )
    parser.add_argument("--out", type=Path, help="generate: output CSV path")
    parser.add_argument("--transactions", type=int, help="generate: number of transactions")
    parser.add_argument(
        "--force", action="store_true", help="generate: overwrite an existing dataset"
    )
    parser.add_argument("--quiet", action="store_true", help="log warnings and errors only")
    parser.add_argument("--version", action="version", version=f"returns-clv {__version__}")
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    overrides: dict[str, Any] = {}
    if args.split:
        overrides.setdefault("split", {})["strategy"] = args.split
    if args.resampler:
        overrides.setdefault("training", {})["resampler"] = args.resampler
    if args.search_iterations:
        overrides.setdefault("training", {})["search_iterations"] = args.search_iterations
    if args.seed is not None:
        overrides.setdefault("project", {})["random_seed"] = args.seed
    if args.data:
        overrides.setdefault("paths", {})["raw_csv"] = str(args.data)
    return load_config(args.config, overrides=overrides or None)


def _run_generate(config: Config, args: argparse.Namespace) -> dict[str, Any]:
    """Generate a synthetic dataset without silently replacing an existing one."""
    import pandas as pd  # noqa: F401  (imported for the type of the returned frame)

    from returns_clv.data.generate import generate_dataset

    destination = Path(args.out) if args.out else config.paths.raw_csv_path
    if destination.exists() and not args.force:
        raise SystemExit(
            f"{destination} already exists. Pass --force to overwrite it, or --out to write "
            "elsewhere. The committed dataset is the one the coursework notebooks use."
        )
    df = generate_dataset(config, n_transactions=args.transactions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    logger.info("wrote %s rows to %s", f"{len(df):,}", destination)
    return {"path": str(destination), "n_transactions": int(len(df))}


def _run_validate(config: Config) -> dict[str, Any]:
    import pandas as pd

    from returns_clv.data.validate import validate_dataset
    from returns_clv.logging_utils import banner

    banner(logger, "stage 1 of 5: data validation")
    df = pd.read_csv(config.paths.raw_csv_path)
    report = validate_dataset(df, config)
    report.log()
    config.paths.ensure_dirs()
    write_json(report.to_dict(), config.paths.reports_path / "validation_report.json")
    (config.paths.reports_path / "validation_report.md").write_text(report.to_markdown() + "\n")
    if not report.passed:
        raise SystemExit(f"validation failed: {[c.name for c in report.failures]}")
    return report.to_dict()


def run_stage(stage: str, config: Config, args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch a single stage. Imports are local so `--help` stays fast."""
    if stage == "generate":
        return _run_generate(config, args)
    if stage == "validate":
        return _run_validate(config)
    if stage == "eda":
        from returns_clv.eda import run_eda

        return run_eda(config)
    if stage == "train":
        from returns_clv.train import run_training

        return run_training(config)
    if stage == "clv":
        from returns_clv.clv import run_clv

        return run_clv(config)
    if stage == "explain":
        from returns_clv.explain import run_explain

        return run_explain(config)
    if stage == "experiments":
        from returns_clv.experiments import run_experiments

        return run_experiments(config)
    raise ValueError(f"unknown stage {stage!r}")  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(logging.WARNING if args.quiet else logging.INFO)
    config = _config_from_args(args)

    logger.info("returns-clv %s | config %s", __version__, args.config)
    stages = PIPELINE if args.stage == "all" else (args.stage,)

    for stage in stages:
        try:
            run_stage(stage, config, args)
        except SystemExit:
            raise
        except Exception:
            logger.exception("stage %r failed", stage)
            return 1

    if args.stage == "all":
        logger.info("pipeline complete - artefacts in %s", config.paths.outputs_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
