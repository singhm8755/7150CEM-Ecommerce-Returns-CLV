"""Command-line interface."""

from __future__ import annotations

import pytest

from returns_clv.cli import build_parser, main


def test_help_lists_every_stage(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    assert "generate" in capsys.readouterr().out


def test_unknown_stage_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["deploy"])


def test_overrides_reach_the_config():
    from returns_clv.cli import _config_from_args

    args = build_parser().parse_args(["train", "--split", "random", "--seed", "99"])
    config = _config_from_args(args)
    assert config.split.strategy == "random"
    assert config.seed == 99


def test_generate_writes_a_dataset(tmp_path):
    destination = tmp_path / "generated.csv"
    exit_code = main(["generate", "--out", str(destination), "--transactions", "300", "--quiet"])
    assert exit_code == 0
    assert destination.exists()
    import pandas as pd

    assert len(pd.read_csv(destination)) == 300


def test_generate_refuses_to_clobber_an_existing_dataset(tmp_path):
    destination = tmp_path / "generated.csv"
    destination.write_text("existing,data\n1,2\n")
    with pytest.raises(SystemExit, match="already exists"):
        main(["generate", "--out", str(destination), "--transactions", "100"])
    assert destination.read_text().startswith("existing")


def test_generate_overwrites_with_force(tmp_path):
    destination = tmp_path / "generated.csv"
    destination.write_text("existing,data\n1,2\n")
    assert main(["generate", "--out", str(destination), "--transactions", "100", "--force"]) == 0
    assert "transaction_id" in destination.read_text().splitlines()[0]


def test_validate_stage_reports_on_a_generated_dataset(tmp_path):
    dataset = tmp_path / "data.csv"
    main(["generate", "--out", str(dataset), "--transactions", "800", "--quiet"])
    assert main(["validate", "--data", str(dataset), "--quiet"]) == 0


def test_a_failing_stage_returns_a_nonzero_exit_code(tmp_path):
    """Training against a dataset that does not exist must fail loudly."""
    assert main(["train", "--data", str(tmp_path / "absent.csv"), "--quiet"]) == 1
