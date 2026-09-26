"""Configuration loading, validation and overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from returns_clv.config import DEFAULT_CONFIG_PATH, Config, load_config


def test_default_config_loads():
    config = load_config()
    assert isinstance(config, Config)
    assert config.features.target == "returned"
    assert len(config.features.all_features) == 10


def test_paths_are_absolute_and_independent_of_cwd(tmp_path: Path):
    config = load_config(root=tmp_path)
    assert Path(config.paths.raw_csv).is_absolute()
    assert Path(config.paths.models_dir).is_relative_to(tmp_path)


def test_overrides_merge_without_replacing_siblings():
    config = load_config(overrides={"training": {"cv_folds": 7}})
    assert config.training.cv_folds == 7
    # An untouched sibling keeps its configured value.
    assert config.training.resampler == "smote"


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        load_config(overrides={"training": {"not_a_setting": 1}})


@pytest.mark.parametrize(
    ("section", "payload", "message"),
    [
        ("split", {"strategy": "sideways"}, "split.strategy"),
        ("split", {"test_size": 1.5}, "between 0 and 1"),
        ("split", {"test_size": 0.5, "val_size": 0.45}, "too little training data"),
        ("training", {"resampler": "magic"}, "training.resampler"),
        ("training", {"calibration": "vibes"}, "training.calibration"),
        ("training", {"cv_folds": 1}, "cv_folds"),
    ],
)
def test_invalid_values_are_rejected(section, payload, message):
    with pytest.raises(ValueError, match=message):
        load_config(overrides={section: payload})


def test_missing_config_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_shipped_config_matches_the_dataclass_schema():
    """The YAML in the repo must not contain keys the code cannot read."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    load_config()  # would raise on an unknown key
    assert set(raw) <= set(Config.__dataclass_fields__)


def test_economics_cost_per_return_components():
    config = load_config()
    economics = config.economics
    value = 200.0
    expected = (
        value * economics.gross_margin_rate
        + economics.return_logistics_cost_gbp
        + value * economics.return_restocking_loss_rate
    )
    assert economics.cost_per_return(value) == pytest.approx(expected)
    # A return always costs more than the margin it forgoes.
    assert economics.cost_per_return(value) > economics.margin_per_kept_order(value)


def test_config_is_found_from_the_working_directory(tmp_path: Path, monkeypatch):
    """A wheel install puts the package in site-packages, far from configs/.

    Resolving the config relative to the module would look inside the Python
    library directory and find nothing, crashing the container at startup. The
    working directory is searched first so the same code works from a checkout,
    a wheel and an image.
    """
    from returns_clv.config import find_config_path

    (tmp_path / "configs").mkdir()
    config_file = tmp_path / "configs" / "default.yaml"
    config_file.write_text(DEFAULT_CONFIG_PATH.read_text())

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RETURNS_CLV_CONFIG", raising=False)
    assert find_config_path() == config_file

    # Relative paths then resolve against that directory, not the install root.
    config = load_config()
    assert Path(config.paths.models_dir) == tmp_path / "models"


def test_config_env_var_takes_priority(tmp_path: Path, monkeypatch):
    from returns_clv.config import CONFIG_ENV_VAR, find_config_path

    elsewhere = tmp_path / "custom.yaml"
    elsewhere.write_text(DEFAULT_CONFIG_PATH.read_text())
    monkeypatch.setenv(CONFIG_ENV_VAR, str(elsewhere))
    assert find_config_path() == elsewhere


def test_missing_config_everywhere_names_what_was_searched(tmp_path: Path, monkeypatch):
    import returns_clv.config as config_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RETURNS_CLV_CONFIG", raising=False)
    monkeypatch.setattr(config_module, "_SOURCE_ROOT", tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match="Searched"):
        config_module.find_config_path()
