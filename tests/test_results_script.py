"""The documentation generator that keeps README numbers in step with artefacts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_results.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("update_results", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["update_results"] = loaded
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def reports(tmp_path: Path, module, monkeypatch) -> Path:
    """A minimal but structurally complete set of pipeline reports."""
    directory = tmp_path / "reports"
    directory.mkdir()
    (directory / "training_summary.json").write_text(
        json.dumps(
            {
                "selected_model": "Logistic Regression",
                "split": {"test": {"n_rows": 18121}},
                "deployed_test_metrics": {
                    "roc_auc": 0.6773,
                    "average_precision": 0.4495,
                    "brier_score": 0.1918,
                    "positive_rate_actual": 0.2969,
                },
                "achievable_ceiling": {"roc_auc": 0.6784, "average_precision": 0.4578},
                "signal_attainment": {"roc_auc_attainment": 0.9938},
                "candidates": [
                    {
                        "model": "Logistic Regression",
                        "cv_score": 0.4635,
                        "fit_seconds": 22.6,
                        "test_at_0.5": {"roc_auc": 0.6773, "average_precision": 0.4495},
                    }
                ],
                "business_impact": [
                    {
                        "policy": "Do nothing",
                        "n_flagged": 0,
                        "flag_rate": 0.0,
                        "intervention_spend_gbp": 0.0,
                        "gross_saving_gbp": 0.0,
                        "net_saving_gbp": 0.0,
                    },
                    {
                        "policy": "Intervene on every order",
                        "n_flagged": 18121,
                        "flag_rate": 1.0,
                        "intervention_spend_gbp": 86074.75,
                        "gross_saving_gbp": 64743.0,
                        "net_saving_gbp": -21331.0,
                    },
                    {
                        "policy": "Model, value-aware rule",
                        "n_flagged": 3915,
                        "flag_rate": 0.216,
                        "intervention_spend_gbp": 18596.25,
                        "gross_saving_gbp": 34495.0,
                        "net_saving_gbp": 15899.0,
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(module, "REPORTS", directory)
    return directory


def test_money_formats_losses_with_a_visible_sign(module):
    assert module.money(15899) == "£15,899"
    assert module.money(-21331) == "-£21,331"


def test_headline_block_reports_the_ceiling_alongside_the_score(module, reports):
    blocks = module.build_blocks()
    assert "headline" in blocks
    assert "0.677" in blocks["headline"]
    assert "0.678" in blocks["headline"]
    assert "99.4%" in blocks["headline"]


def test_policy_block_is_ordered_best_first(module, reports):
    rows = module.build_blocks()["policies"].splitlines()[2:]
    assert "value-aware" in rows[0]
    assert "Intervene on every order" in rows[-1]


def test_missing_reports_yield_no_blocks(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "REPORTS", tmp_path / "empty")
    assert module.build_blocks() == {}


def test_markers_are_replaced_and_reusable(module, tmp_path):
    document = tmp_path / "doc.md"
    document.write_text("intro\n<!-- BEGIN:headline -->\nold content\n<!-- END:headline -->\nend\n")

    assert module.apply(document, {"headline": "first"}) == 1
    assert "first" in document.read_text()
    assert "old content" not in document.read_text()

    # Regenerating must replace, not accumulate.
    module.apply(document, {"headline": "second"})
    text = document.read_text()
    assert "second" in text and "first" not in text
    assert text.count("<!-- BEGIN:headline -->") == 1
    assert text.startswith("intro") and text.rstrip().endswith("end")


def test_unknown_block_is_ignored(module, tmp_path):
    document = tmp_path / "doc.md"
    document.write_text("<!-- BEGIN:headline -->\nx\n<!-- END:headline -->\n")
    assert module.apply(document, {"not_a_block": "y"}) == 0


def test_every_marker_in_the_repo_has_a_builder(module):
    """A marker nobody generates would silently stay as placeholder text."""
    import re

    root = SCRIPT.parents[1]
    markers = set()
    for document in [root / "README.md", *(root / "docs").glob("*.md")]:
        markers |= set(re.findall(r"<!-- BEGIN:([a-z_]+) -->", document.read_text()))

    source = SCRIPT.read_text()
    builders = set(re.findall(r'^\s+"([a-z_]+)": lambda', source, re.M))
    assert markers <= builders, f"markers with no builder: {sorted(markers - builders)}"
