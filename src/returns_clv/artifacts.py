"""Saving and loading the deployable model bundle.

A trained model on its own is not deployable: scoring a transaction also needs
the operating threshold, the exact input columns in the right order, and enough
provenance to tell which dataset and configuration produced it. All of that
travels together in a single :class:`ModelBundle`, which is what the API, the
dashboard and the CLV stage all load. There is therefore exactly one code path
from raw input columns to a flagging decision, in training and in serving alike.
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from returns_clv import __version__
from returns_clv.config import Config
from returns_clv.logging_utils import get_logger

logger = get_logger(__name__)

BUNDLE_FILENAME = "return_risk_model.joblib"
BUNDLE_FORMAT_VERSION = 2


@dataclass
class ModelBundle:
    """Everything needed to score a transaction, plus provenance."""

    pipeline: Any
    threshold: float
    model_name: str
    feature_columns: list[str]
    categorical_levels: dict[str, list[str]]
    threshold_criterion: str = "net_saving"
    # A sample of training rows in transformed feature space. SHAP needs a
    # reference distribution to explain a prediction against: without one, a
    # single-row request is explained against itself and every contribution
    # collapses to zero.
    explainer_background: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    config_summary: dict[str, Any] = field(default_factory=dict)
    economics: dict[str, float] = field(default_factory=dict)
    dataset_fingerprint: str = ""
    trained_at: str = ""
    package_version: str = __version__
    format_version: int = BUNDLE_FORMAT_VERSION
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")
        if not self.trained_at:
            self.trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.environment:
            self.environment = {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            }

    # -- scoring ----------------------------------------------------------
    def prepare(self, records: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        """Coerce input to the exact columns and order the pipeline expects."""
        frame = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise KeyError(f"missing required feature columns: {missing}")
        return frame.loc[:, self.feature_columns].copy()

    def predict_proba(self, records: pd.DataFrame | list[dict[str, Any]]) -> np.ndarray:
        """Probability of return for each input row."""
        return self.pipeline.predict_proba(self.prepare(records))[:, 1]

    def flag(self, records: pd.DataFrame | list[dict[str, Any]]) -> np.ndarray:
        """Boolean flag at the bundle's deployed threshold."""
        return self.predict_proba(records) >= self.threshold

    def flag_value_aware(self, records: pd.DataFrame | list[dict[str, Any]]) -> np.ndarray:
        """Flag rows where intervening has positive expected value.

        The global threshold treats a GBP 20 order and a GBP 500 order alike,
        but the saving from preventing a return scales with order value while
        the intervention costs the same either way. This rule compares the two
        per transaction, and is the policy the evaluation stage shows to be
        worth the most money.
        """
        frame = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records
        if "order_value_gbp" not in frame.columns:
            raise KeyError("value-aware flagging needs an 'order_value_gbp' column")
        margin = self.economics.get("gross_margin_rate", 0.30)
        logistics = self.economics.get("return_logistics_cost_gbp", 4.50)
        restocking = self.economics.get("return_restocking_loss_rate", 0.05)
        cost = self.economics.get("intervention_cost_gbp", 4.75)
        effectiveness = self.economics.get("intervention_effectiveness", 0.25)

        values = frame["order_value_gbp"].to_numpy(dtype=float)
        return_cost = values * margin + logistics + values * restocking
        expected_saving = self.predict_proba(frame) * effectiveness * return_cost
        return expected_saving > cost

    def risk_band(self, probabilities: np.ndarray, low: float, medium: float) -> np.ndarray:
        """Map probabilities to Low/Medium/High risk labels."""
        return np.select(
            [probabilities < low, probabilities < medium],
            ["Low_Risk", "Medium_Risk"],
            default="High_Risk",
        )

    def describe(self) -> dict[str, Any]:
        """Serialisable summary for the API's ``/model`` endpoint."""
        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "threshold_criterion": self.threshold_criterion,
            "trained_at": self.trained_at,
            "package_version": self.package_version,
            "format_version": self.format_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "feature_columns": self.feature_columns,
            "categorical_levels": self.categorical_levels,
            "metrics": self.metrics,
            "economics": self.economics,
            "environment": self.environment,
        }


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Short, stable hash of a dataset, so a model can be tied to its training data."""
    digest = hashlib.sha256()
    digest.update(str(df.shape).encode())
    digest.update(",".join(map(str, df.columns)).encode())
    # Hashing the values directly would be slow on large frames; a summary of the
    # numeric content is enough to detect a different dataset.
    numeric = df.select_dtypes("number")
    digest.update(np.ascontiguousarray(numeric.to_numpy(dtype=float)).tobytes()[:1_000_000])
    return digest.hexdigest()[:16]


def categorical_levels(df: pd.DataFrame, config: Config) -> dict[str, list[str]]:
    """Observed values for each categorical column, for API input validation."""
    return {col: sorted(df[col].dropna().unique().tolist()) for col in config.features.categorical}


def save_bundle(bundle: ModelBundle, directory: Path, filename: str = BUNDLE_FILENAME) -> Path:
    """Persist ``bundle`` with joblib and return the written path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    joblib.dump(bundle, path, compress=3)
    logger.info("saved model bundle to %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def load_bundle(path: Path | str) -> ModelBundle:
    """Load a bundle, with a clear error when training has not been run yet."""
    path = Path(path)
    if path.is_dir():
        path = path / BUNDLE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"no model bundle at {path}. Run `make train` (or `returns-clv train`) first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):  # pragma: no cover
        raise TypeError(f"{path} does not contain a ModelBundle")
    if bundle.format_version != BUNDLE_FORMAT_VERSION:
        logger.warning(
            "bundle format version %s differs from the expected %s",
            bundle.format_version,
            BUNDLE_FORMAT_VERSION,
        )
    return bundle
