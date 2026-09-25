"""Typed configuration loaded from YAML.

Configuration is deliberately explicit rather than a dict-of-dicts: the
dataclasses below give editor completion, fail fast on typos, and document the
contract between pipeline stages in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

T = TypeVar("T")


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "returns-clv"
    random_seed: int = 42


@dataclass(frozen=True)
class PathsConfig:
    raw_csv: str = "data/synthetic_ecommerce.csv"
    models_dir: str = "models"
    outputs_dir: str = "outputs"
    figures_dir: str = "outputs/figures"
    reports_dir: str = "outputs/reports"

    def resolve(self, root: Path) -> PathsConfig:
        """Return a copy with every path made absolute against ``root``."""
        return PathsConfig(
            **{f.name: str((root / getattr(self, f.name)).resolve()) for f in fields(self)}
        )

    @property
    def raw_csv_path(self) -> Path:
        return Path(self.raw_csv)

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir)

    @property
    def outputs_path(self) -> Path:
        return Path(self.outputs_dir)

    @property
    def figures_path(self) -> Path:
        return Path(self.figures_dir)

    @property
    def reports_path(self) -> Path:
        return Path(self.reports_dir)

    def ensure_dirs(self) -> None:
        for path in (self.models_path, self.outputs_path, self.figures_path, self.reports_path):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class GeneratorEffects:
    segment_First_Time: float = 1.4
    segment_Wholesale: float = 0.3
    payment_Cash_on_Delivery: float = 1.5
    device_Mobile: float = 1.1
    low_click_depth: float = 1.2
    short_dwell: float = 1.15


@dataclass(frozen=True)
class DataGenerationConfig:
    n_customers: int = 12_000
    n_transactions: int = 120_000
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"
    category_base_return_rate: dict[str, float] = field(
        default_factory=lambda: {"Fashion": 0.30, "Electronics": 0.15, "Home_Garden": 0.20}
    )
    category_mix: dict[str, float] = field(
        default_factory=lambda: {"Fashion": 0.45, "Electronics": 0.35, "Home_Garden": 0.20}
    )
    payment_mix: dict[str, float] = field(
        default_factory=lambda: {"Credit_Card": 0.60, "PayPal": 0.30, "Cash_on_Delivery": 0.10}
    )
    device_mix: dict[str, float] = field(
        default_factory=lambda: {"Mobile": 0.45, "Desktop": 0.35, "Tablet": 0.20}
    )
    segment_mix: dict[str, float] = field(
        default_factory=lambda: {"First_Time": 0.40, "Repeat": 0.50, "Wholesale": 0.10}
    )
    effects: GeneratorEffects = field(default_factory=GeneratorEffects)
    max_return_probability: float = 0.70


@dataclass(frozen=True)
class FeaturesConfig:
    target: str = "returned"
    date_column: str = "transaction_date"
    id_columns: list[str] = field(default_factory=lambda: ["transaction_id", "customer_id"])
    categorical: list[str] = field(
        default_factory=lambda: [
            "product_category",
            "payment_method",
            "device_type",
            "customer_segment",
        ]
    )
    numeric: list[str] = field(
        default_factory=lambda: [
            "order_value_gbp",
            "click_depth",
            "time_on_page_seconds",
            "product_page_visits",
            "customer_tenure_days",
            "order_frequency_12m",
        ]
    )

    @property
    def all_features(self) -> list[str]:
        return [*self.categorical, *self.numeric]


@dataclass(frozen=True)
class SplitConfig:
    strategy: str = "temporal"
    test_size: float = 0.15
    val_size: float = 0.15

    def __post_init__(self) -> None:
        allowed = {"temporal", "random", "grouped"}
        if self.strategy not in allowed:
            raise ValueError(
                f"split.strategy must be one of {sorted(allowed)}, got {self.strategy!r}"
            )
        if not 0 < self.test_size < 1 or not 0 < self.val_size < 1:
            raise ValueError("split sizes must be between 0 and 1")
        if self.test_size + self.val_size >= 0.9:
            raise ValueError("split.test_size + split.val_size leaves too little training data")


@dataclass(frozen=True)
class TrainingConfig:
    models: list[str] = field(
        default_factory=lambda: ["logistic_regression", "random_forest", "xgboost"]
    )
    resampler: str = "smote"
    cv_folds: int = 3
    search_iterations: int = 8
    scoring: str = "average_precision"
    selection_metric: str = "average_precision"
    calibration: str = "isotonic"
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.resampler not in {"smote", "adasyn", "none"}:
            raise ValueError(
                f"training.resampler must be smote|adasyn|none, got {self.resampler!r}"
            )
        if self.calibration not in {"isotonic", "sigmoid", "none"}:
            raise ValueError("training.calibration must be isotonic|sigmoid|none")
        if self.cv_folds < 2:
            raise ValueError("training.cv_folds must be >= 2")


@dataclass(frozen=True)
class EconomicsConfig:
    """Unit economics driving the cost-sensitive threshold and CLV model."""

    gross_margin_rate: float = 0.30
    return_logistics_cost_gbp: float = 4.50
    return_restocking_loss_rate: float = 0.05
    intervention_cost_gbp: float = 4.75
    intervention_effectiveness: float = 0.25

    def margin_per_kept_order(self, order_value: float) -> float:
        return order_value * self.gross_margin_rate

    def cost_per_return(self, order_value: float) -> float:
        """Total loss when an order is returned: forgone margin plus handling."""
        return (
            order_value * self.gross_margin_rate
            + self.return_logistics_cost_gbp
            + order_value * self.return_restocking_loss_rate
        )


@dataclass(frozen=True)
class RiskBands:
    low: float = 0.25
    medium: float = 0.50


@dataclass(frozen=True)
class ClvConfig:
    horizon_years: int = 3
    discount_rate: float = 0.10
    annual_retention_rate: float = 0.75
    risk_bands: RiskBands = field(default_factory=RiskBands)


@dataclass(frozen=True)
class ExplainabilityConfig:
    shap_sample_size: int = 3000


@dataclass(frozen=True)
class Config:
    """Root configuration object handed to every pipeline stage."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    clv: ClvConfig = field(default_factory=ClvConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)

    @property
    def seed(self) -> int:
        return self.project.random_seed


def _build(cls: type[T], raw: Any) -> T:
    """Recursively construct a (possibly nested) dataclass from plain data."""
    if not is_dataclass(cls):
        return raw  # type: ignore[return-value]
    if raw is None:
        return cls()  # type: ignore[call-arg]
    if not isinstance(raw, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(raw).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in {cls.__name__}; valid keys are {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        # Nested sections are looked up by class rather than by resolving the
        # string annotations that `from __future__ import annotations` produces.
        nested = _NESTED.get((cls, name))
        kwargs[name] = _build(nested, value) if nested is not None else value
    return cls(**kwargs)  # type: ignore[call-arg]


# Explicit nesting map keeps `from __future__ import annotations` (string
# annotations) from forcing runtime type resolution.
_NESTED: dict[tuple[type, str], type] = {
    (Config, "project"): ProjectConfig,
    (Config, "paths"): PathsConfig,
    (Config, "data_generation"): DataGenerationConfig,
    (Config, "features"): FeaturesConfig,
    (Config, "split"): SplitConfig,
    (Config, "training"): TrainingConfig,
    (Config, "economics"): EconomicsConfig,
    (Config, "clv"): ClvConfig,
    (Config, "explainability"): ExplainabilityConfig,
    (DataGenerationConfig, "effects"): GeneratorEffects,
    (ClvConfig, "risk_bands"): RiskBands,
}


def load_config(
    path: str | Path | None = None,
    *,
    root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration from YAML and resolve paths against the project root.

    Args:
        path: YAML file to read. Defaults to ``configs/default.yaml``.
        root: Directory that relative paths are resolved against. Defaults to
            the repository root, so the pipeline behaves the same from any cwd.
        overrides: Nested dict merged over the file contents, e.g.
            ``{"training": {"cv_folds": 2}}``. Used by tests and CLI flags.

    Returns:
        A fully populated, immutable :class:`Config`.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text()) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)

    config = _build(Config, raw)
    resolved_root = (root or PROJECT_ROOT).resolve()
    return Config(
        project=config.project,
        paths=config.paths.resolve(resolved_root),
        data_generation=config.data_generation,
        features=config.features,
        split=config.split,
        training=config.training,
        economics=config.economics,
        clv=config.clv,
        explainability=config.explainability,
    )


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
