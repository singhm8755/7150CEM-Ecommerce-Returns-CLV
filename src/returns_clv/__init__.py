"""Return-risk prediction and customer lifetime value analysis for e-commerce.

Coventry University 7150CEM Data Science Project.

The package is organised as a pipeline:

    generate -> validate -> train -> clv -> explain -> report

Each stage is a module with a single public entry point, driven by the
configuration in ``configs/default.yaml``. See :mod:`returns_clv.cli`.
"""

from returns_clv.config import Config, load_config

__version__ = "1.0.0"
__all__ = ["Config", "load_config", "__version__"]
