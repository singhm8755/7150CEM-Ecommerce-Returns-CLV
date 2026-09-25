#!/usr/bin/env python3
"""Inject measured results from the pipeline into the Markdown documentation.

Numbers in a README go stale the moment the pipeline is rerun, and a stale
number is worse than no number because nothing signals that it is wrong. This
script regenerates every results block from the JSON artefacts in
``outputs/reports`` so the documentation cannot drift from the run that produced
it.

Blocks are delimited in the Markdown by HTML comments:

    <!-- BEGIN:headline -->
    ...generated content, do not edit by hand...
    <!-- END:headline -->

Run with ``make results`` after ``make all``.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
TARGETS = [ROOT / "README.md", ROOT / "docs" / "model_card.md", ROOT / "docs" / "methodology.md"]


def read(name: str) -> dict[str, Any] | None:
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else None


def money(value: float) -> str:
    """Format pounds, keeping the sign visible for losses."""
    return f"-£{abs(value):,.0f}" if value < 0 else f"£{value:,.0f}"


# ---------------------------------------------------------------------------
# Block builders. Each returns Markdown, or None when its inputs are missing.
# ---------------------------------------------------------------------------


def block_headline(training: dict, clv: dict | None) -> str:
    metrics = training["deployed_test_metrics"]
    ceiling = training["achievable_ceiling"]
    attain = training["signal_attainment"]["roc_auc_attainment"]
    impact = training["business_impact"]
    best = max(impact, key=lambda row: row["net_saving_gbp"])
    blanket = next((r for r in impact if r["policy"] == "Intervene on every order"), None)

    lines = [
        "| Result | Value | Context |",
        "| --- | --- | --- |",
        f"| Test ROC-AUC | **{metrics['roc_auc']:.3f}** | "
        f"against an achievable ceiling of {ceiling['roc_auc']:.3f} |",
        f"| Share of achievable signal captured | **{attain:.1%}** | "
        "the remaining gap is irreducible noise |",
        f"| Average precision | {metrics['average_precision']:.3f} | "
        f"no-skill baseline {metrics['positive_rate_actual']:.3f} |",
        f"| Brier score | {metrics['brier_score']:.3f} | "
        "calibrated, so the probabilities can be multiplied by money |",
        f"| Best intervention policy | **{money(best['net_saving_gbp'])}** | "
        f"{best['policy']}, over the "
        f"{training['split']['test']['n_rows']:,}-order test period |",
    ]
    if blanket:
        lines.append(
            f"| Same intervention without a model | {money(blanket['net_saving_gbp'])} | "
            "intervening on every order destroys value |"
        )
    if clv:
        lines.append(
            f"| Portfolio lifetime value | {money(clv['portfolio_clv_gbp'])} | "
            f"{clv['n_customers']:,} customers, "
            f"{clv['assumptions']['horizon_years']}-year horizon |"
        )
    return "\n".join(lines)


def block_models(training: dict) -> str:
    rows = [
        "| Model | CV average precision | Test ROC-AUC | Test average precision | Fit time |",
        "| --- | --- | --- | --- | --- |",
    ]
    for candidate in training["candidates"]:
        cv = candidate["cv_score"]
        test = candidate["test_at_0.5"]
        selected = candidate["model"] == training["selected_model"]
        name = f"**{candidate['model']}**" if selected else candidate["model"]
        cv_text = "—" if cv is None or cv != cv else f"{cv:.4f}"
        rows.append(
            f"| {name} | {cv_text} | {test['roc_auc']:.4f} | "
            f"{test['average_precision']:.4f} | {candidate['fit_seconds']:.0f}s |"
        )
    ceiling = training["achievable_ceiling"]
    rows.append(
        f"| _Achievable ceiling_ | — | _{ceiling['roc_auc']:.4f}_ | "
        f"_{ceiling['average_precision']:.4f}_ | — |"
    )
    return "\n".join(rows)


def block_policies(training: dict) -> str:
    rows = [
        "| Policy | Orders flagged | Intervention spend | Gross saving | **Net** |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(training["business_impact"], key=lambda r: -r["net_saving_gbp"]):
        rows.append(
            f"| {row['policy']} | {row['n_flagged']:,} ({row['flag_rate']:.0%}) | "
            f"{money(row['intervention_spend_gbp'])} | {money(row['gross_saving_gbp'])} | "
            f"**{money(row['net_saving_gbp'])}** |"
        )
    return "\n".join(rows)


def block_clv_segments(clv: dict) -> str:
    rows = [
        "| Risk band | Customers | Mean predicted return rate | Actual return rate | Mean CLV |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for segment in clv["risk_segments"]:
        rows.append(
            f"| {segment['risk_segment'].replace('_', ' ')} | {int(segment['customers']):,} | "
            f"{segment['mean_predicted_return_prob']:.1%} | "
            f"{segment['mean_actual_return_rate']:.1%} | "
            f"{money(segment['mean_clv_gbp'])} |"
        )
    return "\n".join(rows)


def block_experiment_leakage(experiments: dict) -> str:
    leak = experiments["resampling_leakage"]
    metric = leak["metric"].replace("_", " ")
    return "\n".join(
        [
            f"| Protocol | Cross-validated {metric} | Test {metric} | Optimism gap |",
            "| --- | ---: | ---: | ---: |",
            f"| Resampling inside cross-validation (correct) | {leak['correct']['cv_score']:.4f} | "
            f"{leak['correct']['test_score']:.4f} | **{leak['correct']['optimism_gap']:+.4f}** |",
            f"| Resampling before cross-validation (leaky) | {leak['leaky']['cv_score']:.4f} | "
            f"{leak['leaky']['test_score']:.4f} | **{leak['leaky']['optimism_gap']:+.4f}** |",
            "",
            f"Resampling first inflates the reported cross-validated score by "
            f"{leak['inflation_in_reported_cv']:+.4f} without improving the model.",
        ]
    )


def block_experiment_threshold(experiments: dict) -> str:
    result = experiments["threshold_optimism"]
    return "\n".join(
        [
            "| Threshold chosen on | Threshold | F1 reported on the test set |",
            "| --- | ---: | ---: |",
            f"| Validation (correct) | {result['threshold_from_validation']:.2f} | "
            f"{result['reported_f1_honest']:.4f} |",
            f"| Test set (the mistake) | {result['threshold_from_test']:.2f} | "
            f"{result['reported_f1_tuned_on_test']:.4f} |",
            "",
            f"Tuning on the test set overstates F1 by "
            f"{result['overstatement']:+.4f} — an improvement nobody could have "
            "obtained in advance.",
        ]
    )


def block_experiment_ablation(experiments: dict) -> str:
    results = experiments["feature_ablation"]
    rows = [
        "| Model | Raw columns only | With engineered features | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model, scores in results.items():
        rows.append(
            f"| {model.replace('_', ' ').title()} | "
            f"{scores['raw_features_only']['average_precision']:.4f} | "
            f"{scores['with_engineered_features']['average_precision']:.4f} | "
            f"**{scores['average_precision_delta']:+.4f}** |"
        )
    rows.append("")
    rows.append("_Test-set average precision._")
    return "\n".join(rows)


def block_features(explain: dict) -> str:
    rows = ["| Feature | Mean absolute SHAP | Pushes towards |", "| --- | ---: | --- |"]
    for feature in explain["top_features"][:10]:
        rows.append(
            f"| `{feature['feature']}` | {feature['mean_abs_shap']:.4f} | "
            f"{feature['pushes_towards']} |"
        )
    return "\n".join(rows)


def block_validation(validation: dict) -> str:
    rows = ["| Check | Status | Detail |", "| --- | --- | --- |"]
    for check in validation["checks"]:
        detail = check["detail"].replace("|", "/")
        rows.append(f"| `{check['name']}` | {check['status']} | {detail} |")
    return "\n".join(rows)


def build_blocks() -> dict[str, str]:
    training = read("training_summary.json")
    clv = read("clv_summary.json")
    explain = read("explainability.json")
    experiments = read("methodology_experiments.json")
    validation = read("validation_report.json")

    builders: dict[str, Callable[[], str | None]] = {
        "headline": lambda: block_headline(training, clv) if training else None,
        "models": lambda: block_models(training) if training else None,
        "policies": lambda: block_policies(training) if training else None,
        "clv_segments": lambda: block_clv_segments(clv) if clv else None,
        "features": lambda: block_features(explain) if explain else None,
        "validation": lambda: block_validation(validation) if validation else None,
        "experiment_leakage": lambda: (
            block_experiment_leakage(experiments) if experiments else None
        ),
        "experiment_threshold": lambda: (
            block_experiment_threshold(experiments) if experiments else None
        ),
        "experiment_ablation": lambda: (
            block_experiment_ablation(experiments) if experiments else None
        ),
    }
    blocks = {}
    for key, builder in builders.items():
        content = builder()
        if content:
            blocks[key] = content
    return blocks


def apply(path: Path, blocks: dict[str, str]) -> int:
    """Replace every delimited block in ``path``. Returns how many were updated."""
    if not path.exists():
        return 0
    text = path.read_text()
    updated = 0
    for key, content in blocks.items():
        pattern = re.compile(
            rf"(<!-- BEGIN:{re.escape(key)} -->\n).*?(\n<!-- END:{re.escape(key)} -->)",
            re.DOTALL,
        )
        text, count = pattern.subn(lambda m, body=content: m.group(1) + body + m.group(2), text)
        updated += count
    path.write_text(text)
    return updated


def main() -> int:
    blocks = build_blocks()
    if not blocks:
        print("No pipeline reports found. Run `make all` first.", file=sys.stderr)
        return 1

    total = 0
    for target in TARGETS:
        count = apply(target, blocks)
        total += count
        print(f"{target.relative_to(ROOT)}: updated {count} block(s)")

    missing = {key for key in ("headline", "models", "policies") if key not in blocks}
    if missing:
        print(f"warning: no data for {sorted(missing)}", file=sys.stderr)
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
