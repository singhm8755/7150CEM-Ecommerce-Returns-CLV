"""Streamlit dashboard for the return-risk and CLV project.

Four views:

* **Overview**  - headline results and what they mean commercially.
* **Score a transaction** - the model as an analyst would actually use it, with
  a SHAP explanation for every score.
* **Risk & lifetime value** - the customer portfolio, plus a live policy
  simulator that re-runs the CLV arithmetic against whatever unit economics the
  user dials in. The economics are assumptions, so they are controls rather than
  constants buried in a config file.
* **Model performance** - the evaluation figures with the tables behind them.

Every chart is paired with the table it was drawn from, so nothing depends on
reading colour alone.

Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow `streamlit run` without installing
    sys.path.insert(0, str(ROOT / "src"))

from returns_clv.artifacts import load_bundle  # noqa: E402
from returns_clv.config import load_config  # noqa: E402
from returns_clv.logging_utils import read_json  # noqa: E402

st.set_page_config(
    page_title="Return Risk & CLV", page_icon="chart_with_upwards_trend", layout="wide"
)

CONFIG = load_config()
FIGURES = CONFIG.paths.figures_path
REPORTS = CONFIG.paths.reports_path


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model…")
def get_bundle():
    return load_bundle(CONFIG.paths.models_path)


@st.cache_data(show_spinner=False)
def get_report(name: str) -> dict[str, Any] | None:
    path = REPORTS / name
    return read_json(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def get_table(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


def show_figure(name: str, caption: str = "") -> None:
    path = FIGURES / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{name} has not been generated yet. Run `returns-clv all`.")


def require_pipeline() -> bool:
    """Guard every view behind the artefacts it needs."""
    if get_report("training_summary.json") is None:
        st.warning(
            "No pipeline output found. Run `make all` (or `returns-clv all`) to train the model "
            "and generate the reports this dashboard reads."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def view_overview() -> None:
    st.title("Return risk and customer lifetime value")
    st.caption(
        "Predicting which e-commerce orders get returned, and what acting on that prediction "
        "is worth."
    )
    if not require_pipeline():
        return

    training = get_report("training_summary.json")
    clv = get_report("clv_summary.json")
    metrics = training["deployed_test_metrics"]
    ceiling = training["achievable_ceiling"]
    attainment = training["signal_attainment"]["roc_auc_attainment"]

    left, mid, right, far = st.columns(4)
    left.metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}", f"ceiling {ceiling['roc_auc']:.3f}")
    mid.metric(
        "Signal captured",
        f"{attainment:.0%}",
        help="Share of the learnable signal the model recovers",
    )
    right.metric("Average precision", f"{metrics['average_precision']:.3f}")
    far.metric(
        "Brier score", f"{metrics['brier_score']:.3f}", help="Lower is better; measures calibration"
    )

    st.subheader("What the model is worth")
    impact = pd.DataFrame(training["business_impact"])
    best = impact.loc[impact["net_saving_gbp"].idxmax()]
    blanket = impact.loc[impact["policy"] == "Intervene on every order"]

    cols = st.columns(3)
    cols[0].metric("Best policy", best["policy"], f"£{best['net_saving_gbp']:,.0f} net")
    if not blanket.empty:
        cols[1].metric(
            "Acting without a model",
            "Intervene on every order",
            f"£{blanket.iloc[0]['net_saving_gbp']:,.0f} net",
            delta_color="inverse",
        )
    if clv:
        cols[2].metric(
            "Portfolio CLV",
            f"£{clv['portfolio_clv_gbp']:,.0f}",
            f"{clv['n_customers']:,} customers",
        )

    show_figure("07_policy_impact.png")
    with st.expander("Policy comparison table"):
        st.dataframe(impact.set_index("policy").round(2), use_container_width=True)

    st.subheader("Where returns concentrate")
    show_figure("00_return_rate_by_dimension.png")


def view_scoring() -> None:
    st.title("Score a transaction")
    st.caption("The model as it is served by the API, with the reasoning behind each score.")
    try:
        bundle = get_bundle()
    except FileNotFoundError as exc:
        st.warning(str(exc))
        return

    with st.form("transaction"):
        a, b, c = st.columns(3)
        category = a.selectbox("Product category", bundle.categorical_levels["product_category"])
        payment = a.selectbox("Payment method", bundle.categorical_levels["payment_method"])
        device = a.selectbox("Device", bundle.categorical_levels["device_type"])
        segment = b.selectbox("Customer segment", bundle.categorical_levels["customer_segment"])
        order_value = b.number_input("Order value (GBP)", 10.0, 1000.0, 130.0, step=5.0)
        tenure = b.number_input("Customer tenure (days)", 0, 5000, 30)
        clicks = c.slider("Click depth", 1, 10, 3)
        dwell = c.slider("Time on page (seconds)", 5, 300, 40)
        visits = c.slider("Product page visits", 1, 20, 3)
        frequency = c.number_input("Orders in last 12 months", 0, 100, 1)
        submitted = st.form_submit_button("Score transaction", type="primary")

    if not submitted:
        st.info("Set the transaction details above and score it.")
        return

    record = {
        "product_category": category,
        "payment_method": payment,
        "device_type": device,
        "customer_segment": segment,
        "order_value_gbp": order_value,
        "click_depth": clicks,
        "time_on_page_seconds": dwell,
        "product_page_visits": visits,
        "customer_tenure_days": tenure,
        "order_frequency_12m": frequency,
    }
    frame = pd.DataFrame([record])
    probability = float(bundle.predict_proba(frame)[0])
    economics = bundle.economics
    cost_per_return = (
        order_value * economics["gross_margin_rate"]
        + economics["return_logistics_cost_gbp"]
        + order_value * economics["return_restocking_loss_rate"]
    )
    expected_saving = (
        probability * economics["intervention_effectiveness"] * cost_per_return
        - economics["intervention_cost_gbp"]
    )
    band = str(
        bundle.risk_band(
            bundle.predict_proba(frame), CONFIG.clv.risk_bands.low, CONFIG.clv.risk_bands.medium
        )[0]
    )

    left, mid, right = st.columns(3)
    left.metric("Return probability", f"{probability:.1%}")
    mid.metric("Risk band", band.replace("_", " "))
    right.metric("Expected cost if returned", f"£{cost_per_return:,.2f}")

    if expected_saving > 0:
        st.success(
            f"**Intervene.** Expected net saving £{expected_saving:,.2f} per order — the "
            f"£{economics['intervention_cost_gbp']:.2f} intervention pays for itself at this "
            "risk and order value."
        )
    else:
        st.info(
            f"**No action.** Intervening would cost £{abs(expected_saving):,.2f} more than it "
            "saves on an order of this value."
        )

    with st.spinner("Explaining the prediction…"):
        try:
            from returns_clv.explain import explain_single

            contributions = explain_single(bundle, record, top_n=8)
            st.subheader("Why")
            st.dataframe(
                pd.DataFrame(contributions).rename(
                    columns={
                        "feature": "Feature",
                        "contribution": "Contribution",
                        "direction": "Effect",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:  # pragma: no cover - explainer backends vary
            st.caption(f"Explanation unavailable: {exc}")


def view_clv() -> None:
    st.title("Risk bands and lifetime value")
    if not require_pipeline():
        return
    customers = get_table("customer_clv.csv")
    if customers is None:
        st.warning("Run `returns-clv clv` to generate the customer table.")
        return

    summary = get_report("clv_summary.json")
    a, b, c, d = st.columns(4)
    a.metric("Customers", f"{summary['n_customers']:,}")
    b.metric("Portfolio CLV", f"£{summary['portfolio_clv_gbp']:,.0f}")
    c.metric("Median CLV", f"£{summary['median_clv_gbp']:,.0f}")
    d.metric("Loss-making customers", f"{summary['negative_clv_customers']:,}")

    show_figure("08_clv_by_risk_segment.png")
    segments = get_table("clv_by_risk_segment.csv")
    if segments is not None:
        with st.expander("Risk band table"):
            st.dataframe(segments.round(3), use_container_width=True, hide_index=True)

    st.subheader("Policy simulator")
    st.caption(
        "The unit economics below are assumptions, not measurements. Move them and the lifetime "
        "value of the whole customer base is recomputed from the stored per-customer predictions."
    )
    a, b, c = st.columns(3)
    margin = a.slider(
        "Gross margin rate", 0.10, 0.60, float(CONFIG.economics.gross_margin_rate), 0.01
    )
    logistics = a.slider(
        "Logistics cost per return (£)",
        0.0,
        15.0,
        float(CONFIG.economics.return_logistics_cost_gbp),
        0.25,
    )
    restocking = b.slider(
        "Restocking loss rate", 0.0, 0.25, float(CONFIG.economics.return_restocking_loss_rate), 0.01
    )
    intervention_cost = b.slider(
        "Intervention cost (£)", 0.0, 15.0, float(CONFIG.economics.intervention_cost_gbp), 0.25
    )
    effectiveness = c.slider(
        "Intervention effectiveness",
        0.0,
        0.60,
        float(CONFIG.economics.intervention_effectiveness),
        0.01,
    )
    target_band = c.selectbox("Intervene on", ["High_Risk", "Medium_Risk and above", "Everyone"])

    from dataclasses import replace

    economics = replace(
        CONFIG.economics,
        gross_margin_rate=margin,
        return_logistics_cost_gbp=logistics,
        return_restocking_loss_rate=restocking,
        intervention_cost_gbp=intervention_cost,
        intervention_effectiveness=effectiveness,
    )
    targeted = {
        "High_Risk": customers["risk_segment"] == "High_Risk",
        "Medium_Risk and above": customers["risk_segment"].isin(["Medium_Risk", "High_Risk"]),
        "Everyone": pd.Series(True, index=customers.index),
    }[target_band]

    from returns_clv.clv import compute_clv

    baseline = compute_clv(customers, economics, CONFIG.clv)
    with_policy = compute_clv(
        customers, economics, CONFIG.clv, intervention=targeted.to_numpy(), suffix="_sim"
    )
    uplift = float(with_policy["clv_gbp_sim"].sum() - baseline["clv_gbp"].sum())

    left, mid, right = st.columns(3)
    left.metric("Portfolio CLV", f"£{baseline['clv_gbp'].sum():,.0f}")
    mid.metric("Customers targeted", f"{int(targeted.sum()):,}")
    right.metric("CLV change", f"£{uplift:,.0f}", delta=f"{uplift:,.0f}")

    if uplift <= 0:
        st.warning(
            "Under these assumptions the intervention destroys value: it costs more than the "
            "returns it prevents are worth."
        )

    st.subheader("Customer detail")
    st.dataframe(
        baseline[
            [
                "customer_id",
                "segment",
                "risk_segment",
                "n_transactions",
                "avg_order_value_gbp",
                "annual_orders",
                "predicted_return_prob",
                "actual_return_rate",
                "clv_gbp",
            ]
        ]
        .sort_values("clv_gbp", ascending=False)
        .round(3),
        use_container_width=True,
        hide_index=True,
        height=340,
    )


def view_performance() -> None:
    st.title("Model performance")
    if not require_pipeline():
        return
    training = get_report("training_summary.json")

    st.caption(
        f"Protocol: {training['protocol']['split_strategy']} split, "
        f"{training['protocol']['resampler']} resampling inside cross-validation, "
        f"{training['protocol']['calibration']} calibration. The threshold was fitted on "
        f"{training['threshold']['fitted_on']}; the test block was scored once."
    )

    comparison = get_table("model_comparison.csv")
    if comparison is not None:
        st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    with a:
        show_figure("01_roc_curves.png")
        show_figure("03_calibration.png")
    with b:
        show_figure("02_precision_recall.png")
        show_figure("05_threshold_sweep.png")
    show_figure("11_feature_importance.png")

    experiments = get_report("methodology_experiments.json")
    if experiments:
        st.subheader("What the methodology fixes are worth")
        leak = experiments["resampling_leakage"]
        st.write(
            f"Resampling inside cross-validation reports {leak['correct']['cv_score']:.3f} and "
            f"delivers {leak['correct']['test_score']:.3f} on unseen data "
            f"(gap {leak['correct']['optimism_gap']:+.3f}). Resampling beforehand reports "
            f"{leak['leaky']['cv_score']:.3f} and delivers {leak['leaky']['test_score']:.3f} "
            f"(gap {leak['leaky']['optimism_gap']:+.3f})."
        )
        st.json(experiments, expanded=False)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

VIEWS = {
    "Overview": view_overview,
    "Score a transaction": view_scoring,
    "Risk & lifetime value": view_clv,
    "Model performance": view_performance,
}

with st.sidebar:
    st.header("Return Risk & CLV")
    choice = st.radio("View", list(VIEWS), label_visibility="collapsed")
    st.divider()
    try:
        bundle = get_bundle()
        st.caption(
            f"**{bundle.model_name}**  \nthreshold {bundle.threshold:.2f}  \n"
            f"trained {bundle.trained_at[:10]}  \ndata {bundle.dataset_fingerprint}"
        )
    except FileNotFoundError:
        st.caption("No trained model loaded.")
    st.divider()
    st.caption("Coventry University 7150CEM")

VIEWS[choice]()
