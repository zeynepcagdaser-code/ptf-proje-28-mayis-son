#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_DIR = DATA_DIR / "features"
PRED_DIR = DATA_DIR / "predictions"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models" / "regime_aware_regressor"
CVE_PATH = FEATURE_DIR / "supply_demand_curve_features.parquet"
RECON_DAILY_DATE = "2026-06-01"
RECON_DAILY_FEATURES = FEATURE_DIR / f"reconstructed_daily_curve_features_{RECON_DAILY_DATE}.parquet"
RECON_DAILY_REPORT = REPORT_DIR / f"reconstructed_daily_curve_{RECON_DAILY_DATE}.json"
RECON_DAILY_PNG = REPORT_DIR / "curve_debug_examples" / RECON_DAILY_DATE / "daily_2026-06-01_curve.png"

st.set_page_config(page_title="Regime-Aware PTF Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
    .stMetric { background: rgba(20, 20, 20, 0.35); padding: 0.75rem; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Regime-Aware PTF Forecasting Dashboard")


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    from src.utils.io_utils import read_parquet_with_normalized_ts
    return read_parquet_with_normalized_ts(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_models() -> dict[str, object]:
    models = {}
    if (MODEL_DIR / "splitter.joblib").exists():
        models["splitter"] = joblib.load(MODEL_DIR / "splitter.joblib")
    if (MODEL_DIR / "normal_residual_lgb.joblib").exists():
        models["normal"] = joblib.load(MODEL_DIR / "normal_residual_lgb.joblib")
    if (MODEL_DIR / "spike_residual_lgb_custom.joblib").exists():
        models["spike"] = joblib.load(MODEL_DIR / "spike_residual_lgb_custom.joblib")
    return models


def safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def available_curve_dates() -> list[str]:
    dates = []
    if RECON_DAILY_FEATURES.exists():
        dates.append(RECON_DAILY_DATE)
    return dates


def resolve_forecast() -> pd.DataFrame:
    paths = [
        PRED_DIR / "tomorrow_morning_ptf_forecast.csv",
        PRED_DIR / "regime_aware_regressor_predictions.csv",
    ]
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            if "ts_hour" in df.columns:
                df["ts_hour"] = pd.to_datetime(df["ts_hour"], errors="coerce")
                return df.sort_values("ts_hour")
    return pd.DataFrame()


def enrich_forecast_with_curve_proxy(forecast: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty or curve.empty or "ts_hour" not in forecast.columns or "ts_hour" not in curve.columns:
        return forecast
    cf = curve.copy()
    cf["ts_hour"] = pd.to_datetime(cf["ts_hour"], errors="coerce")
    cf["hour_of_day"] = cf["ts_hour"].dt.hour
    proxy_cols = [c for c in [
        "cap_risk_from_curve",
        "marginality_risk_score",
        "low_price_pressure_score",
        "oversupply_curve_pressure",
        "supply_gap_score",
    ] if c in cf.columns]
    if not proxy_cols:
        return forecast
    hourly_proxy = cf.groupby("hour_of_day")[proxy_cols].median(numeric_only=True).reset_index()
    out = forecast.copy()
    out["hour_of_day"] = out["ts_hour"].dt.hour
    out = out.merge(hourly_proxy, on="hour_of_day", how="left", suffixes=("", "_curve_proxy"))
    return out


def regime_color_map() -> dict[str, str]:
    return {
        "negative_zero_pressure": "#4cc9f0",
        "normal": "#adb5bd",
        "tight": "#f9c74f",
        "spike_cap": "#f94144",
    }


def add_regime_bands(fig: go.Figure, forecast: pd.DataFrame) -> None:
    if forecast.empty or "predicted_regime" not in forecast.columns:
        return
    colors = regime_color_map()
    x = forecast["ts_hour"]
    regimes = forecast["predicted_regime"].fillna("normal").tolist()
    if not len(x):
        return
    start = x.iloc[0]
    current = regimes[0]
    for i in range(1, len(x)):
        if regimes[i] != current:
            fig.add_vrect(x0=start, x1=x.iloc[i], fillcolor=colors.get(current, "#666"), opacity=0.12, line_width=0)
            start = x.iloc[i]
            current = regimes[i]
    fig.add_vrect(x0=start, x1=x.iloc[-1], fillcolor=colors.get(current, "#666"), opacity=0.12, line_width=0)


forecast = resolve_forecast()
tomorrow_features = load_parquet(FEATURE_DIR / "tomorrow_morning_features_enriched.parquet")
reasoning = load_parquet(FEATURE_DIR / "market_reasoning_features.parquet")
curve_features = load_parquet(CVE_PATH)
forecast = enrich_forecast_with_curve_proxy(forecast, curve_features)
regime_preds = load_csv(PRED_DIR / "regime_classifier_predictions.csv")
spike_preds = load_csv(PRED_DIR / "spike_cap_detector_predictions.csv")
spike_transition_preds = load_csv(PRED_DIR / "spike_transition_detector_predictions.csv")
backtest_preds = load_csv(PRED_DIR / "regime_aware_regressor_predictions.csv")
metrics = safe_read_json(REPORT_DIR / "regime_aware_regressor_metrics.json")
tomorrow_report = safe_read_json(REPORT_DIR / "tomorrow_forecast_run.json")
models = load_models()

st.sidebar.header("Dashboard Controls")
curve_dates = available_curve_dates()
selected_curve_date = st.sidebar.selectbox(
    "Raw Curve Date",
    curve_dates if curve_dates else [RECON_DAILY_DATE],
    index=0,
)
selected_curve_features_path = FEATURE_DIR / f"reconstructed_daily_curve_features_{selected_curve_date}.parquet"
selected_curve_report_path = REPORT_DIR / f"reconstructed_daily_curve_{selected_curve_date}.json"
selected_curve_png_path = REPORT_DIR / "curve_debug_examples" / selected_curve_date / "daily_2026-06-01_curve.png"
selected_curve_features = load_parquet(selected_curve_features_path)
selected_curve_report = safe_read_json(selected_curve_report_path)

left, right = st.columns([1.3, 1.0])
with left:
    st.subheader("Tomorrow Forecast")
    if forecast.empty:
        st.warning("Forecast output not found.")
    else:
        table_cols = [c for c in [
            "ts_hour",
            "predicted_ptf",
            "persistence_pred",
            "predicted_regime",
            "spike_probability",
            "spike_transition_probability",
            "must_run_supply_proxy",
            "analyst_spike_score",
            "analyst_persistence_break_score",
            "cap_risk_from_curve",
            "marginality_risk_score",
            "low_price_pressure_score",
            "oversupply_curve_pressure",
        ] if c in forecast.columns]
        st.dataframe(forecast[table_cols], use_container_width=True, hide_index=True)
        line = go.Figure()
        line.add_trace(go.Scatter(x=forecast["ts_hour"], y=forecast["predicted_ptf"], mode="lines+markers", name="Predicted PTF"))
        if "persistence_pred" in forecast.columns:
            line.add_trace(go.Scatter(x=forecast["ts_hour"], y=forecast["persistence_pred"], mode="lines+markers", name="Persistence", line=dict(dash="dash")))
        line.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(line, use_container_width=True)
        if any(c in forecast.columns for c in ["cap_risk_from_curve", "marginality_risk_score", "low_price_pressure_score", "oversupply_curve_pressure"]):
            fig = go.Figure()
            for col, name in [
                ("cap_risk_from_curve", "Cap risk from curve"),
                ("marginality_risk_score", "Marginality risk"),
                ("low_price_pressure_score", "Low price pressure"),
                ("oversupply_curve_pressure", "Oversupply curve pressure"),
            ]:
                if col in forecast.columns:
                    fig.add_trace(go.Scatter(x=forecast["ts_hour"], y=forecast[col], mode="lines+markers", name=name))
            fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Model Diagnostics")
    if metrics:
        st.metric("Test MAE", f"{metrics.get('split_metrics', {}).get('test', {}).get('mae', 'n/a')}")
        st.metric("Persistence MAE", f"{metrics.get('split_metrics', {}).get('test', {}).get('persistence_mae', 'n/a')}")
        st.metric("Delta vs Persistence", f"{metrics.get('split_metrics', {}).get('test', {}).get('delta_vs_persistence', 'n/a')}")
    branch_counts = tomorrow_report.get("routing_branch_counts", {})
    if branch_counts:
        st.write("Routing branch counts")
        st.bar_chart(pd.Series(branch_counts))
    if models.get("splitter") is not None:
        feats = load_parquet(FEATURE_DIR / "tomorrow_morning_features_enriched.parquet")
        if not feats.empty and hasattr(models["splitter"], "feature_importances_"):
            cols = json.loads((MODEL_DIR / "feature_columns.json").read_text()) if (MODEL_DIR / "feature_columns.json").exists() else []
            fi = pd.Series(models["splitter"].feature_importances_, index=cols).sort_values(ascending=False)
            st.write("Splitter feature importance")
            st.bar_chart(fi.head(12))
    if not curve_features.empty:
        st.caption(f"Supply-demand curve proxy rows: {len(curve_features):,}")
    else:
        st.warning("Supply-demand curve proxy data not found.")

st.divider()
regime_tab, spike_tab, must_tab, reasoning_tab, curve_tab, raw_curve_tab, backtest_tab = st.tabs([
    "Regime Timeline",
    "Spike Risk",
    "Must-Run / Renewable Pressure",
    "Analyst Reasoning",
    "Supply-Demand Curve Intelligence",
    "Raw Curve Reconstruction",
    "Backtest Viewer",
])

with regime_tab:
    if forecast.empty:
        st.info("No forecast data available.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=forecast["ts_hour"], y=forecast["predicted_ptf"], name="Predicted PTF", mode="lines+markers"), secondary_y=False)
        if "predicted_regime" in forecast.columns:
            regime_numeric = forecast["predicted_regime"].astype("category").cat.codes
            fig.add_trace(go.Scatter(x=forecast["ts_hour"], y=regime_numeric, name="Regime code", mode="lines+markers"), secondary_y=True)
        add_regime_bands(fig, forecast)
        fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(forecast[["ts_hour", "predicted_regime", "predicted_ptf"] + [c for c in ["spike_probability", "spike_transition_probability"] if c in forecast.columns]], use_container_width=True, hide_index=True)

with spike_tab:
    if forecast.empty:
        st.info("No forecast data available.")
    else:
        fig = go.Figure()
        if "spike_probability" in forecast.columns:
            fig.add_trace(go.Bar(x=forecast["ts_hour"], y=forecast["spike_probability"], name="Spike probability"))
        if "spike_transition_probability" in forecast.columns:
            fig.add_trace(go.Scatter(x=forecast["ts_hour"], y=forecast["spike_transition_probability"], mode="lines+markers", name="Spike transition probability"))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#f94144")
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        cap_cols = [c for c in forecast.columns if c in {"ts_hour", "spike_probability", "spike_transition_probability", "predicted_regime"}]
        st.dataframe(forecast[cap_cols], use_container_width=True, hide_index=True)

with must_tab:
    if forecast.empty:
        st.info("No forecast data available.")
    else:
        fig = go.Figure()
        for col, name in [
            ("must_run_supply_proxy", "Must-run proxy"),
            ("renewable_share_of_load", "Renewable share of load"),
            ("solar_oversupply_score", "Solar oversupply score"),
            ("residual_load_after_renewables", "Residual load after renewables"),
        ]:
            if col in forecast.columns:
                fig.add_trace(go.Scatter(x=forecast["ts_hour"], y=forecast[col], mode="lines+markers", name=name))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

with reasoning_tab:
    if forecast.empty:
        st.info("No forecast data available.")
    else:
        cols = [c for c in [
            "ts_hour",
            "analyst_spike_score",
            "analyst_persistence_break_score",
            "analyst_zero_score",
            "analyst_tight_score",
            "analyst_confidence_score",
            "analyst_reason_text",
        ] if c in forecast.columns]
        st.dataframe(forecast[cols], use_container_width=True, hide_index=True)
        if "analyst_reason_text" in forecast.columns:
            st.write("Reason text")
            st.text_area("Analyst reasoning", value="\n".join(forecast["analyst_reason_text"].fillna("").astype(str).head(5)), height=180)

with curve_tab:
    if curve_features.empty:
        st.warning("Supply-demand curve proxy feature file is missing or empty.")
    else:
        cf = curve_features.copy()
        if "ts_hour" in cf.columns:
            cf["ts_hour"] = pd.to_datetime(cf["ts_hour"], errors="coerce")
        st.subheader("PTF Regime Distribution")
        if "target_regime" in cf.columns:
            st.bar_chart(cf["target_regime"].fillna("unknown").value_counts())
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            if {"ts_hour", "cap_risk_from_curve"}.issubset(cf.columns):
                fig.add_trace(go.Scatter(x=cf["ts_hour"], y=cf["cap_risk_from_curve"], mode="lines", name="Cap risk from curve"))
            if {"ts_hour", "marginality_risk_score"}.issubset(cf.columns):
                fig.add_trace(go.Scatter(x=cf["ts_hour"], y=cf["marginality_risk_score"], mode="lines", name="Marginality risk"))
            fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            if {"supply_gap_score", "clearing_price_proxy"}.issubset(cf.columns):
                fig.add_trace(go.Scatter(x=cf["supply_gap_score"], y=cf["clearing_price_proxy"], mode="markers", name="Supply gap vs PTF", opacity=0.55))
            if {"low_price_pressure_score", "clearing_price_proxy"}.issubset(cf.columns):
                fig.add_trace(go.Scatter(x=cf["low_price_pressure_score"], y=cf["clearing_price_proxy"], mode="markers", name="Low price pressure vs PTF", opacity=0.55))
            fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        if "target_regime" in cf.columns:
            spike_or_cap = cf[cf["target_regime"].isin(["spike_cap", "tight"])]
            zero_slice = cf[cf["target_regime"] == "negative_zero_pressure"]
            s1, s2 = st.columns(2)
            with s1:
                st.write("Spike / cap hours")
                spike_cols = [c for c in ["ts_hour", "target_regime", "clearing_price_proxy", "cap_risk_from_curve", "marginality_risk_score", "supply_gap_score", "low_price_pressure_score"] if c in spike_or_cap.columns]
                st.dataframe(spike_or_cap[spike_cols].sort_values("ts_hour"), use_container_width=True, hide_index=True)
            with s2:
                st.write("Zero-pressure hours")
                zero_cols = [c for c in ["ts_hour", "target_regime", "clearing_price_proxy", "oversupply_curve_pressure", "supply_gap_score", "low_price_pressure_score"] if c in zero_slice.columns]
                st.dataframe(zero_slice[zero_cols].sort_values("ts_hour"), use_container_width=True, hide_index=True)

with backtest_tab:
    if backtest_preds.empty:
        st.info("Backtest predictions not found.")
    else:
        b = backtest_preds.copy()
        if "ts_hour" in b.columns:
            b["ts_hour"] = pd.to_datetime(b["ts_hour"], errors="coerce")
        st.dataframe(b.head(50), use_container_width=True, hide_index=True)
        fig = go.Figure()
        if {"price", "pred_price"}.issubset(b.columns):
            fig.add_trace(go.Scatter(x=b["ts_hour"], y=b["price"], mode="lines", name="Actual"))
            fig.add_trace(go.Scatter(x=b["ts_hour"], y=b["pred_price"], mode="lines", name="Predicted"))
        if "persistence_pred" in b.columns:
            fig.add_trace(go.Scatter(x=b["ts_hour"], y=b["persistence_pred"], mode="lines", name="Persistence", line=dict(dash="dash")))
        fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if {"price", "pred_price", "target_regime"}.issubset(b.columns):
            st.write("MAE by regime")
            regime_mae = b.groupby("target_regime").apply(lambda x: float(np.mean(np.abs(x["price"] - x["pred_price"])))).sort_values()
            st.bar_chart(regime_mae)
        if "hour_of_day" in b.columns:
            st.write("MAE by hour")
            hour_mae = b.groupby("hour_of_day").apply(lambda x: float(np.mean(np.abs(x["price"] - x["pred_price"])))).sort_index()
            st.line_chart(hour_mae)

with raw_curve_tab:
    if selected_curve_features.empty:
        st.warning(f"Raw curve reconstruction data not found for {selected_curve_date}.")
    else:
        rc = selected_curve_features.copy()
        if "delivery_hour" in rc.columns:
            rc["delivery_hour"] = pd.to_datetime(rc["delivery_hour"], errors="coerce")
        summary = selected_curve_report or {}
        a, b = st.columns(4)
        a.metric("Selected date", selected_curve_date)
        b.metric("Successful hours", summary.get("hours_successful", "n/a"))
        a.metric("Average error", summary.get("mean_reconstruction_error", "n/a"))
        b.metric("Max error", summary.get("max_reconstruction_error", "n/a"))
        high_error_hours = summary.get("hours_high_error", [])
        if high_error_hours:
            st.write(f"High-error hours: {', '.join(high_error_hours)}")
        st.subheader("Actual vs Reconstructed PTF")
        fig = go.Figure()
        if {"delivery_hour", "mcpPrice"}.issubset(rc.columns):
            fig.add_trace(go.Scatter(x=rc["delivery_hour"], y=rc["mcpPrice"], mode="lines+markers", name="EPİAŞ mcpPrice"))
        if {"delivery_hour", "reconstructed_clearing_price"}.issubset(rc.columns):
            fig.add_trace(go.Scatter(x=rc["delivery_hour"], y=rc["reconstructed_clearing_price"], mode="lines+markers", name="Reconstructed clearing"))
        if {"delivery_hour", "reconstruction_price_error"}.issubset(rc.columns):
            fig.add_trace(go.Bar(x=rc["delivery_hour"], y=rc["reconstruction_price_error"], name="Error", opacity=0.3))
        fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Reconstruction Error Heatmap")
        if {"delivery_hour", "reconstruction_price_error"}.issubset(rc.columns):
            heat = rc.copy()
            heat["hour"] = heat["delivery_hour"].dt.hour
            heat = heat.pivot_table(index="hour", values="reconstruction_price_error", aggfunc="mean").reset_index()
            fig = go.Figure(data=go.Heatmap(z=[heat["reconstruction_price_error"].to_list()], x=heat["hour"].astype(str).tolist(), colorscale="RdBu", zmid=0))
            fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Curve Microstructure Features")
        feature_cols = [c for c in [
            "slope_near_clearing",
            "elasticity_near_clearing",
            "curve_fragility_score",
            "volume_needed_for_100TL_move",
            "volume_needed_for_500TL_move",
            "oversupply_pressure",
            "cap_risk_score",
        ] if c in rc.columns]
        if feature_cols:
            fig = go.Figure()
            for col in feature_cols:
                fig.add_trace(go.Scatter(x=rc["delivery_hour"], y=rc[col], mode="lines+markers", name=col))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Regime View")
        regime_rows = rc.copy()
        for label, cond in [
            ("zero-price", regime_rows["mcpPrice"] <= 0 if "mcpPrice" in regime_rows.columns else pd.Series(False, index=regime_rows.index)),
            ("low-price", regime_rows["mcpPrice"] <= 50 if "mcpPrice" in regime_rows.columns else pd.Series(False, index=regime_rows.index)),
            ("spike", regime_rows["mcpPrice"] >= 4000 if "mcpPrice" in regime_rows.columns else pd.Series(False, index=regime_rows.index)),
        ]:
            sub = regime_rows[cond]
            if not sub.empty and "reconstruction_price_error" in sub.columns:
                st.write(f"{label} hours mean error: {float(sub['reconstruction_price_error'].abs().mean()):.2f}")
        if {"mcpPrice", "reconstructed_clearing_price"}.issubset(rc.columns):
            regime_fig = go.Figure()
            regime_fig.add_trace(go.Scatter(x=rc["delivery_hour"], y=rc["mcpPrice"], mode="lines+markers", name="mcpPrice"))
            regime_fig.add_trace(go.Scatter(x=rc["delivery_hour"], y=rc["reconstructed_clearing_price"], mode="lines+markers", name="reconstructed"))
            regime_fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
            st.plotly_chart(regime_fig, use_container_width=True)

        st.subheader("Debug Plot")
        if selected_curve_png_path.exists():
            st.image(str(selected_curve_png_path), caption=str(selected_curve_png_path.relative_to(PROJECT_ROOT)), use_container_width=True)
        else:
            st.warning(f"Debug plot not found: {selected_curve_png_path}")
