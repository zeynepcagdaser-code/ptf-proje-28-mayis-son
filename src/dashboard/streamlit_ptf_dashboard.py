#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
PTF_PATH = PROJECT_ROOT / "data" / "ptf_dataset.csv"

st.set_page_config(page_title="PTF Tahmin Paneli", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
    .stMetric { background: rgba(30, 30, 30, 0.28); padding: 0.75rem; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PTF Tahmin Paneli")
st.write("Bu panel, saatlik PTF tahminlerini ve model performanslarını izlemek için hazırlandı.")


def _parse_datetime_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series.dtype):
        max_val = int(series.max(skipna=True)) if not series.empty else 0
        if 10**14 <= max_val < 10**16:
            return pd.to_datetime(series, unit="us", errors="coerce")
        if 10**16 <= max_val < 10**19:
            return pd.to_datetime(series, unit="ns", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "ts_hour" in df.columns:
        df["ts_hour"] = _parse_datetime_column(df["ts_hour"])
    if "delivery_hour" in df.columns:
        df["delivery_hour"] = _parse_datetime_column(df["delivery_hour"])
    if "delivery_ts_hour" in df.columns:
        df["delivery_ts_hour"] = _parse_datetime_column(df["delivery_ts_hour"])
    if "anchor_ts_hour" in df.columns:
        df["anchor_ts_hour"] = _parse_datetime_column(df["anchor_ts_hour"])
    return df


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_ptf_actuals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["delivery_ts_hour", "actual_ptf"])
    df = pd.read_csv(path)
    if "date" not in df.columns or "price" not in df.columns:
        return pd.DataFrame(columns=["delivery_ts_hour", "actual_ptf"])
    ts = pd.to_datetime(df["date"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    out = pd.DataFrame(
        {
            "delivery_ts_hour": ts.dt.floor("h"),
            "actual_ptf": pd.to_numeric(df["price"], errors="coerce"),
        }
    )
    return out.dropna(subset=["delivery_ts_hour"]).drop_duplicates("delivery_ts_hour", keep="last")


def load_summary() -> dict[str, dict]:
    out = {}
    for suffix in ["regression", "quantile_0.50", "quantile_0.75", "huber"]:
        path = PRED_DIR / f"ptf_two_stage_latest_diag_{suffix}.json"
        if path.exists():
            out[suffix] = load_json(path)
    return out


def model_label(suffix: str) -> str:
    return {
        "regression": "İki Aşamalı Regresyon",
        "quantile_0.50": "İki Aşamalı Quantile (0.5)",
        "quantile_0.75": "İki Aşamalı Quantile (0.75)",
        "huber": "İki Aşamalı Huber",
    }.get(suffix, suffix)


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"predicted_ptf", "actual_ptf"}.issubset(out.columns):
        out["error"] = out["predicted_ptf"] - out["actual_ptf"]
        out["abs_error"] = out["error"].abs()
        out["error_pct"] = (out["abs_error"] / out["actual_ptf"].replace(0, pd.NA)) * 100
    return out


summary = load_summary()
real_time = load_csv(PRED_DIR / "ptf_realtime_latest.csv")

view = st.sidebar.radio(
    "Görünüm",
    ["Rolling Next-24", "İki Aşamalı Eski Hat"],
)

if view == "Rolling Next-24":
    forecast = load_csv(PRED_DIR / "rolling_ptf_next24_forecast.csv")
    forecast_summary = load_json(PRED_DIR / "rolling_ptf_next24_forecast.json")
    metrics = load_json(PROJECT_ROOT / "reports" / "rolling_ptf_next24_metrics.json")
    actual_cmp = load_csv(PRED_DIR / "rolling_ptf_next24_forecast_2026-06-03_vs_actual.csv")
    actuals = load_ptf_actuals(PTF_PATH)

    st.header("Rolling Next-24 PTF Tahmini")

    if forecast.empty:
        st.warning("rolling_ptf_next24_forecast.csv bulunamadı. Önce rolling pipeline çalıştırılmalı.")
    else:
        forecast = forecast.sort_values("delivery_ts_hour")
        if not actuals.empty:
            forecast = forecast.merge(actuals, on="delivery_ts_hour", how="left")
        if not actual_cmp.empty and "actual_ptf" in actual_cmp.columns:
            cmp_actual = actual_cmp[["hour", "actual_ptf"]].copy()
            forecast["hour"] = forecast["delivery_ts_hour"].dt.strftime("%H:%M")
            forecast = forecast.merge(cmp_actual, on="hour", how="left", suffixes=("", "_cmp"))
            if "actual_ptf" not in forecast.columns:
                forecast["actual_ptf"] = forecast["actual_ptf_cmp"]
            elif "actual_ptf_cmp" in forecast.columns:
                forecast["actual_ptf"] = forecast["actual_ptf"].fillna(forecast["actual_ptf_cmp"])
            forecast = forecast.drop(columns=[c for c in ["actual_ptf_cmp", "hour"] if c in forecast.columns])
        if "actual_ptf" in forecast.columns:
            forecast = add_error_columns(forecast)
        cols = st.columns(5)
        cols[0].metric("Profil", forecast_summary.get("profile", forecast["profile"].iloc[0] if "profile" in forecast else "NA"))
        cols[1].metric("Anchor", str(forecast_summary.get("anchor_ts_hour", forecast["anchor_ts_hour"].iloc[0]))[:16])
        cols[2].metric("Saat", f"{len(forecast)}")
        cols[3].metric("Ortalama PTF", f"{forecast['predicted_ptf'].mean():.2f}")
        cols[4].metric("Aralık", f"{forecast['predicted_ptf'].min():.0f} - {forecast['predicted_ptf'].max():.0f}")

        has_actual = "actual_ptf" in forecast.columns and forecast["actual_ptf"].notna().any()
        fig = go.Figure()
        if has_actual:
            fig.add_trace(
                go.Scatter(
                    x=forecast["delivery_ts_hour"],
                    y=forecast["actual_ptf"],
                    mode="lines+markers",
                    name="Gerçek PTF",
                    line=dict(width=3),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=forecast["delivery_ts_hour"],
                y=forecast["predicted_ptf"],
                mode="lines+markers",
                name="Tahmin PTF",
                line=dict(width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["delivery_ts_hour"],
                y=forecast["baseline_d1_ptf"],
                mode="lines",
                name="D-1 Referans",
                line=dict(width=2, dash="dot"),
            )
        )
        fig.update_layout(
            title="Teslimat Saati Bazında Gerçek ve Tahmin",
            xaxis_title="Teslimat Saati",
            yaxis_title="PTF TL/MWh",
            legend_title_text="Seri",
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

        if has_actual:
            err_fig = px.bar(
                forecast,
                x="delivery_ts_hour",
                y="error",
                labels={"error": "Tahmin - Gerçek (TL/MWh)", "delivery_ts_hour": "Teslimat Saati"},
                title="Saatlik Tahmin Hatası",
            )
            err_fig.add_hline(y=0, line_width=1, line_color="gray")
            st.plotly_chart(err_fig, width="stretch")

        chart_cols = ["predicted_ptf", "baseline_d1_ptf"]
        if "actual_ptf" in forecast.columns and forecast["actual_ptf"].notna().any():
            chart_cols = ["actual_ptf", "predicted_ptf", "baseline_d1_ptf"]

        st.subheader("Saatlik Tahmin")
        table_cols = [
            "delivery_ts_hour",
            "horizon",
            "actual_ptf",
            "predicted_ptf",
            "baseline_d1_ptf",
            "error",
            "abs_error",
            "error_pct",
            "shrinkage_alpha",
        ]
        table_cols = [c for c in table_cols if c in forecast.columns]
        table = forecast[table_cols].copy()
        table = table.rename(
            columns={
                "delivery_ts_hour": "Teslimat Saati",
                "horizon": "H",
                "actual_ptf": "Gerçek PTF",
                "predicted_ptf": "Tahmin PTF",
                "baseline_d1_ptf": "D-1 PTF",
                "error": "Tahmin - Gerçek",
                "abs_error": "Mutlak Hata",
                "error_pct": "% Hata",
                "shrinkage_alpha": "Alpha",
            }
        )
        st.dataframe(table, width="stretch", hide_index=True)

        if metrics:
            selected = metrics.get("selected_profile", "NA")
            result = next((r for r in metrics.get("results", []) if r.get("profile") == selected), None)
            if result:
                st.subheader("Backtest Özeti")
                metric_cols = st.columns(4)
                metric_cols[0].metric("Validation MAE", f"{result['metrics']['validation_model']['mae']:.2f}")
                metric_cols[1].metric("Validation Persistence", f"{result['metrics']['validation_persistence']['mae']:.2f}")
                metric_cols[2].metric("Test MAE", f"{result['metrics']['test_model']['mae']:.2f}")
                metric_cols[3].metric("Test Persistence", f"{result['metrics']['test_persistence']['mae']:.2f}")

        if not actual_cmp.empty:
            actual_cmp = add_error_columns(actual_cmp)
            st.subheader("3 Haziran Gerçek Karşılaştırması")
            cmp_cols = st.columns(4)
            cmp_cols[0].metric("MAE", f"{actual_cmp['abs_error'].mean():.2f}")
            cmp_cols[1].metric("RMSE", f"{(actual_cmp['error'].pow(2).mean() ** 0.5):.2f}")
            cmp_cols[2].metric("Bias", f"{actual_cmp['error'].mean():.2f}")
            cmp_cols[3].metric("Ortalama % Hata", f"{actual_cmp['error_pct'].mean():.2f}%")

            cmp_fig = go.Figure()
            cmp_fig.add_trace(go.Scatter(x=actual_cmp["hour"], y=actual_cmp["actual_ptf"], mode="lines+markers", name="Gerçek PTF", line=dict(width=3)))
            cmp_fig.add_trace(go.Scatter(x=actual_cmp["hour"], y=actual_cmp["predicted_ptf"], mode="lines+markers", name="Tahmin PTF", line=dict(width=3)))
            cmp_fig.add_trace(go.Scatter(x=actual_cmp["hour"], y=actual_cmp["baseline_d1_ptf"], mode="lines", name="D-1 Referans", line=dict(width=2, dash="dot")))
            cmp_fig.update_layout(
                title="3 Haziran: Gerçek ve Tahmin",
                xaxis_title="Saat",
                yaxis_title="PTF TL/MWh",
                legend_title_text="Seri",
                hovermode="x unified",
            )
            st.plotly_chart(cmp_fig, width="stretch")

            cmp_err_fig = px.bar(
                actual_cmp,
                x="hour",
                y="error",
                labels={"hour": "Saat", "error": "Tahmin - Gerçek (TL/MWh)"},
                title="3 Haziran Saatlik Hata",
            )
            cmp_err_fig.add_hline(y=0, line_width=1, line_color="gray")
            st.plotly_chart(cmp_err_fig, width="stretch")

            cmp_table = actual_cmp.rename(
                columns={
                    "hour": "Saat",
                    "actual_ptf": "Gerçek PTF",
                    "predicted_ptf": "Tahmin PTF",
                    "baseline_d1_ptf": "D-1 PTF",
                    "error": "Tahmin - Gerçek",
                    "abs_error": "Mutlak Hata",
                    "error_pct": "% Hata",
                    "shrinkage_alpha": "Alpha",
                }
            )
            st.dataframe(cmp_table, width="stretch", hide_index=True)

else:
    model_variant = st.sidebar.selectbox(
        "Gösterilecek model",
        options=list(summary.keys()) or ["regression"],
        format_func=model_label,
    )

    if not summary:
        st.warning("PTF iki aşamalı diag JSON dosyaları bulunamadı. data/predictions dizinini kontrol edin.")
    else:
        diag = summary[model_variant]
        st.header(model_label(model_variant))
        cols = st.columns(4)
        cols[0].metric("MAE Genel", f"{diag.get('mae_overall', 0):.3f}")
        cols[1].metric("MAE Spike", f"{diag.get('mae_spike', 0):.3f}")
        cols[2].metric("MAE Non-Spike", f"{diag.get('mae_nonspike', 0):.3f}")
        cols[3].metric("% 100 TL Altı", f"{diag.get('accuracy_within_100TL', 0)*100:.2f}%")

        st.markdown("---")
        with st.expander("Diagnostik detayları göster"):
            st.json(diag)

        data_path = PRED_DIR / f"ptf_two_stage_latest_{model_variant}.csv"
        data = load_csv(data_path)
        if data.empty:
            st.warning(f"{data_path.name} bulunamadı veya boş.")
        else:
            st.subheader("Hedef ve Tahmin Karşılaştırması")
            data = data.sort_values("ts_hour")
            sample = data.tail(240)
            if "target_1h" in sample.columns:
                fig = px.line(sample, x="ts_hour", y=["target_1h", "pred"], labels={"value": "PTF TL", "ts_hour": "Zaman"}, title="Son 240 Saat: Gerçek vs Tahmin")
                st.plotly_chart(fig, width="stretch")
            st.subheader("Hata Dağılımı")
            hist = px.histogram(sample, x="abs_err", nbins=40, title="Mutlak Hata Dağılımı")
            st.plotly_chart(hist, width="stretch")

            if "is_spike_true" in sample.columns and "is_spike_pred" in sample.columns:
                spike_tab = sample.groupby(["is_spike_true", "is_spike_pred"]).size().reset_index(name="count")
                st.subheader("Spike sınıflandırma sayıları")
                st.dataframe(spike_tab)

with st.sidebar:
    st.write("## Realtime PTF")
    if real_time.empty:
        st.warning("Realtime PTF tahmin CSV dosyası bulunamadı.")
    else:
        st.write(real_time)
        rt_fig = px.bar(real_time, x="delivery_hour", y="predicted_ptf", color="horizon", title="Realtime PTF Tahminleri")
        st.plotly_chart(rt_fig, width="stretch")
st.markdown("---")
st.caption("PTF tahmin paneli, data/predictions altındaki en güncel PTF sonuçlarına dayanır.")
