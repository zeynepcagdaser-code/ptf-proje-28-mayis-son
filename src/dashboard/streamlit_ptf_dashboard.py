#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = PROJECT_ROOT / "data" / "predictions"

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
st.write("Bu panel, saatlik PTF tahminlerini ve iki aşamalı model performansını izlemek için hazırlandı.")


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "ts_hour" in df.columns:
        df["ts_hour"] = pd.to_datetime(df["ts_hour"], errors="coerce")
    if "delivery_hour" in df.columns:
        df["delivery_hour"] = pd.to_datetime(df["delivery_hour"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


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


summary = load_summary()
real_time = load_csv(PRED_DIR / "ptf_realtime_latest.csv")
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
    cols[0].metric("MAE Genel", f"{diag.get('mae_overall', 'NA'):.3f}")
    cols[1].metric("MAE Spike", f"{diag.get('mae_spike', 'NA'):.3f}")
    cols[2].metric("MAE Non-Spike", f"{diag.get('mae_nonspike', 'NA'):.3f}")
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
        if not data.empty:
            sample = data.tail(240)
            if "target_1h" in sample.columns:
                fig = px.line(sample, x="ts_hour", y=["target_1h", "pred"], labels={"value": "PTF TL", "ts_hour": "Zaman"}, title="Son 240 Saat: Gerçek vs Tahmin")
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("Hata Dağılımı")
            hist = px.histogram(sample, x="abs_err", nbins=40, title="Mutlak Hata Dağılımı")
            st.plotly_chart(hist, use_container_width=True)

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
        st.plotly_chart(rt_fig, use_container_width=True)

st.markdown("---")
st.caption("PTF tahmin paneli, data/predictions altındaki en güncel PTF sonuçlarına dayanır.")
