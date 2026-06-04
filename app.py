"""PTF Tahmin Paneli — Streamlit Community Cloud için ana giriş noktası."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Sayfa yapılandırması
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PTF Tahmin Paneli",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parent
LIVE_DIR     = PROJECT_ROOT / "data" / "live"
METRICS_PATH = PROJECT_ROOT / "models" / "rolling_ptf_next24" / "full_market" / "metrics.json"

# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_forecast() -> pd.DataFrame:
    path = LIVE_DIR / "forecast_next24_latest.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["delivery_ts_hour", "anchor_ts_hour"])
    return df

@st.cache_data(ttl=300)
def load_recent_actuals() -> pd.DataFrame:
    path = LIVE_DIR / "ptf_actuals_30d.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["ts_hour"])
    return df

@st.cache_data(ttl=3600)
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())

@st.cache_data(ttl=300)
def load_forecast_history() -> pd.DataFrame:
    path = LIVE_DIR / "forecast_history_30d.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["delivery_ts_hour"])
    return df

# ---------------------------------------------------------------------------
# Verileri al
# ---------------------------------------------------------------------------
fc      = load_forecast()
actuals = load_recent_actuals()
metrics = load_metrics()
history = load_forecast_history()

# ---------------------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------------------
st.title("⚡ Türkiye PTF Tahmin Paneli")

anchor_str = ""
if not fc.empty and "anchor_ts_hour" in fc.columns:
    anchor_ts = pd.to_datetime(fc["anchor_ts_hour"].iloc[0])
    anchor_str = anchor_ts.strftime("%d %B %Y, %H:%M")
    st.caption(f"Son güncelleme: **{anchor_str}** (Istanbul) · Tahmin ufku: sonraki 24 saat")
else:
    st.caption("Tahmin verisi henüz mevcut değil — GitHub Actions sonraki saatte çalışacak.")

# ---------------------------------------------------------------------------
# Üst metrik kartları
# ---------------------------------------------------------------------------
m_val  = metrics.get("metrics", {}).get("validation_model", {})
m_test = metrics.get("metrics", {}).get("test_model", {})

col1, col2, col3, col4 = st.columns(4)

with col1:
    if not fc.empty:
        mean_fc = fc["predicted_ptf"].mean()
        st.metric("Yarın Ort. Tahmini", f"{mean_fc:,.0f} TL/MWh")
    else:
        st.metric("Yarın Ort. Tahmini", "—")

with col2:
    if not fc.empty:
        peak_row = fc.loc[fc["predicted_ptf"].idxmax()]
        peak_h   = pd.to_datetime(peak_row["delivery_ts_hour"]).strftime("%H:00")
        st.metric("Peak Tahmin", f"{peak_row['predicted_ptf']:,.0f} TL/MWh", delta=f"Saat {peak_h}")
    else:
        st.metric("Peak Tahmin", "—")

with col3:
    val_mae = m_val.get("mae")
    st.metric("Val MAE (2025 Q2)", f"{val_mae:,.0f} TL/MWh" if val_mae else "—")

with col4:
    test_mae = m_test.get("mae")
    st.metric("Test MAE (2026)", f"{test_mae:,.0f} TL/MWh" if test_mae else "—")

st.divider()

# ---------------------------------------------------------------------------
# Ana grafik: Yarın tahmin
# ---------------------------------------------------------------------------
st.subheader("Yarınki PTF Tahmini (Sonraki 24 Saat)")

if not fc.empty:
    fig = go.Figure()

    # Persistence (dünün fiyatı)
    fig.add_trace(go.Scatter(
        x=fc["delivery_ts_hour"],
        y=fc["baseline_d1_ptf"],
        name="Persistence (dün)",
        line=dict(color="#aaaaaa", dash="dot", width=1.5),
        opacity=0.7,
    ))

    # Model tahmini
    fig.add_trace(go.Scatter(
        x=fc["delivery_ts_hour"],
        y=fc["predicted_ptf"],
        name="Model Tahmini",
        line=dict(color="#00b4d8", width=3),
        fill="tozeroy",
        fillcolor="rgba(0,180,216,0.08)",
    ))

    fig.update_layout(
        xaxis_title="Saat (İstanbul)",
        yaxis_title="PTF (TL/MWh)",
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tablo
    with st.expander("Saatlik Detay"):
        show = fc[["delivery_ts_hour", "predicted_ptf", "baseline_d1_ptf", "shrinkage_alpha"]].copy()
        show["delivery_ts_hour"] = pd.to_datetime(show["delivery_ts_hour"]).dt.strftime("%d.%m %H:00")
        show.columns = ["Saat", "Tahmin (TL/MWh)", "Persistence (TL/MWh)", "Model Ağırlığı (α)"]
        show["Tahmin (TL/MWh)"]    = show["Tahmin (TL/MWh)"].round(0).astype(int)
        show["Persistence (TL/MWh)"] = show["Persistence (TL/MWh)"].round(0).astype(int)
        st.dataframe(show, use_container_width=True, hide_index=True)
else:
    st.info("Tahmin verisi bekleniyor — GitHub Actions her saat başında günceller.")

# ---------------------------------------------------------------------------
# Tahmin geçmişi vs gerçek
# ---------------------------------------------------------------------------
if not history.empty and not actuals.empty:
    st.divider()
    st.subheader("Son 30 Günlük Tahmin vs Gerçek PTF")

    # Her teslimat saati için en yakın horizon'ı al (horizon=24)
    hist24 = history[history.get("horizon", pd.Series(24)) == 24] if "horizon" in history.columns else history

    merged = hist24.merge(
        actuals.rename(columns={"ts_hour": "delivery_ts_hour", "price": "actual_ptf"}),
        on="delivery_ts_hour", how="inner"
    )

    if not merged.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=merged["delivery_ts_hour"], y=merged["actual_ptf"],
            name="Gerçek PTF", line=dict(color="#ffd60a", width=1.5),
        ))
        fig2.add_trace(go.Scatter(
            x=merged["delivery_ts_hour"], y=merged["predicted_ptf"],
            name="Tahmin", line=dict(color="#00b4d8", width=1.5),
        ))
        mae_30d = (merged["predicted_ptf"] - merged["actual_ptf"]).abs().mean()
        fig2.update_layout(
            title=f"Son 30 Gün — MAE: {mae_30d:,.0f} TL/MWh",
            xaxis_title="Tarih", yaxis_title="PTF (TL/MWh)",
            height=350, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Model performans detayı
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Model Performans Özeti")

if metrics:
    cv_folds = metrics.get("cv_folds", [])
    meta     = metrics.get("meta", {})

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Cross-Validation (Yıllık Fold)**")
        if cv_folds:
            fold_df = pd.DataFrame(cv_folds)
            fold_df = fold_df.rename(columns={
                "fold": "Dönem", "model_mae": "Model MAE",
                "persistence_mae": "Persistence MAE", "improvement_pct": "İyileşme %"
            })
            fold_df["Model MAE"]       = fold_df["Model MAE"].round(0).astype(int)
            fold_df["Persistence MAE"] = fold_df["Persistence MAE"].round(0).astype(int)
            fold_df["İyileşme %"]      = fold_df["İyileşme %"].round(1).astype(str) + "%"
            st.dataframe(fold_df[["Dönem", "Model MAE", "Persistence MAE", "İyileşme %"]],
                         use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Test Seti Metrikleri (2026)**")
        if m_test:
            rows = [
                ("MAE",        f"{m_test.get('mae',0):,.0f} TL/MWh"),
                ("Median AE",  f"{m_test.get('median_ae',0):,.0f} TL/MWh"),
                ("P90 Hata",   f"{m_test.get('p90_ae',0):,.0f} TL/MWh"),
                ("WAPE",       f"{m_test.get('wape',0):.1f}%"),
                ("Bias",       f"{m_test.get('bias',0):+,.0f} TL/MWh"),
            ]
            perf_df = pd.DataFrame(rows, columns=["Metrik", "Değer"])
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

        st.markdown(f"**Feature sayısı:** {meta.get('feature_count','—')}")
        st.markdown(f"**Eğitim verisi:** {meta.get('rows',0):,} satır")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "LightGBM tabanlı gün öncesi PTF tahmini · "
    "Veri: EPİAŞ Şeffaflık Platformu · "
    "Model: full_market profili, 217 feature · "
    "Otomatik güncelleme: saatte bir (GitHub Actions)"
)
