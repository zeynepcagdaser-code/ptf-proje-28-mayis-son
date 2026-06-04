"""PTF Tahmin Paneli — Streamlit Community Cloud."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="PTF Tahmin Paneli",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

ROOT         = Path(__file__).resolve().parent
LIVE_DIR     = ROOT / "data" / "live"
METRICS_PATH = ROOT / "models" / "rolling_ptf_next24" / "full_market" / "metrics.json"

# ── Veri yükleme ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_csv(name: str) -> pd.DataFrame:
    p = LIVE_DIR / name
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in df.columns:
        if "ts_hour" in c or "ts" == c:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

fc       = load_csv("forecast_next24_latest.csv")
history  = load_csv("forecast_vs_actual_all.csv")
metrics  = load_metrics()

# ── Başlık ────────────────────────────────────────────────────────────────────

st.title("⚡ Türkiye PTF Tahmin Paneli")

if not fc.empty and "anchor_ts_hour" in fc.columns:
    anchor = pd.to_datetime(fc["anchor_ts_hour"].iloc[0])
    st.caption(
        f"Son güncelleme: **{anchor.strftime('%d %B %Y %H:%M')}** (İstanbul) · "
        "Saatlik güncelleme: GitHub Actions"
    )

# ── Üst metrik kartları ────────────────────────────────────────────────────────

m_val  = metrics.get("metrics", {}).get("validation_model", {})
m_test = metrics.get("metrics", {}).get("test_model", {})

c1, c2, c3, c4 = st.columns(4)
with c1:
    v = fc["predicted_ptf"].mean() if not fc.empty else None
    st.metric("Bugün Ort. Tahmin", f"{v:,.0f} TL/MWh" if v else "—")
with c2:
    if not fc.empty:
        row = fc.loc[fc["predicted_ptf"].idxmax()]
        h = pd.to_datetime(row["delivery_ts_hour"]).strftime("%H:00")
        st.metric("Peak Saat", f"{row['predicted_ptf']:,.0f} TL/MWh", delta=f"Saat {h}")
    else:
        st.metric("Peak Saat", "—")
with c3:
    v = m_val.get("mae")
    st.metric("Val MAE (2025 Q2)", f"{v:,.0f} TL/MWh" if v else "—")
with c4:
    v = m_test.get("mae")
    st.metric("Test MAE (2026)", f"{v:,.0f} TL/MWh" if v else "—")

st.divider()

# ── Bölüm 1: Bugünün tahmini ──────────────────────────────────────────────────

st.subheader("Bugünkü PTF Tahmini (Sonraki 24 Saat)")

if not fc.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc["delivery_ts_hour"], y=fc["baseline_d1_ptf"],
        name="Persistence (dün)", line=dict(color="#888", dash="dot", width=1.5), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=fc["delivery_ts_hour"], y=fc["predicted_ptf"],
        name="Model Tahmini",
        line=dict(color="#00b4d8", width=3),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.07)",
    ))
    fig.update_layout(
        height=360, hovermode="x unified",
        xaxis_title="Saat", yaxis_title="PTF (TL/MWh)",
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(200,200,200,0.15)"),
        yaxis=dict(gridcolor="rgba(200,200,200,0.15)"),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Saatlik detay tablosu"):
        tbl = fc[["delivery_ts_hour","predicted_ptf","baseline_d1_ptf","shrinkage_alpha"]].copy()
        tbl["delivery_ts_hour"] = pd.to_datetime(tbl["delivery_ts_hour"]).dt.strftime("%d.%m %H:00")
        tbl.columns = ["Saat","Tahmin (TL/MWh)","Persistence (TL/MWh)","Model Ağırlığı α"]
        tbl["Tahmin (TL/MWh)"] = tbl["Tahmin (TL/MWh)"].round(0).astype(int)
        tbl["Persistence (TL/MWh)"] = tbl["Persistence (TL/MWh)"].round(0).astype(int)
        st.dataframe(tbl, hide_index=True, use_container_width=True)
else:
    st.info("Tahmin verisi bekleniyor — GitHub Actions her saat başında günceller.")

# ── Bölüm 2: Tahmin vs Gerçek (2026) ─────────────────────────────────────────

st.divider()
st.subheader("2026 Tahmin vs Gerçekleşen PTF")

if not history.empty:
    # Sidebar filtre
    years_avail = sorted(history["ts_hour"].dt.year.dropna().unique().astype(int), reverse=True)
    col_filter, col_agg = st.columns([2, 2])
    with col_filter:
        selected_years = st.multiselect(
            "Yıl seç", years_avail,
            default=[y for y in [2026, 2025] if y in years_avail]
        )
    with col_agg:
        agg_mode = st.radio("Gruplama", ["Günlük ort.", "Saatlik"], horizontal=True)

    sub = history[history["ts_hour"].dt.year.isin(selected_years)].copy() if selected_years else history.copy()

    if agg_mode == "Günlük ort.":
        sub["date"] = sub["ts_hour"].dt.date
        grp = sub.groupby(["date","split"])[["actual_ptf","predicted_ptf"]].mean().reset_index()
        grp["date"] = pd.to_datetime(grp["date"])
        x_col = "date"
    else:
        grp = sub
        x_col = "ts_hour"

    fig2 = go.Figure()

    # Gerçekleşen PTF
    for sp, color in [("test","#ffd60a"), ("validation","#f4a261"), ("train","#adb5bd")]:
        mask = grp["split"] == sp
        if mask.any():
            label_map = {"test":"Gerçek PTF (Test 2026)", "validation":"Gerçek PTF (Val 2025 Q2)", "train":"Gerçek PTF (Eğitim)"}
            fig2.add_trace(go.Scatter(
                x=grp.loc[mask, x_col], y=grp.loc[mask, "actual_ptf"],
                name=label_map[sp],
                line=dict(color=color, width=1.5 if sp=="test" else 1),
                opacity=1 if sp in ("test","validation") else 0.4,
            ))

    # Tahmin (test + val)
    mask_tv = grp["split"].isin(["test","validation"])
    if mask_tv.any():
        fig2.add_trace(go.Scatter(
            x=grp.loc[mask_tv, x_col], y=grp.loc[mask_tv, "predicted_ptf"],
            name="Model Tahmini",
            line=dict(color="#00b4d8", width=2, dash="dot"),
        ))

    # MAE hesapla
    test_rows = sub[sub["split"]=="test"].dropna(subset=["actual_ptf","predicted_ptf"])
    mae_str = f"Test MAE: {(test_rows['predicted_ptf']-test_rows['actual_ptf']).abs().mean():,.0f} TL/MWh" if len(test_rows) else ""

    fig2.update_layout(
        title=f"PTF Tahmin vs Gerçekleşen — {agg_mode}  {mae_str}",
        height=420, hovermode="x unified",
        xaxis_title="Tarih", yaxis_title="PTF (TL/MWh)",
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(200,200,200,0.15)"),
        yaxis=dict(gridcolor="rgba(200,200,200,0.15)"),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Aylık MAE tablosu (sadece test/val)
    with st.expander("Aylık MAE detayı"):
        err_df = sub[sub["split"].isin(["test","validation"])].copy()
        err_df["ym"] = err_df["ts_hour"].dt.to_period("M").astype(str)
        err_df["abs_err"] = (err_df["predicted_ptf"] - err_df["actual_ptf"]).abs()
        monthly = err_df.groupby("ym").agg(
            MAE=("abs_err","mean"),
            Gerçek_ort=("actual_ptf","mean"),
            Tahmin_ort=("predicted_ptf","mean"),
            Saat=("abs_err","count")
        ).round(0).astype(int)
        st.dataframe(monthly, use_container_width=True)
else:
    st.info("Geçmiş tahmin verisi oluşturuluyor...")

# ── Bölüm 3: Model performans ─────────────────────────────────────────────────

st.divider()
st.subheader("Model Performans Özeti")

if metrics:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Cross-Validation Foldları**")
        folds = metrics.get("cv_folds", [])
        if folds:
            fdf = pd.DataFrame(folds)[["fold","model_mae","persistence_mae","improvement_pct"]]
            fdf.columns = ["Dönem","Model MAE","Persistence MAE","İyileşme %"]
            fdf["Model MAE"] = fdf["Model MAE"].round(0).astype(int)
            fdf["Persistence MAE"] = fdf["Persistence MAE"].round(0).astype(int)
            fdf["İyileşme %"] = fdf["İyileşme %"].round(1).astype(str) + "%"
            st.dataframe(fdf, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("**Test Metrikleri (2026)**")
        if m_test:
            rows = [
                ("MAE",       f"{m_test.get('mae',0):,.0f} TL/MWh"),
                ("Median AE", f"{m_test.get('median_ae',0):,.0f} TL/MWh"),
                ("P90 Hata",  f"{m_test.get('p90_ae',0):,.0f} TL/MWh"),
                ("WAPE",      f"{m_test.get('wape',0):.1f}%"),
                ("Bias",      f"{m_test.get('bias',0):+,.0f} TL/MWh"),
            ]
            st.dataframe(pd.DataFrame(rows, columns=["Metrik","Değer"]),
                         hide_index=True, use_container_width=True)
        meta = metrics.get("meta", {})
        st.caption(f"Feature: {meta.get('feature_count','—')} · Eğitim: {meta.get('rows',0):,} satır · Profil: full_market")

st.divider()
st.caption(
    "LightGBM tabanlı gün öncesi PTF tahmini · "
    "Veri: EPİAŞ · Model: 217 feature, 700 estimator · "
    "Güncelleme: saatte bir"
)
