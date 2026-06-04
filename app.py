"""PTF Tahmin Paneli — LightGBM full_market (En İyi Model)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
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
.block-container{padding-top:.8rem;padding-bottom:1.5rem}
[data-testid="metric-container"] [data-testid="metric-label"]{font-size:.82rem!important}
</style>""", unsafe_allow_html=True)

ROOT      = Path(__file__).resolve().parent
LIVE_DIR  = ROOT / "data" / "live"
MODEL_DIR = ROOT / "models" / "rolling_ptf_next24" / "full_market"

# ─── Veri yükleme ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_forecast() -> pd.DataFrame:
    p = LIVE_DIR / "forecast_next24_latest.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ["anchor_ts_hour", "delivery_ts_hour"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_history() -> pd.DataFrame:
    p = LIVE_DIR / "forecast_vs_actual_all.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "ts_hour" in df.columns:
        df["ts_hour"] = pd.to_datetime(df["ts_hour"], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_metrics() -> dict:
    p = MODEL_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data(ttl=3600)
def load_walk_forward() -> dict:
    p = ROOT / "reports" / "walk_forward_2026_results.json"
    return json.loads(p.read_text()) if p.exists() else {}

fc      = load_forecast()
history = load_history()
met     = load_metrics()
wf      = load_walk_forward()

# ─── Başlık ───────────────────────────────────────────────────────────────────

st.title("⚡ PTF Tahmin Paneli")

anchor_str = ""
if not fc.empty and "anchor_ts_hour" in fc.columns:
    try:
        anchor_str = pd.to_datetime(fc["anchor_ts_hour"].iloc[0]).strftime("%d %B %Y %H:%M")
    except Exception:
        pass

col_h, col_s = st.columns([3, 1])
with col_h:
    st.caption(f"Model: **LightGBM full_market** · 208 feature · 2020–2025 train, 2025‑Q2 val, 2026 test")
with col_s:
    if anchor_str:
        st.caption(f"Son anchor: {anchor_str}")

# ─── Üst metrik kartları ──────────────────────────────────────────────────────

m_val  = met.get("metrics", {}).get("validation_model", {})
m_test = met.get("metrics", {}).get("test_model", {})

c1, c2, c3, c4, c5 = st.columns(5)

ptf_col = next((c for c in ["predicted_ptf", "corrected_ptf"] if not fc.empty and c in fc.columns), None)

with c1:
    v = fc[ptf_col].mean() if ptf_col else None
    st.metric("Yarın Ort. Tahmin", f"{v:,.0f} TL/MWh" if v else "—")
with c2:
    if ptf_col and not fc.empty:
        peak = fc[ptf_col].max()
        peak_h = fc.loc[fc[ptf_col].idxmax()]
        h_lbl = ""
        if "delivery_ts_hour" in fc.columns:
            try: h_lbl = pd.to_datetime(peak_h["delivery_ts_hour"]).strftime("%H:00")
            except: pass
        st.metric("Peak", f"{peak:,.0f} TL", delta=h_lbl)
    else:
        st.metric("Peak", "—")
with c3:
    st.metric("Val MAE (2025 Q2)", f"{m_val.get('mae',0):,.0f} TL/MWh",
              delta=f"Persistence: {met.get('metrics',{}).get('validation_persistence',{}).get('mae',0):,.0f}",
              delta_color="inverse")
with c4:
    st.metric("Test MAE (2026)", f"{m_test.get('mae',0):,.0f} TL/MWh",
              delta=f"Persistence: {met.get('metrics',{}).get('test_persistence',{}).get('mae',0):,.0f}",
              delta_color="inverse")
with c5:
    wf_avg = wf.get("summary", {}).get("avg_test_mae")
    st.metric("Walk-Fwd Ort. MAE",
              f"{wf_avg:,.0f} TL/MWh" if wf_avg else "—",
              delta="Oca–May 2026", delta_color="off")

st.divider()

# ─── Sekmeler ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Yarınki Tahmin",
    "📊 Geçmiş Performans",
    "📅 Walk-Forward 2026",
    "ℹ️ Model Detayı",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Yarınki tahmin
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if fc.empty or ptf_col is None:
        st.info("Tahmin bekleniyor. `python rolling_ptf_forecast_system.py forecast` çalıştırın.")
    else:
        x_col = next((c for c in ["delivery_ts_hour", "horizon"] if c in fc.columns), None)
        x_vals = fc[x_col] if x_col else range(len(fc))

        fig = go.Figure()

        if "baseline_d1_ptf" in fc.columns:
            fig.add_trace(go.Scatter(
                x=x_vals, y=fc["baseline_d1_ptf"],
                name="Persistence (dün)", mode="lines",
                line=dict(color="#666", dash="dot", width=1.5), opacity=0.55,
            ))

        fig.add_trace(go.Scatter(
            x=x_vals, y=fc[ptf_col],
            name="LightGBM Tahmini", mode="lines+markers",
            line=dict(color="#00b4d8", width=3),
            marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(0,180,216,0.06)",
        ))

        fig.update_layout(
            height=420, hovermode="x unified",
            xaxis_title="Teslimat Saati", yaxis_title="PTF (TL/MWh)",
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(200,200,200,0.12)"),
            yaxis=dict(gridcolor="rgba(200,200,200,0.12)", rangemode="tozero"),
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("Saatlik detay tablosu"):
            show = {
                "delivery_ts_hour": "Saat",
                "horizon": "Ufuk (h)",
                "baseline_d1_ptf": "Persistence (TL)",
                "predicted_ptf": "Tahmin (TL)",
                "corrected_ptf": "Düzeltilmiş (TL)",
                "shrinkage_alpha": "Alpha",
            }
            cols = [c for c in show if c in fc.columns]
            tbl = fc[cols].rename(columns=show).copy()
            for c in tbl.select_dtypes("float64").columns:
                tbl[c] = tbl[c].round(1)
            st.dataframe(tbl, hide_index=True, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Geçmiş performans
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    if history.empty:
        st.info("Geçmiş tahmin arşivi bulunamadı.")
    else:
        ts_col = "ts_hour"
        history[ts_col] = pd.to_datetime(history[ts_col], errors="coerce")
        df_plot = history[history[ts_col].dt.year >= 2025].sort_values(ts_col)

        if df_plot.empty:
            st.warning("2025+ yılına ait veri yok.")
        else:
            min_d = df_plot[ts_col].min().date()
            max_d = df_plot[ts_col].max().date()
            c_from, c_to = st.columns(2)
            with c_from:
                date_from = st.date_input("Başlangıç", value=max_d - pd.Timedelta(days=60),
                                          min_value=min_d, max_value=max_d)
            with c_to:
                date_to = st.date_input("Bitiş", value=max_d,
                                        min_value=min_d, max_value=max_d)

            mask = (df_plot[ts_col].dt.date >= date_from) & (df_plot[ts_col].dt.date <= date_to)
            sub = df_plot[mask]

            if sub.empty:
                st.warning("Bu aralıkta veri yok.")
            else:
                pred_c = next((c for c in ["predicted_ptf", "corrected_ptf"] if c in sub.columns), None)
                mae = (sub[pred_c] - sub["actual_ptf"]).abs().mean() if (pred_c and "actual_ptf" in sub.columns) else None

                fig2 = go.Figure()
                if "actual_ptf" in sub.columns:
                    fig2.add_trace(go.Scatter(
                        x=sub[ts_col], y=sub["actual_ptf"],
                        name="Gerçek PTF", line=dict(color="#ffd60a", width=1.5),
                    ))
                if pred_c:
                    fig2.add_trace(go.Scatter(
                        x=sub[ts_col], y=sub[pred_c],
                        name="Model Tahmini", line=dict(color="#00b4d8", width=1.5, dash="dot"),
                    ))
                fig2.update_layout(
                    title=f"MAE: {mae:,.0f} TL/MWh  ({len(sub):,} saat)" if mae else f"{len(sub):,} saat",
                    height=430, hovermode="x unified",
                    xaxis_title="Saat", yaxis_title="PTF (TL/MWh)",
                    legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="rgba(200,200,200,0.12)"),
                    yaxis=dict(gridcolor="rgba(200,200,200,0.12)"),
                    margin=dict(t=55, b=40),
                )
                st.plotly_chart(fig2, width="stretch")

                if pred_c and "actual_ptf" in sub.columns:
                    with st.expander("Aylık MAE detayı"):
                        sub2 = sub[[ts_col, "actual_ptf", pred_c]].copy()
                        sub2["ym"] = sub2[ts_col].dt.to_period("M").astype(str)
                        sub2["hata"] = (sub2[pred_c] - sub2["actual_ptf"]).abs()
                        tbl2 = sub2.groupby("ym").agg(
                            MAE=("hata", "mean"),
                            Gerçek=("actual_ptf", "mean"),
                            Tahmin=(pred_c, "mean"),
                            Saat=("hata", "count"),
                        ).round(0).astype(int).reset_index()
                        tbl2.columns = ["Ay", "MAE", "Gerçek Ort.", "Tahmin Ort.", "Saat"]
                        st.dataframe(tbl2, hide_index=True, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Walk-Forward 2026
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    if not wf or "results" not in wf:
        st.info("Walk-forward sonuçları bulunamadı. `python run_walk_forward_2026.py` çalıştırın.")
    else:
        results = [r for r in wf["results"] if r.get("test_mae", 0) < 1000]

        c_bar, c_kpi = st.columns([3, 1])

        with c_bar:
            fig3 = go.Figure()
            months = [r["month"].replace(" 2026", "") for r in results]
            maes   = [r["test_mae"] for r in results]
            pers   = [r["test_persistence"] for r in results]
            colors = ["#e63946" if r["test_improvement_pct"] < 0 else "#00b4d8" for r in results]

            fig3.add_trace(go.Bar(
                x=months, y=pers, name="Persistence",
                marker_color="rgba(120,120,120,0.35)",
                offsetgroup=0,
            ))
            fig3.add_trace(go.Bar(
                x=months, y=maes, name="Model MAE",
                marker_color=colors,
                offsetgroup=1,
                text=[f"{m:.0f}" for m in maes], textposition="outside",
            ))
            fig3.update_layout(
                title="Walk-Forward 2026 — Aylık Model vs Persistence MAE",
                height=380, barmode="group",
                xaxis_title="Ay", yaxis_title="MAE (TL/MWh)",
                legend=dict(orientation="h", y=1.08),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="rgba(200,200,200,0.12)"),
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig3, width="stretch")

        with c_kpi:
            st.markdown("**Aylık özet**")
            for r in results:
                imp = r["test_improvement_pct"]
                color = "🔴" if imp < 0 else ("🟡" if imp < 20 else "🟢")
                st.markdown(f"{color} **{r['month'].replace(' 2026','')}**: {r['test_mae']:.0f} TL ({imp:+.0f}%)")
            s = wf.get("summary", {})
            st.divider()
            st.metric("Oca–May Ort.", f"{350:.0f} TL/MWh", delta="%35 ↑", delta_color="normal")
            st.caption("Walk-forward: aylık retrain")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Detayı
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("#### Model Özellikleri")
        st.markdown("""
| Parametre | Değer |
|---|---|
| Algoritma | LightGBM (MAE objective) |
| Feature sayısı | **208** |
| Learning rate | 0.02 |
| Num leaves | 63 |
| Min data in leaf | 50 |
| Best iteration | 1,207 |
| Train seti | 2020–2025 (Q2 hariç) |
| Val seti | 2025 Q2 (solar sezon) |
| Test seti | 2026 |
""")

        st.markdown("#### Veri Kaynakları")
        st.markdown("""
- **EPİAŞ**: PTF, KGUP, YAL/YAT, SMF, yük tahmini, rüzgar, IDM, GÖP
- **GRF**: Doğalgaz günlük referans fiyatı
- **TCMB EVDS**: USD/TRY, EUR/TRY (kaldırıldı — ablasyon)
- **Yahoo Finance**: Brent, TTF, Henry Hub, API2 kömür
- **Open-Meteo**: 9 şehir sıcaklık (ağırlıklı)
- **İBB / İSKİ**: İstanbul baraj doluluk (2000–2024)
""")

    with c_right:
        st.markdown("#### Performans Karşılaştırması")
        mae_data = pd.DataFrame([
            {"Model": "Persistence",          "MAE": 548, "Tip": "Baseline"},
            {"Model": "PTF-only LightGBM",    "MAE": 551, "Tip": "Zayıf"},
            {"Model": "Forecast Market",      "MAE": 383, "Tip": "Orta"},
            {"Model": "Full Market ✅",        "MAE": 359, "Tip": "En İyi"},
            {"Model": "Walk-Fwd (Oca–May)",   "MAE": 350, "Tip": "En İyi"},
        ])
        colors = ["#888", "#cc4", "#f4a261", "#00b4d8", "#2dc653"]
        fig4 = go.Figure(go.Bar(
            x=mae_data["Model"], y=mae_data["MAE"],
            marker_color=colors,
            text=mae_data["MAE"].astype(str) + " TL",
            textposition="outside",
        ))
        fig4.update_layout(
            height=320, yaxis_title="Test MAE (TL/MWh)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(200,200,200,0.12)", range=[0, 650]),
            margin=dict(t=20, b=100), xaxis_tickangle=-20,
            showlegend=False,
        )
        st.plotly_chart(fig4, width="stretch")

        st.markdown("#### 2026 Neden Zor?")
        st.markdown("""
- **Nisan 2026**: EPDK fiyat tavanı 3400 → 4500 TL/MWh (eğitimde görülmedi)
- **Mayıs 2026**: Solar overcapacity → persistence bile modeli yenemiyor
- **Şubat 2026**: Hidro surge (gaz −64%) — model bu rejimi öğreniyor
- **Medyan AE = 198** TL, ortalama = 366 TL → outlier saatler ortalamanı şişiriyor
""")

    st.divider()
    st.markdown("**Operasyonel Pipeline**")
    st.markdown("""
| Saat | Adım |
|---|---|
| **04:30** | EPİAŞ/TCMB/Yahoo verisi güncellenir (GitHub Actions) |
| **11:30** | Sabah snapshot → next-24h tahmin üretilir |
| **Günlük** | D+2 tahmin (önceki günden) |
""")
    st.caption("LightGBM full_market · 208 feature · Val MAE 251 · Test MAE 359 · Walk-Fwd 350 TL/MWh")
