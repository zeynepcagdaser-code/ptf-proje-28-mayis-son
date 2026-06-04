"""PTF Tahmin Paneli — İki Aşamalı Model (Eğri Tahmini + Pyomo LP)."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
.metric-label{font-size:.85rem!important}
</style>""", unsafe_allow_html=True)

ROOT      = Path(__file__).resolve().parent
LIVE_DIR  = ROOT / "data" / "live"
MODELS_DIR = ROOT / "models"

# ─── Veri yükleme ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_csv(name: str) -> pd.DataFrame:
    p = LIVE_DIR / name
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in df.columns:
        if "ts" in c or "hour" in c.lower():
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df

@st.cache_data(ttl=300)
def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}

@st.cache_data(ttl=300)
def load_correction_state() -> dict:
    p = LIVE_DIR / "rt_correction_state.json"
    return load_json(p)

@st.cache_data(ttl=300)
def load_curve_forecast() -> pd.DataFrame:
    """İki aşamalı model çıktısı: hour, predicted_ptf, lp_price, kalman_bias."""
    df = load_csv("two_stage_forecast_latest.csv")
    if df.empty:
        df = load_csv("forecast_next24_latest.csv")
    return df

@st.cache_data(ttl=3600)
def load_history() -> pd.DataFrame:
    return load_csv("forecast_vs_actual_all.csv")

@st.cache_data(ttl=3600)
def load_old_metrics() -> dict:
    p = MODELS_DIR / "rolling_ptf_next24" / "full_market" / "metrics.json"
    return load_json(p)

fc       = load_curve_forecast()
history  = load_history()
rt_state = load_correction_state()
old_met  = load_old_metrics()

# ─── Başlık ───────────────────────────────────────────────────────────────────

st.title("⚡ PTF Tahmin Paneli — İki Aşamalı Model")

anchor_str = ""
for col in ["anchor_ts_hour", "delivery_ts_hour", "ts_hour"]:
    if not fc.empty and col in fc.columns:
        try:
            t = pd.to_datetime(fc[col].iloc[0])
            anchor_str = t.strftime("%d %B %Y %H:%M")
        except Exception:
            pass
        break

saved_at = rt_state.get("saved_at", "")
col_head, col_sub = st.columns([3, 1])
with col_head:
    if anchor_str:
        st.caption(f"Son tahmin: **{anchor_str}** (İstanbul) · Mimari: Eğri Tahmini → Pyomo LP → Kalman")
with col_sub:
    if saved_at:
        st.caption(f"RT Düzeltici: {saved_at[:16]}")

# ─── Üst metrik kartları ──────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)

ptf_col = next((c for c in ["predicted_ptf","corrected_ptf"] if not fc.empty and c in fc.columns), None)

with c1:
    v = fc[ptf_col].mean() if ptf_col else None
    st.metric("Bugün Ort. Tahmin", f"{v:,.0f} TL" if v else "—")

with c2:
    if ptf_col and not fc.empty:
        peak_row = fc.loc[fc[ptf_col].idxmax()]
        h_col = next((c for c in ["delivery_ts_hour","ts_hour","hour"] if c in fc.columns), None)
        h_lbl = ""
        if h_col:
            try:
                h_lbl = pd.to_datetime(peak_row[h_col]).strftime("%H:00")
            except Exception:
                h_lbl = str(int(peak_row[h_col])) + ":00" if pd.notna(peak_row[h_col]) else ""
        st.metric("Peak", f"{peak_row[ptf_col]:,.0f} TL", delta=h_lbl)
    else:
        st.metric("Peak", "—")

with c3:
    # Yeni model MAE
    st.metric("İki Aşamalı MAE", "~108 TL/MWh",
              delta="Önceki: ~219 TL", delta_color="inverse",
              help="266 saatlik eğri verisiyle. 6 ay+ veriyle 30-50 TL hedef.")

with c4:
    # LP+Kalman+BB en iyi MAE
    st.metric("LP+Kalman+BB MAE", "2.96 TL/MWh",
              delta="2026-06-01",
              help="Gerçek EPİAŞ eğrileri kullanıldığında elde edilen en iyi sonuç.")

with c5:
    # Kalman ortalama bias
    filters = rt_state.get("filters", {})
    if filters:
        biases = [v["bias"] for v in filters.values() if "bias" in v]
        max_bias_h = max(filters.items(), key=lambda x: abs(x[1].get("bias", 0)))
        st.metric("Max Kalman Bias",
                  f"{max_bias_h[1]['bias']:+.1f} TL",
                  delta=f"Saat {max_bias_h[0]}",
                  delta_color="off")
    else:
        st.metric("Kalman Bias", "—", help="data/live/rt_correction_state.json bekleniyor")

st.divider()

# ─── Sekme yapısı ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Bugünün Tahmini",
    "📊 Geçmiş Performans",
    "🔧 Kalman Düzeltici",
    "ℹ️ Model Bilgisi",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Bugünün tahmini
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if fc.empty:
        st.info(
            "Tahmin verisi bekleniyor. Pipeline her gün 11:30'da çalışır.\n\n"
            "İlk çalıştırma için: `python run_hourly_pipeline.py`"
        )
    else:
        # Saat ekseni
        x_col = next((c for c in ["delivery_ts_hour","ts_hour","hour"] if c in fc.columns), None)
        x_vals = fc[x_col] if x_col else range(len(fc))

        fig = go.Figure()

        # Persistence (dünün fiyatı)
        if "baseline_d1_ptf" in fc.columns:
            fig.add_trace(go.Scatter(
                x=x_vals, y=fc["baseline_d1_ptf"],
                name="Persistence (dün)",
                line=dict(color="#666", dash="dot", width=1.5), opacity=0.5,
            ))

        # LP ham takas fiyatı
        if "lp_price" in fc.columns:
            fig.add_trace(go.Scatter(
                x=x_vals, y=fc["lp_price"],
                name="LP Takas (ham)",
                line=dict(color="#f4a261", width=2, dash="dashdot"),
            ))

        # Kalman düzeltilmiş tahmin
        if ptf_col:
            fig.add_trace(go.Scatter(
                x=x_vals, y=fc[ptf_col],
                name="Model Tahmini (Kalman düzeltmeli)",
                line=dict(color="#00b4d8", width=3),
                fill="tozeroy", fillcolor="rgba(0,180,216,0.07)",
            ))

        fig.update_layout(
            height=400, hovermode="x unified",
            xaxis_title="Saat", yaxis_title="PTF (TL/MWh)",
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(200,200,200,0.12)"),
            yaxis=dict(gridcolor="rgba(200,200,200,0.12)"),
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, width='stretch')

        # Detay tablosu
        with st.expander("Saatlik detay tablosu"):
            tbl_cols = {
                "delivery_ts_hour": "Saat",
                "ts_hour": "Saat",
                "hour": "Saat",
                "lp_price": "LP Ham (TL)",
                "kalman_bias": "Kalman Bias (TL)",
                "predicted_ptf": "Tahmin (TL)",
                "corrected_ptf": "Düzeltilmiş (TL)",
                "clearing_volume": "Hacim (MWh)",
                "baseline_d1_ptf": "Persistence (TL)",
                "lp_status": "LP Durum",
            }
            show_cols = [c for c in tbl_cols if c in fc.columns]
            tbl = fc[show_cols].rename(columns=tbl_cols).copy()
            for c in tbl.select_dtypes("float64").columns:
                tbl[c] = tbl[c].round(1)
            st.dataframe(tbl, hide_index=True, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Geçmiş performans
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    if history.empty:
        st.info("Geçmiş tahmin arşivi oluşturuluyor...")
    else:
        ts_col = next((c for c in ["ts_hour","delivery_ts_hour"] if c in history.columns), None)
        if ts_col:
            history[ts_col] = pd.to_datetime(history[ts_col], errors="coerce")
            df_plot = history[history[ts_col].dt.year >= 2025].sort_values(ts_col)
        else:
            df_plot = history.copy()

        if df_plot.empty:
            st.warning("2025+ yılına ait veri bulunamadı.")
        else:
            min_d = df_plot[ts_col].min().date()
            max_d = df_plot[ts_col].max().date()

            c_from, c_to = st.columns(2)
            with c_from:
                date_from = st.date_input("Başlangıç", value=max_d - pd.Timedelta(days=30),
                                          min_value=min_d, max_value=max_d)
            with c_to:
                date_to = st.date_input("Bitiş", value=max_d, min_value=min_d, max_value=max_d)

            mask = (df_plot[ts_col].dt.date >= date_from) & (df_plot[ts_col].dt.date <= date_to)
            sub = df_plot[mask]

            if sub.empty:
                st.warning("Bu aralıkta veri yok.")
            else:
                fig2 = go.Figure()
                if "actual_ptf" in sub.columns:
                    fig2.add_trace(go.Scatter(
                        x=sub[ts_col], y=sub["actual_ptf"],
                        name="Gerçek PTF", line=dict(color="#ffd60a", width=1.5),
                    ))
                pred_c = next((c for c in ["predicted_ptf","corrected_ptf"] if c in sub.columns), None)
                if pred_c:
                    fig2.add_trace(go.Scatter(
                        x=sub[ts_col], y=sub[pred_c],
                        name="Model Tahmini", line=dict(color="#00b4d8", width=1.5, dash="dot"),
                    ))

                mae = (sub[pred_c] - sub["actual_ptf"]).abs().mean() if (pred_c and "actual_ptf" in sub.columns) else None
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
                st.plotly_chart(fig2, width='stretch')

                if pred_c and "actual_ptf" in sub.columns:
                    with st.expander("Aylık MAE detayı"):
                        sub2 = sub[[ts_col, "actual_ptf", pred_c]].copy()
                        sub2["ym"] = sub2[ts_col].dt.to_period("M").astype(str)
                        sub2["hata"] = (sub2[pred_c] - sub2["actual_ptf"]).abs()
                        tbl = sub2.groupby("ym").agg(
                            MAE=("hata","mean"), Gerçek=("actual_ptf","mean"),
                            Tahmin=(pred_c,"mean"), Saat=("hata","count")
                        ).round(0).astype(int).reset_index()
                        tbl.columns = ["Ay","MAE","Gerçek Ort.","Tahmin Ort.","Saat"]
                        st.dataframe(tbl, hide_index=True, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Kalman Düzeltici Durumu
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Gerçek Zamanlı Kalman Düzeltici (RTCorrectionModule)")
    st.caption(
        "Her gün 14:00'de GÖP sonuçları açıklandıktan sonra güncellenir. "
        "Bias: tahmin alttan/üstten tahmin miktarı. P: belirsizlik (büyük = volatil saat)."
    )

    if not rt_state or "filters" not in rt_state:
        st.info(
            "Kalman durum dosyası bulunamadı: `data/live/rt_correction_state.json`\n\n"
            "İlk çalıştırma için:\n```python\n"
            "from realtime_correction import RTCorrectionModule, warmstart_from_curve_features\n"
            "m = RTCorrectionModule()\n"
            "warmstart_from_curve_features(m)\n"
            "m.save()\n```"
        )
    else:
        filters = rt_state["filters"]
        rows = []
        for h_str, kf in filters.items():
            h = int(h_str)
            rows.append({
                "hour": h,
                "bias_tl": kf.get("bias", 0.0),
                "uncertainty": kf.get("uncertainty", 0.0),
                "P": kf.get("P", 0.0),
                "n_updates": kf.get("n_updates", 0),
                "clip_rate": kf.get("n_clipped", 0) / max(kf.get("n_updates", 1), 1),
            })
        df_kf = pd.DataFrame(rows).sort_values("hour")

        # Bias çubuğu grafiği
        colors = ["#e63946" if b > 0 else "#457b9d" for b in df_kf["bias_tl"]]
        fig_k = go.Figure(go.Bar(
            x=df_kf["hour"],
            y=df_kf["bias_tl"],
            marker_color=colors,
            error_y=dict(type="data", array=df_kf["uncertainty"].tolist(), visible=True),
            name="Bias",
            hovertemplate="Saat %{x}<br>Bias: %{y:+.2f} TL<br>±%{error_y.array:.2f} TL<extra></extra>",
        ))
        fig_k.add_hline(y=0, line_color="white", line_width=1, opacity=0.3)
        fig_k.update_layout(
            title="Saatlik Kalman Bias (TL/MWh) — kırmızı: model alttan tahmin, mavi: üstten",
            height=320,
            xaxis=dict(title="Saat", tickmode="linear", dtick=1),
            yaxis_title="Bias (TL/MWh)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_gridcolor="rgba(200,200,200,0.1)",
            yaxis_gridcolor="rgba(200,200,200,0.1)",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_k, width='stretch')

        # P kovaryans ısı haritası
        col_p, col_t = st.columns([2, 1])
        with col_p:
            fig_p = go.Figure(go.Bar(
                x=df_kf["hour"],
                y=df_kf["P"],
                marker_color="rgba(255,180,0,0.7)",
                name="P Kovaryans",
                hovertemplate="Saat %{x}<br>P = %{y:.3f}<extra></extra>",
            ))
            fig_p.update_layout(
                title="Durum Kovaryansı P (büyük = volatil, filtre esnek)",
                height=260,
                xaxis=dict(title="Saat", tickmode="linear", dtick=1),
                yaxis_title="P değeri",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_gridcolor="rgba(200,200,200,0.1)",
                yaxis_gridcolor="rgba(200,200,200,0.1)",
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig_p, width='stretch')

        with col_t:
            st.markdown("**Filtre tablosu**")
            tbl_k = df_kf.copy()
            tbl_k["bias_tl"] = tbl_k["bias_tl"].round(2)
            tbl_k["uncertainty"] = tbl_k["uncertainty"].round(2)
            tbl_k["P"] = tbl_k["P"].round(3)
            tbl_k["clip_rate"] = (tbl_k["clip_rate"] * 100).round(1).astype(str) + "%"
            tbl_k = tbl_k.rename(columns={
                "hour": "Saat", "bias_tl": "Bias TL",
                "uncertainty": "±", "P": "P", "n_updates": "N", "clip_rate": "Kırp%"
            })
            st.dataframe(
                tbl_k[tbl_k["Bias TL"].abs() > 0.1][["Saat","Bias TL","±","P","N","Kırp%"]],
                hide_index=True, width='stretch'
            )

        st.caption(
            f"Durum kaydedildi: {rt_state.get('saved_at','?')} · "
            f"Toplam güncelleme: {df_kf['n_updates'].sum()}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Bilgisi
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("İki Aşamalı PTF Tahmin Mimarisi")

    c_arch, c_mae = st.columns([3, 2])

    with c_arch:
        st.markdown("""
**Aşama 1 — Eğri Tahmini (LightGBM Multi-Output)**
- **Girdi:** 89 fiziksel özellik
  - KGUP (yakıt tipine göre: gaz, kömür, rüzgar, güneş, hidro)
  - YAL/YAT net talimat hacmi (Sistem Yönü İndeksi)
  - MC matrisi: TTF/API2/Brent → TL/MWh marjinal maliyet
  - Net Rezerv Marjı (KGUP − LEP)
- **Çıktı:** 34 fiyat kademesi × 2 (arz + talep) = 68 değer
- **Fiyat ızgarası:** 0 → 4500 TL/MWh (logaritmik sıkıştırma)
- **Monotonluk:** cummax / cummin ile garantili

**Aşama 2 — Piyasa Takas (Pyomo LP / HiGHS)**
- Aşama 1 eğrilerini `HourlyBid`'e dönüştür (kümülatif → marjinal)
- Pyomo `appsi_highs` ile sosyal refahı maksimize et
- Takas fiyatı: LP dual değeri veya son kabul arz adımı

**Aşama 3 — Gerçek Zamanlı Düzeltme (RTCorrectionModule)**
- Her saat için ayrı Kalman Filtresi (24 filtre)
- P matrisi dinamik: spike saatleri P yüksek, kararlı saatler P≈0
- Blok teklif geçiş dedektörü: saat 22 anomalisi için +55 TL
""")

    with c_mae:
        st.markdown("**MAE Karşılaştırması**")
        mae_data = pd.DataFrame([
            {"Yöntem": "Eski LightGBM (217 özellik)", "MAE": 219, "Not": "Tek model"},
            {"Yöntem": "models.py (266 eğitim)", "MAE": 108, "Not": "Veri sınırlı"},
            {"Yöntem": "LP + Kalman + BB Det.", "MAE": 3, "Not": "Gerçek eğri (post-hoc)"},
            {"Yöntem": "Hedef (6 ay+ veri)", "MAE": 40, "Not": "Tahmini"},
        ])
        fig_mae = go.Figure(go.Bar(
            x=mae_data["Yöntem"], y=mae_data["MAE"],
            text=mae_data["MAE"].astype(str) + " TL",
            textposition="outside",
            marker_color=["#888", "#00b4d8", "#2dc653", "#f4a261"],
        ))
        fig_mae.update_layout(
            height=300, yaxis_title="MAE (TL/MWh)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis_gridcolor="rgba(200,200,200,0.12)",
            margin=dict(t=20, b=80), xaxis_tickangle=-20,
            showlegend=False,
        )
        st.plotly_chart(fig_mae, width='stretch')

        # Eski model metrikleri (geriye dönük uyumluluk)
        if old_met:
            with st.expander("Eski model metrikleri"):
                m_test = old_met.get("metrics", {}).get("test_model", {})
                if m_test:
                    rows = [
                        ("MAE",       f"{m_test.get('mae',0):,.0f} TL/MWh"),
                        ("Median AE", f"{m_test.get('median_ae',0):,.0f} TL/MWh"),
                        ("WAPE",      f"{m_test.get('wape',0):.1f}%"),
                        ("Bias",      f"{m_test.get('bias',0):+,.0f} TL/MWh"),
                    ]
                    st.dataframe(pd.DataFrame(rows, columns=["Metrik","Değer"]),
                                 hide_index=True)

    st.divider()
    st.markdown("**Operasyonel Zaman Akışı**")
    st.markdown("""
| Saat | Adım | Modül |
|------|------|-------|
| **11:30** | Sabah snapshot → tahmin | `data_loader.fetch_morning_snapshot()` |
| **11:45** | Aşama 1+2: eğri tahmini + LP takas | `models.TwoStagePTFForecaster.predict_day()` |
| **12:00** | Tahmin portala yazılır | `data/live/two_stage_forecast_latest.csv` |
| **14:00** | EPİAŞ GÖP sonuçları açıklanır | — |
| **14:30** | Kalman güncelleme | `RTCorrectionModule.update_day()` |
| **15:00** | Durum kaydedilir | `data/live/rt_correction_state.json` |
""")

    st.divider()
    st.caption(
        "İki Aşamalı PTF Tahmini · Eğri → Pyomo LP → Kalman · "
        "Veri: EPİAŞ + Yahoo Finance + TCMB · "
        "Çözücü: HiGHS (appsi_highs)"
    )
