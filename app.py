# app.py
import pandas as pd
import pytz
import requests
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ===========================
# Configurações principais
# ===========================
st.set_page_config(page_title="Market Breadth",
                   layout="wide", page_icon="📈")
API_KEY = "3CImfjoxNd98om3uhS89X4lmlp4Mrp3H"
TZ = pytz.timezone("America/Sao_Paulo")
ASSETS = ['AUDUSD', 'CADUSD', 'CHFUSD', 'CNHUSD', 'CZKUSD', 'DKKUSD', 'EURUSD',
          'GBPUSD', 'HUFUSD', 'HKDUSD', 'JPYUSD', 'NOKUSD', 'NZDUSD', 'PLNUSD',
          'SEKUSD', 'SGDUSD', 'TRYUSD', 'XAUUSD', 'XAGUSD', 'ZARUSD']
NUM_CANDLES = 120  # últimos 120 candles
TOTAL_ASSETS = len(ASSETS)
LOWER_LINE = round(TOTAL_ASSETS * 0.1)
UPPER_LINE = round(TOTAL_ASSETS * 0.9)

# Atualização automática a cada 60s
st_autorefresh(interval=60 * 1000, key="refresh")

# ===========================
# Menu lateral
# ===========================
st.sidebar.title("Configurações")
MA_INPUT = st.sidebar.text_input(
    "Períodos", "50,72,200")
MA_PERIODS = [int(x.strip())
              for x in MA_INPUT.split(",") if x.strip().isdigit()]

TIMEFRAME = st.sidebar.radio("Timeframe", ["1min", "5min", "15min", "1h"])
if TIMEFRAME == "1min":
    BASE_URL = "https://financialmodelingprep.com/stable/historical-chart/1min"
elif TIMEFRAME == "5min":
    BASE_URL = "https://financialmodelingprep.com/stable/historical-chart/5min"
elif TIMEFRAME == "15min":
    BASE_URL = "https://financialmodelingprep.com/stable/historical-chart/15min"
else:
    BASE_URL = "https://financialmodelingprep.com/stable/historical-chart/1hour"

# ===========================
# Funções auxiliares
# ===========================


@st.cache_data(ttl=60)
def get_data(symbol: str, base_url: str) -> pd.DataFrame | None:
    try:
        r = requests.get(
            f"{base_url}?symbol={symbol}&apikey={API_KEY}", timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        if not j:
            return None
        df = pd.DataFrame(j)
        df["date"] = pd.to_datetime(df["date"])
        # Corrigir fuso: soma 1 hora para ajustar BRT
        df["date"] = df["date"] + pd.Timedelta(hours=1)
        df = df.set_index("date").sort_index()
        df = df.tail(NUM_CANDLES)
        return df
    except Exception:
        return None


def build_combined_close(symbols, base_url):
    frames = []
    for s in symbols:
        df = get_data(s, base_url)
        if df is None or df.empty:
            continue
        frames.append(df["close"].rename(f"{s}_Close"))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1)
    return combined


# ===========================
# Tabs
# ===========================
tab1, tab2 = st.tabs(["📊 Gráficos", "📑 Resumo de Mercado"])

# ===========================
# TAB 1 — Gráficos
# ===========================
with tab1:
    st.title("Market Breadth")
    st.caption(
        f"Períodos configuráveis — Timeframe: {TIMEFRAME} — Últimos {NUM_CANDLES} candles")

    with st.spinner(f"Buscando dados {TIMEFRAME}..."):
        combined = build_combined_close(ASSETS, BASE_URL)

    if combined.empty:
        st.error("Nenhum dado disponível.")
        st.stop()

    # Calcular EMAs e flags "acima"
    for s in ASSETS:
        close_col = f"{s}_Close"
        if close_col not in combined.columns:
            continue
        for p in MA_PERIODS:
            ema_col = f"{s}_EMA{p}"
            above_col = f"{s}_Above_EMA{p}"
            combined[ema_col] = combined[close_col].ewm(
                span=p, adjust=False).mean()
            combined[above_col] = combined[close_col] > combined[ema_col]

    # Contagem de ativos acima de cada EMA
    counts = {}
    for p in MA_PERIODS:
        filtered_cols = combined.filter(like=f"_Above_EMA{p}")
        if filtered_cols.empty:
            counts[p] = pd.Series(dtype=int)
        else:
            counts[p] = filtered_cols.sum(axis=1)

    # Gráficos empilhados verticalmente
    now_local = datetime.now(TZ)
    for idx, p in enumerate(MA_PERIODS):
        series = counts[p]
        fig = go.Figure()
        if not series.empty:
            # Linha principal suave com sombreamento
            fig.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line=dict(
                    width=2, color=f"rgb({50+idx*60},{100+idx*30},{200-idx*50})"),
                fill="tozeroy",
                fillcolor=f"rgba({50+idx*60},{100+idx*30},{200-idx*50},0.1)"
            ))
            # Último ponto destacado
            fig.add_trace(go.Scatter(
                x=[series.index[-1]],
                y=[series.values[-1]],
                mode="markers+text",
                text=[series.values[-1]],
                textposition="top right",
                marker=dict(size=10, color="red")
            ))
        # Linhas horizontais discretas adaptadas ao total de ativos
        fig.add_hline(y=LOWER_LINE, line_dash="dot", line_color="gray",
                      annotation_text=f"{LOWER_LINE}", annotation_position="bottom right")
        fig.add_hline(y=UPPER_LINE, line_dash="dot", line_color="gray",
                      annotation_text=f"{UPPER_LINE}", annotation_position="bottom right")
        fig.update_layout(
            template="plotly_white",
            height=350,
            title=dict(
                text=f"{p} — Última atualização: {now_local.strftime('%H:%M:%S')}", font=dict(size=14)),
            margin=dict(l=10, r=10, t=40, b=20),
            xaxis=dict(title="Data/Hora (BRT)", showgrid=False, nticks=8),
            yaxis=dict(title="Ativos acima", showgrid=False,
                       gridcolor="rgba(200,200,200,0.3)", dtick=1, rangemode="tozero")
        )
        st.plotly_chart(fig, use_container_width=True)

# ===========================
# TAB 2 — Resumo de Mercado
# ===========================
with tab2:
    st.subheader("Resumo atual")
    latest_counts = {f"EMA{p}": int(
        counts[p].iloc[-1]) if not counts[p].empty else 0 for p in MA_PERIODS}
    st.table(pd.DataFrame.from_dict(latest_counts,
             orient="index", columns=["Ativos acima"]))

    st.subheader("Ativos atualmente acima de cada EMA")
    latest_row = combined.iloc[-1] if not combined.empty else pd.Series()
    for p in MA_PERIODS:
        above_cols = [
            c for c in combined.columns if c.endswith(f"_Above_EMA{p}")]
        currently = [c.replace("_Above_EMA" + str(p), "")
                     for c in above_cols if latest_row.get(c, False)]
        st.markdown(f"**EMA{p}** — {len(currently)} pares")
        if currently:
            st.write(", ".join(currently))
        else:
            st.write("_Nenhum par acima desta EMA no momento_")

    st.caption("Feito com Streamlit • Dados via FinancialModelingPrep")


