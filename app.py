import pandas as pd
import pytz
import requests
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===========================
# Configurações principais
# ===========================
st.set_page_config(page_title="Market Breadth",
                   layout="wide", page_icon="📈")
API_KEY = "3CImfjoxNd98om3uhS89X4lmlp4Mrp3H"
TZ = pytz.timezone("America/Sao_Paulo")

# --- CESTA DE ATIVOS E PESOS ---
# Foco na força do Dólar (USD/XXX). Os pesos somam 100.
ASSET_WEIGHTS = {
    'DX-Y.NYB': 20, 'USDJPY': 20, 'USDCHF': 10, 'USDCAD': 10, 'USDCNH': 10,
    'USDSEK': 5,  'USDNOK': 5,  'USDMXN': 5,  'USDSGD': 5,  'USDZAR': 2,
    'USDHKD': 2,  'USDPLN': 2,  'USDCZK': 1,  'USDDKK': 1,  'USDHUF': 1
}
ASSETS = list(ASSET_WEIGHTS.keys())
NUM_CANDLES_DISPLAY = 120  # Número de velas para mostrar nos gráficos

# Atualização automática a cada 60s
st_autorefresh(interval=60 * 1000, key="refresh")

# ===========================
# Menu lateral
# ===========================
st.sidebar.title("Configurações")
MA_INPUT = st.sidebar.text_input(
    "Períodos das Médias Móveis", "9,21,72,200")
MA_PERIODS = [int(x.strip())
              for x in MA_INPUT.split(",") if x.strip().isdigit()]

TIMEFRAME = st.sidebar.radio("Timeframe", ["1min", "5min", "15min", "1h"])

st.sidebar.header("Parâmetros do Z-Score")
Z_SCORE_WINDOW = st.sidebar.slider("Janela de Cálculo do Z-Score", 50, 500, 200)


# ===========================
# Funções auxiliares
# ===========================

@st.cache_data(ttl=60)
def get_single_data(symbol: str, timeframe: str, candles_to_fetch: int) -> pd.DataFrame | None:
    """Busca dados para um único ativo com fuso horário corrigido."""
    try:
        base_url = f"https://financialmodelingprep.com/api/v3/historical-chart/{timeframe}/{symbol}"
        r = requests.get(base_url, params={"apikey": API_KEY}, timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        if not j:
            return None
        df = pd.DataFrame(j).iloc[::-1]
        df["date"] = pd.to_datetime(df["date"])
        df['date'] = df['date'].dt.tz_localize('US/Eastern').dt.tz_convert(TZ)
        df = df.set_index("date")
        df = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        return df.tail(candles_to_fetch)
    except Exception:
        return None

def build_combined_data(symbols: list, timeframe: str, candles_to_fetch: int) -> pd.DataFrame:
    """Busca dados para todos os ativos em paralelo para maior velocidade."""
    with st.spinner(f"A buscar dados para {len(symbols)} ativos..."):
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_symbol = {executor.submit(get_single_data, s, timeframe, candles_to_fetch): s for s in symbols}
            frames = []
            for future in as_completed(future_to_symbol):
                df = future.result()
                if df is not None and not df.empty:
                    df.columns = [f"{future_to_symbol[future]}_{col}" for col in df.columns]
                    frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined_df = pd.concat(frames, axis=1)
    return combined_df.ffill().dropna()

# ===========================
# Lógica Principal e Cálculos
# ===========================
st.title("Market Breadth: Força do Dólar")
st.caption(f"Timeframe: {TIMEFRAME} | Última atualização: {datetime.now(TZ).strftime('%H:%M:%S')}")

candles_to_fetch = (max(MA_PERIODS) if MA_PERIODS else 200) + NUM_CANDLES_DISPLAY + Z_SCORE_WINDOW
combined = build_combined_data(ASSETS, TIMEFRAME, candles_to_fetch)

if combined.empty:
    st.error("Nenhum dado disponível. Verifique a API ou os símbolos dos ativos.")
    st.stop()

# --- Cálculos de EMAs e Condições ---
for s in ASSETS:
    close_col = f"{s}_close"
    if close_col not in combined.columns: continue
    for p in MA_PERIODS:
        above_col = f"{s}_Above_EMA{p}"
        ema_val = combined[close_col].ewm(span=p, adjust=False).mean()
        combined[above_col] = combined[close_col] > ema_val

# --- Cálculos de Amplitude, Z-Score, ROC e Aceleração ---
weighted_counts = {}
z_scores = {}
rocs = {}
accelerations = {}

for p in MA_PERIODS:
    total_weight_series = pd.Series(0.0, index=combined.index)
    for s in ASSETS:
        above_col = f"{s}_Above_EMA{p}"
        if above_col in combined.columns:
            weight = ASSET_WEIGHTS.get(s, 0)
            total_weight_series += combined[above_col].astype(int) * weight
    
    # Armazena a série ponderada
    weighted_counts[p] = total_weight_series
    
    # Cálculo do Z-Score
    mean = total_weight_series.rolling(window=Z_SCORE_WINDOW).mean()
    std = total_weight_series.rolling(window=Z_SCORE_WINDOW).std()
    z_scores[p] = (total_weight_series - mean) / std
    
    # Cálculo do ROC (Velocidade - 1ª Derivada)
    rocs[p] = total_weight_series.diff()
    
    # Cálculo da Aceleração (2ª Derivada)
    accelerations[p] = rocs[p].diff()


# ===========================
# Visualização em Abas
# ===========================
tab1, tab2 = st.tabs(["📊 Gráficos", "📑 Resumo de Mercado"])

with tab1:
    # --- MODELO PRINCIPAL: AMPLITUDE PONDERADA ---
    st.header("Modelo Principal: Amplitude Ponderada pela Liquidez")
    st.markdown("Mede a força total do sentimento do mercado, ponderada pela liquidez de cada ativo.")

    for p in MA_PERIODS:
        series = weighted_counts[p].tail(NUM_CANDLES_DISPLAY)
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", fill="tozeroy", name=f'EMA {p} Ponderada'))
        fig_w.add_hline(y=10, line_dash="dot", line_color="green", annotation_text="Extremo Venda (10%)")
        fig_w.add_hline(y=90, line_dash="dot", line_color="red", annotation_text="Extremo Compra (90%)")
        fig_w.update_layout(
            title=f'Força Ponderada Acima da EMA {p}',
            yaxis=dict(title="Força Ponderada (0-100)", range=[0, 100]),
            height=350, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_w, use_container_width=True)

    st.divider()

    # --- IMPLEMENTAÇÃO 2: Z-SCORE DA AMPLITUDE ---
    st.header("Implementação 2: Z-Score da Amplitude Ponderada")
    st.markdown("Mede quão estatisticamente incomum (em desvios padrão) está a leitura atual da amplitude.")

    for p in MA_PERIODS:
        series = z_scores[p].tail(NUM_CANDLES_DISPLAY)
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=f'Z-Score EMA {p}', line=dict(color='orange')))
        fig_z.add_hline(y=-2, line_dash="dot", line_color="green", annotation_text="Extremo de Pânico (-2σ)")
        fig_z.add_hline(y=2, line_dash="dot", line_color="red", annotation_text="Extremo de Euforia (+2σ)")
        fig_z.update_layout(
            title=f'Z-Score da Amplitude (Base EMA {p})',
            yaxis=dict(title="Desvios Padrão (σ)", range=[-3.5, 3.5]),
            height=350, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_z, use_container_width=True)

    st.divider()

    # --- IMPLEMENTAÇÃO 3: VELOCIDADE E ACELERAÇÃO ---
    st.header("Implementação 3: Velocidade e Aceleração da Amplitude")
    st.markdown("Mede a velocidade (1ª Derivada/ROC) e aceleração (2ª Derivada) da Força Ponderada. Picos indicam impulsos ou clímax. A desaceleração pode sinalizar exaustão.")

    for p in MA_PERIODS:
        # Gráfico de Velocidade (ROC)
        roc_series = rocs[p].tail(NUM_CANDLES_DISPLAY)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Bar(x=roc_series.index, y=roc_series.values, name='ROC', marker_color=['green' if v >= 0 else 'red' for v in roc_series.values]))
        fig_roc.update_layout(
            title=f'Velocidade (ROC) da Amplitude (Base EMA {p})',
            yaxis_title="Variação de Força",
            height=250, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Gráfico de Aceleração
        accel_series = accelerations[p].tail(NUM_CANDLES_DISPLAY)
        fig_accel = go.Figure()
        fig_accel.add_trace(go.Bar(x=accel_series.index, y=accel_series.values, name='Aceleração', marker_color=['#1f77b4' if v >= 0 else '#ff7f0e' for v in accel_series.values]))
        fig_accel.update_layout(
            title=f'Aceleração da Amplitude (Base EMA {p})',
            yaxis_title="Variação da Velocidade",
            height=250, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_accel, use_container_width=True)


with tab2:
    st.header("Resumo Atual")
    
    # --- Resumo Ponderado e Z-Score ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Força Ponderada")
        latest_counts_weighted = {f"EMA {p}": f"{int(weighted_counts[p].iloc[-1])}%" for p in MA_PERIODS if not weighted_counts[p].empty}
        st.table(pd.DataFrame.from_dict(latest_counts_weighted, orient="index", columns=["Força Atual"]))
    with col2:
        st.subheader("Z-Score")
        latest_z_scores = {f"EMA {p}": f"{z_scores[p].iloc[-1]:.2f} σ" for p in MA_PERIODS if not z_scores[p].empty}
        st.table(pd.DataFrame.from_dict(latest_z_scores, orient="index", columns=["Nível de Extremo Atual"]))

    # --- Resumo ROC e Aceleração ---
    st.subheader("Dinâmica Atual (ROC & Aceleração)")
    summary_data = {}
    for p in MA_PERIODS:
        if not rocs[p].empty and not accelerations[p].empty:
            summary_data[f"EMA {p}"] = {
                "Velocidade (ROC)": f"{rocs[p].iloc[-1]:.2f}",
                "Aceleração": f"{accelerations[p].iloc[-1]:.2f}"
            }
    if summary_data:
        st.table(pd.DataFrame.from_dict(summary_data, orient='index'))


st.caption("Feito com Streamlit • Dados via FinancialModelingPrep")

