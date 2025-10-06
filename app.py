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

# --- CESTA DE ATIVOS E PESOS PARA A IMPLEMENTAÇÃO 1 ---
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
        df = pd.DataFrame(j).iloc[::-1] # API retorna do mais recente para o mais antigo
        df["date"] = pd.to_datetime(df["date"])
        # CORREÇÃO DE TIMEZONE: Localiza como fuso de NY e converte para São Paulo
        df['date'] = df['date'].dt.tz_localize('US/Eastern').dt.tz_convert(TZ)
        df = df.set_index("date")
        df = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        return df.tail(candles_to_fetch)
    except Exception as e:
        # st.sidebar.error(f"Erro em {symbol}: {e}") # Para depuração
        return None

def build_combined_data(symbols: list, timeframe: str, candles_to_fetch: int) -> pd.DataFrame:
    """Busca dados para todos os ativos em paralelo para maior velocidade."""
    combined_df = pd.DataFrame()
    with st.spinner(f"A buscar dados para {len(symbols)} ativos..."):
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_symbol = {executor.submit(get_single_data, s, timeframe, candles_to_fetch): s for s in symbols}
            frames = []
            for future in as_completed(future_to_symbol):
                df = future.result()
                if df is not None and not df.empty:
                    # Renomeia colunas para evitar conflitos
                    df.columns = [f"{future_to_symbol[future]}_{col}" for col in df.columns]
                    frames.append(df)
    if frames:
        combined_df = pd.concat(frames, axis=1)
        # Preenche lacunas e alinha os dados para estabilidade
        combined_df = combined_df.ffill().dropna()
    return combined_df

# ===========================
# Lógica Principal e Cálculos
# ===========================
st.title("Market Breadth: Força do Dólar")
st.caption(f"Timeframe: {TIMEFRAME} | Última atualização: {datetime.now(TZ).strftime('%H:%M:%S')}")

# Busca um histórico maior para garantir o cálculo correto das médias
candles_to_fetch = (max(MA_PERIODS) if MA_PERIODS else 200) + NUM_CANDLES_DISPLAY
combined = build_combined_data(ASSETS, TIMEFRAME, candles_to_fetch)

if combined.empty:
    st.error("Nenhum dado disponível. Verifique a API ou os símbolos dos ativos.")
    st.stop()

# --- Cálculos de EMAs e Condições ---
for s in ASSETS:
    close_col = f"{s}_close"
    if close_col not in combined.columns:
        continue
    for p in MA_PERIODS:
        above_col = f"{s}_Above_EMA{p}"
        ema_val = combined[close_col].ewm(span=p, adjust=False).mean()
        combined[above_col] = combined[close_col] > ema_val

# ===========================
# Visualização em Abas
# ===========================
tab1, tab2 = st.tabs(["📊 Gráficos", "📑 Resumo de Mercado"])

with tab1:
    # --- MODELO ORIGINAL: CONTAGEM SIMPLES ---
    st.header("Modelo Original: Contagem Simples de Ativos")
    st.markdown("Cada ativo tem o mesmo peso (valor = 1). O eixo Y mostra o número de ativos acima da média.")

    counts = {}
    for p in MA_PERIODS:
        filtered_cols = combined.filter(like=f"_Above_EMA{p}")
        counts[p] = filtered_cols.sum(axis=1).tail(NUM_CANDLES_DISPLAY)

    for p in MA_PERIODS:
        series = counts[p]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", fill="tozeroy", name=f'EMA {p}'))
        fig.update_layout(
            title=f'Contagem de Ativos Acima da EMA {p}',
            yaxis_title="Nº de Ativos",
            height=350,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- IMPLEMENTAÇÃO 1: AMPLITUDE PONDERADA ---
    st.header("Implementação 1: Amplitude Ponderada pela Liquidez")
    st.markdown("Cada ativo contribui com o seu peso. O eixo Y vai de 0 a 100, representando a força total do sentimento do mercado.")

    weighted_counts = {}
    for p in MA_PERIODS:
        total_weight_series = pd.Series(0, index=combined.index)
        for s in ASSETS:
            above_col = f"{s}_Above_EMA{p}"
            if above_col in combined.columns:
                weight = ASSET_WEIGHTS.get(s, 0)
                total_weight_series += combined[above_col].astype(int) * weight
        weighted_counts[p] = total_weight_series.tail(NUM_CANDLES_DISPLAY)

    for p in MA_PERIODS:
        series = weighted_counts[p]
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", fill="tozeroy", name=f'EMA {p} Ponderada'))
        fig_w.add_hline(y=10, line_dash="dot", line_color="green", annotation_text="Extremo Venda (10%)")
        fig_w.add_hline(y=90, line_dash="dot", line_color="red", annotation_text="Extremo Compra (90%)")
        fig_w.update_layout(
            title=f'Força Ponderada Acima da EMA {p}',
            yaxis=dict(title="Força Ponderada (0-100)", range=[0, 100]),
            height=350,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_w, use_container_width=True)


with tab2:
    st.header("Resumo Atual")
    
    # --- Resumo Original ---
    st.subheader("Contagem Simples")
    latest_counts_simple = {f"EMA {p}": int(counts[p].iloc[-1]) if not counts[p].empty else 0 for p in MA_PERIODS}
    st.metric("Total de Ativos na Cesta", f"{len(ASSETS)}")
    st.table(pd.DataFrame.from_dict(latest_counts_simple, orient="index", columns=["Nº de Ativos Acima"]))

    # --- Resumo Ponderado ---
    st.subheader("Força Ponderada")
    latest_counts_weighted = {f"EMA {p}": f"{int(weighted_counts[p].iloc[-1])}%" if not weighted_counts[p].empty else "0%" for p in MA_PERIODS}
    st.metric("Força Máxima Possível", "100%")
    st.table(pd.DataFrame.from_dict(latest_counts_weighted, orient="index", columns=["Força Atual"]))

st.caption("Feito com Streamlit • Dados via FinancialModelingPrep")

