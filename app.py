import pandas as pd
import pytz
import requests
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# =============================================================================
# CONFIGURAÇÕES GLOBAIS E CONSTANTES
# =============================================================================
st.set_page_config(
    page_title="Intraday Forex Breadth",
    layout="wide",
    page_icon="🌊"
)

API_KEY = "3CImfjoxNd98om3uhS89X4lmlp4Mrp3H" 
TZ = pytz.timezone("America/Sao_Paulo")

# --- CESTA DE ATIVOS FOCADA: FORÇA ANTI-DÓLAR ---
# A lógica agora mede a força coordenada de vários ativos CONTRA o Dólar (USD).
# Para pares USD/XXX, uma queda (close < open) significa força anti-dólar.
# Para pares XXX/USD, uma subida (close > open) significa força anti-dólar.

# Ativos onde a SUBIDA representa fraqueza do Dólar
ANTI_DOLLAR_ASSETS = {
    'EURUSD': 24, 'GBPUSD': 10, 'AUDUSD': 6, 'NZDUSD': 2, 'XAUUSD': 8, 'XAGUSD': 3
}
# Ativos onde a QUEDA representa fraqueza do Dólar
PRO_DOLLAR_ASSETS = {
    'USDJPY': 13, 'USDCHF': 4, 'USDCAD': 5, 'USDCNH': 4
}
ALL_ASSETS = {**ANTI_DOLLAR_ASSETS, **PRO_DOLLAR_ASSETS}
TOTAL_WEIGHT = sum(ALL_ASSETS.values())

# Atualização automática
st_autorefresh(interval=60 * 1000, key="refresh")

# =============================================================================
# MENU LATERAL (SIDEBAR)
# =============================================================================
st.sidebar.title("Configurações do Painel")
TIMEFRAME = st.sidebar.radio(
    "Timeframe de Análise",
    ["5min", "15min"],
    index=0,
    captions=["Alta Frequência", "Média Frequência"]
)

CANDLES_TO_DISPLAY = st.sidebar.slider(
    "Histórico no Gráfico (Nº de Velas)", 
    min_value=50, max_value=500, value=150,
    help="Define quantas velas recentes serão exibidas nos gráficos. Menos velas para focar na ação mais recente."
)

st.sidebar.header("Parâmetros dos Indicadores")
Z_SCORE_WINDOW = st.sidebar.slider("Janela do Z-Score", 50, 500, 200)
ATR_PERIOD = st.sidebar.slider("Período do ATR", 10, 30, 14)
ENERGY_THRESHOLD = st.sidebar.slider("Limiar de 'Energia' da Vela", 1.0, 3.0, 1.5, 0.1)

# =============================================================================
# FUNÇÕES DE BUSCA E PROCESSAMENTO DE DADOS
# =============================================================================

@st.cache_data(ttl=60)
def get_single_pair_data(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Busca dados OHLCV para um único par de moedas."""
    candles_to_fetch = 600 # Busca um histórico maior para a estabilidade dos cálculos
    try:
        base_url = f"https://financialmodelingprep.com/api/v3/historical-chart/{timeframe}/{symbol}"
        r = requests.get(base_url, params={"apikey": API_KEY}, timeout=15)
        if r.status_code != 200: return None
        data = r.json()
        if not data: return None

        df = pd.DataFrame(data).iloc[::-1]
        df['date'] = pd.to_datetime(df['date'])
        # CORREÇÃO DE TIMEZONE: Localiza os dados como UTC e converte para o fuso horário de São Paulo
        df['date'] = df['date'].dt.tz_localize('UTC').dt.tz_convert(TZ)
        df = df.set_index('date')
        df = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        return df.tail(candles_to_fetch)
    except Exception:
        return None

def fetch_all_data_parallel(symbols: list, timeframe: str) -> dict:
    """Busca dados para todos os símbolos em paralelo para acelerar o carregamento."""
    data = {}
    with st.spinner("A buscar e a processar dados de mercado..."):
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_symbol = {executor.submit(get_single_pair_data, symbol, timeframe): symbol for symbol in symbols}
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        data[symbol] = result
                except Exception:
                    pass
    return data

def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Calcula o Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

# =============================================================================
# ESTRUTURA PRINCIPAL DA APLICAÇÃO
# =============================================================================
st.title("🌊 Painel de Controlo de Amplitude Intraday: Força Anti-Dólar")
last_update_time = datetime.now(TZ).strftime('%H:%M:%S')
st.caption(f"Analisando {len(ALL_ASSETS)} pares no timeframe de {TIMEFRAME} | Última atualização: {last_update_time} (Horário de Brasília)")

all_data = fetch_all_data_parallel(list(ALL_ASSETS.keys()), TIMEFRAME)

if not all_data:
    st.error("Não foi possível obter dados de mercado. A API pode estar indisponível. Tente novamente.")
    st.stop()

# --- Cálculos dos Painéis ---
# PAINEL 1: Força Central (Amplitude Ponderada Anti-Dólar)
breadth_components = []
for symbol, df in all_data.items():
    weight = ALL_ASSETS.get(symbol, 0)
    if symbol in ANTI_DOLLAR_ASSETS:
        # Para EURUSD, etc., subida é força anti-dólar
        condition = (df['close'] > df['open']).astype(int) * weight
    else: # PRO_DOLLAR_ASSETS
        # Para USDJPY, etc., queda é força anti-dólar
        condition = (df['close'] < df['open']).astype(int) * weight
    breadth_components.append(condition.rename(symbol))
    
breadth_weighted = pd.concat(breadth_components, axis=1).sum(axis=1)

# Z-Score
breadth_mean = breadth_weighted.rolling(window=Z_SCORE_WINDOW).mean()
breadth_std = breadth_weighted.rolling(window=Z_SCORE_WINDOW).std()
breadth_zscore = (breadth_weighted - breadth_mean) / breadth_std

# PAINEL 2: Agressão e Velocidade
aggression_buyer_anti_usd = []
aggression_seller_anti_usd = []
for symbol, df in all_data.items():
    weight = ALL_ASSETS.get(symbol, 0)
    df['atr'] = calculate_atr(df, ATR_PERIOD)
    df['energy'] = (df['high'] - df['low']) / df['atr']
    is_high_energy = df['energy'] > ENERGY_THRESHOLD

    if symbol in ANTI_DOLLAR_ASSETS:
        buyer_agg = ((df['close'] > df['open']) & is_high_energy).astype(int) * weight
        seller_agg = ((df['close'] < df['open']) & is_high_energy).astype(int) * weight
    else: # PRO_DOLLAR_ASSETS
        buyer_agg = ((df['close'] < df['open']) & is_high_energy).astype(int) * weight
        seller_agg = ((df['close'] > df['open']) & is_high_energy).astype(int) * weight
        
    aggression_buyer_anti_usd.append(buyer_agg.rename(symbol))
    aggression_seller_anti_usd.append(seller_agg.rename(symbol))
    
aggression_buyer = pd.concat(aggression_buyer_anti_usd, axis=1).sum(axis=1)
aggression_seller = pd.concat(aggression_seller_anti_usd, axis=1).sum(axis=1)
    
# ROC
breadth_roc = breadth_weighted.diff()

# PAINEL 3: Ambiente de Risco (RORO)
returns = []
for symbol, df in all_data.items():
    ret = df['close'].pct_change()
    if symbol in PRO_DOLLAR_ASSETS:
        ret = -ret # Inverte o retorno para alinhar com a ótica anti-dólar
    returns.append(ret)
    
returns_df = pd.concat(returns, axis=1)
dispersion = returns_df.std(axis=1)
cohesion_index = (1 / dispersion).rolling(window=20).mean()

# --- Fatiar dados para exibição ---
breadth_weighted_display = breadth_weighted.tail(CANDLES_TO_DISPLAY)
breadth_zscore_display = breadth_zscore.tail(CANDLES_TO_DISPLAY)
aggression_buyer_display = aggression_buyer.tail(CANDLES_TO_DISPLAY)
aggression_seller_display = aggression_seller.tail(CANDLES_TO_DISPLAY)
breadth_roc_display = breadth_roc.tail(CANDLES_TO_DISPLAY)
cohesion_index_display = cohesion_index.tail(CANDLES_TO_DISPLAY)

# --- Visualização dos Painéis ---
st.header("Painel de Controlo de Amplitude")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Motor: Força Central")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=breadth_weighted_display.index, y=breadth_weighted_display.values, name='Amplitude Ponderada', line=dict(color='royalblue', width=2)))
    fig1.update_layout(title='Força Ponderada Anti-Dólar', height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=breadth_zscore_display.index, y=breadth_zscore_display.values, name='Z-Score', line=dict(color='orange')))
    fig_z.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Extremo Euforia")
    fig_z.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="Extremo Pânico")
    fig_z.update_layout(title='Nível de Extremo (Z-Score)', yaxis_title='Desvios Padrão', height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    st.plotly_chart(fig_z, use_container_width=True)

with col2:
    st.subheader("Tacómetro: Agressão")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=aggression_buyer_display.index, y=aggression_buyer_display.values, name='Agressão Anti-Dólar', marker_color='green'))
    fig2.add_trace(go.Bar(x=aggression_seller_display.index, y=aggression_seller_display.values, name='Agressão Pró-Dólar', marker_color='red'))
    fig2.update_layout(barmode='relative', title='Clímax de Agressão Ponderado', height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=breadth_roc_display.index, y=breadth_roc_display.values, name='ROC', line=dict(color='purple'), fill='tozeroy'))
    fig_roc.add_hline(y=0, line_dash="dash", line_color="grey")
    fig_roc.update_layout(title='Velocidade da Amplitude (ROC)', height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    st.plotly_chart(fig_roc, use_container_width=True)
    
with col3:
    st.subheader("Radar: Ambiente RORO")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=cohesion_index_display.index, y=cohesion_index_display.values, name='Coesão', line=dict(color='teal')))
    fig3.update_layout(title='Índice de Coesão (Medo Risk-Off)', height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

