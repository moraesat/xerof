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

# --- CESTAS DE ATIVOS ---
# Ativos onde a SUBIDA representa fraqueza do Dólar (Pesos ajustados para somar 100)
ANTI_DOLLAR_ASSETS = {
    'EURUSD': 45, 'GBPUSD': 19, 'AUDUSD': 11, 'NZDUSD': 4, 'XAUUSD': 15, 'XAGUSD': 6
}
# Ativos onde a SUBIDA representa força do Dólar (Pesos ajustados para somar 100)
PRO_DOLLAR_ASSETS = {
    'USDJPY': 50, 'USDCHF': 15, 'USDCAD': 20, 'USDCNH': 15
}
ALL_ASSETS_COMBINED = {**ANTI_DOLLAR_ASSETS, **PRO_DOLLAR_ASSETS}

# Atualização automática
st_autorefresh(interval=60 * 1000, key="refresh")

# =============================================================================
# MENU LATERAL (SIDEBAR)
# =============================================================================
st.sidebar.title("Configurações do Painel")

ASSET_BASKET_CHOICE = st.sidebar.selectbox(
    "Selecionar Cesta de Ativos",
    ["Visão Combinada (Anti-Dólar)", "Força Anti-Dólar (XXX/USD)", "Força Pró-Dólar (USD/XXX)"]
)

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
# LÓGICA DE SELEÇÃO DE ATIVOS
# =============================================================================
if ASSET_BASKET_CHOICE == "Força Anti-Dólar (XXX/USD)":
    selected_assets = ANTI_DOLLAR_ASSETS
    dashboard_title = "🌊 Painel de Controlo: Força Anti-Dólar (Cesta XXX/USD)"
    is_combined_view = False
elif ASSET_BASKET_CHOICE == "Força Pró-Dólar (USD/XXX)":
    selected_assets = PRO_DOLLAR_ASSETS
    dashboard_title = "🌊 Painel de Controlo: Força Pró-Dólar (Cesta USD/XXX)"
    is_combined_view = False
else: # Visão Combinada
    selected_assets = ALL_ASSETS_COMBINED
    dashboard_title = "🌊 Painel de Controlo: Amplitude Combinada (Visão Anti-Dólar)"
    is_combined_view = True

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
        # CORREÇÃO DE TIMEZONE: Localiza os dados como 'US/Eastern' (fuso de NY) e converte para São Paulo.
        df['date'] = df['date'].dt.tz_localize('US/Eastern').dt.tz_convert(TZ)
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
st.title(dashboard_title)
last_update_time = datetime.now(TZ).strftime('%H:%M:%S')
st.caption(f"Analisando {len(selected_assets)} pares no timeframe de {TIMEFRAME} | Última atualização: {last_update_time} (Horário de Brasília)")

all_data = fetch_all_data_parallel(list(selected_assets.keys()), TIMEFRAME)

if not all_data:
    st.error("Não foi possível obter dados de mercado. A API pode estar indisponível. Tente novamente.")
    st.stop()

# --- Cálculos dos Painéis ---
breadth_components = []
aggression_buyer_components = []
aggression_seller_components = []
returns_components = []

for symbol, df in all_data.items():
    weight = selected_assets.get(symbol, 0)
    
    # --- Lógica de Força ---
    if is_combined_view:
        if symbol in ANTI_DOLLAR_ASSETS:
            strength_condition = (df['close'] > df['open'])
        else: # PRO_DOLLAR_ASSETS
            strength_condition = (df['close'] < df['open']) # Invertido para visão anti-dólar
    else: # Visão de Cesta Simples
        strength_condition = (df['close'] > df['open'])
        
    breadth_components.append(strength_condition.astype(int).rename(symbol) * weight)
    
    # --- Lógica de Agressão ---
    df['atr'] = calculate_atr(df, ATR_PERIOD)
    df['energy'] = (df['high'] - df['low']) / df['atr']
    is_high_energy = df['energy'] > ENERGY_THRESHOLD
    
    buyer_aggression = (strength_condition & is_high_energy).astype(int) * weight
    seller_aggression = (~strength_condition & is_high_energy).astype(int) * weight
    aggression_buyer_components.append(buyer_aggression.rename(symbol))
    aggression_seller_components.append(seller_aggression.rename(symbol))

    # --- Lógica de Retornos (para Coesão) ---
    ret = df['close'].pct_change()
    if is_combined_view and symbol in PRO_DOLLAR_ASSETS:
        ret = -ret
    returns_components.append(ret)

# PAINEL 1: Força Central
breadth_weighted = pd.concat(breadth_components, axis=1).sum(axis=1)
breadth_mean = breadth_weighted.rolling(window=Z_SCORE_WINDOW).mean()
breadth_std = breadth_weighted.rolling(window=Z_SCORE_WINDOW).std()
breadth_zscore = (breadth_weighted - breadth_mean) / breadth_std

# PAINEL 2: Agressão e Velocidade
aggression_buyer = pd.concat(aggression_buyer_components, axis=1).sum(axis=1)
aggression_seller = pd.concat(aggression_seller_components, axis=1).sum(axis=1)
breadth_roc = breadth_weighted.diff()

# PAINEL 3: Ambiente de Risco (RORO)
returns_df = pd.concat(returns_components, axis=1)
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
    fig1.update_layout(title='Força Ponderada da Cesta', height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
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
    fig2.add_trace(go.Bar(x=aggression_buyer_display.index, y=aggression_buyer_display.values, name='Agressão Compradora', marker_color='green'))
    fig2.add_trace(go.Bar(x=aggression_seller_display.index, y=aggression_seller_display.values, name='Agressão Vendedora', marker_color='red'))
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


