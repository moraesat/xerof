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
    page_title="Intraday Forex Breadth Dashboard",
    layout="wide",
    page_icon="🌊"
)

API_KEY = "3CImfjoxNd98om3uhS89X4lmlp4Mrp3H" 
TZ = pytz.timezone("America/Sao_Paulo")

# --- LISTA DE PARES DE FOREX E SEUS PESOS DE LIQUIDEZ (Baseado no volume de negociação global) ---
# Fonte: BIS Triennial Central Bank Survey
FOREX_PAIRS_WEIGHTS = {
    'EURUSD': 24, 'USDJPY': 13, 'GBPUSD': 10, 'AUDUSD': 6, 'USDCAD': 5,
    'USDCNH': 4, 'USDCHF': 4, 'NZDUSD': 2, 'USDSEK': 1, 'USDNOK': 1,
    'USDSGD': 1, 'USDMXN': 1, 'USDZAR': 1, 'EURJPY': 3, 'EURGBP': 2,
    'AUDJPY': 2, 'CADJPY': 1, 'CHFJPY': 1
}
TOTAL_WEIGHT = sum(FOREX_PAIRS_WEIGHTS.values())

# Atualização automática
st_autorefresh(interval=60 * 1000, key="refresh") # Atualiza a cada 60 segundos

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

st.sidebar.header("Parâmetros dos Indicadores")
Z_SCORE_WINDOW = st.sidebar.slider("Janela do Z-Score", 50, 500, 200)
ATR_PERIOD = st.sidebar.slider("Período do ATR", 10, 30, 14)
ENERGY_THRESHOLD = st.sidebar.slider("Limiar de 'Energia' da Vela", 1.0, 3.0, 1.5, 0.1)
EMA_DIST_PERIOD = st.sidebar.slider("Período da EMA (Distribuição)", 50, 200, 200)

# =============================================================================
# FUNÇÕES DE BUSCA E PROCESSAMENTO DE DADOS
# =============================================================================

@st.cache_data(ttl=60)
def get_single_pair_data(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Busca dados OHLCV para um único par de moedas."""
    candles_to_fetch = 500 # Puxar um histórico maior para cálculos de Z-score e EMA
    try:
        base_url = f"https://financialmodelingprep.com/api/v3/historical-chart/{timeframe}/{symbol}"
        r = requests.get(base_url, params={"apikey": API_KEY}, timeout=15)
        if r.status_code != 200: return None
        data = r.json()
        if not data: return None

        df = pd.DataFrame(data).iloc[::-1] # API retorna do mais recente para o mais antigo
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.tz_localize('UTC').dt.tz_convert(TZ)
        df = df.set_index('date')
        df = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        return df.tail(candles_to_fetch)
    except Exception:
        return None

def fetch_all_data_parallel(symbols: list, timeframe: str) -> dict:
    """Busca dados para todos os símbolos em paralelo para acelerar o carregamento."""
    data = {}
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
st.title("🌊 Painel de Controlo de Amplitude Intraday para Forex")
st.caption(f"Analisando {len(FOREX_PAIRS_WEIGHTS)} pares no timeframe de {TIMEFRAME} | Última atualização: {datetime.now(TZ).strftime('%H:%M:%S')}")

# --- Busca e Preparação dos Dados ---
with st.spinner("A buscar e a processar dados de mercado em tempo real..."):
    all_data = fetch_all_data_parallel(list(FOREX_PAIRS_WEIGHTS.keys()), TIMEFRAME)

    if not all_data:
        st.error("Não foi possível obter dados de mercado. A API pode estar indisponível. Tente novamente em breve.")
        st.stop()

    # --- Cálculos dos Painéis ---

    # PAINEL 1: Força Central
    breadth_weighted_values = []
    for symbol, df in all_data.items():
        weight = FOREX_PAIRS_WEIGHTS.get(symbol, 0)
        condition = (df['close'] > df['open']).astype(int) * weight
        breadth_weighted_values.append(condition.rename(symbol))
    
    breadth_weighted = pd.concat(breadth_weighted_values, axis=1).sum(axis=1)
    
    # Z-Score
    breadth_mean = breadth_weighted.rolling(window=Z_SCORE_WINDOW).mean()
    breadth_std = breadth_weighted.rolling(window=Z_SCORE_WINDOW).std()
    breadth_zscore = (breadth_weighted - breadth_mean) / breadth_std

    # PAINEL 2: Agressão e Velocidade
    aggression_buyer = []
    aggression_seller = []
    for symbol, df in all_data.items():
        weight = FOREX_PAIRS_WEIGHTS.get(symbol, 0)
        df['atr'] = calculate_atr(df, ATR_PERIOD)
        df['energy'] = (df['high'] - df['low']) / df['atr']
        
        is_buyer_aggression = ((df['close'] > df['open']) & (df['energy'] > ENERGY_THRESHOLD)).astype(int) * weight
        is_seller_aggression = ((df['close'] < df['open']) & (df['energy'] > ENERGY_THRESHOLD)).astype(int) * weight
        
        aggression_buyer.append(is_buyer_aggression.rename(symbol))
        aggression_seller.append(is_seller_aggression.rename(symbol))
        
    aggression_buyer_weighted = pd.concat(aggression_buyer, axis=1).sum(axis=1)
    aggression_seller_weighted = pd.concat(aggression_seller, axis=1).sum(axis=1)
    
    # ROC
    breadth_roc = breadth_weighted.diff()

    # PAINEL 3: Ambiente de Risco (RORO)
    returns = pd.concat([df['close'].pct_change() for df in all_data.values()], axis=1)
    dispersion = returns.std(axis=1)
    cohesion_index = (1 / dispersion).rolling(window=20).mean() # Inverso da dispersão = coesão
    
    # Distribuição
    dist_from_ema = []
    for symbol, df in all_data.items():
        ema = df['close'].ewm(span=EMA_DIST_PERIOD, adjust=False).mean()
        dist = ((df['close'] - ema) / ema) * 100
        dist_from_ema.append(dist.iloc[-1])


# --- Visualização dos Painéis ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Motor: Força Central")
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=breadth_weighted.index, y=breadth_weighted.values, name='Amplitude Ponderada', line=dict(color='royalblue', width=2)))
    fig1.update_layout(title='Amplitude Ponderada pela Liquidez ($BREADTH_W)', yaxis_title='Força Ponderada', height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig1, use_container_width=True)
    
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=breadth_zscore.index, y=breadth_zscore.values, name='Z-Score', line=dict(color='orange')))
    fig_z.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Extremo de Euforia")
    fig_z.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="Extremo de Pânico")
    fig_z.update_layout(title='Z-Score do $BREADTH_W (Nível de Extremo)', yaxis_title='Desvios Padrão', height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_z, use_container_width=True)

with col2:
    st.header("Tacómetro: Agressão")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=aggression_buyer_weighted.index, y=aggression_buyer_weighted.values, name='Agressão Compradora', marker_color='green'))
    fig2.add_trace(go.Bar(x=aggression_seller_weighted.index, y=aggression_seller_weighted.values, name='Agressão Vendedora', marker_color='red'))
    fig2.update_layout(barmode='stack', title='Clímax de Agressão Ponderado (Energia)', yaxis_title='Força da Agressão', height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=breadth_roc.index, y=breadth_roc, name='ROC', line=dict(color='purple'), fill='tozeroy'))
    fig_roc.add_hline(y=0, line_dash="dash", line_color="grey")
    fig_roc.update_layout(title='Velocidade da Amplitude (ROC)', yaxis_title='Variação', height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_roc, use_container_width=True)
    
with col3:
    st.header("Radar: Ambiente RORO")
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=cohesion_index.index, y=cohesion_index.values, name='Coesão', line=dict(color='teal')))
    fig3.update_layout(title='Índice de Coesão (Medo Risk-Off)', yaxis_title='Coesão (1 / Dispersão)', height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=dist_from_ema, name='Distribuição', marker_color='darkgrey'))
    fig_dist.add_vline(x=0, line_dash="dash", line_color="blue", annotation_text=f"EMA {EMA_DIST_PERIOD}")
    fig_dist.update_layout(title=f'Perfil de Distribuição vs EMA {EMA_DIST_PERIOD}', xaxis_title='% de Distância da EMA', height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_dist, use_container_width=True)

# --- Guia de Interpretação ---
st.header("Guia Rápido de Interpretação")
st.markdown("""
- **Sinal de Reversão para Compra (Fundo):** Procure por um **Z-Score < -2** (Pânico) acompanhado de um **pico na Agressão Vendedora** (Clímax) e um **pico no Índice de Coesão** (Medo RORO). O gatilho de timing ocorre quando a Agressão Vendedora desaparece e o ROC começa a subir.
- **Sinal de Reversão para Venda (Topo):** Procure por um **Z-Score > +2** (Euforia) com um **pico na Agressão Compradora**. A fraqueza é confirmada quando o **ROC começa a cair** mesmo com a amplitude ainda alta (divergência de velocidade).
- **Análise de Contexto:** O **Perfil de Distribuição** mostra a saúde da tendência. Uma distribuição deslocada para a direita confirma a força geral, enquanto um recuo da mediana pode sinalizar fraqueza interna.
""")

