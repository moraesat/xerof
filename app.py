import pandas as pd
import pytz
import requests
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ===========================
# Configurações principais
# ===========================
st.set_page_config(page_title="Market Breadth Dashboard",
                   layout="wide", page_icon="⚔️")
API_KEY = "3CImfjoxNd98om3uhS89X4lmlp4Mrp3H"
TZ = pytz.timezone("America/Sao_Paulo")

# --- CESTAS DE ATIVOS E PESOS ---
# Cesta Risk-Off: Mede a força do Dólar (USD/XXX). Pesos somam 100.
RISK_OFF_ASSETS = {
    'DX-Y.NYB': 20, 'USDJPY': 20, 'USDCHF': 10, 'USDCAD': 10, 'USDCNH': 10,
    'USDSEK': 5,  'USDNOK': 5,  'USDMXN': 5,  'USDSGD': 5,  'USDZAR': 2,
    'USDHKD': 2,  'USDPLN': 2,  'USDCZK': 1,  'USDDKK': 1,  'USDHUF': 1
}

# Cesta Risk-On: Mede a fraqueza do Dólar (XXX/USD). Pesos somam 100.
RISK_ON_ASSETS = {
    'EURUSD': 27, 'JPYUSD': 14, 'GBPUSD': 11, 'AUDUSD': 7, 'CADUSD': 6,
    'CNYUSD': 4,  'CHFUSD': 4,  'XAUUSD': 17, 'XAGUSD': 6, 'NZDUSD': 2,
    'HKDUSD': 1,  'KRWUSD': 1
}

ALL_UNIQUE_ASSETS = list(set(RISK_OFF_ASSETS.keys()) | set(RISK_ON_ASSETS.keys()))
NUM_CANDLES_DISPLAY = 120

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

st.sidebar.header("Parâmetros dos Indicadores")
Z_SCORE_WINDOW = st.sidebar.slider("Janela Z-Score (Amplitude)", 50, 500, 200)
ATR_PERIOD = st.sidebar.slider("Período do ATR", 10, 30, 14)
ENERGY_THRESHOLD = st.sidebar.slider("Limiar de 'Energia'", 1.0, 3.0, 1.5, 0.1)
CLIMAX_Z_WINDOW = st.sidebar.slider("Janela Z-Score (Clímax)", 50, 200, 100)
MOMENTUM_PERIOD = st.sidebar.slider("Período ROC (Momentum)", 10, 50, 21)
MOMENTUM_Z_WINDOW = st.sidebar.slider("Janela Z-Score (Momentum)", 50, 200, 100)

# ===========================
# Funções de Cálculo e Busca
# ===========================

@st.cache_data(ttl=60)
def get_single_data(symbol: str, timeframe: str, candles_to_fetch: int) -> pd.DataFrame | None:
    try:
        base_url = f"https://financialmodelingprep.com/api/v3/historical-chart/{timeframe}/{symbol}"
        r = requests.get(base_url, params={"apikey": API_KEY}, timeout=10)
        if r.status_code != 200: return None
        df = pd.DataFrame(r.json()).iloc[::-1]
        df["date"] = pd.to_datetime(df["date"])
        df['date'] = df['date'].dt.tz_localize('US/Eastern').dt.tz_convert(TZ)
        df = df.set_index("date")
        df = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        return df.tail(candles_to_fetch)
    except Exception:
        return None

def build_combined_data(symbols: list, timeframe: str, candles_to_fetch: int) -> pd.DataFrame:
    with st.spinner(f"A buscar dados para {len(symbols)} ativos..."):
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_symbol = {executor.submit(get_single_data, s, timeframe, candles_to_fetch): s for s in symbols}
            frames = [future.result().rename(columns=lambda c: f"{future_to_symbol[future]}_{c}") for future in as_completed(future_to_symbol) if future.result() is not None]
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1).ffill().dropna()

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)
    return tr.rolling(period).mean()

def calculate_zscore(series: pd.Series, window: int) -> pd.Series:
    return (series - series.rolling(window=window).mean()) / series.rolling(window=window).std()

def calculate_breadth_metrics(asset_weights: dict, combined_data: pd.DataFrame, is_risk_on: bool):
    """Calcula todas as métricas de amplitude para uma cesta de ativos."""
    metrics = {}
    
    # --- Cálculos Base ---
    weighted_counts = {p: pd.Series(0.0, index=combined_data.index) for p in MA_PERIODS}
    aggression_buyer = pd.Series(0.0, index=combined_data.index)
    aggression_seller = pd.Series(0.0, index=combined_data.index)
    momentum_components = []

    for s, weight in asset_weights.items():
        close_col, open_col, high_col, low_col = f"{s}_close", f"{s}_open", f"{s}_high", f"{s}_low"
        if close_col not in combined_data.columns: continue

        strength_condition = (combined_data[close_col] > combined_data[open_col]) if is_risk_on else (combined_data[close_col] < combined_data[open_col])

        # Agressão
        atr = calculate_atr(combined_data[high_col], combined_data[low_col], combined_data[close_col], ATR_PERIOD)
        is_high_energy = (combined_data[high_col] - combined_data[low_col]) / atr > ENERGY_THRESHOLD
        aggression_buyer += (strength_condition & is_high_energy).astype(int) * weight
        aggression_seller += (~strength_condition & is_high_energy).astype(int) * weight
        
        # Amplitude Ponderada
        for p in MA_PERIODS:
            ema_val = combined_data[close_col].ewm(span=p, adjust=False).mean()
            above_ema = (combined_data[close_col] > ema_val) if is_risk_on else (combined_data[close_col] < ema_val)
            weighted_counts[p] += above_ema.astype(int) * weight

        # Momentum
        roc = combined_data[close_col].pct_change(periods=MOMENTUM_PERIOD)
        if not is_risk_on: roc = -roc # Inverte para alinhar
        normalized_momentum = calculate_zscore(roc, MOMENTUM_Z_WINDOW)
        momentum_components.append(normalized_momentum * weight)

    metrics['weighted_counts'] = weighted_counts
    metrics['aggression_buyer'] = aggression_buyer
    metrics['aggression_seller'] = aggression_seller
    metrics['buyer_climax_zscore'] = calculate_zscore(aggression_buyer, CLIMAX_Z_WINDOW)
    metrics['seller_climax_zscore'] = calculate_zscore(aggression_seller, CLIMAX_Z_WINDOW)
    metrics['aggregate_momentum_index'] = pd.concat(momentum_components, axis=1).sum(axis=1)
    
    # --- Cálculos Derivados ---
    metrics['z_scores'], metrics['rocs'], metrics['accelerations'] = {}, {}, {}
    for p in MA_PERIODS:
        series = weighted_counts[p]
        metrics['z_scores'][p] = calculate_zscore(series, Z_SCORE_WINDOW)
        metrics['rocs'][p] = series.diff()
        metrics['accelerations'][p] = metrics['rocs'][p].diff()

    return metrics

def display_charts(column, metrics, title_prefix):
    """Exibe todos os gráficos para uma cesta de métricas em uma coluna do Streamlit."""
    column.header(title_prefix)

    # Gráfico 1: Amplitude Ponderada
    for p, series in metrics['weighted_counts'].items():
        fig = go.Figure(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, mode="lines", fill="tozeroy"))
        fig.update_layout(title=f'Força Ponderada (EMA {p})', yaxis=dict(range=[0, 100]), height=250, margin=dict(t=30, b=10, l=10, r=10))
        column.plotly_chart(fig, use_container_width=True)
    
    # Gráfico 2: Z-Score da Amplitude
    for p, series in metrics['z_scores'].items():
        fig = go.Figure(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, line=dict(color='orange')))
        fig.add_hline(y=2, line_dash="dot", line_color="red"); fig.add_hline(y=-2, line_dash="dot", line_color="green")
        fig.update_layout(title=f'Nível de Extremo (Z-Score EMA {p})', yaxis=dict(range=[-3.5, 3.5]), height=250, margin=dict(t=30, b=10, l=10, r=10))
        column.plotly_chart(fig, use_container_width=True)

    # Gráfico 3: Indicador de Clímax
    buyer_series = metrics['buyer_climax_zscore'].tail(NUM_CANDLES_DISPLAY).clip(lower=0)
    seller_series = metrics['seller_climax_zscore'].tail(NUM_CANDLES_DISPLAY).clip(lower=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=buyer_series.index, y=buyer_series.values, name='Clímax Comprador', marker_color='green'))
    fig.add_trace(go.Bar(x=seller_series.index, y=seller_series.values, name='Clímax Vendedor', marker_color='red'))
    fig.add_hline(y=2, line_dash="dot", line_color="black")
    fig.update_layout(barmode='relative', title='Indicador de Clímax de Agressão', height=250, margin=dict(t=30, b=10, l=10, r=10))
    column.plotly_chart(fig, use_container_width=True)

    # Gráfico 4: Índice de Momentum
    series = metrics['aggregate_momentum_index'].tail(NUM_CANDLES_DISPLAY)
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, line=dict(color='#636EFA'), fill='tozeroy'))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(title='Índice de Momentum Agregado', height=250, margin=dict(t=30, b=10, l=10, r=10))
    column.plotly_chart(fig, use_container_width=True)


# ===========================
# Lógica Principal da Aplicação
# ===========================
st.title("⚔️ Painel de Batalha: Risk-Off vs. Risk-On")
st.caption(f"Timeframe: {TIMEFRAME} | Última atualização: {datetime.now(TZ).strftime('%H:%M:%S')}")

candles_to_fetch = (max(MA_PERIODS) if MA_PERIODS else 200) + NUM_CANDLES_DISPLAY + max(Z_SCORE_WINDOW, MOMENTUM_Z_WINDOW, CLIMAX_Z_WINDOW)
combined = build_combined_data(ALL_UNIQUE_ASSETS, TIMEFRAME, candles_to_fetch)

if combined.empty:
    st.error("Nenhum dado disponível. Verifique a API ou os símbolos dos ativos.")
    st.stop()

# --- Calcular métricas para ambas as cestas ---
metrics_risk_off = calculate_breadth_metrics(RISK_OFF_ASSETS, combined, is_risk_on=False)
metrics_risk_on = calculate_breadth_metrics(RISK_ON_ASSETS, combined, is_risk_on=True)

# --- Visualização ---
tab1, tab2 = st.tabs(["📊 Gráficos Comparativos", "📑 Resumo de Mercado"])

with tab1:
    col1, col2 = st.columns(2)
    display_charts(col1, metrics_risk_off, "Risk-Off (Força do Dólar)")
    display_charts(col2, metrics_risk_on, "Risk-On (Fraqueza do Dólar)")

with tab2:
    st.header("Resumo Numérico Atual")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk-Off (Força do Dólar)")
        latest_weighted = {f"EMA {p}": f"{int(metrics_risk_off['weighted_counts'][p].iloc[-1])}%" for p in MA_PERIODS}
        st.table(pd.DataFrame.from_dict(latest_weighted, orient='index', columns=["Força Atual"]))
        latest_climax = {"Clímax Comprador": f"{metrics_risk_off['buyer_climax_zscore'].iloc[-1]:.2f} σ", "Clímax Vendedor": f"{metrics_risk_off['seller_climax_zscore'].iloc[-1]:.2f} σ"}
        st.table(pd.DataFrame.from_dict(latest_climax, orient='index', columns=["Nível Atual"]))
        latest_momentum = {"Momentum Agregado": f"{metrics_risk_on['aggregate_momentum_index'].iloc[-1]:.2f}"}
        st.table(pd.DataFrame.from_dict(latest_momentum, orient='index', columns=['Valor Atual']))

    with col2:
        st.subheader("Risk-On (Fraqueza do Dólar)")
        latest_weighted = {f"EMA {p}": f"{int(metrics_risk_on['weighted_counts'][p].iloc[-1])}%" for p in MA_PERIODS}
        st.table(pd.DataFrame.from_dict(latest_weighted, orient='index', columns=["Força Atual"]))
        latest_climax = {"Clímax Comprador": f"{metrics_risk_on['buyer_climax_zscore'].iloc[-1]:.2f} σ", "Clímax Vendedor": f"{metrics_risk_on['seller_climax_zscore'].iloc[-1]:.2f} σ"}
        st.table(pd.DataFrame.from_dict(latest_climax, orient='index', columns=["Nível Atual"]))
        latest_momentum = {"Momentum Agregado": f"{metrics_risk_on['aggregate_momentum_index'].iloc[-1]:.2f}"}
        st.table(pd.DataFrame.from_dict(latest_momentum, orient='index', columns=['Valor Atual']))

st.caption("Feito com Streamlit • Dados via FinancialModelingPrep")

