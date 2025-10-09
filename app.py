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

# Cesta Risk-On: Mede a fraqueza do Dólar (XXX/USD). Ativos de baixo volume removidos. Pesos somam 100.
RISK_ON_ASSETS = {
    'EURUSD': 38, 'GBPUSD': 16, 'AUDUSD': 10, 'XAUUSD': 24, 'XAGUSD': 9, 'NZDUSD': 3
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
CONVICTION_THRESHOLD = st.sidebar.slider("Filtro de Convicção (ATR)", 0.0, 1.0, 0.2, 0.05, help="Distância mínima (em ATRs) da média para um ativo ser contado. Filtra o 'samba'.")
Z_SCORE_WINDOW = st.sidebar.slider("Janela Z-Score (Amplitude)", 50, 500, 200)
ATR_PERIOD = st.sidebar.slider("Período do ATR", 10, 30, 14)
ENERGY_THRESHOLD = st.sidebar.slider("Limiar de 'Energia'", 1.0, 3.0, 1.5, 0.1)
CLIMAX_Z_WINDOW = st.sidebar.slider("Janela Z-Score (Clímax)", 50, 200, 100)
MOMENTUM_PERIOD = st.sidebar.slider("Período ROC (Momentum)", 10, 50, 21)
MOMENTUM_Z_WINDOW = st.sidebar.slider("Janela Z-Score (Momentum)", 50, 200, 100)
VOLUME_MA_PERIOD = st.sidebar.slider("Janela Média de Volume (VFI)", 10, 50, 20)


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

def calculate_breadth_metrics(asset_weights: dict, combined_data: pd.DataFrame):
    """Calcula todas as métricas de amplitude para uma cesta de ativos."""
    metrics = {}
    
    # --- Dicionários de armazenamento ---
    metrics['weighted_counts'] = {p: pd.Series(0.0, index=combined_data.index) for p in MA_PERIODS}
    metrics['qualified_counts'] = {p: pd.Series(0.0, index=combined_data.index) for p in MA_PERIODS}
    metrics['weighted_distance_indices'] = {p: pd.Series(0.0, index=combined_data.index) for p in MA_PERIODS}
    metrics['volume_force_indices'] = {p: pd.Series(0.0, index=combined_data.index) for p in MA_PERIODS} # NOVO
    aggression_buyer = pd.Series(0.0, index=combined_data.index)
    aggression_seller = pd.Series(0.0, index=combined_data.index)
    momentum_components = []

    for s, weight in asset_weights.items():
        close_col, open_col, high_col, low_col, vol_col = f"{s}_close", f"{s}_open", f"{s}_high", f"{s}_low", f"{s}_volume"
        if close_col not in combined_data.columns: continue

        strength_condition = (combined_data[close_col] > combined_data[open_col])
        atr = calculate_atr(combined_data[high_col], combined_data[low_col], combined_data[close_col], ATR_PERIOD)
        atr_safe = atr.replace(0, np.nan)
        
        is_high_energy = (combined_data[high_col] - combined_data[low_col]) / atr_safe > ENERGY_THRESHOLD
        aggression_buyer += (strength_condition & is_high_energy).astype(int) * weight
        aggression_seller += (~strength_condition & is_high_energy).astype(int) * weight
        
        volume_ma = combined_data[vol_col].rolling(window=VOLUME_MA_PERIOD).mean()
        volume_strength = (combined_data[vol_col] / volume_ma.replace(0, np.nan)).fillna(1)

        for p in MA_PERIODS:
            ema_val = combined_data[close_col].ewm(span=p, adjust=False).mean()
            
            above_ema = (combined_data[close_col] > ema_val)
            metrics['weighted_counts'][p] += above_ema.astype(int) * weight

            normalized_distance = ((combined_data[close_col] - ema_val) / atr_safe).fillna(0)
            is_significant_above = normalized_distance > CONVICTION_THRESHOLD
            metrics['qualified_counts'][p] += is_significant_above.astype(int) * weight
            metrics['weighted_distance_indices'][p] += normalized_distance * weight
            
            # Cálculo do VFI
            volume_force = normalized_distance * volume_strength
            metrics['volume_force_indices'][p] += volume_force * weight
        
        roc = combined_data[close_col].pct_change(periods=MOMENTUM_PERIOD)
        normalized_momentum = calculate_zscore(roc, MOMENTUM_Z_WINDOW)
        momentum_components.append(normalized_momentum * weight)

    metrics['aggression_buyer'] = aggression_buyer
    metrics['aggression_seller'] = aggression_seller
    metrics['buyer_climax_zscore'] = calculate_zscore(aggression_buyer, CLIMAX_Z_WINDOW)
    metrics['seller_climax_zscore'] = calculate_zscore(aggression_seller, CLIMAX_Z_WINDOW)
    metrics['aggregate_momentum_index'] = pd.concat(momentum_components, axis=1).sum(axis=1)
    
    # --- Cálculos Derivados ---
    metrics['z_scores'], metrics['rocs'], metrics['accelerations'] = {}, {}, {}
    metrics['conviction_zscore'] = {}
    metrics['qualified_zscore'] = {} 
    for p in MA_PERIODS:
        series_wc = metrics['weighted_counts'][p]
        metrics['z_scores'][p] = calculate_zscore(series_wc, Z_SCORE_WINDOW)
        metrics['rocs'][p] = series_wc.diff()
        metrics['accelerations'][p] = series_wc.diff().diff()
        
        conviction_index = (series_wc / 100) * metrics['weighted_distance_indices'][p]
        metrics['conviction_zscore'][p] = calculate_zscore(conviction_index, Z_SCORE_WINDOW)

        series_qc = metrics['qualified_counts'][p]
        metrics['qualified_zscore'][p] = calculate_zscore(series_qc, Z_SCORE_WINDOW) 

    return metrics

def display_charts(column, metrics, title_prefix, theme_colors):
    """Exibe todos os gráficos para uma cesta de métricas em uma coluna do Streamlit."""
    column.header(title_prefix)
    
    # Gráficos anteriores...
    for p, series in metrics['weighted_counts'].items():
        fig = go.Figure(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, mode="lines", fill="tozeroy", line_color=theme_colors['main'], opacity=0.7))
        fig.update_layout(title=f'Força Ponderada (Contagem Simples EMA {p})', yaxis=dict(range=[0, 100]), height=250, margin=dict(t=30, b=10, l=10, r=10), template="plotly_dark")
        column.plotly_chart(fig, use_container_width=True)

    for p, series in metrics['qualified_counts'].items():
        fig = go.Figure(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, mode="lines", fill="tozeroy", line_color=theme_colors['qualified']))
        fig.update_layout(title=f'Força Qualificada (Filtro EMA {p})', yaxis=dict(range=[0, 100]), height=250, margin=dict(t=30, b=10, l=10, r=10), template="plotly_dark")
        column.plotly_chart(fig, use_container_width=True)
        
    for p, series in metrics['qualified_zscore'].items():
        fig = go.Figure(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, line=dict(color=theme_colors['accent'])))
        fig.add_hline(y=2, line_dash="dot", line_color="white", opacity=0.5); fig.add_hline(y=-2, line_dash="dot", line_color="white", opacity=0.5)
        fig.update_layout(title=f'Z-Score da Força Qualificada (EMA {p})', yaxis=dict(range=[-3.5, 3.5]), height=250, margin=dict(t=30, b=10, l=10, r=10), template="plotly_dark")
        column.plotly_chart(fig, use_container_width=True)

    if MA_PERIODS:
        p_short = MA_PERIODS[0]
        roc_series = metrics['rocs'][p_short].tail(NUM_CANDLES_DISPLAY)
        fig_roc = go.Figure(go.Bar(x=roc_series.index, y=roc_series.values, marker_color=['green' if v >= 0 else 'red' for v in roc_series.values]))
        fig_roc.update_layout(title=f'Velocidade (Contagem Simples EMA {p_short})', height=200, margin=dict(t=30, b=10, l=10, r=10), template="plotly_dark")
        column.plotly_chart(fig_roc, use_container_width=True)
        
        accel_series = metrics['accelerations'][p_short].tail(NUM_CANDLES_DISPLAY)
        fig_accel = go.Figure(go.Bar(x=accel_series.index, y=accel_series.values, marker_color=['#1f77b4' if v >= 0 else '#ff7f0e' for v in accel_series.values]))
        fig_accel.update_layout(title=f'Aceleração (Contagem Simples EMA {p_short})', height=200, margin=dict(t=30, b=10, l=10, r=10), template="plotly_dark")
        column.plotly_chart(fig_accel, use_container_width=True)

    buyer_series = metrics['buyer_climax_zscore'].tail(NUM_CANDLES_DISPLAY).clip(lower=0)
    seller_series = metrics['seller_climax_zscore'].tail(NUM_CANDLES_DISPLAY).clip(lower=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=buyer_series.index, y=buyer_series.values, name='Clímax Comprador', marker_color='green'))
    fig.add_trace(go.Bar(x=seller_series.index, y=seller_series.values, name='Clímax Vendedor', marker_color='red'))
    fig.add_hline(y=3, line_dash="dot", line_color="white", annotation_text="Limiar de Clímax (+3σ)")
    fig.update_layout(barmode='relative', title='Indicador de Clímax de Agressão', height=250, margin=dict(t=30, b=10, l=10, r=10), template="plotly_dark")
    column.plotly_chart(fig, use_container_width=True)

    series = metrics['aggregate_momentum_index'].tail(NUM_CANDLES_DISPLAY)
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, line=dict(color=theme_colors['momentum']), fill='tozeroy'))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(title='Índice de Momentum Agregado', height=250, margin=dict(t=30, b=10, l=10, r=10), template="plotly_dark")
    column.plotly_chart(fig, use_container_width=True)
    
    for p, series in metrics['conviction_zscore'].items():
        fig = go.Figure(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, line=dict(color=theme_colors['conviction_z'])))
        fig.add_hline(y=2, line_dash="dot", line_color="white", opacity=0.5); fig.add_hline(y=-2, line_dash="dot", line_color="white", opacity=0.5)
        fig.update_layout(title=f'Z-Score da Convicção (EMA {p})', yaxis=dict(range=[-3.5, 3.5]), height=250, margin=dict(t=30, b=10, l=10, r=10), template="plotly_dark")
        column.plotly_chart(fig, use_container_width=True)
        
    # Gráfico 8: Índice de Força de Volume (VFI) (NOVO)
    for p, series in metrics['volume_force_indices'].items():
        fig = go.Figure(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, mode="lines", line_color=theme_colors['vfi'], fill='tozeroy'))
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(title=f'Índice de Força de Volume (VFI EMA {p})', yaxis_title="Força (Dist*Vol)", height=250, margin=dict(t=30, b=10, l=10, r=10), template="plotly_dark")
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
metrics_risk_off = calculate_breadth_metrics(RISK_OFF_ASSETS, combined)
metrics_risk_on = calculate_breadth_metrics(RISK_ON_ASSETS, combined)

# --- Definição dos Temas de Cores ---
risk_off_colors = {'main': '#E74C3C', 'accent': '#F1948A', 'momentum': '#D98880', 'qualified': '#FFA07A', 'conviction_z': '#F5B041', 'vfi': '#E67E22'}
risk_on_colors = {'main': '#2ECC71', 'accent': '#ABEBC6', 'momentum': '#76D7C4', 'qualified': '#87CEEB', 'conviction_z': '#5DADE2', 'vfi': '#3498DB'}


# --- Visualização ---
col1, col2 = st.columns(2)
display_charts(col1, metrics_risk_off, "Risk-Off (Força do Dólar)", risk_off_colors)
display_charts(col2, metrics_risk_on, "Risk-On (Fraqueza do Dólar)", risk_on_colors)


st.caption("Feito com Streamlit • Dados via FinancialModelingPrep")

