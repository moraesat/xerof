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
st.set_page_config(page_title="Painel Avançado XAUUSD",
                   layout="wide", page_icon="🥇")
API_KEY = "3CImfjoxNd98om3uhS89X4lmlp4Mrp3H"
TZ = pytz.timezone("America/Sao_Paulo")

# --- CESTAS DE ATIVOS E PESOS ESTÁTICOS ---
RISK_OFF_ASSETS = {
    'DX-Y.NYB': 20, 'USDJPY': 20, 'USDCHF': 10, 'USDCAD': 10, 'USDCNH': 10,
    'USDSEK': 5,  'USDNOK': 5,  'USDMXN': 5,  'USDSGD': 5,  'USDZAR': 2,
    'USDHKD': 2,  'USDPLN': 2,  'USDCZK': 1,  'USDDKK': 1,  'USDHUF': 1
}
RISK_ON_ASSETS = {
    'EURUSD': 38, 'GBPUSD': 16, 'AUDUSD': 10, 'XAUUSD': 24, 'XAGUSD': 9, 'NZDUSD': 3
}
ALL_UNIQUE_ASSETS = list(set(RISK_OFF_ASSETS.keys()) | set(RISK_ON_ASSETS.keys()))
NUM_CANDLES_DISPLAY = 120

# Atualização automática a cada 20s
st_autorefresh(interval=20 * 1000, key="refresh")

# ===========================
# Parâmetros Padrão Otimizados (Modo Avançado)
# ===========================
CONVICTION_THRESHOLD = 0.2
Z_SCORE_WINDOW = 100
ATR_PERIOD = 14
ENERGY_THRESHOLD = 1.5
CLIMAX_Z_WINDOW = 100
SHADOW_TO_BODY_RATIO = 2.0
MOMENTUM_PERIOD = 21
MOMENTUM_Z_WINDOW = 100
VOLUME_MA_PERIOD = 20
CORRELATION_WINDOW = 100

ALL_CHARTS_LIST = [
    'Indicador de Divergência de Agressão',
    'Força Qualificada (Filtro)', 'Z-Score da Força Qualificada',
    'Indicador de Clímax de Agressão', 'Indicador de Clímax de Rejeição',
    'Índice de Momentum Agregado', 'Índice de Força de Volume (VFI)'
]

# ===========================
# Menu lateral (Simplificado)
# ===========================
st.sidebar.title("Configurações Gerais")
if st.sidebar.button("Atualizar Agora 🔄"):
    st.rerun()

MA_INPUT = st.sidebar.text_input("Períodos das Médias Móveis", "9,21")
MA_PERIODS = [int(x.strip()) for x in MA_INPUT.split(",") if x.strip().isdigit()]
SELECTED_CHARTS = st.sidebar.multiselect("Gráficos a Exibir", ALL_CHARTS_LIST, default=ALL_CHARTS_LIST)

# ===========================
# Funções de Cálculo e Busca
# ===========================

@st.cache_data(ttl=20)
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
    data = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_symbol = {executor.submit(get_single_data, s, timeframe, candles_to_fetch): s for s in symbols}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            result = future.result()
            if result is not None and not result.empty:
                data[symbol] = result
    if not data: return pd.DataFrame()
    
    base_index = data.get('XAUUSD', next(iter(data.values()))).index
    aligned_data = {symbol: df.reindex(base_index, method='ffill') for symbol, df in data.items()}
    
    frames = [df.rename(columns=lambda c: f"{symbol}_{c}") for symbol, df in aligned_data.items()]
    return pd.concat(frames, axis=1).dropna()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_zscore(series: pd.Series, window: int) -> pd.Series:
    return (series - series.rolling(window=window).mean()) / series.rolling(window=window).std()

def calculate_breadth_metrics(asset_weights: dict, combined_data: pd.DataFrame, is_dynamic_weights=False):
    metrics = {}
    metrics['qualified_counts'] = {p: pd.Series(0.0, index=combined_data.index) for p in MA_PERIODS}
    metrics['volume_force_indices'] = {p: pd.Series(0.0, index=combined_data.index) for p in MA_PERIODS}
    momentum_components = []

    for s, weight in asset_weights.items():
        close_col, open_col, high_col, low_col, vol_col = f"{s}_close", f"{s}_open", f"{s}_high", f"{s}_low", f"{s}_volume"
        if close_col not in combined_data.columns: continue

        if isinstance(weight, pd.Series):
            weight = weight.reindex(combined_data.index, method='ffill').fillna(0)

        atr = calculate_atr(combined_data[high_col], combined_data[low_col], combined_data[close_col], ATR_PERIOD).replace(0, np.nan)
        
        volume_ma = combined_data[vol_col].rolling(window=VOLUME_MA_PERIOD).mean().replace(0, np.nan)
        volume_strength = (combined_data[vol_col] / volume_ma).fillna(1)

        for p in MA_PERIODS:
            ema_val = combined_data[close_col].ewm(span=p, adjust=False).mean()
            normalized_distance = ((combined_data[close_col] - ema_val) / atr).fillna(0)
            metrics['qualified_counts'][p] += (normalized_distance > CONVICTION_THRESHOLD).astype(int) * weight
            metrics['volume_force_indices'][p] += (normalized_distance * volume_strength) * weight
        
        roc = combined_data[close_col].pct_change(periods=MOMENTUM_PERIOD)
        normalized_momentum = calculate_zscore(roc, MOMENTUM_Z_WINDOW)
        momentum_components.append(normalized_momentum * weight)

    metrics['aggregate_momentum_index'] = pd.concat(momentum_components, axis=1).sum(axis=1) if momentum_components else pd.Series(0.0, index=combined_data.index)
    
    metrics['qualified_zscore'] = {} 
    for p in MA_PERIODS:
        metrics['qualified_zscore'][p] = calculate_zscore(metrics['qualified_counts'][p], Z_SCORE_WINDOW) 

    return metrics

def display_charts(container, metrics, theme_colors, overlay_price_data, selected_charts, key_prefix, is_dynamic_weights=False):
    
    def create_fig_with_overlay(title):
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", height=250, margin=dict(t=30, b=20, l=20, r=40),
            title=dict(text=title, x=0.01, font=dict(size=14)),
            yaxis2=dict(overlaying='y', side='right', showgrid=False, showticklabels=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    if 'Indicador de Divergência de Agressão' in selected_charts and not overlay_price_data.empty:
        # Lógica de Divergência permanece baseada na amplitude geral
        # ... (código existente da divergência) ...

    if 'Força Qualificada (Filtro)' in selected_charts:
        # ... (código existente) ...

    if 'Z-Score da Força Qualificada' in selected_charts:
        # ... (código existente) ...
        
    # --- NOVOS INDICADORES DE CLÍMAX (XAUUSD-ONLY) ---
    if 'Indicador de Clímax de Agressão' in selected_charts:
        atr = calculate_atr(overlay_price_data['high'], overlay_price_data['low'], overlay_price_data['close'], ATR_PERIOD)
        energy = (overlay_price_data['high'] - overlay_price_data['low']) / atr.replace(0, np.nan)
        is_high_energy = energy > ENERGY_THRESHOLD
        
        buyer_aggression = (overlay_price_data['close'] > overlay_price_data['open']) & is_high_energy
        seller_aggression = (overlay_price_data['close'] < overlay_price_data['open']) & is_high_energy
        
        fig = create_fig_with_overlay('Clímax de Agressão (Apenas XAUUSD)')
        fig.add_trace(go.Bar(x=overlay_price_data.index[buyer_aggression], y=energy[buyer_aggression], name='Agressão Compradora', marker_color='green'))
        fig.add_trace(go.Bar(x=overlay_price_data.index[seller_aggression], y=energy[seller_aggression], name='Agressão Vendedora', marker_color='red'))
        fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name='XAUUSD', yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
        fig.update_layout(barmode='relative', yaxis_title='Energia (ATR)')
        container.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_climax_agg_xau")

    if 'Indicador de Clímax de Rejeição' in selected_charts:
        body = abs(overlay_price_data['close'] - overlay_price_data['open']).replace(0, 0.00001)
        upper_shadow = overlay_price_data['high'] - overlay_price_data[['open', 'close']].max(axis=1)
        lower_shadow = overlay_price_data[['open', 'close']].min(axis=1) - overlay_price_data['low']
        
        is_buyer_rejection = lower_shadow > body * SHADOW_TO_BODY_RATIO
        is_seller_rejection = upper_shadow > body * SHADOW_TO_BODY_RATIO
        
        fig = create_fig_with_overlay('Clímax de Rejeição (Apenas XAUUSD)')
        fig.add_trace(go.Bar(x=overlay_price_data.index[is_buyer_rejection], y=(lower_shadow / body)[is_buyer_rejection], name='Rejeição Compradora', marker_color='lime'))
        fig.add_trace(go.Bar(x=overlay_price_data.index[is_seller_rejection], y=(upper_shadow / body)[is_seller_rejection], name='Rejeição Vendedora', marker_color='pink'))
        fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name='XAUUSD', yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
        fig.update_layout(barmode='relative', yaxis_title='Rácio Sombra/Corpo')
        container.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_climax_rej_xau")

    # ... (Restante do código de display dos outros gráficos)

# ===========================
# Lógica Principal da Aplicação
# ===========================
st.title("🥇 Painel de Análise Avançada XAUUSD")

placeholder = st.empty()
xauusd_basket = list(set(ALL_UNIQUE_ASSETS) - {'XAUUSD', 'XAGUSD'})

def process_timeframe(timeframe):
    candles_to_fetch = (max(MA_PERIODS) if MA_PERIODS else 200) + NUM_CANDLES_DISPLAY + max(Z_SCORE_WINDOW, MOMENTUM_Z_WINDOW, CLIMAX_Z_WINDOW, CORRELATION_WINDOW)
    combined_data = build_combined_data(ALL_UNIQUE_ASSETS, timeframe, candles_to_fetch)
    if combined_data.empty or 'XAUUSD_close' not in combined_data.columns:
        return timeframe, None, None, None
    
    dynamic_weights = {}
    latest_correlations = {}
    if len(combined_data) > CORRELATION_WINDOW:
        ref_returns = combined_data['XAUUSD_close'].pct_change()
        for s in xauusd_basket:
            asset_returns = combined_data.get(f"{s}_close").pct_change()
            if asset_returns is not None:
                correlation = ref_returns.rolling(window=CORRELATION_WINDOW).corr(asset_returns)
                dynamic_weights[s] = correlation.fillna(0)
                if not correlation.empty:
                    latest_correlations[s] = correlation.iloc[-1]

    if not dynamic_weights:
        return timeframe, None, None, None

    metrics = calculate_breadth_metrics(dynamic_weights, combined_data, is_dynamic_weights=True)
    overlay_data = combined_data[[f"XAUUSD_open", f"XAUUSD_high", f"XAUUSD_low", f"XAUUSD_close"]].tail(NUM_CANDLES_DISPLAY)
    overlay_data.columns = ['open', 'high', 'low', 'close']
    return timeframe, metrics, overlay_data, latest_correlations

results = {}
with st.spinner("A processar dados multi-timeframe..."):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_timeframe, tf) for tf in ['1min', '5min']]
        for future in as_completed(futures):
            tf, metrics, overlay_data, correlations = future.result()
            if metrics:
                results[tf] = {'metrics': metrics, 'overlay': overlay_data, 'correlations': correlations}

if '1min' in results:
    now = datetime.now(TZ)
    last_candle_time = results['1min']['overlay'].index[-1]
    delay_minutes = (now - last_candle_time).total_seconds() / 60
    if delay_minutes < 2:
        placeholder.success(f"🟢 Dados FRESCOS (Atraso de {delay_minutes:.1f} min)")
    else:
        placeholder.warning(f"🟠 ATENÇÃO: Atraso nos dados de {delay_minutes:.1f} min")

else:
    st.error("Não foi possível carregar os dados. Verifique a API.")
    st.stop()


# --- VISUALIZAÇÃO ---
tab1, tab5, tab_corr = st.tabs(["Análise de 1 Minuto", "Análise de 5 Minutos", "Matriz de Correlação"])
corr_colors = {'main': '#FFD700', 'accent': '#FFFACD', 'momentum': '#F0E68C', 'qualified': '#EEE8AA', 'conviction_z': '#FFECB3', 'vfi': '#FFC107', 'overlay': 'rgba(255, 255, 255, 0.6)'}

with tab1:
    if '1min' in results:
        display_charts(st, results['1min']['metrics'], "Análise de 1 Minuto", corr_colors, results['1min']['overlay'], SELECTED_CHARTS, "1min_charts", is_dynamic_weights=True)

with tab5:
    if '5min' in results:
        display_charts(st, results['5min']['metrics'], "Análise de 5 Minutos", corr_colors, results['5min']['overlay'], SELECTED_CHARTS, "5min_charts", is_dynamic_weights=True)

with tab_corr:
    # ... (código da aba de correlação, idêntico à versão anterior)

