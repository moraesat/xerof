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
    'Força Ponderada (Contagem)', 'Força Qualificada (Filtro)', 'Z-Score da Força Qualificada',
    'Velocidade e Aceleração', 'Indicador de Clímax de Agressão', 'Indicador de Clímax de Rejeição',
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

def calculate_dynamic_correlation_weights(asset_list, reference_asset_symbol, combined_data, window):
    weights = {}
    ref_returns = combined_data[f"{reference_asset_symbol}_close"].pct_change()
    for asset in asset_list:
        asset_returns = combined_data.get(f"{asset}_close", pd.Series(dtype=float)).pct_change()
        if not asset_returns.empty:
            weights[asset] = ref_returns.rolling(window=window).corr(asset_returns)
    return weights

def calculate_breadth_metrics(asset_weights: dict, combined_data: pd.DataFrame, is_dynamic_weights=False):
    metrics = {}
    metrics['weighted_counts'] = {p: pd.Series(0.0, index=combined_data.index) for p in MA_PERIODS}
    metrics['qualified_counts'] = {p: pd.Series(0.0, index=combined_data.index) for p in MA_PERIODS}
    metrics['volume_force_indices'] = {p: pd.Series(0.0, index=combined_data.index) for p in MA_PERIODS}
    momentum_components = []

    for s, weight in asset_weights.items():
        close_col, open_col = f"{s}_close", f"{s}_open"
        if close_col not in combined_data.columns: continue

        if isinstance(weight, pd.Series):
            weight = weight.reindex(combined_data.index, method='ffill').fillna(0)

        atr = calculate_atr(combined_data[f"{s}_high"], combined_data[f"{s}_low"], combined_data[close_col], ATR_PERIOD).replace(0, np.nan)
        volume_ma = combined_data[f"{s}_volume"].rolling(window=VOLUME_MA_PERIOD).mean().replace(0, np.nan)
        volume_strength = (combined_data[f"{s}_volume"] / volume_ma).fillna(1)

        for p in MA_PERIODS:
            ema_val = combined_data[close_col].ewm(span=p, adjust=False).mean()
            above_ema = (combined_data[close_col] > ema_val)
            metrics['weighted_counts'][p] += above_ema.astype(int) * weight
            normalized_distance = ((combined_data[close_col] - ema_val) / atr).fillna(0)
            metrics['qualified_counts'][p] += (normalized_distance > CONVICTION_THRESHOLD).astype(int) * weight
            metrics['volume_force_indices'][p] += (normalized_distance * volume_strength) * weight
        
        roc = combined_data[close_col].pct_change(periods=MOMENTUM_PERIOD)
        normalized_momentum = calculate_zscore(roc, MOMENTUM_Z_WINDOW)
        momentum_components.append(normalized_momentum * weight)

    metrics['aggregate_momentum_index'] = pd.concat(momentum_components, axis=1).sum(axis=1) if momentum_components else pd.Series(0.0, index=combined_data.index)
    
    # 2. Pré-Cálculos XAUUSD (Eventos - Usados nos gatilhos)
    xau_close, xau_open, xau_high, xau_low = combined_data['XAUUSD_close'], combined_data['XAUUSD_open'], combined_data['XAUUSD_high'], combined_data['XAUUSD_low']
    xau_atr = calculate_atr(xau_high, xau_low, xau_close, ATR_PERIOD).replace(0, np.nan)
    xau_energy = (xau_high - xau_low) / xau_atr
    xau_body = abs(xau_close - xau_open).replace(0, 0.00001)

    metrics['xau_aggression_buyer'] = xau_energy.where((xau_close > xau_open) & (xau_energy > ENERGY_THRESHOLD), 0)
    metrics['xau_aggression_seller'] = xau_energy.where((xau_close < xau_open) & (xau_energy > ENERGY_THRESHOLD), 0)
    metrics['xau_buyer_climax_zscore'] = calculate_zscore(metrics['xau_aggression_buyer'], CLIMAX_Z_WINDOW)
    metrics['xau_seller_climax_zscore'] = calculate_zscore(metrics['xau_aggression_seller'], CLIMAX_Z_WINDOW)
    
    xau_upper_shadow = xau_high - combined_data[['XAUUSD_open', 'XAUUSD_close']].max(axis=1)
    xau_lower_shadow = combined_data[['XAUUSD_open', 'XAUUSD_close']].min(axis=1) - xau_low
    
    metrics['xau_rejection_buyer'] = (xau_lower_shadow / xau_body).where(xau_lower_shadow > xau_body * SHADOW_TO_BODY_RATIO, 0)
    metrics['xau_rejection_seller'] = (xau_upper_shadow / xau_body).where(xau_upper_shadow > xau_body * SHADOW_TO_BODY_RATIO, 0)

    # --- Adicionar ROC/Accel do XAUUSD-ONLY ---
    metrics['xau_roc'] = {}
    metrics['xau_accel'] = {}
    for p in MA_PERIODS:
        xau_close = combined_data['XAUUSD_close']
        metrics['xau_roc'][p] = xau_close.diff()
        metrics['xau_accel'][p] = xau_close.diff().diff()

    # --- Cálculos Finais ---
    metrics['rocs'], metrics['accelerations'] = {}, {}
    metrics['qualified_zscore'] = {} 
    for p in MA_PERIODS:
        metrics['rocs'][p] = metrics['weighted_counts'][p].diff()
        metrics['accelerations'][p] = metrics['rocs'][p].diff()
        metrics['qualified_zscore'][p] = calculate_zscore(metrics['qualified_counts'][p], Z_SCORE_WINDOW) 

    return metrics

def display_charts(column, metrics, title_prefix, theme_colors, overlay_price_data, selected_charts, key_prefix, is_dynamic_weights=False):
    
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
        buyer_climax = metrics['xau_buyer_climax_zscore'] > 1
        seller_climax = metrics['xau_seller_climax_zscore'] > 1
        candle_is_up = overlay_price_data['close'] > overlay_price_data['open']

        confirmation_buy = buyer_climax & candle_is_up
        confirmation_sell = seller_climax & ~candle_is_up
        divergence_buy = seller_climax & candle_is_up
        divergence_sell = buyer_climax & ~candle_is_up

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=overlay_price_data.index, open=overlay_price_data['open'], high=overlay_price_data['high'], low=overlay_price_data['low'], close=overlay_price_data['close'], name="XAUUSD",
                                     increasing_line_color='green', decreasing_line_color='red'))
        fig.add_trace(go.Scatter(x=overlay_price_data[confirmation_buy].index, y=overlay_price_data[confirmation_buy]['low'], mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Confirmação Compra'))
        fig.add_trace(go.Scatter(x=overlay_price_data[confirmation_sell].index, y=overlay_price_data[confirmation_sell]['high'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Confirmação Venda'))
        fig.add_trace(go.Scatter(x=overlay_price_data[divergence_buy].index, y=overlay_price_data[divergence_buy]['low'], mode='markers', marker=dict(symbol='diamond', color='cyan', size=10), name='Divergência Compra'))
        fig.add_trace(go.Scatter(x=overlay_price_data[divergence_sell].index, y=overlay_price_data[divergence_sell]['high'], mode='markers', marker=dict(symbol='diamond', color='magenta', size=10), name='Divergência Venda'))
        fig.update_layout(title='Indicador de Clímax e Resultado', height=350, margin=dict(t=30, b=40, l=10, r=10), template="plotly_dark", xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="right", x=1))
        column.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_divergence")

    if 'Força Ponderada (Contagem)' in selected_charts:
        for p, series in metrics['weighted_counts'].items():
            fig = create_fig_with_overlay(f'Força Ponderada (Contagem EMA {p})')
            fig.add_trace(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, name='Força', mode="lines", fill="tozeroy", line_color=theme_colors['main'], opacity=0.7))
            fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name='XAUUSD', yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
            if not is_dynamic_weights: fig.update_layout(yaxis=dict(range=[0, 100]))
            column.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_wc_{p}")
            
    if 'Força Qualificada (Filtro)' in selected_charts:
        for p, series in metrics['qualified_counts'].items():
            fig = create_fig_with_overlay(f'Força Qualificada (Filtro EMA {p})')
            fig.add_trace(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, name='Qualificada', mode="lines", fill="tozeroy", line_color=theme_colors['qualified']))
            fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name='XAUUSD', yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
            if not is_dynamic_weights: fig.update_layout(yaxis=dict(range=[0, 100]))
            column.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_qc_{p}")

    if 'Z-Score da Força Qualificada' in selected_charts:
        for p, series in metrics['qualified_zscore'].items():
            fig = create_fig_with_overlay(f'Z-Score da Força Qualificada (EMA {p})')
            fig.add_trace(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, name='Z-Score', line=dict(color=theme_colors['accent'])))
            fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name='XAUUSD', yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
            fig.add_hline(y=2, line_dash="dot", line_color="white", opacity=0.5); fig.add_hline(y=-2, line_dash="dot", line_color="white", opacity=0.5)
            fig.update_layout(yaxis=dict(range=[-3.5, 3.5]))
            column.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_zqc_{p}")

    if 'Velocidade e Aceleração' in selected_charts and MA_PERIODS:
        p_short = MA_PERIODS[0]
        # XAUUSD-ONLY ROC/Aceleração
        xau_roc = metrics['xau_roc'][p_short].tail(NUM_CANDLES_DISPLAY)
        xau_accel = metrics['xau_accel'][p_short].tail(NUM_CANDLES_DISPLAY)

        fig_roc = create_fig_with_overlay(f'Velocidade (ROC XAUUSD {p_short})')
        fig_roc.add_trace(go.Bar(x=xau_roc.index, y=xau_roc.values, name='ROC', marker_color=['green' if v >= 0 else 'red' for v in xau_roc.values]))
        fig_roc.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name='XAUUSD', yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
        fig_roc.update_layout(height=200)
        column.plotly_chart(fig_roc, use_container_width=True, key=f"{key_prefix}_roc")

        fig_accel = create_fig_with_overlay(f'Aceleração (XAUUSD {p_short})')
        fig_accel.add_trace(go.Bar(x=xau_accel.index, y=xau_accel.values, name='Aceleração', marker_color=['#1f77b4' if v >= 0 else '#ff7f0e' for v in xau_accel.values]))
        fig_accel.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name='XAUUSD', yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
        fig_accel.update_layout(height=200)
        column.plotly_chart(fig_accel, use_container_width=True, key=f"{key_prefix}_accel")

    if 'Indicador de Clímax de Agressão' in selected_charts:
        buyer_series = metrics['xau_buyer_climax_zscore'].tail(NUM_CANDLES_DISPLAY).clip(lower=0)
        seller_series = metrics['xau_seller_climax_zscore'].tail(NUM_CANDLES_DISPLAY).clip(lower=0)
        fig = create_fig_with_overlay('Clímax de Agressão (XAUUSD)')
        fig.add_trace(go.Bar(x=buyer_series.index, y=buyer_series.values, name='Clímax Comprador', marker_color='green'))
        fig.add_trace(go.Bar(x=seller_series.index, y=seller_series.values, name='Clímax Vendedor', marker_color='red'))
        fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name='XAUUSD', yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
        fig.add_hline(y=3, line_dash="dot", line_color="white")
        fig.update_layout(barmode='relative')
        column.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_climax_agg")

    if 'Indicador de Clímax de Rejeição' in selected_charts:
        buyer_series = metrics['xau_rejection_buyer'].tail(NUM_CANDLES_DISPLAY)
        seller_series = metrics['xau_rejection_seller'].tail(NUM_CANDLES_DISPLAY)
        fig = create_fig_with_overlay('Clímax de Rejeição (XAUUSD)')
        fig.add_trace(go.Bar(x=buyer_series.index, y=buyer_series.values, name='Rejeição Compradora', marker_color='lime'))
        fig.add_trace(go.Bar(x=seller_series.index, y=seller_series.values, name='Rejeição Vendedora', marker_color='pink'))
        fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name='XAUUSD', yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
        fig.update_layout(barmode='relative')
        column.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_climax_rej")

    if 'Índice de Momentum Agregado' in selected_charts:
        series = metrics['aggregate_momentum_index'].tail(NUM_CANDLES_DISPLAY)
        fig = create_fig_with_overlay('Índice de Momentum Agregado')
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name='Momentum', line=dict(color=theme_colors['momentum']), fill='tozeroy'))
        fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name='XAUUSD', yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        column.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_momentum")
    
    if 'Índice de Força de Volume (VFI)' in selected_charts:
        for p, series in metrics['volume_force_indices'].items():
            fig = create_fig_with_overlay(f'Índice de Força de Volume (VFI EMA {p})')
            fig.add_trace(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, name='VFI', mode="lines", line_color=theme_colors['vfi'], fill='tozeroy'))
            fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name='XAUUSD', yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
            fig.add_hline(y=0, line_dash="dash", line_color="grey")
            column.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_vfi_{p}")

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
        # Cestas para o cálculo de correlação alinhada
        risk_on_basket = list(set(RISK_ON_ASSETS.keys()) - {'XAUUSD', 'XAGUSD'})
        risk_off_basket = list(RISK_OFF_ASSETS.keys())
        
        # Correlação para Risk-On (esperado > 0)
        for s in risk_on_basket:
            asset_returns = combined_data.get(f"{s}_close").pct_change()
            if asset_returns is not None:
                correlation = ref_returns.rolling(window=CORRELATION_WINDOW).corr(asset_returns)
                dynamic_weights[s] = correlation.fillna(0)
                if not correlation.empty: latest_correlations[s] = correlation.iloc[-1]
        
        # Correlação para Risk-Off (esperado < 0, então invertemos)
        for s in risk_off_basket:
            asset_returns = combined_data.get(f"{s}_close").pct_change()
            if asset_returns is not None:
                correlation = ref_returns.rolling(window=CORRELATION_WINDOW).corr(asset_returns)
                dynamic_weights[s] = correlation.fillna(0) * -1 # Inverte para alinhar
                if not correlation.empty: latest_correlations[s] = correlation.iloc[-1]

    if not dynamic_weights:
        return timeframe, None, None, None

    metrics = calculate_breadth_metrics(dynamic_weights, combined_data, is_dynamic_weights=True)
    overlay_data = combined_data[[f"XAUUSD_open", f"XAUUSD_high", f"XAUUSD_low", f"XAUUSD_close"]].tail(NUM_CANDLES_DISPLAY)
    overlay_data.columns = ['open', 'high', 'low', 'close']
    
    # --- Adicionar Métricas XAUUSD-Only ---
    xau_close, xau_open, xau_high, xau_low = overlay_data['close'], overlay_data['open'], overlay_data['high'], overlay_data['low']
    xau_atr = calculate_atr(xau_high, xau_low, xau_close, ATR_PERIOD).replace(0, np.nan)
    xau_energy = (xau_high - xau_low) / xau_atr
    xau_body = abs(xau_close - xau_open).replace(0, 0.00001)

    metrics['xau_aggression_buyer'] = xau_energy.where((xau_close > xau_open) & (xau_energy > ENERGY_THRESHOLD), 0)
    metrics['xau_aggression_seller'] = xau_energy.where((xau_close < xau_open) & (xau_energy > ENERGY_THRESHOLD), 0)
    metrics['xau_buyer_climax_zscore'] = calculate_zscore(metrics['xau_aggression_buyer'], CLIMAX_Z_WINDOW)
    metrics['xau_seller_climax_zscore'] = calculate_zscore(metrics['xau_aggression_seller'], CLIMAX_Z_WINDOW)
    
    xau_upper_shadow = xau_high - combined_data[['XAUUSD_open', 'XAUUSD_close']].max(axis=1)
    xau_lower_shadow = combined_data[['XAUUSD_open', 'XAUUSD_close']].min(axis=1) - xau_low
    
    metrics['xau_rejection_buyer'] = (xau_lower_shadow / xau_body).where(xau_lower_shadow > xau_body * SHADOW_TO_BODY_RATIO, 0)
    metrics['xau_rejection_seller'] = (xau_upper_shadow / xau_body).where(xau_upper_shadow > xau_body * SHADOW_TO_BODY_RATIO, 0)

    # --- Adicionar ROC/Accel do XAUUSD-ONLY ---
    metrics['xau_roc'] = {}
    metrics['xau_accel'] = {}
    for p in MA_PERIODS:
        xau_close = combined_data['XAUUSD_close']
        metrics['xau_roc'][p] = xau_close.diff()
        metrics['xau_accel'][p] = xau_close.diff().diff()
    
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
    st.header("Matriz de Correlação vs. XAUUSD (Baseado no Timeframe de 1 Minuto)")
    st.markdown("Mostra a correlação móvel mais recente de cada ativo com o XAUUSD.")
    if '1min' in results:
        correlations = results['1min']['correlations']
        latest_corr_values = {asset: value for asset, value in correlations.items() if pd.notna(value)}

        risk_on_basket_corr = list(set(RISK_ON_ASSETS.keys()) - {'XAUUSD', 'XAGUSD'})
        risk_off_basket_corr = list(RISK_OFF_ASSETS.keys())

        risk_on_corr_data = {asset: corr for asset, corr in latest_corr_values.items() if asset in risk_on_basket_corr}
        risk_off_corr_data = {asset: corr for asset, corr in latest_corr_values.items() if asset in risk_off_basket_corr}

        df_risk_on = pd.DataFrame(list(risk_on_corr_data.items()), columns=['Ativo', 'Correlação']).sort_values(by='Correlação', ascending=False).set_index('Ativo')
        df_risk_off = pd.DataFrame(list(risk_off_corr_data.items()), columns=['Ativo', 'Correlação']).sort_values(by='Correlação', ascending=False).set_index('Ativo')

        col1_corr, col2_corr = st.columns(2)
        with col1_corr:
            st.subheader("Ativos da Cesta Risk-On")
            st.dataframe(df_risk_on.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.2f}"), use_container_width=True)
        with col2_corr:
            st.subheader("Ativos da Cesta Risk-Off")
            st.dataframe(df_risk_off.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.2f}"), use_container_width=True)
    else:
        st.warning("Dados de correlação para o timeframe de 1 minuto ainda não estão disponíveis. Aguarde a próxima atualização.")


st.caption("Feito com Streamlit • Dados via FinancialModelingPrep")
