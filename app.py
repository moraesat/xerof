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
# Menu lateral
# ===========================
st.sidebar.title("Configurações Gerais")
if st.sidebar.button("Atualizar Agora 🔄"):
    st.rerun()

MA_INPUT = st.sidebar.text_input("Períodos das Médias Móveis", "9,21")
MA_PERIODS = [int(x.strip()) for x in MA_INPUT.split(",") if x.strip().isdigit()]
TIMEFRAME = st.sidebar.radio("Timeframe", ["1min", "5min", "15min", "1h"])

st.sidebar.header("Parâmetros dos Indicadores")
CONVICTION_THRESHOLD = st.sidebar.slider("Filtro de Convicção (ATR)", 0.0, 1.0, 0.2, 0.05)
Z_SCORE_WINDOW = st.sidebar.slider("Janela Z-Score (Amplitude)", 50, 500, 200)
ATR_PERIOD = st.sidebar.slider("Período do ATR", 10, 30, 14)
ENERGY_THRESHOLD = st.sidebar.slider("Limiar de 'Energia'", 1.0, 3.0, 1.5, 0.1)
CLIMAX_Z_WINDOW = st.sidebar.slider("Janela Z-Score (Clímax)", 50, 200, 100)
SHADOW_TO_BODY_RATIO = st.sidebar.slider("Rácio Sombra/Corpo (Rejeição)", 1.0, 5.0, 2.0, 0.1)
MOMENTUM_PERIOD = st.sidebar.slider("Período ROC (Momentum)", 10, 50, 21)
MOMENTUM_Z_WINDOW = st.sidebar.slider("Janela Z-Score (Momentum)", 50, 200, 100)
VOLUME_MA_PERIOD = st.sidebar.slider("Janela Média de Volume (VFI)", 10, 50, 20)
CORRELATION_WINDOW = st.sidebar.slider("Janela de Correlação (XAUUSD)", 50, 200, 100)

ALL_CHARTS_LIST = [
    'Indicador de Divergência de Agressão',
    'Força Ponderada (Contagem)', 'Força Qualificada (Filtro)', 'Z-Score da Força Qualificada',
    'Velocidade e Aceleração', 'Indicador de Clímax de Agressão', 'Indicador de Clímax de Rejeição',
    'Índice de Momentum Agregado', 'Z-Score da Convicção', 'Índice de Força de Volume (VFI)'
]

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
    with st.spinner(f"A buscar dados para {len(symbols)} ativos..."):
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_symbol = {executor.submit(get_single_data, s, timeframe, candles_to_fetch): s for s in symbols}
            frames = [future.result().rename(columns=lambda c: f"{future_to_symbol[future]}_{c}") for future in as_completed(future_to_symbol) if future.result() is not None]
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1).ffill().dropna()

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
    for metric_name in ['weighted_counts', 'qualified_counts', 'weighted_distance_indices', 'volume_force_indices']:
        metrics[metric_name] = {p: pd.Series(0.0, index=combined_data.index) for p in MA_PERIODS}
    
    aggression_buyer = pd.Series(0.0, index=combined_data.index)
    aggression_seller = pd.Series(0.0, index=combined_data.index)
    rejection_buyer = pd.Series(0.0, index=combined_data.index)
    rejection_seller = pd.Series(0.0, index=combined_data.index)
    momentum_components = []

    for s in asset_weights.keys():
        weight = asset_weights[s]
        close_col, open_col, high_col, low_col, vol_col = f"{s}_close", f"{s}_open", f"{s}_high", f"{s}_low", f"{s}_volume"
        if close_col not in combined_data.columns: continue

        strength_condition = (combined_data[close_col] > combined_data[open_col])
        atr = calculate_atr(combined_data[high_col], combined_data[low_col], combined_data[close_col], ATR_PERIOD).replace(0, np.nan)
        
        is_high_energy = (combined_data[high_col] - combined_data[low_col]) / atr > ENERGY_THRESHOLD
        aggression_buyer += (strength_condition & is_high_energy).astype(int) * weight
        aggression_seller += (~strength_condition & is_high_energy).astype(int) * weight
        
        body = abs(combined_data[close_col] - combined_data[open_col]).replace(0, 0.00001)
        upper_shadow = combined_data[high_col] - combined_data[[open_col, close_col]].max(axis=1)
        lower_shadow = combined_data[[open_col, close_col]].min(axis=1) - combined_data[low_col]
        
        rejection_buyer += (lower_shadow > body * SHADOW_TO_BODY_RATIO).astype(int) * weight
        rejection_seller += (upper_shadow > body * SHADOW_TO_BODY_RATIO).astype(int) * weight

        volume_ma = combined_data[vol_col].rolling(window=VOLUME_MA_PERIOD).mean().replace(0, np.nan)
        volume_strength = (combined_data[vol_col] / volume_ma).fillna(1)

        for p in MA_PERIODS:
            ema_val = combined_data[close_col].ewm(span=p, adjust=False).mean()
            above_ema = (combined_data[close_col] > ema_val)
            metrics['weighted_counts'][p] += above_ema.astype(int) * weight

            normalized_distance = ((combined_data[close_col] - ema_val) / atr).fillna(0)
            is_significant_above = normalized_distance > CONVICTION_THRESHOLD
            metrics['qualified_counts'][p] += is_significant_above.astype(int) * weight
            metrics['weighted_distance_indices'][p] += normalized_distance * weight
            
            volume_force = normalized_distance * volume_strength
            metrics['volume_force_indices'][p] += volume_force * weight
        
        roc = combined_data[close_col].pct_change(periods=MOMENTUM_PERIOD)
        normalized_momentum = calculate_zscore(roc, MOMENTUM_Z_WINDOW)
        momentum_components.append(normalized_momentum * weight)

    metrics['aggression_buyer'] = aggression_buyer
    metrics['aggression_seller'] = aggression_seller
    metrics['rejection_buyer'] = rejection_buyer
    metrics['rejection_seller'] = rejection_seller
    metrics['buyer_climax_zscore'] = calculate_zscore(aggression_buyer, CLIMAX_Z_WINDOW)
    metrics['seller_climax_zscore'] = calculate_zscore(aggression_seller, CLIMAX_Z_WINDOW)
    metrics['aggregate_momentum_index'] = pd.concat(momentum_components, axis=1).sum(axis=1) if momentum_components else pd.Series(0.0, index=combined_data.index)
    
    metrics['z_scores'], metrics['rocs'], metrics['accelerations'] = {}, {}, {}
    metrics['conviction_zscore'] = {}
    metrics['qualified_zscore'] = {} 
    for p in MA_PERIODS:
        series_wc = metrics['weighted_counts'][p]
        metrics['z_scores'][p] = calculate_zscore(series_wc, Z_SCORE_WINDOW)
        metrics['rocs'][p] = series_wc.diff()
        metrics['accelerations'][p] = series_wc.diff().diff()
        
        conviction_index = (series_wc / (100 if not is_dynamic_weights else 1)) * metrics['weighted_distance_indices'][p]
        metrics['conviction_zscore'][p] = calculate_zscore(conviction_index, Z_SCORE_WINDOW)

        series_qc = metrics['qualified_counts'][p]
        metrics['qualified_zscore'][p] = calculate_zscore(series_qc, Z_SCORE_WINDOW) 

    return metrics

def display_charts(column, metrics, title_prefix, theme_colors, overlay_price_data, selected_charts, overlay_asset, is_dynamic_weights=False):
    column.header(title_prefix)
    summaries = {
        'Indicador de Divergência de Agressão': "Sinaliza quando o esforço do mercado (agressão) diverge do resultado no preço do ativo.",
        'Força Ponderada (Contagem)': "Confirma se a maioria do mercado apoia a direção do ativo.",
        'Força Qualificada (Filtro)': "Filtra o ruído e confirma se o movimento do ativo tem convicção.",
        'Z-Score da Força Qualificada': "Alerta para exaustão ou pontos de viragem no ativo quando atinge extremos.",
        'Velocidade e Aceleração': "Mede a 'explosão' de um movimento; um pico de velocidade confirma um breakout no ativo.",
        'Indicador de Clímax de Agressão': "Sinaliza a capitulação (golpe final) de uma das pontas.",
        'Indicador de Clímax de Rejeição': "Sinaliza a absorção e a resposta da ponta contrária após um clímax.",
        'Índice de Momentum Agregado': "Mostra a saúde da tendência; divergências com o preço do ativo sinalizam fraqueza.",
        'Z-Score da Convicção': "Identifica extremos de euforia/pânico (Contagem * Distância), ideal para reversões no ativo.",
        'Índice de Força de Volume (VFI)': "Valida um movimento no ativo com participação institucional (Distância * Volume)."
    }
    
    def create_fig_with_overlay(title):
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", height=250, margin=dict(t=50, b=20, l=20, r=40),
            title=dict(text=title, x=0.01),
            yaxis2=dict(title=overlay_asset, overlaying='y', side='right', showgrid=False, showticklabels=False, zeroline=False, color=theme_colors['overlay']),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    # --- Gráficos ---
    if 'Indicador de Divergência de Agressão' in selected_charts:
        column.markdown(f"<p style='font-size:12px; color:grey;'><b>{overlay_asset}:</b> {summaries['Indicador de Divergência de Agressão']}</p>", unsafe_allow_html=True)
        
        p_short = MA_PERIODS[0] if MA_PERIODS else None
        
        if p_short and not overlay_price_data.empty:
            asset_is_up = overlay_price_data['close'] > overlay_price_data['open']
            
            buyer_climax_is_high = metrics['buyer_climax_zscore'] > 1
            seller_climax_is_high = metrics['seller_climax_zscore'] > 1
            
            total_aggression = metrics['aggression_buyer'] + metrics['aggression_seller']
            total_aggression_safe = total_aggression.replace(0, np.nan)
            
            buyer_aggression_is_dominant = (metrics['aggression_buyer'] / total_aggression_safe) > 0.7
            seller_aggression_is_dominant = (metrics['aggression_seller'] / total_aggression_safe) > 0.7

            context_is_strong = metrics['qualified_zscore'][p_short] > 1
            context_is_weak = metrics['qualified_zscore'][p_short] < -1

            topo_divergence = buyer_climax_is_high & buyer_aggression_is_dominant & ~asset_is_up & context_is_strong
            topo_points = overlay_price_data[topo_divergence]
            
            fundo_divergence = seller_climax_is_high & seller_aggression_is_dominant & asset_is_up & context_is_weak
            fundo_points = overlay_price_data[fundo_divergence]

            fig_div = go.Figure()
            fig_div.add_trace(go.Candlestick(x=overlay_price_data.index, open=overlay_price_data['open'], high=overlay_price_data['high'], low=overlay_price_data['low'], close=overlay_price_data['close'], name=overlay_asset))
            fig_div.add_trace(go.Scatter(x=topo_points.index, y=topo_points['high'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Div. de Topo'))
            fig_div.add_trace(go.Scatter(x=fundo_points.index, y=fundo_points['low'], mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Div. de Fundo'))
            fig_div.update_layout(title='Indicador de Divergência de Agressão', height=300, margin=dict(t=30, b=10, l=10, r=10), template="plotly_dark", xaxis_rangeslider_visible=False)
            column.plotly_chart(fig_div, use_container_width=True, key=f"{title_prefix}_div_chart")
        else:
            column.warning("Insira pelo menos um período de MA para o Indicador de Divergência.")

    if 'Força Ponderada (Contagem)' in selected_charts:
        column.markdown(f"<p style='font-size:12px; color:grey;'><b>{overlay_asset}:</b> {summaries['Força Ponderada (Contagem)']}</p>", unsafe_allow_html=True)
        for i, (p, series) in enumerate(metrics['weighted_counts'].items()):
            fig = create_fig_with_overlay(f'Força Ponderada (Contagem EMA {p})')
            fig.add_trace(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, name='Força', mode="lines", fill="tozeroy", line_color=theme_colors['main'], opacity=0.7))
            fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name=overlay_asset, yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
            if not is_dynamic_weights: fig.update_layout(yaxis=dict(range=[0, 100]))
            column.plotly_chart(fig, use_container_width=True, key=f"{title_prefix}_wc_{p}_{i}")

    if 'Força Qualificada (Filtro)' in selected_charts:
        column.markdown(f"<p style='font-size:12px; color:grey;'><b>{overlay_asset}:</b> {summaries['Força Qualificada (Filtro)']}</p>", unsafe_allow_html=True)
        for i, (p, series) in enumerate(metrics['qualified_counts'].items()):
            fig = create_fig_with_overlay(f'Força Qualificada (Filtro EMA {p})')
            fig.add_trace(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, name='Qualificada', mode="lines", fill="tozeroy", line_color=theme_colors['qualified']))
            fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name=overlay_asset, yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
            if not is_dynamic_weights: fig.update_layout(yaxis=dict(range=[0, 100]))
            column.plotly_chart(fig, use_container_width=True, key=f"{title_prefix}_qc_{p}_{i}")
    
    if 'Z-Score da Força Qualificada' in selected_charts:
        column.markdown(f"<p style='font-size:12px; color:grey;'><b>{overlay_asset}:</b> {summaries['Z-Score da Força Qualificada']}</p>", unsafe_allow_html=True)
        for i, (p, series) in enumerate(metrics['qualified_zscore'].items()):
            fig = create_fig_with_overlay(f'Z-Score da Força Qualificada (EMA {p})')
            fig.add_trace(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, name='Z-Score', line=dict(color=theme_colors['accent'])))
            fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name=overlay_asset, yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
            fig.add_hline(y=2, line_dash="dot", line_color="white", opacity=0.5); fig.add_hline(y=-2, line_dash="dot", line_color="white", opacity=0.5)
            fig.update_layout(yaxis=dict(range=[-3.5, 3.5]))
            column.plotly_chart(fig, use_container_width=True, key=f"{title_prefix}_zqc_{p}_{i}")

    if 'Velocidade e Aceleração' in selected_charts and MA_PERIODS:
        column.markdown(f"<p style='font-size:12px; color:grey;'><b>{overlay_asset}:</b> {summaries['Velocidade e Aceleração']}</p>", unsafe_allow_html=True)
        p_short = MA_PERIODS[0]
        roc_series = metrics['rocs'][p_short].tail(NUM_CANDLES_DISPLAY)
        fig_roc = create_fig_with_overlay(f'Velocidade (ROC EMA {p_short})')
        fig_roc.add_trace(go.Bar(x=roc_series.index, y=roc_series.values, name='ROC', marker_color=['green' if v >= 0 else 'red' for v in roc_series.values]))
        fig_roc.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name=overlay_asset, yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
        fig_roc.update_layout(height=200)
        column.plotly_chart(fig_roc, use_container_width=True, key=f"{title_prefix}_roc_chart")
        
    if 'Indicador de Clímax de Agressão' in selected_charts:
        column.markdown(f"<p style='font-size:12px; color:grey;'><b>{overlay_asset}:</b> {summaries['Indicador de Clímax de Agressão']}</p>", unsafe_allow_html=True)
        buyer_series = metrics['buyer_climax_zscore'].tail(NUM_CANDLES_DISPLAY).clip(lower=0)
        seller_series = metrics['seller_climax_zscore'].tail(NUM_CANDLES_DISPLAY).clip(lower=0)
        fig_climax = create_fig_with_overlay('Indicador de Clímax de Agressão')
        fig_climax.add_trace(go.Bar(x=buyer_series.index, y=buyer_series.values, name='Clímax Comprador', marker_color='green'))
        fig_climax.add_trace(go.Bar(x=seller_series.index, y=seller_series.values, name='Clímax Vendedor', marker_color='red'))
        fig_climax.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name=overlay_asset, yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
        fig_climax.add_hline(y=3, line_dash="dot", line_color="white", annotation_text="Limiar de Clímax (+3σ)")
        fig_climax.update_layout(barmode='relative')
        column.plotly_chart(fig_climax, use_container_width=True, key=f"{title_prefix}_climax_agg_chart")

    if 'Indicador de Clímax de Rejeição' in selected_charts:
        column.markdown(f"<p style='font-size:12px; color:grey;'><b>{overlay_asset}:</b> {summaries['Indicador de Clímax de Rejeição']}</p>", unsafe_allow_html=True)
        buyer_series = metrics['rejection_buyer'].tail(NUM_CANDLES_DISPLAY)
        seller_series = metrics['rejection_seller'].tail(NUM_CANDLES_DISPLAY)
        fig_rej = create_fig_with_overlay('Indicador de Clímax de Rejeição')
        fig_rej.add_trace(go.Bar(x=buyer_series.index, y=buyer_series.values, name='Rejeição Compradora', marker_color='lime'))
        fig_rej.add_trace(go.Bar(x=seller_series.index, y=seller_series.values, name='Rejeição Vendedora', marker_color='pink'))
        fig_rej.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name=overlay_asset, yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
        fig_rej.update_layout(barmode='relative')
        column.plotly_chart(fig_rej, use_container_width=True, key=f"{title_prefix}_climax_rej_chart")

    if 'Índice de Momentum Agregado' in selected_charts:
        column.markdown(f"<p style='font-size:12px; color:grey;'><b>{overlay_asset}:</b> {summaries['Índice de Momentum Agregado']}</p>", unsafe_allow_html=True)
        series = metrics['aggregate_momentum_index'].tail(NUM_CANDLES_DISPLAY)
        fig_mom = create_fig_with_overlay('Índice de Momentum Agregado')
        fig_mom.add_trace(go.Scatter(x=series.index, y=series.values, name='Momentum', line=dict(color=theme_colors['momentum']), fill='tozeroy'))
        fig_mom.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name=overlay_asset, yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
        fig_mom.add_hline(y=0, line_dash="dash", line_color="grey")
        column.plotly_chart(fig_mom, use_container_width=True, key=f"{title_prefix}_momentum_chart")
    
    if 'Z-Score da Convicção' in selected_charts:
        column.markdown(f"<p style='font-size:12px; color:grey;'><b>{overlay_asset}:</b> {summaries['Z-Score da Convicção']}</p>", unsafe_allow_html=True)
        for i, (p, series) in enumerate(metrics['conviction_zscore'].items()):
            fig = create_fig_with_overlay(f'Z-Score da Convicção (EMA {p})')
            fig.add_trace(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, name='Convicção', line=dict(color=theme_colors['conviction_z'])))
            fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name=overlay_asset, yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
            fig.add_hline(y=2, line_dash="dot", line_color="white", opacity=0.5); fig.add_hline(y=-2, line_dash="dot", line_color="white", opacity=0.5)
            fig.update_layout(yaxis=dict(range=[-3.5, 3.5]))
            column.plotly_chart(fig, use_container_width=True, key=f"{title_prefix}_zc_{p}_{i}")
        
    if 'Índice de Força de Volume (VFI)' in selected_charts:
        column.markdown(f"<p style='font-size:12px; color:grey;'><b>{overlay_asset}:</b> {summaries['Índice de Força de Volume (VFI)']}</p>", unsafe_allow_html=True)
        for i, (p, series) in enumerate(metrics['volume_force_indices'].items()):
            fig = create_fig_with_overlay(f'Índice de Força de Volume (VFI EMA {p})')
            fig.add_trace(go.Scatter(x=series.tail(NUM_CANDLES_DISPLAY).index, y=series.tail(NUM_CANDLES_DISPLAY).values, name='VFI', mode="lines", line_color=theme_colors['vfi'], fill='tozeroy'))
            fig.add_trace(go.Scatter(x=overlay_price_data['close'].index, y=overlay_price_data['close'].values, name=overlay_asset, yaxis='y2', line=dict(color=theme_colors['overlay'], width=1.5, dash='dot')))
            fig.add_hline(y=0, line_dash="dash", line_color="grey")
            column.plotly_chart(fig, use_container_width=True, key=f"{title_prefix}_vfi_{p}_{i}")

# ===========================
# Lógica Principal da Aplicação
# ===========================
st.title("⚔️ Painel de Batalha de Amplitude")

# --- Indicador de Frescura dos Dados ---
placeholder = st.empty()

main_tab_placeholder, xauusd_tab_placeholder = st.tabs(["Painel de Batalha Principal", "🥇 Análise Específica XAUUSD"])

with main_tab_placeholder:
    st.header("Painel Principal (Ponderado por Liquidez)")
    
    candles_to_fetch_main = (max(MA_PERIODS) if MA_PERIODS else 200) + NUM_CANDLES_DISPLAY + max(Z_SCORE_WINDOW, MOMENTUM_Z_WINDOW, CLIMAX_Z_WINDOW, CORRELATION_WINDOW)
    combined_main = build_combined_data(ALL_UNIQUE_ASSETS, TIMEFRAME, candles_to_fetch_main)
    
    if combined_main.empty:
        st.error("Nenhum dado disponível para o Painel Principal.")
    else:
        now = datetime.now(TZ)
        last_candle_time = combined_main.index[-1]
        delay_minutes = (now - last_candle_time).total_seconds() / 60
        if delay_minutes < 2:
            placeholder.success(f"🟢 Dados FRESCOS (Atraso de {delay_minutes:.1f} min)")
        else:
            placeholder.warning(f"🟠 ATENÇÃO: Atraso nos dados de {delay_minutes:.1f} min")

        st.sidebar.header("Visualização (Principal)")
        selected_overlay_main = st.sidebar.selectbox('Ativo para Sobreposição', ['XAUUSD', 'EURUSD', 'GBPUSD'], key='overlay_main')
        selected_charts_main = st.sidebar.multiselect("Gráficos a Exibir", ALL_CHARTS_LIST, default=ALL_CHARTS_LIST, key='charts_main')
        
        overlay_price_data_main = pd.DataFrame()
        if f"{selected_overlay_main}_close" in combined_main.columns:
            overlay_price_data_main = combined_main[[f"{selected_overlay_main}_open", f"{selected_overlay_main}_high", f"{selected_overlay_main}_low", f"{selected_overlay_main}_close"]].tail(NUM_CANDLES_DISPLAY)
            overlay_price_data_main.columns = ['open', 'high', 'low', 'close']
        
        metrics_risk_off = calculate_breadth_metrics(RISK_OFF_ASSETS, combined_main)
        metrics_risk_on = calculate_breadth_metrics(RISK_ON_ASSETS, combined_main)
        
        risk_off_colors = {'main': '#E74C3C', 'accent': '#F1948A', 'momentum': '#D98880', 'qualified': '#FFA07A', 'conviction_z': '#F5B041', 'vfi': '#E67E22', 'overlay': 'rgba(255, 215, 0, 0.5)'}
        risk_on_colors = {'main': '#2ECC71', 'accent': '#ABEBC6', 'momentum': '#76D7C4', 'qualified': '#87CEEB', 'conviction_z': '#5DADE2', 'vfi': '#3498DB', 'overlay': 'rgba(255, 215, 0, 0.5)'}
        
        col1, col2 = st.columns(2)
        display_charts(col1, metrics_risk_off, "Risk-Off (Força do Dólar)", risk_off_colors, overlay_price_data_main, selected_charts_main, selected_overlay_main)
        display_charts(col2, metrics_risk_on, "Risk-On (Fraqueza do Dólar)", risk_on_colors, overlay_price_data_main, selected_charts_main, selected_overlay_main)

with xauusd_tab_placeholder:
    st.header("Índice de Confirmação para o Ouro (Ponderado por Correlação)")
    st.markdown("Esta análise mede se o comportamento de outros ativos do mercado apoia ou contradiz o movimento atual do Ouro. Os pesos são a correlação dinâmica de cada ativo com o XAUUSD.")
    
    xauusd_basket = list(set(ALL_UNIQUE_ASSETS) - {'XAUUSD', 'XAGUSD'})
    
    if 'combined_main' in locals() and not combined_main.empty and 'XAUUSD_close' in combined_main.columns:
        
        st.sidebar.header("Visualização (XAUUSD)")
        selected_charts_xauusd = st.sidebar.multiselect("Gráficos a Exibir", ALL_CHARTS_LIST, default=ALL_CHARTS_LIST, key='charts_xauusd')

        dynamic_weights = calculate_dynamic_correlation_weights(xauusd_basket, 'XAUUSD', combined_main, CORRELATION_WINDOW)
        
        metrics_xauusd_corr = calculate_breadth_metrics(dynamic_weights, combined_main, is_dynamic_weights=True)
        
        xauusd_price_data_tab2 = combined_main[[f"XAUUSD_open", f"XAUUSD_high", f"XAUUSD_low", f"XAUUSD_close"]].tail(NUM_CANDLES_DISPLAY)
        xauusd_price_data_tab2.columns = ['open', 'high', 'low', 'close']
        
        corr_colors = {'main': '#FFD700', 'accent': '#FFFACD', 'momentum': '#F0E68C', 'qualified': '#EEE8AA', 'conviction_z': '#FFECB3', 'vfi': '#FFC107', 'overlay': 'rgba(255, 255, 255, 0.6)'}

        display_charts(st, metrics_xauusd_corr, "Índice de Confirmação (Correlação com XAUUSD)", corr_colors, xauusd_price_data_tab2, selected_charts_xauusd, "XAUUSD", is_dynamic_weights=True)
    else:
        st.warning("Dados do XAUUSD não disponíveis para calcular a correlação. Verifique se o ativo está na cesta principal.")

st.caption("Feito com Streamlit • Dados via FinancialModelingPrep")

