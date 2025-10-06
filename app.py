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

st.sidebar.header("Parâmetros de Agressão")
ATR_PERIOD = st.sidebar.slider("Período do ATR", 10, 30, 14)
ENERGY_THRESHOLD = st.sidebar.slider("Limiar de 'Energia' da Vela", 1.0, 3.0, 1.5, 0.1)
CLIMAX_Z_WINDOW = st.sidebar.slider("Janela Z-Score do Clímax", 50, 200, 100)

st.sidebar.header("Parâmetros de Momentum")
MOMENTUM_PERIOD = st.sidebar.slider("Período do ROC (Momentum)", 10, 50, 21)
MOMENTUM_Z_WINDOW = st.sidebar.slider("Janela Z-Score do Momentum", 50, 200, 100)

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

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Calcula o Average True Range (ATR)."""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

def calculate_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calcula o Z-Score de uma série."""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std

# ===========================
# Lógica Principal e Cálculos
# ===========================
st.title("Market Breadth: Força do Dólar")
st.caption(f"Timeframe: {TIMEFRAME} | Última atualização: {datetime.now(TZ).strftime('%H:%M:%S')}")

candles_to_fetch = (max(MA_PERIODS) if MA_PERIODS else 200) + NUM_CANDLES_DISPLAY + Z_SCORE_WINDOW + MOMENTUM_Z_WINDOW
combined = build_combined_data(ASSETS, TIMEFRAME, candles_to_fetch)

if combined.empty:
    st.error("Nenhum dado disponível. Verifique a API ou os símbolos dos ativos.")
    st.stop()

# --- Cálculos de EMAs, Condições, Agressão e Momentum ---
weighted_counts = {p: pd.Series(0.0, index=combined.index) for p in MA_PERIODS}
aggression_buyer = pd.Series(0.0, index=combined.index)
aggression_seller = pd.Series(0.0, index=combined.index)
momentum_components = []

for s in ASSETS:
    close_col, open_col, high_col, low_col = f"{s}_close", f"{s}_open", f"{s}_high", f"{s}_low"
    if close_col not in combined.columns: continue
    
    weight = ASSET_WEIGHTS.get(s, 0)

    # Cálculo de Agressão
    atr = calculate_atr(combined[high_col], combined[low_col], combined[close_col], ATR_PERIOD)
    energy = (combined[high_col] - combined[low_col]) / atr
    is_high_energy = energy > ENERGY_THRESHOLD
    aggression_buyer += ((combined[close_col] > combined[open_col]) & is_high_energy).astype(int) * weight
    aggression_seller += ((combined[close_col] < combined[open_col]) & is_high_energy).astype(int) * weight

    # Cálculo de Amplitude Ponderada
    for p in MA_PERIODS:
        above_col = f"{s}_Above_EMA{p}"
        ema_val = combined[close_col].ewm(span=p, adjust=False).mean()
        combined[above_col] = combined[close_col] > ema_val
        weighted_counts[p] += combined[above_col].astype(int) * weight
    
    # Cálculo de Momentum Individual Normalizado
    roc = combined[close_col].pct_change(periods=MOMENTUM_PERIOD)
    normalized_momentum = calculate_zscore(roc, MOMENTUM_Z_WINDOW)
    momentum_components.append(normalized_momentum.rename(s) * weight)

# Cálculo do Índice de Momentum Agregado
aggregate_momentum_index = pd.concat(momentum_components, axis=1).sum(axis=1)

# Cálculo do Z-Score de Clímax de Agressão
buyer_climax_zscore = calculate_zscore(aggression_buyer, CLIMAX_Z_WINDOW)
seller_climax_zscore = calculate_zscore(aggression_seller, CLIMAX_Z_WINDOW)


# --- Cálculos de Z-Score, ROC e Aceleração ---
z_scores, rocs, accelerations = {}, {}, {}
for p in MA_PERIODS:
    series = weighted_counts[p]
    z_scores[p] = calculate_zscore(series, Z_SCORE_WINDOW)
    rocs[p] = series.diff()
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
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", fill="tozeroy", name=f'EMA {p} Ponderada'))
        fig.add_hline(y=10, line_dash="dot", line_color="green", annotation_text="Extremo Venda (10%)")
        fig.add_hline(y=90, line_dash="dot", line_color="red", annotation_text="Extremo Compra (90%)")
        fig.update_layout(title=f'Força Ponderada Acima da EMA {p}', yaxis=dict(title="Força (0-100)", range=[0, 100]), height=350, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- IMPLEMENTAÇÃO 2: Z-SCORE ---
    st.header("Implementação 2: Z-Score da Amplitude Ponderada")
    st.markdown("Mede quão estatisticamente incomum (em desvios padrão) está a leitura atual da amplitude.")
    for p in MA_PERIODS:
        series = z_scores[p].tail(NUM_CANDLES_DISPLAY)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=f'Z-Score EMA {p}', line=dict(color='orange')))
        fig.add_hline(y=-2, line_dash="dot", line_color="green", annotation_text="Extremo de Pânico (-2σ)")
        fig.add_hline(y=2, line_dash="dot", line_color="red", annotation_text="Extremo de Euforia (+2σ)")
        fig.update_layout(title=f'Z-Score da Amplitude (Base EMA {p})', yaxis=dict(title="Desvios Padrão (σ)", range=[-3.5, 3.5]), height=350, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- IMPLEMENTAÇÃO 3: VELOCIDADE E ACELERAÇÃO ---
    st.header("Implementação 3: Velocidade e Aceleração da Amplitude")
    if MA_PERIODS:
        p = MA_PERIODS[0] # Foco apenas na média mais curta
        st.markdown(f"Mede a dinâmica da Força Ponderada com base na **EMA {p}**. Picos indicam impulsos ou clímax. A desaceleração pode sinalizar exaustão.")
        
        roc_series = rocs[p].tail(NUM_CANDLES_DISPLAY)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Bar(x=roc_series.index, y=roc_series.values, name='ROC', marker_color=['green' if v >= 0 else 'red' for v in roc_series.values]))
        fig_roc.update_layout(title=f'Velocidade (ROC) da Amplitude', yaxis_title="Variação de Força", height=250, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_roc, use_container_width=True)

        accel_series = accelerations[p].tail(NUM_CANDLES_DISPLAY)
        fig_accel = go.Figure()
        fig_accel.add_trace(go.Bar(x=accel_series.index, y=accel_series.values, name='Aceleração', marker_color=['#1f77b4' if v >= 0 else '#ff7f0e' for v in accel_series.values]))
        fig_accel.update_layout(title=f'Aceleração da Amplitude', yaxis_title="Variação da Velocidade", height=250, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_accel, use_container_width=True)
    else:
        st.warning("Insira pelo menos um período de média móvel para calcular a Velocidade e Aceleração.")
    
    st.divider()

    # --- IMPLEMENTAÇÃO 4: AMPLITUDE DE AGRESSÃO ---
    st.header("Implementação 4: Amplitude de Agressão")
    st.markdown("Mede a força de movimentos explosivos no mercado, sinalizando o início de um impulso ou um clímax de exaustão.")
    st.latex(r''' \text{Energia da Vela} = \frac{(\text{Máxima} - \text{Mínima})}{\text{ATR}(\text{Período})} ''')
    
    agg_buyer_series = aggression_buyer.tail(NUM_CANDLES_DISPLAY)
    agg_seller_series = aggression_seller.tail(NUM_CANDLES_DISPLAY)
    fig_agg = go.Figure()
    fig_agg.add_trace(go.Bar(x=agg_buyer_series.index, y=agg_buyer_series.values, name='Agressão Compradora', marker_color='green'))
    fig_agg.add_trace(go.Bar(x=agg_seller_series.index, y=agg_seller_series.values, name='Agressão Vendedora', marker_color='red'))
    fig_agg.update_layout(barmode='relative', title='Clímax de Agressão Ponderado', yaxis_title="Força da Agressão (0-100)", height=350, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_agg, use_container_width=True)
    
    st.divider()

    # --- IMPLEMENTAÇÃO 5: ÍNDICE DE MOMENTUM AGREGADO ---
    st.header("Implementação 5: Índice de Momentum Agregado")
    st.markdown("Mede a saúde da tendência. Um índice positivo e a subir indica um momentum de mercado forte e generalizado.")
    st.latex(r''' \text{Índice Agregado} = \sum (\text{Momentum Normalizado}_i \times \text{Peso}_i) ''')

    momentum_series = aggregate_momentum_index.tail(NUM_CANDLES_DISPLAY)
    fig_mom = go.Figure()
    fig_mom.add_trace(go.Scatter(x=momentum_series.index, y=momentum_series.values, name='Momentum Agregado', line=dict(color='#636EFA'), fill='tozeroy'))
    fig_mom.add_hline(y=0, line_dash="dash", line_color="grey")
    fig_mom.update_layout(title='Índice de Momentum Agregado Ponderado', yaxis_title="Força do Momentum", height=350, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_mom, use_container_width=True)

    st.divider()
    
    # --- IMPLEMENTAÇÃO 6: INDICADOR DE CLÍMAX DE VOLUME ---
    st.header("Implementação 6: Indicador de Clímax de Agressão")
    st.markdown("Identifica quando um pico de agressão é estatisticamente extremo (acima de +2σ), sinalizando um provável clímax de exaustão ou iniciação.")
    st.latex(r''' \text{Clímax}_i = \frac{\text{Agressão}_i - \text{Média}(\text{Agressão}_i)}{\text{DesvioPadrão}(\text{Agressão}_i)} ''')
    
    buyer_climax_series = buyer_climax_zscore.tail(NUM_CANDLES_DISPLAY)
    seller_climax_series = seller_climax_zscore.tail(NUM_CANDLES_DISPLAY)
    fig_climax = go.Figure()
    fig_climax.add_trace(go.Bar(x=buyer_climax_series.index, y=buyer_climax_series.values, name='Clímax Comprador', marker_color='green'))
    fig_climax.add_trace(go.Bar(x=seller_climax_series.index, y=seller_climax_series.values, name='Clímax Vendedor', marker_color='red'))
    fig_climax.add_hline(y=2, line_dash="dot", line_color="black", annotation_text="Limiar de Clímax (+2σ)")
    fig_climax.update_layout(barmode='relative', title='Indicador de Clímax (Z-Score da Agressão)', yaxis_title="Desvios Padrão (σ)", height=350, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_climax, use_container_width=True)


with tab2:
    st.header("Resumo Atual")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Força Ponderada")
        latest_counts_weighted = {f"EMA {p}": f"{int(weighted_counts[p].iloc[-1])}%" for p in MA_PERIODS if not weighted_counts[p].empty}
        st.table(pd.DataFrame.from_dict(latest_counts_weighted, orient="index", columns=["Força Atual"]))
        
        st.subheader("Agressão Atual")
        latest_aggression = { "Agressão Compradora": f"{int(aggression_buyer.iloc[-1])}%", "Agressão Vendedora": f"{int(aggression_seller.iloc[-1])}%" }
        st.table(pd.DataFrame.from_dict(latest_aggression, orient="index", columns=["Força Atual"]))

    with col2:
        st.subheader("Z-Score")
        latest_z_scores = {f"EMA {p}": f"{z_scores[p].iloc[-1]:.2f} σ" for p in MA_PERIODS if not z_scores[p].empty}
        st.table(pd.DataFrame.from_dict(latest_z_scores, orient="index", columns=["Nível de Extremo Atual"]))

        st.subheader("Dinâmica Atual")
        if MA_PERIODS:
            p = MA_PERIODS[0]
            latest_dynamics = { f"Velocidade (ROC EMA {p})": f"{rocs[p].iloc[-1]:.2f}", f"Aceleração (EMA {p})": f"{accelerations[p].iloc[-1]:.2f}" }
            st.table(pd.DataFrame.from_dict(latest_dynamics, orient="index", columns=["Valor Atual"]))

    with col3:
        st.subheader("Saúde da Tendência")
        latest_momentum = {"Momentum Agregado": f"{aggregate_momentum_index.iloc[-1]:.2f}"}
        st.table(pd.DataFrame.from_dict(latest_momentum, orient='index', columns=['Valor Atual']))

        st.subheader("Clímax de Agressão")
        latest_climax = {
            "Clímax Comprador": f"{buyer_climax_zscore.iloc[-1]:.2f} σ",
            "Clímax Vendedor": f"{seller_climax_zscore.iloc[-1]:.2f} σ"
        }
        st.table(pd.DataFrame.from_dict(latest_climax, orient='index', columns=['Nível Atual']))

st.caption("Feito com Streamlit • Dados via FinancialModelingPrep")

