import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from yahooquery import Ticker          # <--  yahooquery instead of yfinance
import plotly.graph_objs as go
import pandas as pd
import ta
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
server = app.server

# ─────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────
app.layout = dbc.Container([
    # Header
    dbc.NavbarSimple(
        brand="Stock Dashboard",
        color="dark brown",
        dark=True,
    ),

    # Symbol input
    dbc.Row([
        dbc.Col(
            dbc.InputGroup([
                dbc.Input(id='stock-input',
                          placeholder='Enter stock symbol',
                          value='AAPL',
                          debounce=False),
            ]),
            width=4,
        ),
    ], justify='center', className="my-3"),

    # Time-range selector
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Time Range:"),
            dcc.Dropdown(
                id='time-range',
                options=[
                    {'label': '6 months', 'value': '6mo'},
                    {'label': '1 year',   'value': '1y'},
                    {'label': '2 years',  'value': '2y'},
                    {'label': '5 years',  'value': '5y'},
                    {'label': 'All',      'value': 'max'}
                ],
                value='1y',
                clearable=False
            )
        ], width=4),
    ], justify='center', className="my-3"),

    # NEW -- Interval selector
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Interval:"),
            dcc.Dropdown(
                id='interval',
                options=[
                    {'label': 'Daily',   'value': '1d'},
                    {'label': 'Weekly',  'value': '1wk'},
                    {'label': 'Monthly', 'value': '1mo'},
                ],
                value='1d',
                clearable=False
            )
        ], width=4),
    ], justify='center', className="my-3"),

    # Analyze button
    dbc.Row([
        dbc.Col(
            dbc.Button("Analyze Stock",
                       id='analyze-button',
                       n_clicks=0,
                       color="primary"),
            width="auto"
        )
    ], justify="center", className="my-3"),

    # === Charts ===
    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='candlestick-chart'))), width=12)], className="mb-4"),
    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='sma-ema-chart'))), width=12)], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='support-resistance-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='rsi-chart'))),              width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='bollinger-bands-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='macd-chart'))),            width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='stochastic-oscillator-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='obv-chart'))),                     width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='atr-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='cci-chart'))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='mfi-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='cmf-chart'))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='fi-chart'))),  width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='fibonacci-retracement-chart'))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='ichimoku-cloud-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='vwap-chart'))),            width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='adl-chart'))),    width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='adx-di-chart'))), width=6),
    ], className="mb-4"),

    # Metrics explanation (unchanged)
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Explanation of Metrics", className="card-title"),
                    html.Ul([
                        html.Li("Candlestick Chart: Displays the open, high, low, and close prices of a security for a specific time period."),
                        html.Li("SMA (Simple Moving Average): The average stock price over a specific time period, smoothing price data."),
                        html.Li("EMA (Exponential Moving Average): A moving average that reacts more quickly to recent price changes."),
                        html.Li("Support & Resistance: Price levels where a stock tends to reverse direction."),
                        html.Li("RSI (Relative Strength Index): Measures the speed and change of price movements. Values over 70 indicate overbought, under 30 indicate oversold."),
                        html.Li("Bollinger Bands: Uses standard deviations to measure volatility and relative price levels."),
                        html.Li("MACD (Moving Average Convergence Divergence): Shows the relationship between two moving averages to detect momentum."),
                        html.Li("Stochastic Oscillator: Compares closing prices to their price range over time to indicate overbought/oversold conditions."),
                        html.Li("OBV (On-Balance Volume): A volume-based indicator predicting price movement based on cumulative volume."),
                        html.Li("ATR (Average True Range): Measures volatility by averaging the true range of stock prices over time."),
                        html.Li("CCI (Commodity Channel Index): Identifies cyclical trends by comparing current prices to historical averages."),
                        html.Li("MFI (Money Flow Index): Combines price and volume to indicate buying and selling pressure."),
                        html.Li("CMF (Chaikin Money Flow): Measures buying and selling pressure over time, indicating accumulation or distribution."),
                        html.Li("FI (Force Index): Combines price and volume to show the strength of buy/sell signals."),
                        html.Li("Fibonacci Retracement: Predicts potential reversal levels by using high and low price points."),
                        html.Li("Ichimoku Cloud: Defines support, resistance, trend direction, and momentum using various components."),
                        html.Li("VWAP (Volume Weighted Average Price): A benchmark that shows the average price weighted by volume."),
                        html.Li("ADL (Accumulation/Distribution Line): A cumulative indicator to assess whether a stock is being accumulated or distributed."),
                        html.Li("ADX & DI: Measures trend strength and direction using the Average Directional Index and Directional Indicators."),
                    ], className="text-muted")
                ])
            ),
            width=12
        )
    ], className="mb-4"),

    # Footer
    dbc.Row([
        dbc.Col(html.Footer("Stock Dashboard © 2024 By Salman",
                            className="text-center text-muted"))
    ], className="mt-4")
], fluid=True)

# ─────────────────────────────────────────────
#  Callback
# ─────────────────────────────────────────────
@app.callback(
    [Output('candlestick-chart',          'figure'),
     Output('sma-ema-chart',              'figure'),
     Output('support-resistance-chart',   'figure'),
     Output('rsi-chart',                  'figure'),
     Output('bollinger-bands-chart',      'figure'),
     Output('macd-chart',                 'figure'),
     Output('stochastic-oscillator-chart','figure'),
     Output('obv-chart',                  'figure'),
     Output('atr-chart',                  'figure'),
     Output('cci-chart',                  'figure'),
     Output('mfi-chart',                  'figure'),
     Output('cmf-chart',                  'figure'),
     Output('fi-chart',                   'figure'),
     Output('fibonacci-retracement-chart','figure'),
     Output('ichimoku-cloud-chart',       'figure'),
     Output('vwap-chart',                 'figure'),
     Output('adl-chart',                  'figure'),
     Output('adx-di-chart',               'figure')],
    Input('analyze-button', 'n_clicks'),
    State('stock-input',  'value'),
    State('time-range',   'value'),
    State('interval',     'value')               # <-- NEW
)
def update_graphs(n_clicks, ticker, time_range, interval):
    if not n_clicks:
        empty = go.Figure().update_layout(
            title="Click 'Analyze Stock' to display the analysis",
            template='plotly_dark')
        return (empty,) * 18

    # Auto-append '.SR' for Saudi tickers entered as digits
    if ticker.isdigit():
        ticker += '.SR'

    try:
        # Use yahoo_fin to fetch 1y (approx. 252 trading days) of data
        df = si.get_data(
            ticker,
            interval=interval,
            headers={'User-agent': 'Mozilla/5.0'}
        ).tail(252)

        if df is None or df.empty:
            raise ValueError(f"No data for {ticker}")
        
        df = df.reset_index()
        df.columns = [c.capitalize() for c in df.columns]
        df = df.query("Volume != 0")

        df['Date'] = df['Index'].dt.strftime('%Y-%m-%d')
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
        df.drop(columns=['Index'], inplace=True)

        # Capitalize column names to match yahooquery convention
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adjclose': 'adjclose',
            'Volume': 'volume'
        }, inplace=True)

    except Exception as e:
        print(f"Error fetching data: {e}")
        empty = go.Figure().update_layout(
            title=f"No data for {ticker} ({time_range}, {interval})",
            template='plotly_dark')
        return (empty,) * 18


    # === Indicators (all lower-case column names from yahooquery) ===
    df['SMA_20']  = df['close'].rolling(20).mean()
    df['SMA_50']  = df['close'].rolling(50).mean()
    df['SMA_200'] = df['close'].rolling(200).mean()

    df['EMA_20']  = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50']  = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()

    pivot = (df['high'] + df['low'] + df['close']) / 3
    df['Pivot_Point'] = pivot
    df['Support_1']   = 2 * pivot - df['high']
    df['Resistance_1']= 2 * pivot - df['low']
    df['Support_2']   = pivot - (df['high'] - df['low'])
    df['Resistance_2']= pivot + (df['high'] - df['low'])

    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=50).rsi()

    ma20 = df['close'].rolling(20).mean()
    std20= df['close'].rolling(20).std()
    df['Upper_band'] = ma20 + 2 * std20
    df['Lower_band'] = ma20 - 2 * std20

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['%K'] = stoch.stoch()
    df['%D'] = stoch.stoch_signal()

    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['VWAP']= (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['CCI'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()

    df['ADL'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'],
                                                df['close'], df['volume']).acc_dist_index()
    df['SMA_ADL_20']  = df['ADL'].rolling(20).mean()
    df['SMA_ADL_50']  = df['ADL'].rolling(50).mean()
    df['SMA_ADL_200'] = df['ADL'].rolling(200).mean()

    df['MFI'] = ta.volume.MFIIndicator(df['high'], df['low'],
                                       df['close'], df['volume']).money_flow_index()
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
                df['high'], df['low'], df['close'], df['volume'], window=20
               ).chaikin_money_flow()
    df['FI']  = ta.volume.ForceIndexIndicator(df['close'], df['volume']).force_index()

    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    df['ADX'] = adx.adx(); df['DI+'] = adx.adx_pos(); df['DI-'] = adx.adx_neg()

    # Fibonacci levels
    max_p, min_p = df['high'].max(), df['low'].min()
    diff = max_p - min_p
    fib = {
        '0.0%': max_p,
        '23.6%': max_p - 0.236 * diff,
        '38.2%': max_p - 0.382 * diff,
        '50.0%': max_p - 0.5   * diff,
        '61.8%': max_p - 0.618 * diff,
        '100.0%': min_p,
    }

    # Ichimoku
    df['Tenkan_sen']   = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['Kijun_sen']    = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['Senkou_span_a']= ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    df['Senkou_span_b']= ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    df['Chikou_span']  = df['close'].shift(-26)

    # ───────────── Figures (same as before) ─────────────
    candlestick_fig = go.Figure(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Candlestick'))
    candlestick_fig.add_trace(go.Bar(
        x=df.index, y=df['volume'], name='Volume',
        marker_color='rgba(52,152,219,0.5)', yaxis='y2'))
    candlestick_fig.update_layout(
        title=f'{ticker} Candlestick',
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        template='plotly_dark')

    sma_ema_fig = go.Figure()
    sma_ema_fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close'))
    for col in ['SMA_20','SMA_50','SMA_200','EMA_20','EMA_50','EMA_200']:
        sma_ema_fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    sma_ema_fig.update_layout(title=f'{ticker} SMA & EMA', template='plotly_dark')

    support_resistance_fig = go.Figure()
    for col, style in [('Pivot_Point','dash'),('Support_1','dot'),
                       ('Resistance_1','dot'),('Support_2','dot'),
                       ('Resistance_2','dot')]:
        support_resistance_fig.add_trace(
            go.Scatter(x=df.index, y=df[col], name=col.replace('_',' '),
                       line=dict(dash=style)))
    support_resistance_fig.update_layout(
        title=f'{ticker} Support & Resistance', template='plotly_dark')

    rsi_fig = go.Figure(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
    for lvl,color in [(70,'Red'),(30,'Green')]:
        rsi_fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1],
                          y0=lvl, y1=lvl, line=dict(color=color, dash='dash'))
    rsi_fig.update_layout(title=f'{ticker} RSI', template='plotly_dark')

    bollinger_bands_fig = go.Figure()
    for col in ['close','Upper_band','Lower_band']:
        bollinger_bands_fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    bollinger_bands_fig.update_layout(
        title=f'{ticker} Bollinger Bands', template='plotly_dark')

    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'))
    macd_fig.update_layout(title=f'{ticker} MACD', template='plotly_dark')

    stochastic_fig = go.Figure()
    stochastic_fig.add_trace(go.Scatter(x=df.index, y=df['%K'], name='%K'))
    stochastic_fig.add_trace(go.Scatter(x=df.index, y=df['%D'], name='%D'))
    stochastic_fig.update_layout(
        title=f'{ticker} Stochastic Oscillator', template='plotly_dark')

    obv_fig  = go.Figure(go.Scatter(x=df.index, y=df['OBV'], name='OBV'))
    obv_fig.update_layout(title=f'{ticker} OBV', template='plotly_dark')

    atr_fig  = go.Figure(go.Scatter(x=df.index, y=df['ATR'], name='ATR'))
    atr_fig.update_layout(title=f'{ticker} ATR', template='plotly_dark')

    cci_fig  = go.Figure(go.Scatter(x=df.index, y=df['CCI'], name='CCI'))
    for lvl,color in [(100,'Red'),(-100,'Green')]:
        cci_fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1],
                          y0=lvl, y1=lvl, line=dict(color=color, dash='dash'))
    cci_fig.update_layout(title=f'{ticker} CCI', template='plotly_dark')

    mfi_fig  = go.Figure(go.Scatter(x=df.index, y=df['MFI'], name='MFI'))
    for lvl,color in [(80,'Red'),(20,'Green')]:
        mfi_fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1],
                          y0=lvl, y1=lvl, line=dict(color=color, dash='dash'))
    mfi_fig.update_layout(title=f'{ticker} MFI', template='plotly_dark')

    cmf_fig  = go.Figure(go.Scatter(x=df.index, y=df['CMF'], name='CMF'))
    cmf_fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1],
                      y0=0, y1=0, line=dict(color='Red', dash='dash'))
    cmf_fig.update_layout(title=f'{ticker} CMF', template='plotly_dark')

    fi_fig   = go.Figure(go.Scatter(x=df.index, y=df['FI'], name='FI'))
    fi_fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1],
                     y0=0, y1=0, line=dict(color='Red', dash='dash'))
    fi_fig.update_layout(title=f'{ticker} Force Index', template='plotly_dark')

    fib_fig  = go.Figure(go.Scatter(x=df.index, y=df['close'], name='Close'))
    for label,price in fib.items():
        fib_fig.add_trace(go.Scatter(
            x=[df.index[0], df.index[-1]], y=[price, price],
            name=f'Fib {label}', line=dict(dash='dash')))
    fib_fig.update_layout(
        title=f'{ticker} Fibonacci Retracement', template='plotly_dark')

    ichimoku_fig = go.Figure(go.Scatter(x=df.index, y=df['close'], name='Close'))
    for col in ['Tenkan_sen','Kijun_sen','Senkou_span_a','Senkou_span_b','Chikou_span']:
        ichimoku_fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    ichimoku_fig.update_layout(
        title=f'{ticker} Ichimoku Cloud', template='plotly_dark')

    vwap_fig = go.Figure()
    vwap_fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close'))
    vwap_fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'],  name='VWAP'))
    vwap_fig.update_layout(title=f'{ticker} VWAP', template='plotly_dark')

    adl_fig = go.Figure()
    adl_fig.add_trace(go.Scatter(x=df.index, y=df['ADL'], name='ADL'))
    for col in ['SMA_ADL_20','SMA_ADL_50','SMA_ADL_200']:
        adl_fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    adl_fig.update_layout(title=f'{ticker} ADL', template='plotly_dark')

    adx_fig = go.Figure()
    adx_fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX'))
    adx_fig.add_trace(go.Scatter(x=df.index, y=df['DI-'], name='DI-'))
    adx_fig.add_trace(go.Scatter(x=df.index, y=df['DI+'], name='DI+'))
    adx_fig.update_layout(title=f'{ticker} ADX & DI', template='plotly_dark')

    # Return all 18 figs
    return (candlestick_fig, sma_ema_fig, support_resistance_fig, rsi_fig,
            bollinger_bands_fig, macd_fig, stochastic_fig, obv_fig,
            atr_fig, cci_fig, mfi_fig, cmf_fig, fi_fig, fib_fig,
            ichimoku_fig, vwap_fig, adl_fig, adx_fig)

# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run_server(debug=True)
