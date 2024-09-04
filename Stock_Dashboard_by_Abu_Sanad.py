
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import ta
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings("ignore")

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
server = app.server

# Define the layout of the app
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Stock Dashboard",
        brand_href="#",
        color="dark brown",  # Fixed typo here
        dark=True,
    ),
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.Input(id='stock-input', placeholder='Enter stock symbol', value='1303', debounce=False),
                dbc.InputGroupText(''),
            ]),
        ], width=4),
    ], justify='center', className="my-3"),
    
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Time Range:"),
            dcc.Dropdown(
                id='time-range',
                options=[
                    {'label': '6 months', 'value': '6mo'},
                    {'label': '1 year', 'value': '1y'},
                    {'label': '2 years', 'value': '2y'},
                    {'label': '3 years', 'value': '3mo'},
                    {'label': 'All', 'value': 'max'}
                ],
                value='1y',  # default value
                clearable=False
            )
        ], width=4),
    ], justify='center', className="my-3"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='candlestick-chart')
                ])
            ]),
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='sma-ema-chart')
                ])
            ]),
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='support-resistance-chart')
                ])
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='rsi-chart')
                ])
            ]),
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='bollinger-bands-chart')
                ])
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='macd-chart')
                ])
            ]),
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='stochastic-oscillator-chart')
                ])
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='obv-chart')
                ])
            ]),
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='atr-chart')
                ])
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='cci-chart')
                ])
            ]),
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='mfi-chart')
                ])
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='cmf-chart')
                ])
            ]),
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='fi-chart')
                ])
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='fibonacci-retracement-chart')
                ])
            ]),
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='ichimoku-cloud-chart')
                ])
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='vwap-chart')
                ])
            ]),
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='adl-chart')
                ])
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='adx-di-chart')
                ])
            ]),
        ], width=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Footer("Stock Dashboard © 2024 By Salman", className="text-center text-muted")
        ])
    ], className="mt-4")
], fluid=True)

# Define the callback to update the graphs
@app.callback(
    [Output('candlestick-chart', 'figure'),
     Output('sma-ema-chart', 'figure'),
     Output('support-resistance-chart', 'figure'),
     Output('rsi-chart', 'figure'),
     Output('bollinger-bands-chart', 'figure'),
     Output('macd-chart', 'figure'),
     Output('stochastic-oscillator-chart', 'figure'),
     Output('obv-chart', 'figure'),
     Output('atr-chart', 'figure'),
     Output('cci-chart', 'figure'),
     Output('mfi-chart', 'figure'),
     Output('cmf-chart', 'figure'),
     Output('fi-chart', 'figure'),
     Output('fibonacci-retracement-chart', 'figure'),
     Output('ichimoku-cloud-chart', 'figure'),
     Output('vwap-chart', 'figure'),
     Output('adl-chart', 'figure'),
     Output('adx-di-chart', 'figure')],
    [Input('stock-input', 'value'), Input('time-range', 'value')]  # Added time_range input
)
def update_graphs(ticker, time_range):
    # Check if the ticker is an integer, append '.SR' if it is
    if ticker.isdigit():
        ticker += '.SR'

    # Fetch stock data with the selected time range
    df = yf.download(ticker, period=time_range)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    pivot_point = (df['High'] + df['Low'] + df['Close']) / 3
    df['Pivot_Point'] = pivot_point
    df['Support_1'] = 2 * pivot_point - df['High']
    df['Resistance_1'] = 2 * pivot_point - df['Low']
    df['Support_2'] = pivot_point - (df['High'] - df['Low'])
    df['Resistance_2'] = pivot_point + (df['High'] - df['Low'])

    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=50).rsi()

    df['20_day_ma'] = df['Close'].rolling(window=20).mean().round(2)
    df['20_day_std'] = df['Close'].rolling(window=20).std().round(2)
    df['Upper_band'] = df['20_day_ma'] + (df['20_day_std']*2).round(2)
    df['Lower_band'] = df['20_day_ma'] - (df['20_day_std']*2).round(2)
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean().round(2)
    exp2 = df['Close'].ewm(span=26, adjust=False).mean().round(2)
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean().round(2)
    df['MACD'] = macd
    df['MACD_Signal'] = signal

    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['%K'] = stoch.stoch()
    df['%D'] = stoch.stoch_signal()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
    
    df['ADL'] = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
    df['SMA_ADL_20'] = df['ADL'].rolling(window=20).mean()
    df['SMA_ADL_50'] = df['ADL'].rolling(window=50).mean()
    df['SMA_ADL_200'] = df['ADL'].rolling(window=200).mean()

    df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=20).chaikin_money_flow()
    df['FI'] = ta.volume.ForceIndexIndicator(df['Close'], df['Volume']).force_index()

    # Calculate ADX and DI+ and DI-
    adx_indicator = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx_indicator.adx()
    df['DI+'] = adx_indicator.adx_pos()
    df['DI-'] = adx_indicator.adx_neg()

    # Calculate Fibonacci retracement levels
    max_price = df['High'].max()
    min_price = df['Low'].min()
    diff = max_price - min_price

    levels = {
        '0.0%': max_price,
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50.0%': max_price - 0.5 * diff,
        '61.8%': max_price - 0.618 * diff,
        '100.0%': min_price,
    }

    # Calculate Ichimoku Cloud components
    df['Tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['Kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['Senkou_span_a'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    df['Senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['Chikou_span'] = df['Close'].shift(-26)

    # Candlestick Chart
    candlestick_fig = go.Figure(go.Candlestick(x=df.index,
                                               open=df['Open'],
                                               high=df['High'],
                                               low=df['Low'],
                                               close=df['Close'],
                                               name='Candlestick'))
    candlestick_fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(52, 152, 219, 0.5)', yaxis='y2'))
    candlestick_fig.update_layout(
        title=f'{ticker} Candlestick Chart',
        yaxis_title='Stock Price',
        xaxis_title='Date',
        template='plotly_dark',
        yaxis2=dict(title='Volume', overlaying='y', side='right')
    )

    # SMA & EMA Chart
    sma_ema_fig = go.Figure()
    sma_ema_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    sma_ema_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'))
    sma_ema_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'))
    sma_ema_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='SMA 200'))
    sma_ema_fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20'))
    sma_ema_fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50'))
    sma_ema_fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='EMA 200'))
    sma_ema_fig.update_layout(
        title=f'{ticker} SMA & EMA',
        yaxis_title='Stock Price',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # Support & Resistance Chart
    support_resistance_fig = go.Figure()
    support_resistance_fig.add_trace(go.Scatter(x=df.index, y=df['Pivot_Point'], mode='lines', name='Pivot Point', line=dict(dash='dash')))
    support_resistance_fig.add_trace(go.Scatter(x=df.index, y=df['Support_1'], mode='lines', name='Support 1', line=dict(dash='dot')))
    support_resistance_fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_1'], mode='lines', name='Resistance 1', line=dict(dash='dot')))
    support_resistance_fig.add_trace(go.Scatter(x=df.index, y=df['Support_2'], mode='lines', name='Support 2', line=dict(dash='dot')))
    support_resistance_fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_2'], mode='lines', name='Resistance 2', line=dict(dash='dot')))
    support_resistance_fig.update_layout(
        title=f'{ticker} Support & Resistance',
        yaxis_title='Stock Price',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # RSI Chart
    rsi_fig = go.Figure(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    rsi_fig.add_shape(
        type='line',
        x0=df.index[0],
        y0=70,
        x1=df.index[-1],
        y1=70,
        line=dict(
            color='Red',
            width=2,
            dash='dash',
        )
    )
    rsi_fig.add_shape(
        type='line',
        x0=df.index[0],
        y0=30,
        x1=df.index[-1],
        y1=30,
        line=dict(
            color='Green',
            width=2,
            dash='dash',
        )
    )
    rsi_fig.update_layout(
        title=f'{ticker} RSI',
        yaxis_title='RSI',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # Bollinger Bands Chart
    bollinger_bands_fig = go.Figure()
    bollinger_bands_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    bollinger_bands_fig.add_trace(go.Scatter(x=df.index, y=df['Upper_band'], mode='lines', name='Upper Band'))
    bollinger_bands_fig.add_trace(go.Scatter(x=df.index, y=df['Lower_band'], mode='lines', name='Lower Band'))
    bollinger_bands_fig.update_layout(
        title=f'{ticker} Bollinger Bands',
        yaxis_title='Stock Price',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # MACD Chart
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='MACD Signal'))
    macd_fig.update_layout(
        title=f'{ticker} MACD',
        yaxis_title='MACD',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # Stochastic Oscillator Chart
    stochastic_oscillator_fig = go.Figure()
    stochastic_oscillator_fig.add_trace(go.Scatter(x=df.index, y=df['%K'], mode='lines', name='%K'))
    stochastic_oscillator_fig.add_trace(go.Scatter(x=df.index, y=df['%D'], mode='lines', name='%D'))
    stochastic_oscillator_fig.update_layout(
        title=f'{ticker} Stochastic Oscillator',
        yaxis_title='Stochastic Oscillator',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # On-Balance Volume (OBV) Chart
    obv_fig = go.Figure(go.Scatter(x=df.index, y=df['OBV'], mode='lines', name='OBV'))
    obv_fig.update_layout(
        title=f'{ticker} On-Balance Volume',
        yaxis_title='OBV',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # Average True Range (ATR) Chart
    atr_fig = go.Figure(go.Scatter(x=df.index, y=df['ATR'], mode='lines', name='ATR'))
    atr_fig.update_layout(
        title=f'{ticker} Average True Range',
        yaxis_title='ATR',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # Commodity Channel Index (CCI) Chart
    cci_fig = go.Figure(go.Scatter(x=df.index, y=df['CCI'], mode='lines', name='CCI'))
    cci_fig.add_shape(
        type='line',
        x0=df.index[0],
        y0=100,
        x1=df.index[-1],
        y1=100,
        line=dict(
            color='Red',
            width=2,
            dash='dash',
        )
    )
    cci_fig.add_shape(
        type='line',
        x0=df.index[0],
        y0=-100,
        x1=df.index[-1],
        y1=-100,
        line=dict(
            color='Green',
            width=2,
            dash='dash',
        )
    )
    cci_fig.update_layout(
        title=f'{ticker} Commodity Channel Index',
        yaxis_title='CCI',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # Money Flow Index (MFI) Chart
    mfi_fig = go.Figure(go.Scatter(x=df.index, y=df['MFI'], mode='lines', name='MFI'))
    mfi_fig.add_shape(
        type='line',
        x0=df.index[0],
        y0=80,
        x1=df.index[-1],
        y1=80,
        line=dict(
            color='Red',
            width=2,
            dash='dash',
        )
    )
    mfi_fig.add_shape(
        type='line',
        x0=df.index[0],
        y0=20,
        x1=df.index[-1],
        y1=20,
        line=dict(
            color='Green',
            width=2,
            dash='dash',
        )
    )
    mfi_fig.update_layout(
        title=f'{ticker} Money Flow Index',
        yaxis_title='MFI',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # Chaikin Money Flow (CMF) Chart
    cmf_fig = go.Figure(go.Scatter(x=df.index, y=df['CMF'], mode='lines', name='CMF'))
    cmf_fig.add_shape(
        type='line',
        x0=df.index[0],
        y0=0,
        x1=df.index[-1],
        y1=0,
        line=dict(
            color='Red',
            width=2,
            dash='dash',
        )
    )
    cmf_fig.update_layout(
        title=f'{ticker} Chaikin Money Flow',
        yaxis_title='CMF',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # Force Index (FI) Chart
    fi_fig = go.Figure(go.Scatter(x=df.index, y=df['FI'], mode='lines', name='FI'))
    fi_fig.add_shape(
        type='line',
        x0=df.index[0],
        y0=0,
        x1=df.index[-1],
        y1=0,
        line=dict(
            color='Red',
            width=2,
            dash='dash',
        )
    )
    fi_fig.update_layout(
        title=f'{ticker} Force Index',
        yaxis_title='FI',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # Fibonacci Retracement Chart
    fibonacci_retracement_fig = go.Figure(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    for level in levels:
        fibonacci_retracement_fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[levels[level], levels[level]],
                                     mode='lines', name=f'Fibonacci {level}', line=dict(dash='dash')))
    fibonacci_retracement_fig.update_layout(
        title=f'{ticker} Fibonacci Retracement',
        yaxis_title='Stock Price',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # Ichimoku Cloud Chart
    ichimoku_cloud_fig = go.Figure(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    ichimoku_cloud_fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan_sen'], mode='lines', name='Tenkan-sen'))
    ichimoku_cloud_fig.add_trace(go.Scatter(x=df.index, y=df['Kijun_sen'], mode='lines', name='Kijun-sen'))
    ichimoku_cloud_fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_span_a'], mode='lines', name='Senkou Span A'))
    ichimoku_cloud_fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_span_b'], mode='lines', name='Senkou Span B'))
    ichimoku_cloud_fig.add_trace(go.Scatter(x=df.index, y=df['Chikou_span'], mode='lines', name='Chikou Span'))
    ichimoku_cloud_fig.update_layout(
        title=f'{ticker} Ichimoku Cloud',
        yaxis_title='Stock Price',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # VWAP Chart
    vwap_fig = go.Figure(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    vwap_fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP'))
    vwap_fig.update_layout(
        title=f'{ticker} VWAP',
        yaxis_title='Stock Price',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # ADL & SMA ADL Chart
    adl_fig = go.Figure()
    adl_fig.add_trace(go.Scatter(x=df.index, y=df['ADL'], mode='lines', name='ADL'))
    adl_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_ADL_20'], mode='lines', name='SMA ADL 20'))
    adl_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_ADL_50'], mode='lines', name='SMA ADL 50'))
    adl_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_ADL_200'], mode='lines', name='SMA ADL 200'))
    adl_fig.update_layout(
        title=f'{ticker} ADL & SMA ADL',
        yaxis_title='ADL',
        xaxis_title='Date',
        template='plotly_dark'
    )

    # ADX & DI Chart
    adx_di_fig = go.Figure()
    adx_di_fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX'))
    adx_di_fig.add_trace(go.Scatter(x=df.index, y=df['DI-'], mode='lines', name='DI-'))
    adx_di_fig.add_trace(go.Scatter(x=df.index, y=df['DI+'], mode='lines', name='DI+'))
    adx_di_fig.update_layout(
        title=f'{ticker} ADX & DI',
        yaxis_title='Indicator',
        xaxis_title='Date',
        template='plotly_dark'
    )

    return (candlestick_fig, sma_ema_fig, support_resistance_fig, rsi_fig, bollinger_bands_fig, macd_fig, stochastic_oscillator_fig,
            obv_fig, atr_fig, cci_fig, mfi_fig, cmf_fig, fi_fig, fibonacci_retracement_fig, ichimoku_cloud_fig, vwap_fig, adl_fig, adx_di_fig)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
