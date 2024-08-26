# Plotting Price Levels, Supports, and Resistances

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import ta
import warnings
warnings.filterwarnings("ignore")

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the app
app.layout = html.Div([
    html.H1("Stock Dashboard", style={'textAlign': 'center'}),
    dcc.Input(id='stock-input', type='text', placeholder='Enter stock symbol', value='AAPL', debounce=True),
    dcc.Dropdown(
        id='chart-type-dropdown',
        options=[
            {'label': 'Candlestick Chart', 'value': 'candlestick'},
            {'label': 'SMA & EMA', 'value': 'sma-ema'},
            {'label': 'Support & Resistance', 'value': 'support-resistance'},
            {'label': 'RSI', 'value': 'rsi'},
            {'label': 'Bollinger Bands', 'value': 'bollinger-bands'},
            {'label': 'MACD', 'value': 'macd'},
            {'label': 'Stochastic Oscillator', 'value': 'stochastic oscillator'},
            {'label': 'On-Balance Volume', 'value': 'OBV'},
            {'label': 'Average True Range', 'value': 'ATR'},
            {'label': 'Commodity Channel Index', 'value': 'CCI'},
            {'label': 'Money Flow Index', 'value': 'MFI'},
            {'label': 'Chaikin Money Flow', 'value': 'CMF'},
            {'label': 'Force Index', 'value': 'FI'},
            {'label': 'Fibonacci Retracement', 'value': 'fibonacci-retracement'},
            {'label': 'Ichimoku Cloud', 'value': 'ichimoku-cloud'},
            {'label': 'VWAP', 'value': 'vwap'},
            {'label': 'ADX & DI', 'value': 'adx-di'},
            {'label': 'ADL & SMA ADL', 'value': 'adl'}
        ],
        value='candlestick',
        style={'margin-top': '10px'}
    ),
    dcc.Graph(id='stock-graph'),
])

# Define the callback to update the graph
@app.callback(
    Output('stock-graph', 'figure'),
    [Input('stock-input', 'value'), Input('chart-type-dropdown', 'value')]
)
def update_graph(ticker, graph_type):
    # Check if the ticker is an integer, append '.SR' if it is
    if ticker.isdigit():
        ticker += '.SR'

    df = yf.download(ticker)
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

    fig = go.Figure()

    if graph_type == 'candlestick':
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name='Candlestick'))
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(52, 152, 219, 0.5)', yaxis='y2'))
        fig.update_layout(
            title=f'{ticker} Candlestick Chart',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            template='plotly_dark',
            yaxis2=dict(title='Volume', overlaying='y', side='right')
        )
    elif graph_type == 'sma-ema':
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='SMA 200'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='EMA 200'))
        fig.update_layout(
            title=f'{ticker} SMA & EMA',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'support-resistance':
        fig.add_trace(go.Scatter(x=df.index, y=df['Pivot_Point'], mode='lines', name='Pivot Point', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Support_1'], mode='lines', name='Support 1', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_1'], mode='lines', name='Resistance 1', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Support_2'], mode='lines', name='Support 2', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_2'], mode='lines', name='Resistance 2', line=dict(dash='dot')))
        fig.update_layout(
            title=f'{ticker} Support & Resistance',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'rsi':
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
        fig.add_shape(
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
        fig.add_shape(
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
        fig.update_layout(
            title=f'{ticker} RSI',
            yaxis_title='RSI',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'bollinger-bands':
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_band'], mode='lines', name='Upper Band'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_band'], mode='lines', name='Lower Band'))
        fig.update_layout(
            title=f'{ticker} Bollinger Bands',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'macd':
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='MACD Signal'))
        fig.update_layout(
            title=f'{ticker} MACD',
            yaxis_title='MACD',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'stochastic oscillator':
        fig.add_trace(go.Scatter(x=df.index, y=df['%K'], mode='lines', name='%K'))
        fig.add_trace(go.Scatter(x=df.index, y=df['%D'], mode='lines', name='%D'))
        fig.update_layout(
            title=f'{ticker} Stochastic Oscillator',
            yaxis_title='Stochastic Oscillator',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'OBV':
        fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], mode='lines', name='OBV'))
        fig.update_layout(
            title=f'{ticker} On-Balance Volume',
            yaxis_title='OBV',
            xaxis_title='Date',
            template='plotly_dark'
        )    
    elif graph_type == 'ATR':
        fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], mode='lines', name='ATR'))
        fig.update_layout(
            title=f'{ticker} Average True Range',
            yaxis_title='ATR',
            xaxis_title='Date',
            template='plotly_dark'
        )    
    elif graph_type == 'CCI':
        fig.add_trace(go.Scatter(x=df.index, y=df['CCI'], mode='lines', name='CCI'))
        fig.add_shape(
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
        fig.add_shape(
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
        fig.update_layout(
            title=f'{ticker} Commodity Channel Index',
            yaxis_title='CCI',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'MFI':
        fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], mode='lines', name='MFI'))
        fig.add_shape(
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
        fig.add_shape(
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
        fig.update_layout(
            title=f'{ticker} Money Flow Index',
            yaxis_title='MFI',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'CMF':
        fig.add_trace(go.Scatter(x=df.index, y=df['CMF'], mode='lines', name='CMF'))
        fig.add_shape(
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
        fig.update_layout(
            title=f'{ticker} Chaikin Money Flow',
            yaxis_title='CMF',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'FI':
        fig.add_trace(go.Scatter(x=df.index, y=df['FI'], mode='lines', name='FI'))
        fig.add_shape(
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
        fig.update_layout(
            title=f'{ticker} Force Index',
            yaxis_title='FI',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'fibonacci-retracement':
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
        for level in levels:
            fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[levels[level], levels[level]],
                                     mode='lines', name=f'Fibonacci {level}', line=dict(dash='dash')))
        fig.update_layout(
            title=f'{ticker} Fibonacci Retracement',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'ichimoku-cloud':
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan_sen'], mode='lines', name='Tenkan-sen'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Kijun_sen'], mode='lines', name='Kijun-sen'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_span_a'], mode='lines', name='Senkou Span A'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_span_b'], mode='lines', name='Senkou Span B'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Chikou_span'], mode='lines', name='Chikou Span'))

        fig.update_layout(
            title=f'{ticker} Ichimoku Cloud',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'vwap':
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP'))
        fig.update_layout(
            title=f'{ticker} VWAP',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'adl':
        fig.add_trace(go.Scatter(x=df.index, y=df['ADL'], mode='lines', name='ADL'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_ADL_20'], mode='lines', name='SMA ADL 20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_ADL_50'], mode='lines', name='SMA ADL 50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_ADL_200'], mode='lines', name='SMA ADL 200'))
        fig.update_layout(
            title=f'{ticker} ADL & SMA ADL',
            yaxis_title='ADL',
            xaxis_title='Date',
            template='plotly_dark'
        )
    elif graph_type == 'adx-di':
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX'))
        fig.add_trace(go.Scatter(x=df.index, y=df['DI+'], mode='lines', name='DI+'))
        fig.add_trace(go.Scatter(x=df.index, y=df['DI-'], mode='lines', name='DI-'))
        fig.update_layout(
            title=f'{ticker} ADX & DI',
            yaxis_title='Indicator',
            xaxis_title='Date',
            template='plotly_dark'
        )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
