# 📁 app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from xgboost import XGBClassifier
from joblib import dump, load
import os
from alpaca_trade_api.rest import REST, TimeInForce
from sklearn.metrics import accuracy_score

# Load or train model
MODEL_FILE = 'xgb_model.joblib'

# Alpaca API (replace with your actual keys)
from config import API_KEY, SECRET_KEY, BASE_URL
api = REST(API_KEY, SECRET_KEY, BASE_URL)

def get_data(stock, period='60d', interval='15m'):
    df = yf.download(stock, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

def add_features(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['ema'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['sma_10'] = ta.trend.SMAIndicator(df['Close'], window=10).sma_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['signal'] = (df['sma_10'] > df['sma_50']).astype(int)  # Strategy: SMA crossover
    df.dropna(inplace=True)
    return df

def label_data(df):
    df['target'] = df['signal']
    return df

def train_model(df):
    X = df[['rsi', 'macd', 'ema', 'sma_10', 'sma_50']]
    y = df['target']
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    dump(model, MODEL_FILE)
    return model

def predict_signal(df, model):
    latest = df.iloc[-1][['rsi', 'macd', 'ema', 'sma_10', 'sma_50']].values.reshape(1, -1)
    return model.predict(latest)[0]

def backtest_model(df, model):
    df = df.copy()
    X = df[['rsi', 'macd', 'ema', 'sma_10', 'sma_50']]
    y = df['target']
    df['prediction'] = model.predict(X)
    accuracy = accuracy_score(y, df['prediction'])
    return accuracy, df

def place_order(stock, side, qty=1, stop_loss_pct=None, take_profit_pct=None):
    try:
        current_price = yf.Ticker(stock).history(period='1d')['Close'].iloc[-1]
        stop_loss_price = None
        take_profit_price = None

        if stop_loss_pct:
            stop_loss_price = round(current_price * (1 - stop_loss_pct / 100), 2) if side == 'buy' else round(current_price * (1 + stop_loss_pct / 100), 2)

        if take_profit_pct:
            take_profit_price = round(current_price * (1 + take_profit_pct / 100), 2) if side == 'buy' else round(current_price * (1 - take_profit_pct / 100), 2)

        order = api.submit_order(
            symbol=stock,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc',
            order_class='bracket',
            stop_loss={'stop_price': stop_loss_price} if stop_loss_price else None,
            take_profit={'limit_price': take_profit_price} if take_profit_price else None
        )
        st.success(f"{side.upper()} order placed for {stock} with stop-loss at {stop_loss_price} and take-profit at {take_profit_price}")
    except Exception as e:
        st.error(f"Order failed: {e}")

def trade_job():
    stock = "TSLA"  # Replace with dynamic input if needed
    qty = 1
    stop_loss_pct = 3.0
    take_profit_pct = 6.0

    df = get_data(stock)
    df = add_features(df)
    df = label_data(df)

    if os.path.exists(MODEL_FILE):
        model = load(MODEL_FILE)
    else:
        model = train_model(df)

    signal = predict_signal(df, model)
    accuracy, backtest_df = backtest_model(df, model)

    if signal == 1:
        place_order(stock, 'buy', qty, stop_loss_pct, take_profit_pct)
    else:
        place_order(stock, 'sell', qty, stop_loss_pct, take_profit_pct)

    st.write(f"Backtest Accuracy: {round(accuracy * 100, 2)}%")

# Setup scheduler to run every 15 minutes
scheduler = BackgroundScheduler()
scheduler.add_job(trade_job, 'interval', minutes=15)
scheduler.start()

# Streamlit UI
st.title("📈 AI Stock Day Trading Dashboard (XGBoost Model)")

# Display current positions and PnL
display_positions()
display_pnl()
