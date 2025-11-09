import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
import pandas_ta as ta

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Page config
st.set_page_config(
    page_title="Indian Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("Indian Stock Price Prediction App")

# Sidebar
st.sidebar.header("Stock Selection")

# Add .NS to ticker for NSE stocks
nse_stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Larsen & Toubro": "LT.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
}

ticker = st.sidebar.selectbox("Select Stock", list(nse_stocks.keys()))
ticker = nse_stocks[ticker]

# ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS, TCS.NS)", "RELIANCE.NS")
period = st.sidebar.selectbox("Select Time Period", ["1y", "2y", "5y"], index=0)

# Fetch data
@st.cache_data
def fetch_data(ticker, period):
    try:
        data = yf.download(ticker, period=period, auto_adjust=True)
        if data.empty:
            st.error("No data found for this ticker. Please check the symbol.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Safe conversion to float
def safe_float(value):
    try:
        if pd.isna(value):
            return 0.0
        if isinstance(value, (pd.Series, pd.DataFrame)):
            if value.empty:
                return 0.0
            return float(value.iloc[0])
        if value is None:
            return 0.0
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return 0.0

# Calculate technical indicators
def calculate_indicators(df):
    df = df.copy()

    # Ensure Close is numeric and filled
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')

    # SMAs / EMA
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI (safe fallback)
    try:
        df['RSI'] = ta.rsi(df['Close'], length=14)
    except:
        df['RSI'] = 50  # neutral fallback

    # MACD (robust handling)
    try:
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
    except:
        df['MACD'] = 0
        df['MACD_signal'] = 0

    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + (2 * df['BB_std'])
    df['BB_lower'] = df['BB_middle'] - (2 * df['BB_std'])

    # Fill remaining NaN AFTER indicator calc
    df = df.fillna(0)

    return df


# Model training functions
def train_prophet(df):
    try:
        df_prophet = df[['Close']].reset_index()
        df_prophet.columns = ['ds', 'y']
        
        # Convert y values to float
        df_prophet['y'] = df_prophet['y'].astype(float)
        
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        
        future_dates = model.make_future_dataframe(periods=30)
        forecast = model.predict(future_dates)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        st.error(f"Prophet model error: {str(e)}")
        return None

def train_random_forest(df):
    try:
        df['Target'] = df['Close'].shift(-30)
        df['Date_num'] = (df.index - df.index[0]).days
        
        features = ['Date_num', 'Open', 'High', 'Low', 'Close', 'Volume']
        X = df[features][:-30].astype(float)
        y = df['Target'].dropna().astype(float)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        future_dates = pd.DataFrame(index=pd.date_range(start=df.index[-1], periods=31)[1:])
        future_dates['Date_num'] = (future_dates.index - df.index[0]).days
        future_dates['Open'] = safe_float(df['Open'].iloc[-1])
        future_dates['High'] = safe_float(df['High'].iloc[-1])
        future_dates['Low'] = safe_float(df['Low'].iloc[-1])
        future_dates['Close'] = safe_float(df['Close'].iloc[-1])
        future_dates['Volume'] = safe_float(df['Volume'].iloc[-1])
        
        predictions = model.predict(future_dates[features].astype(float))
        return future_dates.index, predictions
    except Exception as e:
        st.error(f"Random Forest model error: {str(e)}")
        return None, None

def train_lstm(df):
    try:
        # Convert Close prices to float and reshape
        close_data = df[['Close']].astype(float)
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)
        
        X = []
        y = []
        for i in range(60, len(scaled_data)-30):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i+30, 0])
        X, y = np.array(X), np.array(y)
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        last_60_days = scaled_data[-60:]
        X_pred = last_60_days.reshape(1, 60, 1)
        pred_scaled = model.predict(X_pred)
        pred = scaler.inverse_transform(np.repeat(pred_scaled, scaled_data.shape[1], axis=1))[0, 0]
        
        return df.index[-1] + timedelta(days=30), float(pred)
    except Exception as e:
        st.error(f"LSTM model error: {str(e)}")
        return None, None

# Main app
data = fetch_data(ticker, period)

# Train models BEFORE tabs
with st.spinner("Training models..."):
    # Prophet
    forecast = train_prophet(data)
    prophet_pred = forecast['yhat'].iloc[-1] if forecast is not None else None

    # Random Forest
    rf_dates, rf_pred = train_random_forest(data)
    rf_prediction = rf_pred[-1] if rf_pred is not None else None

    # LSTM
    lstm_date, lstm_pred = train_lstm(data)


if data is not None:
    # ✅ Flatten multi-index columns (if any)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = calculate_indicators(data)

    required_cols = ["SMA20", "SMA50", "EMA20", "RSI", "BB_upper", "BB_lower"]
    missing = [c for c in required_cols if c not in data.columns]

    if missing:
        st.error(f"Indicator calculation failed. Missing: {missing}")
        st.stop()

    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Technical Indicators", "Model Comparison", "Analysis"])
    
    with tab1:
        st.subheader("Price Overview")
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                            open=data['Open'],
                                            high=data['High'],
                                            low=data['Low'],
                                            close=data['Close'])])
        fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, width="stretch")
        
        # Basic Stats
        col1, col2, col3, col4 = st.columns(4)

        st.write("### Forecast Chart Comparison")

        fig_pred = go.Figure()

        # Actual Historical Close Price
        fig_pred.add_trace(go.Scatter(
            x=data.index, y=data['Close'], mode='lines', name='Actual Price'
        ))

        # Prophet Forecast Curve
        if forecast is not None:
            fig_pred.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prophet Forecast'
            ))

        # Random Forest Future Curve
        if rf_dates is not None and rf_pred is not None:
            fig_pred.add_trace(go.Scatter(
                x=rf_dates, y=rf_pred, mode='lines', name='Random Forest Forecast'
            ))

        # LSTM Single Future Point
        if lstm_pred is not None and lstm_date is not None:
            fig_pred.add_trace(go.Scatter(
                x=[lstm_date], y=[lstm_pred], mode='markers+text',
                text=["LSTM (30d)"], textposition="top center",
                name='LSTM Prediction', marker=dict(size=12)
            ))

        fig_pred.update_layout(title="Predicted Future Prices vs Actual",
                            xaxis_title="Date", yaxis_title="Price")

        st.plotly_chart(fig_pred, use_container_width=True)



        if len(data) > 0:
            current_price = safe_float(data['Close'].iloc[-1])
            open_price = safe_float(data['Open'].iloc[-1])
            high_price = safe_float(data['High'].max())
            low_price = safe_float(data['Low'].min())
            
            col1.metric("Current Price", f"₹{current_price:.2f}")
            col2.metric("Day Change", f"₹{(current_price - open_price):.2f}")
            col3.metric("52W High", f"₹{high_price:.2f}")
            col4.metric("52W Low", f"₹{low_price:.2f}")
        else:
            col1.metric("Current Price", "N/A")
            col2.metric("Day Change", "N/A")
            col3.metric("52W High", "N/A")
            col4.metric("52W Low", "N/A")
    
    with tab2:
        st.subheader("Technical Indicators")
        
        # SMA Plot
        fig_sma = go.Figure()
        fig_sma.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Price", mode='lines'))
        fig_sma.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name="SMA20", mode='lines'))
        fig_sma.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name="SMA50", mode='lines'))
        fig_sma.update_layout(title="Simple Moving Averages")
        st.plotly_chart(fig_sma, width="stretch")
        
        # RSI Plot
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", mode='lines'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(title="Relative Strength Index (RSI)")
        st.plotly_chart(fig_rsi, width="stretch")
        
        # Bollinger Bands
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Price", mode='lines'))
        fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name="Upper Band", mode='lines'))
        fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_middle'], name="Middle Band", mode='lines'))
        fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name="Lower Band", mode='lines'))
        fig_bb.update_layout(title="Bollinger Bands")
        st.plotly_chart(fig_bb, width="stretch")
    
    with tab3:
        st.subheader("Model Predictions")
        
        with st.spinner('Training models...'):
            # Prophet
            try:
                forecast = train_prophet(data)
                prophet_pred = forecast['yhat'].iloc[-1]
            except Exception as e:
                st.error(f"Prophet model error: {e}")
                prophet_pred = None
            
            # Random Forest
            try:
                rf_dates, rf_pred = train_random_forest(data)
                rf_prediction = rf_pred[-1]
            except Exception as e:
                st.error(f"Random Forest model error: {e}")
                rf_prediction = None
            
            # LSTM
            try:
                lstm_date, lstm_pred = train_lstm(data)
            except Exception as e:
                st.error(f"LSTM model error: {e}")
                lstm_pred = None
        
        # Display predictions
        col1, col2, col3 = st.columns(3)
        
        if prophet_pred is not None:
            col1.metric("Prophet Prediction (30 days)", f"₹{prophet_pred:.2f}")
        
        if rf_prediction is not None:
            col2.metric("Random Forest Prediction (30 days)", f"₹{rf_prediction:.2f}")
        
        if lstm_pred is not None:
            col3.metric("LSTM Prediction (30 days)", f"₹{lstm_pred:.2f}")
    
    with tab4:
        st.subheader("Technical Analysis")
        
        # Generate insights
        if len(data) > 0:
            try:
                # Safely get the latest values
                current_price = safe_float(data['Close'].iloc[-1])
                sma20 = safe_float(data['SMA20'].iloc[-1])
                sma50 = safe_float(data['SMA50'].iloc[-1])
                rsi = safe_float(data['RSI'].iloc[-1])
                
                analysis = []
                
                # Only add trend analysis if we have valid SMAs
                if sma20 > 0 and sma50 > 0:
                    if sma20 > sma50:
                        analysis.append("The short-term trend is upward (SMA20 is above SMA50).")
                    else:
                        analysis.append("The short-term trend is downward (SMA20 is below SMA50).")

                else:
                    analysis.append("⚠️ Insufficient data for trend analysis")
                
                # Only add RSI analysis if we have valid RSI
                if 0 <= rsi <= 100:
                    if rsi > 70:
                        analysis.append("The stock appears overbought (RSI > 70). This may suggest a pullback.")
                    elif rsi < 30:
                        analysis.append("The stock appears oversold (RSI < 30). This may indicate a potential rebound.")
                    else:
                        analysis.append("The RSI indicates neutral momentum.")


                else:
                    analysis.append("⚠️ RSI data unavailable")
                
                for insight in analysis:
                    st.write(insight)
                
                # Current Metrics with validation
                st.write("### Current Technical Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['RSI', 'SMA20', 'SMA50', 'Current Price'],
                    'Value': [
                        f"{rsi:.2f}" if 0 <= rsi <= 100 else "N/A",
                        f"₹{sma20:.2f}" if sma20 > 0 else "N/A",
                        f"₹{sma50:.2f}" if sma50 > 0 else "N/A",
                        f"₹{current_price:.2f}" if current_price > 0 else "N/A"
                    ]
                })
            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")
                st.warning("Unable to generate complete analysis due to data issues.")
        else:
            st.warning("No data available for analysis.")
        st.table(metrics_df)