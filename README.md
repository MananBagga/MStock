# Indian Stock Price Prediction App

This is a Streamlit-based web application for predicting Indian stock prices using multiple machine learning models.

## Features

- Real-time stock data fetching using yfinance
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Three prediction models:
  - Prophet
  - Random Forest
  - LSTM Neural Network
- Interactive charts using Plotly
- Technical analysis insights
- Dark theme UI

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the app using:
```bash
streamlit run app.py
```

## Input

- Enter an Indian stock symbol with .NS suffix (e.g., RELIANCE.NS, TCS.NS)
- Select the time period for analysis

## Outputs

1. Overview Tab:
   - Candlestick chart
   - Current price and key statistics

2. Technical Indicators Tab:
   - SMA (Simple Moving Average)
   - RSI (Relative Strength Index)
   - Bollinger Bands

3. Model Comparison Tab:
   - 30-day predictions from three models
   - Performance metrics

4. Analysis Tab:
   - Technical analysis insights
   - Current technical metrics

## Dependencies

- streamlit
- yfinance
- pandas
- numpy
- plotly
- scikit-learn
- prophet
- tensorflow
- pandas-ta