import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
#from datetime import datetime, timedelta
import os
import sys

import os
import sys


# ... continue with your script ...

def extract_features():

    return_period = 5
    
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['TSLA', 'NVDA', 'META', 'GOLD']
    ccy_tickers = ['DTWEXBGS', 'DEXCHUS']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']
    
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    #stk_data = web.DataReader(stk_tickers, 'yahoo')
    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)

    return_period = 5
    target_stock = 'NVDA' # Your chosen target
    
    # 1. Target Variable
    Y = np.log(stk_data.loc[:, ('Adj Close', target_stock)]).diff(return_period).shift(-return_period)
    Y.name = target_stock + '_Future'
    
    # 2. Correlated Assets
    X1 = np.log(stk_data.loc[:, ('Adj Close', [s for s in stk_tickers if s != target_stock])]).diff(return_period)
    X1.columns = X1.columns.droplevel() # Removes the 'Adj Close' multi-index level
    
    # 3. Macro Features
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)
    
    # 4. CUSTOM FEATURES - Explicitly naming them to avoid duplicates
    X_mom = np.log(stk_data.loc[:, ('Adj Close', target_stock)]).diff(return_period)
    X_mom.name = 'Momentum'
    
    X_range = (stk_data.loc[:, ('High', target_stock)] - stk_data.loc[:, ('Low', target_stock)]) / stk_data.loc[:, ('Close', target_stock)]
    X_range.name = 'Day_Range'
    
    X_vol = X_mom.rolling(window=10).std()
    X_vol.name = 'Volatility'
    
    X_trend = X_mom.rolling(window=20).mean()
    X_trend.name = 'Trend'
    
    # Combine all features
    X = pd.concat([X1, X2, X3, X_mom, X_range, X_vol, X_trend], axis=1)
    
    # Create final dataset
    dataset = pd.concat([Y, X], axis=1)
    
    # --- THE FIXES ---
    # Fix 1: Remove duplicate columns (if any)
    dataset = dataset.loc[:, ~dataset.columns.duplicated()]
    
    # Fix 2: Remove duplicate rows/dates (if any) 
    dataset = dataset[~dataset.index.duplicated(keep='first')]
    
    # Drop NAs and downsample
    dataset = dataset.dropna().iloc[::return_period, :]

    Y = dataset.iloc[:, 0]
    X = dataset.iloc[:, 1:]
    dataset.index.name = 'Date'
    #dataset.to_csv(r"./test_data.csv")
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    features = features.iloc[:,1:]
    return features


def get_bitcoin_historical_prices(days = 60):
    
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily' # Ensure we get daily granularity
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df




