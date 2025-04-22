import yfinance as yf
import pandas as pd
import ta  

def download_stock_data(ticker='TSLA', start='2015-01-01', end='2024-12-31'):
    # Explicitly set auto_adjust to False to maintain backward compatibility
    data = yf.download(ticker, start=start, end=end, auto_adjust=False)
    data = data.dropna()
    return data

def add_technical_indicators(df):
    df = df.copy()
    close_series = df['Close'].iloc[:,0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
    
    print(f"[DEBUG] close_series shape: {close_series.shape}, type: {type(close_series)}")
    
    # SMA & EMA
    sma = ta.trend.SMAIndicator(close=close_series, window=20)
    df['SMA_20'] = sma.sma_indicator()
    
    ema = ta.trend.EMAIndicator(close=close_series, window=20)
    df['EMA_20'] = ema.ema_indicator()
    
    # RSI
    rsi = ta.momentum.RSIIndicator(close=close_series, window=14)
    df['RSI_14'] = rsi.rsi()
    
    # MACD
    macd = ta.trend.MACD(close=close_series)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=close_series, window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    return df.dropna()

if __name__ == '__main__':
    ticker = 'AAPL'
    data = download_stock_data(ticker)
    data_with_indicators = add_technical_indicators(data)
    
    print(data_with_indicators.tail())
    data_with_indicators.to_csv(f'{ticker}_processed.csv')