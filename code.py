import xgboost as xgb
import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas_ta as ta
from glob import glob
from pathlib import Path
from numpy.typing import NDArray
from typing import List, Dict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

def process_files(file_path):
    dfs = []
    for file in file_path:
        df = pd.read_csv(file)
        df['Ticker'] = Path(file).stem
        df['Date'] = pd.to_datetime(df['Date'])
        dfs.append(df)
    main_df = pd.concat(np.array(dfs), ignore_index=True)
    main_df.sort_values(by=['Ticker', 'Date'], inplace=True)
    return main_df

def engineer_features(df):
    if len(df) < 20:
        return df
    df['Log Return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
    df['RSI'] = ta.rsi(df['Adj Close'], length=14)
    df['Volatility'] = df['Log Return'].rolling(20).std()
    vol_mean = df['Volume'].rolling(20).mean()
    vol_std = df['Volume'].rolling(20).std()
    pop_mean = df['Volume'].rolling(200).mean()
    df['Vol_T'] = (vol_mean - pop_mean) / (vol_std/np.sqrt(20))
    df['Target'] = (df['Log Return'].shift(-1) > 0).astype(int)
    return df

if __name__ == "__main__":
    df = process_files(stocks_files)
    df = df.groupby('Ticker', group_keys=False).apply(engineer_features)
    df.dropna(inplace=True)
    
    unique_dates = df['Date'].sort_values().unique()
    split_date = unique_dates[int(len(unique_dates) * 0.8)]


    train_df = df[df['Date'] < split_date]
    test_df = df[df['Date'] >= split_date]

    features = ['RSI', 'Volatility', 'Vol_T', 'Log Return']

    X_train = train_df[features]
    y_train = train_df['Target']
    X_test = test_df[features]
    y_test = test_df['Target']

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    precision = precision_score(y_test, preds)
    accuracy = model.score(X_test, y_test)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")

