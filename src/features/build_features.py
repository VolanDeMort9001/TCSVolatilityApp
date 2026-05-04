import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv


load_dotenv()

PRICES_PATH = os.getenv("DATA_DIR") + "/data/raw/combined_prices.csv"
SENTIMENT_PATH = os.getenv("DATA_DIR") + "/data/preprocessed/news_sentiment.csv"
OUTPUT_PATH = os.getenv("DATA_DIR") + "/data/preprocessed/final_dataset.csv"
LAG_DAYS = int(os.getenv("LAG_DAYS"))

def load_data():
    prices = pd.read_csv(PRICES_PATH)
    news = pd.read_csv(SENTIMENT_PATH)
    return prices, news

def merge_data(prices, news):
    df = pd.merge(prices, news, on="date", how="left")
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0)
    df["sentiment_std"] = df["sentiment_std"].fillna(0)
    df["news_count"] = df["news_count"].fillna(0)
    return df

def add_features(df, days: int = 10):
    for lag in range(1, days + 1):
        df[f"sentiment_mean_lag{lag}"] = df["sentiment_mean"].shift(lag)
        df[f"sentiment_std_lag{lag}"] = df["sentiment_std"].shift(lag)
        df[f"news_count_lag{lag}"] = df["news_count"].shift(lag)
        df[f"TCSG_return_lag{lag}"] = df["log_return_TCSG"].shift(lag)
        df[f"MOEX_return_lag{lag}"] = df["log_return_MOEX"].shift(lag)
        df[f"MOEX_volatility_lag{lag}"] = df["volatility_MOEX"].shift(lag)
    return df

def clean(df):
    df = df.dropna()
    return df

def main():
    print("Загрузка данных")
    prices, news = load_data()

    print("Объединение таблиц")
    df = merge_data(prices, news)

    print("Добавление новых фичей")
    df = add_features(df, LAG_DAYS)

    print("Очистка пустых строк")
    df = clean(df)

    print(df.head())

    print(f"\nРазмер датасета: {df.shape}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()