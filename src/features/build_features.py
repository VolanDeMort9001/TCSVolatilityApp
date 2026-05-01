import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv


load_dotenv()

PRICES_PATH = os.getenv("DATA_DIR") + "/data/raw/moex_prices.csv"
SENTIMENT_PATH = os.getenv("DATA_DIR") + "/data/preprocessed/news_sentiment.csv"
OUTPUT_PATH = os.getenv("DATA_DIR") + "/data/preprocessed/final_dataset.csv"

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

def add_features(df):
    df["sentiment_mean_lag1"] = df["sentiment_mean"].shift(1)
    df["sentiment_std_lag1"] = df["sentiment_std"].shift(1)
    df["news_count_lag1"] = df["news_count"].shift(1)
    df["return_lag1"] = df["log_return"].shift(1)
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
    df = add_features(df)

    print("Очистка пустых строк")
    df = clean(df)

    print(df.head())

    print(f"\nРазмер датасета: {df.shape}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Сохранено: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()