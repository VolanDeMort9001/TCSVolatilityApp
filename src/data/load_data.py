import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

DATA_DIR = "/Users/aleknazarov9001/PycharmProjects/TCSVolatilityApp/data/raw/"
TICKER = "TCSG.ME"
MOEX_TICKER = "IMOEX.ME"

START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')


def load_price_data(ticker: str) -> pd.DataFrame:
    """
    Загружает исторические данные по тикеру через yfinance
    """
    data = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        progress=False
    )
    if data.empty:
        raise ValueError(f"Нет данных для тикера {ticker}")

    data = data.reset_index()

    # Оставляем нужные колонки
    data = data[["Date", "Close", "Volume"]]
    data.columns = ["date", "price", "volume"]
    return data


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Считает лог-доходности
    """
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    return df


def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Считает скользящую волатильность
    """
    df["volatility"] = df["log_return"].rolling(window=window).std()
    return df


def prepare_price_dataset(ticker: str, name: str) -> pd.DataFrame:
    """
    Полный пайплайн:
    загрузка -> доходности -> волатильность
    """
    df = load_price_data(ticker)
    df = compute_returns(df)
    df = compute_volatility(df)
    df["asset"] = name

    return df


def save_data(df: pd.DataFrame, filename: str):
    """
    Сохраняет DataFrame в CSV
    """
    path = os.path.join(DATA_DIR, filename)
    print(path)
    df.to_csv(path, index=False)


def main():
    moex_df = prepare_price_dataset(MOEX_TICKER, "MOEX")
    tcs_df = prepare_price_dataset(TICKER, "TCSG")

    save_data(tcs_df, "tcs_prices.csv")
    save_data(moex_df, "moex_prices.csv")

    combined = pd.concat([tcs_df, moex_df], axis=0)
    save_data(combined, "combined_prices.csv")


if __name__ == "__main__":
    main()