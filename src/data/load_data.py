import os
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import requests

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR") + "/data/raw"
OLD_TICKER = "TCSG"
NEW_TICKER = "T"
MOEX_TICKER = "MOEX"

START_DATE = "2019-09-01"
END_DATE = "2026-04-01"    #datetime.today().strftime('%Y-%m-%d')

def load_moex_data(security: str, name: str):
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{security}.json"

    all_data = []
    start = 0

    while True:
        params = {
            "from": START_DATE,
            "till": END_DATE,
            "start": start,
            "iss.meta": "off",
            "iss.only": "history",
            "history.columns": "TRADEDATE,CLOSE,VOLUME"
        }

        response = requests.get(url, params=params)
        data = response.json()

        history = data["history"]["data"]
        columns = data["history"]["columns"]

        if not history:
            break

        df_part = pd.DataFrame(history, columns=columns)
        all_data.append(df_part)

        print(f"Загружено строк: {len(df_part)} (start={start})")

        start += 100

    df = pd.concat(all_data, ignore_index=True)

    df.columns = ["date", f"price_{name}", f"volume_{name}"]
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")
    df = df.dropna()

    return df


def load_price_data(ticker: str, name: str) -> pd.DataFrame:
    """
    Загружает исторические данные по тикеру через yfinance
    """
    data = load_moex_data(ticker, name)
    if data.empty:
        raise ValueError(f"Нет данных для тикера {ticker}")

    data = data.reset_index()
    return data


def compute_returns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Считает лог-доходности
    """
    df[f"log_return_{ticker}"] = np.log(df[f"price_{ticker}"] / df[f"price_{ticker}"].shift(1))
    return df


def compute_volatility(df: pd.DataFrame, ticker: str, window: int = 20) -> pd.DataFrame:
    """
    Считает скользящую волатильность
    """
    df[f"volatility_{ticker}"] = df[f"log_return_{ticker}"].rolling(window=window).std()
    return df


def prepare_price_dataset(ticker: str, name: str) -> pd.DataFrame:
    """
    Полный пайплайн:
    загрузка -> доходности -> волатильность
    """
    df = load_price_data(ticker, name)
    df = compute_returns(df, name)
    df = compute_volatility(df, name)

    return df


def save_data(df: pd.DataFrame, filename: str):
    """
    Сохраняет DataFrame в CSV
    """
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)


def main():
    moex_df = prepare_price_dataset(MOEX_TICKER, "MOEX")
    moex_df = moex_df.drop_duplicates(subset=["date"])
    moex_df = moex_df.drop(columns=["index"], errors="ignore")

    tcs_df = pd.concat([prepare_price_dataset(OLD_TICKER, OLD_TICKER), prepare_price_dataset(NEW_TICKER, OLD_TICKER)], axis=0)
    tcs_df = tcs_df.drop(columns=["index"], errors="ignore")

    save_data(tcs_df, "tcs_prices.csv")
    save_data(moex_df, "moex_prices.csv")

    combined = pd.merge(tcs_df, moex_df, on="date", how="left")
    save_data(combined, "combined_prices.csv")


if __name__ == "__main__":
    main()