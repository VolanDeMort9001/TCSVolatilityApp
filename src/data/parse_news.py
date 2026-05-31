import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, UTC
import trafilatura
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket


socket.setdefaulttimeout(30)

load_dotenv()

API_KEY = os.getenv("API_KEY")
DATA_DIR = os.path.join(os.getenv("DATA_DIR", "."), "data", "raw")
BASE_URL = os.getenv("BASE_URL")
OUTPUT_FILE = "news_raw.csv"

QUERY = """
(
"Т-Банк" OR
"T-Bank" OR
"TCS Group" OR
"ТКС Групп" OR
"Тинькофф"
)
"""

DATE_FROM = (datetime.now(UTC) - timedelta(days=31)).strftime("%Y-%m-%d")
DATE_TO = (datetime.now(UTC)).strftime("%Y-%m-%d")

PAGE_SIZE = 100

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
}


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_news(page: int):
    params = {
        "q": QUERY,
        "language": "ru",
        "from": DATE_FROM,
        "to": DATE_TO,
        "sortBy": "publishedAt",
        "pageSize": PAGE_SIZE,
        "page": page,
        "apiKey": API_KEY,
    }

    response = requests.get(
        BASE_URL,
        params=params,
        headers=REQUEST_HEADERS,
        timeout=30
    )

    print(
        f"Page={page} "
        f"Status={response.status_code}"
    )

    if response.status_code != 200:
        print(response.text)
        response.raise_for_status()

    return response.json()



session = requests.Session()

retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

session.mount(
    "https://",
    HTTPAdapter(max_retries=retries)
)

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 "
        "(Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Connection": "close"
})


def extract_full_text(url: str):
    try:
        response = session.get(
            url,
            timeout=(5, 30),
            allow_redirects=True,
            verify=True
        )

        response.raise_for_status()

        html = response.text

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_images=False,
            favor_precision=True
        )

        if text:
            text = " ".join(text.split())

        return text

    except Exception as e:

        print(
            f"\nОшибка парсинга:\n"
            f"{url}\n"
            f"{type(e).__name__}: {e}\n"
        )

        return None


def parse_articles(data):

    rows = []

    for article in data.get("articles", []):

        rows.append({
            "date": article.get("publishedAt")[:10],
            "title": article.get("title"),
            "source": article.get("source", {}).get("name"),
            "author": article.get("author"),
            "url": article.get("url"),
            "url_to_image": article.get("urlToImage"),
        })

    return rows


def collect_news():

    all_rows = []

    page = 1

    while True:

        data = fetch_news(page)

        articles = parse_articles(data)

        if not articles:
            print("Статьи закончились")
            break

        all_rows.extend(articles)

        print(
            f"Получено {len(articles)} статей"
        )

        if len(articles) < PAGE_SIZE:
            break

        page += 1

        time.sleep(1)

    df = pd.DataFrame(all_rows)

    if not df.empty:

        df.drop_duplicates(
            subset=["url"],
            inplace=True
        )

        df.reset_index(
            drop=True,
            inplace=True
        )

    return df


def enrich_with_full_text(df):

    texts = []

    total = len(df)

    for idx, url in enumerate(df["url"]):

        print(
            f"[{idx + 1}/{total}] "
            f"Скачивание текста"
        )

        text = extract_full_text(url)

        texts.append(text)

    df["text"] = texts

    return df


def filter_bad_articles(df):

    mask = (
        df["text"]
        .notna()
        &
        (df["text"].str.len() > 500)
    )

    return df[mask].copy()


def save_news(df):

    filepath = os.path.join(
        DATA_DIR,
        OUTPUT_FILE
    )

    df.to_csv(
        filepath,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nСохранено:\n{filepath}"
    )


def main():

    if not API_KEY:
        raise ValueError(
            "API_KEY не найден в .env"
        )

    ensure_data_dir()

    print("Получение ссылок из NewsAPI...")
    df = collect_news()

    print(
        f"\nНайдено статей: {len(df)}"
    )

    print(
        "\nСкачивание полных текстов..."
    )

    df = enrich_with_full_text(df)

    print(
        "\nУдаление пустых статей..."
    )

    df = filter_bad_articles(df)

    print(
        f"\nОсталось статей: {len(df)}"
    )

    save_news(df)


if __name__ == "__main__":
    main()