import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

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

DATE_FROM = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
DATE_TO = datetime.utcnow().strftime("%Y-%m-%d")

PAGE_SIZE = 100


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

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        BASE_URL,
        params=params,
        headers=headers,
        timeout=30
    )

    print(
        f"Page={page} "
        f"Status={response.status_code}"
    )

    if response.status_code != 200:
        print(response.text)
        print(response.headers)
        response.raise_for_status()

    return response.json()


def parse_articles(data):
    rows = []

    for article in data.get("articles", []):

        rows.append(
            {
                "published_at": article.get("publishedAt"),
                "title": article.get("title"),
                "description": article.get("description"),
                "content": article.get("content"),
                "source": article.get("source", {}).get("name"),
                "author": article.get("author"),
                "url": article.get("url"),
                "url_to_image": article.get("urlToImage"),
            }
        )

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

        total_results = data.get("totalResults", 0)

        print(
            f"Получено {len(articles)} статей "
            f"(всего API сообщает {total_results})"
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

    return df


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

    print(f"\nСохранено: {filepath}")


def main():

    if not API_KEY:
        raise ValueError(
            "NEWS_API_KEY не найден в .env"
        )

    ensure_data_dir()

    df = collect_news()

    print(f"\nВсего статей: {len(df)}")

    save_news(df)


if __name__ == "__main__":
    main()