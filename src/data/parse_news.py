import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
DATA_DIR = os.getenv("DATA_DIR") + "/data/raw"
BASE_URL = os.getenv("BASE_URL")
OUTPUT_FILE = "news_raw.csv"
KEYWORDS = "TCS Group"
DATE_FROM = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")

MAX_PAGES = 34


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def fetch_news(page=1):
    params = {
        "q": KEYWORDS,
        "lang": "en",
        "from": DATE_FROM,
        "sortby": "publishedAt",
        "max": 10,
        "page": page,
        "apikey": API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    print(f"STATUS: {response.status_code}")
    if response.status_code != 200:
        print(response.text)
        raise Exception("Ошибка API")
    return response.json()


def parse_articles(data):
    articles = []

    for item in data.get("articles", []):
        articles.append({
            "date": item.get("publishedAt", "")[:100],
            "title": item.get("title", ""),
            "text": (item.get("description") or "") + " " + (item.get("content") or ""),
            "source": item.get("source", {}).get("name", "")
        })

    return articles



def collect_news(max_pages=MAX_PAGES):
    all_articles = []

    for page in range(1, max_pages + 1):
        print(f"\nЗапрос страницы {page}")

        data = fetch_news(page=page)
        parsed = parse_articles(data)

        if not parsed:
            print("Нет данных, останавливаемся")
            break

        all_articles.extend(parsed)
        time.sleep(1)

    df = pd.DataFrame(all_articles)

    if df.empty:
        print("⚠️ Нет данных")

    return df

def save_news(df):
    path = os.path.join(DATA_DIR, OUTPUT_FILE)
    df.to_csv(path, index=False)
    print(f"Сохранено: {path}")



def main():
    ensure_data_dir()
    df = collect_news()
    print(f"\nСобрано новостей: {len(df)}")
    save_news(df)

if __name__ == "__main__":
    main()