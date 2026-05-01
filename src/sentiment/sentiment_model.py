import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from dotenv import load_dotenv


load_dotenv()

INPUT_PATH = os.getenv("DATA_DIR") + "raw/news_raw.csv"
OUTPUT_PATH = os.getenv("DATA_DIR") + "preprocessed/news_sentiment.csv"

def load_model():
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )
    return sentiment_model

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("\n", " ")
    text = text.strip()
    return text

def compute_sentiment(df, model):
    sentiments = []
    scores = []

    for text in tqdm(df["text"]):
        if not text:
            sentiments.append("neutral")
            scores.append(0.0)
            continue

        try:
            result = model(text)[0]
            label = result["label"]
            score = result["score"]
            if label == "positive":
                sentiments.append("positive")
                scores.append(score)
            elif label == "negative":
                sentiments.append("negative")
                scores.append(-score)
            else:
                sentiments.append("neutral")
                scores.append(0.0)

        except Exception:
            sentiments.append("neutral")
            scores.append(0.0)

    df["sentiment"] = sentiments
    df["sentiment_score"] = scores
    return df

def aggregate_daily(df):
    df["date"] = pd.to_datetime(df["date"]).dt.date
    daily = df.groupby("date").agg({
        "sentiment_score": ["mean", "std", "count"]
    })

    daily.columns = [
        "sentiment_mean",
        "sentiment_std",
        "news_count"
    ]

    daily = daily.reset_index()
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    return daily

def save(df):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Сохранено: {OUTPUT_PATH}")


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Всего новостей: {len(df)}")
    df["text"] = df["text"].apply(clean_text)
    model = load_model()
    df = compute_sentiment(df, model)
    daily_features = aggregate_daily(df)
    print(daily_features.head(20))
    save(daily_features)


if __name__ == "__main__":
    main()