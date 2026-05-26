import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
import os
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from src.models.train import PredictingModel


load_dotenv()
DATA_PATH = os.getenv("DATA_DIR") + "/data/preprocessed/final_dataset.csv"
LAG_DAYS = int(os.getenv("LAG_DAYS"))
PREDICT_DAYS = int(os.getenv("PREDICT_DAYS"))
st.set_page_config(
    page_title="TCS Volatility Predictor",
    layout="wide"
)


@st.cache_resource
def load_model():
    model = PredictingModel(
        path=DATA_PATH,
        lag=LAG_DAYS
    )
    model.fit()
    return model


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df

model = load_model()
df = load_data()

st.title("TCS Volatility Predictor")
st.markdown("""
Веб-приложение для прогнозирования волатильности акций TCS Group
на основе:
- новостного sentiment analysis
- исторических данных TCSG и индекса MOEX
""")
st.subheader("Метрики модели")

preds = model.evaluate(show_metrics=False)
rmse = np.sqrt(np.mean((model.y_test - preds) ** 2))
mae = np.mean(np.abs(model.y_test - preds))
col1, col2 = st.columns(2)
with col1:
    st.metric("RMSE", f"{rmse:.6f}")
with col2:
    st.metric("MAE", f"{mae:.6f}")

st.subheader(f"Прогноз на следующие {PREDICT_DAYS} дней")
future_preds = model.predict_volatility(days=PREDICT_DAYS)
future_df = pd.DataFrame({
    "Day": np.arange(1, PREDICT_DAYS + 1),
    "Predicted Volatility": future_preds
})
st.dataframe(future_df)
forecast_fig = px.line(
    future_df,
    x="Day",
    y="Predicted Volatility",
    markers=True,
    title="Прогноз волатильности"
)
st.plotly_chart(forecast_fig, use_container_width=True)

st.subheader("Историческая волатильность TCSG")
if "date" in df.columns:
    hist_fig = px.line(
        df,
        x="date",
        y="volatility_TCSG",
        title="Историческая волатильность TCSG"
    )
    st.plotly_chart(hist_fig, use_container_width=True)

st.subheader("Sentiment новостей")
if "date" in df.columns:
    sentiment_fig = px.line(
        df,
        x="date",
        y="sentiment_mean",
        title="Средний sentiment новостей"
    )
    st.plotly_chart(sentiment_fig, use_container_width=True)

st.subheader("Важность признаков")
importance_df = pd.DataFrame({
    "feature": model.X.columns,
    "importance": model.model.feature_importances_
})
importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)
importance_fig = px.bar(
    importance_df.head(15),
    x="importance",
    y="feature",
    orientation="h",
    title="Top-15 признаков"
)
st.plotly_chart(importance_fig, use_container_width=True)

with st.expander("Показать датасет"):

    st.dataframe(df.tail(100))

with st.expander("Предсказания на test set"):

    compare_df = pd.DataFrame({
        "real": model.y_test.values,
        "predicted": preds
    })

    st.dataframe(compare_df.head(50))

    compare_fig = px.line(
        compare_df.head(100),
        title="Real vs Predicted Volatility"
    )

    st.plotly_chart(compare_fig, use_container_width=True)
