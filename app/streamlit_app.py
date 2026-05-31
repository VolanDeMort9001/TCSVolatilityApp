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
def load_model(model_name: str = "RFR"):
    model = PredictingModel(
        path=DATA_PATH,
        lag=LAG_DAYS,
        model_name=model_name
    )
    model.fit()
    return model


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df

RFR_model = load_model(model_name="RFR")
df = load_data()

st.title("TCS Volatility Predictor")
st.markdown("""
Веб-приложение для прогнозирования волатильности акций TCS Group
на основе:
- новостного sentiment analysis
- исторических данных TCSG и индекса MOEX
""")


st.subheader("Метрики модели RandomForestRegressor")

RFR_preds = RFR_model.evaluate(show_metrics=False)
RFR_rmse = np.sqrt(np.mean((RFR_model.y_test - RFR_preds) ** 2))
RFR_mae = np.mean(np.abs(RFR_model.y_test - RFR_preds))
col1, col2 = st.columns(2)
with col1:
    st.metric("RMSE", f"{RFR_rmse:.6f}")
with col2:
    st.metric("MAE", f"{RFR_mae:.6f}")

st.subheader(f"Прогноз на следующие {PREDICT_DAYS} дней на RandomForestRegressor")
RFR_future_preds = RFR_model.predict_volatility(days=PREDICT_DAYS)
RFR_future_df = pd.DataFrame({
    "Day": np.arange(1, PREDICT_DAYS + 1),
    "Predicted Volatility": RFR_future_preds
})

st.dataframe(RFR_future_df)
RFR_forecast_fig = px.line(
    RFR_future_df,
    x="Day",
    y="Predicted Volatility",
    markers=True,
    title="Прогноз волатильности"
)
st.plotly_chart(RFR_forecast_fig, use_container_width=True)


GBR_model = load_model(model_name="GBR")

st.subheader("Метрики модели GradientBoostingRegressor")

GBR_preds = GBR_model.evaluate(show_metrics=False)
GBR_rmse = np.sqrt(np.mean((GBR_model.y_test - GBR_preds) ** 2))
GBR_mae = np.mean(np.abs(GBR_model.y_test - GBR_preds))
col1, col2 = st.columns(2)
with col1:
    st.metric("RMSE", f"{GBR_rmse:.6f}")
with col2:
    st.metric("MAE", f"{GBR_mae:.6f}")

st.subheader(f"Прогноз на следующие {PREDICT_DAYS} дней на GradientBoostingRegressor")
GBR_future_preds = GBR_model.predict_volatility(days=PREDICT_DAYS)
GBR_future_df = pd.DataFrame({
    "Day": np.arange(1, PREDICT_DAYS + 1),
    "Predicted Volatility": GBR_future_preds
})

st.dataframe(GBR_future_df)
GBR_forecast_fig = px.line(
    GBR_future_df,
    x="Day",
    y="Predicted Volatility",
    markers=True,
    title="Прогноз волатильности"
)
st.plotly_chart(GBR_forecast_fig, use_container_width=True)


AdaRegression_model = load_model(model_name="AdaRegression")

st.subheader("Метрики модели GradientBoostingRegressor")

AdaRegression_preds = AdaRegression_model.evaluate(show_metrics=False)
AdaRegression_rmse = np.sqrt(np.mean((AdaRegression_model.y_test - AdaRegression_preds) ** 2))
AdaRegression_mae = np.mean(np.abs(AdaRegression_model.y_test - AdaRegression_preds))
col1, col2 = st.columns(2)
with col1:
    st.metric("RMSE", f"{AdaRegression_rmse:.6f}")
with col2:
    st.metric("MAE", f"{AdaRegression_mae:.6f}")

st.subheader(f"Прогноз на следующие {PREDICT_DAYS} дней на GradientBoostingRegressor")
AdaRegression_future_preds = AdaRegression_model.predict_volatility(days=PREDICT_DAYS)
AdaRegression_future_df = pd.DataFrame({
    "Day": np.arange(1, PREDICT_DAYS + 1),
    "Predicted Volatility": AdaRegression_future_preds
})

st.dataframe(AdaRegression_future_df)
AdaRegression_forecast_fig = px.line(
    AdaRegression_future_df,
    x="Day",
    y="Predicted Volatility",
    markers=True,
    title="Прогноз волатильности"
)
st.plotly_chart(AdaRegression_forecast_fig, use_container_width=True)


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

st.subheader("Важность признаков в RandomForestRegressor")
RFR_importance_df = pd.DataFrame({
    "feature": RFR_model.X.columns,
    "importance": RFR_model.model.feature_importances_
})
RFR_importance_df = RFR_importance_df.sort_values(
    "importance",
    ascending=False
)
RFR_importance_fig = px.bar(
    RFR_importance_df.head(15),
    x="importance",
    y="feature",
    orientation="h",
    title="Top-15 признаков"
)
st.plotly_chart(RFR_importance_fig, use_container_width=True)

st.subheader("Важность признаков в GradientBoostRegressor")
GBR_importance_df = pd.DataFrame({
    "feature": GBR_model.X.columns,
    "importance": GBR_model.model.feature_importances_
})
GBR_importance_df = GBR_importance_df.sort_values(
    "importance",
    ascending=False
)
GBR_importance_fig = px.bar(
    GBR_importance_df.head(15),
    x="importance",
    y="feature",
    orientation="h",
    title="Top-15 признаков"
)
st.plotly_chart(GBR_importance_fig, use_container_width=True)

st.subheader("Важность признаков в AdaBoostRegressor")
AdaRegression_importance_df = pd.DataFrame({
    "feature": AdaRegression_model.X.columns,
    "importance": AdaRegression_model.model.feature_importances_
})
AdaRegression_importance_df = AdaRegression_importance_df.sort_values(
    "importance",
    ascending=False
)
AdaRegression_importance_fig = px.bar(
    AdaRegression_importance_df.head(15),
    x="importance",
    y="feature",
    orientation="h",
    title="Top-15 признаков"
)
st.plotly_chart(AdaRegression_importance_fig, use_container_width=True)

with st.expander("Показать датасет"):

    st.dataframe(df.tail(100))

with st.expander("Предсказания на test set в RandomForestRegressor"):

    RFR_compare_df = pd.DataFrame({
        "real": RFR_model.y_test.values,
        "predicted": RFR_preds
    })

    st.dataframe(RFR_compare_df.head(50))

    RFR_compare_fig = px.line(
        RFR_compare_df.head(100),
        title="Real vs Predicted Volatility"
    )

    st.plotly_chart(RFR_compare_fig, use_container_width=True)

with st.expander("Предсказания на test set в GradientBoostRegressor"):
    GBR_compare_df = pd.DataFrame({
        "real": GBR_model.y_test.values,
        "predicted": GBR_preds
    })

    st.dataframe(GBR_compare_df.head(50))

    GBR_compare_fig = px.line(
        GBR_compare_df.head(100),
        title="Real vs Predicted Volatility"
    )

    st.plotly_chart(GBR_compare_fig, use_container_width=True)

with st.expander("Предсказания на test set в AdaBoostRegressorRFR"):
    AdaRegression_compare_df = pd.DataFrame({
        "real": AdaRegression_model.y_test.values,
        "predicted": AdaRegression_preds
    })

    st.dataframe(AdaRegression_compare_df.head(50))

    AdaRegression_compare_fig = px.line(
        AdaRegression_compare_df.head(100),
        title="Real vs Predicted Volatility"
    )

    st.plotly_chart(AdaRegression_compare_fig, use_container_width=True)



