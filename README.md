# TCSVolatilityApp

# data/

data/raw/tcs_prices.csv - сырые данные цен акций TCS Group

data/raw/moex_prices.csv - сырые данные цен индекса MOEX

data/raw/combined_prices.csv - сырые объединенные данные цен индекса MOEX и TCS Group

data/raw/news_raw.csv - данные из новостей (по сути, необработанные новости)

data/preprocessed/news_sentiment.csv - обработанные значения коэффициента настроения статей из news_raw.csv

data/preprocessed/dataset.csv - хранение финального датасета для обучения модели

# notebooks/

notebooks/eda.ipynb - ноутбук для анализа распределений, составления графиков, чтобы проверять гипотезы перед исследованием и иметь хорошую визуализацию.

notebooks/experiments.ipynb - ноутбук для экспериментов с данными, обучение различных моделей.

# src/

# src/data/

load_data.py - загрузка финансовых данных в prices.csv

parse_news.py - загрузка данных новостей, фильтрация по ключевым словам, очистка HTML, сохранение в news_raw.csv

# src/sentiment/

sentiment_model.py - анализ текста с помощью API, обработка текста, получение sentiment score.

# src/models/

train.py - централизованное обучение моделей

predict.py - предсказание волатильности в модели через веб-приложение

# src/features/

build_features.py - создание фичей, подготовка к обучению

# app/

app.py - загрузка веб-приложения, создание графиков, запуск прогноза
