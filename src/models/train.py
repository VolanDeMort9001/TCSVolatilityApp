import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit


load_dotenv()
DATA_PATH = os.getenv("DATA_DIR") + "/data/preprocessed/final_dataset.csv"
LAG_DAYS = int(os.getenv("LAG_DAYS"))
PREDICT_DAYS = int(os.getenv("PREDICT_DAYS"))


class PredictingModel:
    def __init__(self, path: str, train_size=0.9, lag: int = 10):
        self.df = pd.read_csv(path)
        self.tscv = TimeSeriesSplit(n_splits=8)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.y = self.df["volatility_TCSG"]
        self.lag = lag
        columns = []
        for lag in range(1, lag + 1):
            columns.append(f"sentiment_mean_lag{lag}")
            columns.append(f"sentiment_std_lag{lag}")
            columns.append(f"news_count_lag{lag}")
            columns.append(f"TCSG_return_lag{lag}")
            columns.append(f"MOEX_return_lag{lag}")
            columns.append(f"MOEX_volatility_lag{lag}")
        self.X = self.df[columns]
        split_size = int(len(self.X) * train_size)
        self.X_train = self.X[:split_size]
        self.X_test = self.X[split_size:]
        self.y_train = self.y[:split_size]
        self.y_test = self.y[split_size:]


    def fit(self):
        val_scores = []
        for i, (train_index, val_index) in enumerate(self.tscv.split(self.X_train)):
            X_train, X_val = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
            y_train, y_val = self.y_train.iloc[train_index].values.reshape(-1), self.y_train.iloc[val_index].values.reshape(-1)
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_val)
            val_rmse = mean_squared_error(y_val, preds)
            val_scores.append(np.sqrt(val_rmse))
        print(f"Ошибки RMSE на валидационных фолдах: {val_scores}")
        return self.model

    def evaluate(self, show_metrics: bool = True):
        preds = self.model.predict(self.X_test)

        if show_metrics:
            mse = mean_squared_error(self.y_test, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, preds)
            print("\nМетрики:")
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE:  {mae:.6f}")

        return preds

    def predict_volatility(self, days: int = 5):
        volatilities = []
        X = self.X
        for day in range(1, days + 1):
            pred = self.model.predict(pd.DataFrame(X.iloc[[len(X) - 1]]))
            volatilities.append(pred[0])
            new_day = dict()
            for lag in range(1, self.lag + 1):
                if len(X) - lag < len(self.df):
                    new_day[f"sentiment_mean_lag{lag}"] = [self.df.at[len(X) - lag, "sentiment_mean"]]
                    new_day[f"sentiment_std_lag{lag}"] = [self.df.at[len(X) - lag, "sentiment_std"]]
                    new_day[f"news_count_lag{lag}"] = [self.df.at[len(X) - lag, "news_count"]]
                    new_day[f"TCSG_return_lag{lag}"] = [self.df.at[len(X) - lag, "log_return_TCSG"]]
                    new_day[f"MOEX_return_lag{lag}"] = [self.df.at[len(X) - lag, "log_return_MOEX"]]
                    new_day[f"MOEX_volatility_lag{lag}"] = [self.df.at[len(X) - lag, "volatility_MOEX"]]
                else:
                    new_day[f"sentiment_mean_lag{lag}"] = [0]
                    new_day[f"sentiment_std_lag{lag}"] = [0]
                    new_day[f"news_count_lag{lag}"] = [0]
                    new_day[f"TCSG_return_lag{lag}"] = [0]
                    new_day[f"MOEX_return_lag{lag}"] = [0]
                    new_day[f"MOEX_volatility_lag{lag}"] = [0]
            new_day = pd.DataFrame(new_day)
            X = pd.concat([X, new_day], axis=0)
        return volatilities



    def show_feature_importance(self):
        importances = self.model.feature_importances_

        print("\nВажность признаков:")
        for name, val in zip(self.X.columns, importances):
            print(f"{name}: {val:.4f}")

def main():
    model = PredictingModel(DATA_PATH, lag=LAG_DAYS)
    model.fit()

    model.show_feature_importance()

    print(f"\nПредсказание волатильности на следующие {PREDICT_DAYS} дней:", model.predict_volatility(PREDICT_DAYS))

if __name__ == "__main__":
    main()