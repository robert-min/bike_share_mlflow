import csv
import dvc.api
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import warnings
import mlflow.sklearn
from utils import fetch_logged_data


url = 'https://github.com/robert-min/bike_share_mlflow'
branch = 'feature/eda-process'
experiment_name = "origin"
ml_cols = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity',
           'windspeed', 'year', 'month', 'day', 'hour', 'dow', 'woy']
model = ("LinearRegression", LinearRegression())

def dvc_open(path):
    with dvc.api.open(
        path = path,  ## 데이터 경로
        repo = url,  ## github repo 경로,
        rev =  branch ## 현재는 branch 기준
    ) as f:
        return pd.read_csv(f, sep=",")

def find_experiment_id(experiment_name):
    current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
    return current_experiment["experiment_id"]

def main():
    train_df = dvc_open('data/train.csv')
    X_train, X_test, y_train, y_test = train_test_split(train_df[ml_cols], train_df["count_log"]
                                                        , test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.sklearn.autolog()
    
    pipe = Pipeline([("scaler", StandardScaler()), model])
    
    experiment_id = find_experiment_id(experiment_name)
    with mlflow.start_run(experiment_id = experiment_id, run_name = model[0]) as run:
        pipe.fit(X_train, y_train)
        print("Logged data and model in run: {}".format(run.info.run_id))

    # show logged data
    for key, data in fetch_logged_data(run.info.run_id).items():
        print("\n---------- logged {} ----------".format(key))
        print(data)
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()