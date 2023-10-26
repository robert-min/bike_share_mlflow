import mlflow
import warnings

import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import fetch_logged_data, dvc_open, find_experiment_id


url = 'https://github.com/robert-min/bike_share_mlflow'
branch = 'feature/feature-eng'
ml_cols = [
           'season', 'holiday', 'workingday', 'weather', 'temp',
           'atemp', 'humidity', 'windspeed', 'day', 'month',
           'year', 'hour', 'dow', 'woy', 'peaktime', 'fit', 'humid'
]
model = [("LinearRegression", LinearRegression()), ("Lasso", Lasso()), ("RandomForestRegressor", RandomForestRegressor())]



def main():
    train_df = dvc_open('data/train.csv', url, branch) 

    X_train, X_test, y_train, y_test = train_test_split(train_df[ml_cols], train_df["count_log"]
                                                        , test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.sklearn.autolog()
    
    
    for i in range(len(model)):
    
        pipe = Pipeline([("scaler", StandardScaler()), model[i]])
        
        experiment_id = find_experiment_id(branch)
        with mlflow.start_run(experiment_id = experiment_id, run_name = model[i][0]) as run:
            pipe.fit(X_train, y_train)
            print("Logged data and model in run: {}".format(run.info.run_id))

        # show logged data
        for key, data in fetch_logged_data(run.info.run_id).items():
            print("\n---------- logged {} ----------".format(key))
            print(data)
    

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()