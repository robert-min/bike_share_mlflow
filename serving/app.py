import pickle
import numpy as np
from flask import Flask, jsonify, request

model = pickle.load(open("./build/model.pkl", "rb"))

app = Flask(__name__)

@app.route("/count/predict", methods=["POST"])
def predict_count():
    req = request.get_json()
    
    ml_cols = [
           'season', 'holiday', 'workingday', 'weather', 'temp',
           'atemp', 'humidity', 'windspeed', 'day', 'month',
           'year', 'hour', 'dow', 'woy', 'peaktime', 'fit', 'humid'
    ]
    
    X_test = [req[col_name] for col_name in ml_cols]
    X_test = np.array(X_test)
    X_test = X_test.reshape(1, -1)
    
    y_pred = model.predict(X_test)
    resp = jsonify(y_pred.tolist())
    return resp, 200
    

if __name__ == "__main__":
    
    app.run(port=8080, debug=True)