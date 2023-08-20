from flask import Flask, request, jsonify
import numpy as np
import pickle
import joblib

RFC_MODEL = joblib.load("models/RFC_MODEL.pkl")
SCLAER_MODEL = joblib.load("models/SCALER_MODEL.pkl")


def return_prediction(model, scaler, sample_json):
    a = list(sample_json.values())
    a = [float(i) for i in a]
    data = SCLAER_MODEL.transform([a])
    result = RFC_MODEL.predict(data)
    labels = ["Insomnia", "None", "Sleep Apnea"]
    return labels[result[0]]


app = Flask(__name__)


@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'


@app.route('/prediction', methods=['POST'])
def predict_flower():
    # RECIEVE THE REQUEST
    content = request.json

    # PRINT THE DATA PRESENT IN THE REQUEST
    print("[INFO] Request: ", content)

    # PREDICT THE CLASS USING HELPER FUNCTION
    results = return_prediction(model=RFC_MODEL,
                                scaler=SCLAER_MODEL,
                                sample_json=content)

    # PRINT THE RESULT
    print("[INFO] Responce: ", results)

    # SEND THE RESULT AS JSON OBJECT
    return jsonify(results)


if __name__ == '__main__':
    app.run("0.0.0.0")
