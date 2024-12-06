import numpy as np
from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)
model = load('model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    test_data = np.array(request.json).reshape(1, -1)
    prediction = int(model.predict(test_data)[0])
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
