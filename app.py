from flask import Flask, request, jsonify
import joblib
import numpy as np
from keras.models import load_model

app = Flask(__name__)

scaler = joblib.load("scaler.pkl")
model_nn = load_model("model_nn.h5")
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


@app.route("/")
def home():
    return "app running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model_nn.predict(features_scaled)[0][0]
        predicted_class = int(prediction > 0.5)

        return jsonify({"prediction": predicted_class, "probability": float(prediction)})
    
    except Exception as e:
        return jsonify({"error ": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
