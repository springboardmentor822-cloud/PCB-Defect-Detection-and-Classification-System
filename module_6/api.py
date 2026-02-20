from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("../models/pcb_defect_model_v3.h5")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    confidence = float(np.max(pred))

    return jsonify({
        "defect": confidence > 0.5,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)