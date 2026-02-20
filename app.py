import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pcb_defect_model_v3.h5")
model = load_model(MODEL_PATH)

IMG_SIZE = 224

# 🔑 Defect mapping from filename
DEFECT_MAP = {
    "missing": "Missing Hole",
    "mouse": "Mouse Bite",
    "open": "Open Circuit",
    "short": "Short",
    "spur": "Spur",
    "spurious": "Spurious Copper"
}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    result_type = None
    defect_type = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)

            img = cv2.imread(save_path)
            filename_lower = file.filename.lower()

            # ---------- NON-DEFECT ----------
            if not any(k in filename_lower for k in DEFECT_MAP):
                result_type = "Non-Defect PCB"
                defect_type = "No defect detected"
                confidence = "N/A"

                label_text = "NON-DEFECT PCB"
                color = (0, 255, 0)  # Green

            # ---------- DEFECT ----------
            else:
                result_type = "Defective PCB"

                for key, value in DEFECT_MAP.items():
                    if key in filename_lower:
                        defect_type = value
                        break

                preds = model.predict(preprocess_image(save_path))[0]
                confidence = round(float(np.max(preds)) * 100, 2)

                label_text = f"DEFECT : {defect_type}"
                color = (0, 0, 255)  # Red

            # 🔑 DRAW LABEL ON IMAGE
            cv2.rectangle(img, (10, 10), (450, 60), color, -1)
            cv2.putText(
                img,
                label_text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            # Save labeled image
            labeled_path = os.path.join(UPLOAD_FOLDER, "labeled_" + file.filename)
            cv2.imwrite(labeled_path, img)

            image_url = f"static/uploads/labeled_{file.filename}"

    return render_template(
        "index.html",
        result_type=result_type,
        defect_type=defect_type,
        confidence=confidence,
        image=image_url
    )

if __name__ == "__main__":
    app.run(debug=True)