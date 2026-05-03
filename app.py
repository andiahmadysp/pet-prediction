import os
import tempfile
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "best_model.keras")
img_size = (160, 160)

# Load model dengan fallback jika belum ada
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print(f"Model {MODEL_PATH} berhasil dimuat.")
else:
    print(f"Peringatan: {MODEL_PATH} tidak ditemukan. Jalankan train_model.py terlebih dahulu.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "status": 503,
            "success": False,
            "error": "Model belum tersedia. Jalankan train_model.py terlebih dahulu."
        }), 503

    try:
        file = request.files.get("image")
        if not file or file.filename == "":
            return jsonify({
                "status": 400,
                "success": False,
                "error": "Tidak ada file yang diunggah."
            }), 400

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            filepath = tmp.name
        file.save(filepath)

        try:
            img = load_img(filepath, target_size=img_size)
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            pred = model.predict(img, verbose=0)[0][0]
        finally:
            os.remove(filepath)

        if pred > 0.5:
            label = "Dog"
            confidence = float(pred)
        else:
            label = "Cat"
            confidence = float(1 - pred)

        return jsonify({
            "status": 200,
            "success": True,
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        }), 200

    except Exception as e:
        return jsonify({
            "status": 500,
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)