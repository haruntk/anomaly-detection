from flask import Flask, request, jsonify
from flask_cors import CORS
from models import padim  # şimdilik sadece PaDiM
from utils.preprocess import load_image, preprocess_image

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form.get("model", "padim")
    image_file = request.files['image']

    # Görüntüyü yükle
    image = load_image(image_file)
    
    # Görüntüyü ön işleme adımlarından geçir
    # Arka plan kaldırma, normalizasyon ve boyut ayarlama
    processed_image = preprocess_image(image, remove_background=True)

    if model_type == "padim":
        result = padim.predict(processed_image)
    else:
        return jsonify({"error": "Model desteklenmiyor"}), 400

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)