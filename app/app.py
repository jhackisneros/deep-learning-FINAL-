# Proyecto Deep-Learning
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
import io
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.preprocessing import preprocess_image
from utils.qr_utils import generate_qr_from_data  # ⚠ cambiado
from utils.export_utils import export_predictions_to_csv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('app', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Cargar modelo si existe ---
MODEL_PATH = os.path.join('models', 'mnist_compiled_model.keras')
model = None
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
else:
    print(f"Warning: modelo no encontrado en {MODEL_PATH}. Rutas de predicción estarán inactivas.")

# --- Log de predicciones ---
PRED_LOG = os.path.join('app', 'predictions.json')
if not os.path.exists(PRED_LOG):
    with open(PRED_LOG, 'w') as f:
        json.dump([], f)

def save_prediction_local(record: dict):
    """Guarda la predicción localmente en predictions.json"""
    with open(PRED_LOG, 'r+', encoding='utf-8') as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            data = []
        data.insert(0, record)  # newest first
        fh.seek(0)
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.truncate()

# ---------------- Rutas principales ----------------
@app.route('/')
def index():
    return render_template('index.html')

# ---------------- Rutas de autenticación ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        return redirect(url_for('index'))
    return render_template('register.html')

# ---------------- Rutas de predicción ----------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    img_input = data['image']
    user = data.get('user', None)

    try:
        arr = preprocess_image(img_input, target_size=(28,28), flatten=True)
        pred = model.predict(arr.reshape(1, -1))
        label = int(np.argmax(pred))
        confidence = float(np.max(pred))

        record = {
            'time': datetime.utcnow().isoformat(),
            'user': user,
            'pred': label,
            'confidence': confidence,
        }
        save_prediction_local(record)
        return jsonify({'pred': label, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500

    results = []
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files provided'}), 400

    for f in files:
        try:
            img_bytes = f.read()
            arr = preprocess_image(img_bytes, target_size=(28,28), flatten=True)
            pred = model.predict(arr.reshape(1, -1))
            label = int(np.argmax(pred))
            confidence = float(np.max(pred))
            record = {
                'time': datetime.utcnow().isoformat(),
                'filename': f.filename,
                'pred': label,
                'confidence': confidence,
            }
            save_prediction_local(record)
            results.append(record)
        except Exception as e:
            results.append({'filename': getattr(f, 'filename', None), 'error': str(e)})

    return jsonify(results)

@app.route('/history', methods=['GET'])
def history():
    user = request.args.get('user')
    pred = request.args.get('pred')
    limit = int(request.args.get('limit', 50))

    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    if user:
        data = [d for d in data if d.get('user') == user]
    if pred is not None:
        try:
            pval = int(pred)
            data = [d for d in data if d.get('pred') == pval]
        except ValueError:
            pass

    return jsonify(data[:limit])

@app.route('/export', methods=['GET'])
def export():
    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    csv_bytes = export_predictions_to_csv(data)
    return send_file(
        io.BytesIO(csv_bytes),
        mimetype='text/csv',
        as_attachment=True,
        download_name='predictions_export.csv'
    )

# ---------------- Página de estadísticas ----------------
@app.route('/stats')
def stats():
    """Renderiza stats.html con todas las predicciones"""
    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        history = json.load(fh)
    return render_template("stats.html", history=history)

# ---------------- QR que contiene predicciones ----------------
@app.route('/generate_qr', methods=['GET'])
def generate_qr():
    """Genera un QR con los datos actuales de predicciones"""
    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        history = json.load(fh)
    # ⚡ Aquí el QR contiene directamente los datos de predicción
    img_bytes = generate_qr_from_data(history)
    return send_file(io.BytesIO(img_bytes), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
