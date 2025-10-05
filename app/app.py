# Proyecto Deep-Learning
import sys
import os
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, jsonify, render_template, send_file, session
import io
import json
from datetime import datetime
import numpy as np
from tensorflow import keras
from utils.preprocessing import preprocess_image
from utils.qr_utils import generate_qr_from_data
from utils.export_utils import export_predictions_to_csv

app = Flask(__name__)
app.secret_key = "super_secret_key"  # ⚡ necesario para manejar sesión anónima
app.config['UPLOAD_FOLDER'] = os.path.join('app', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Cargar modelo si existe ---
MODEL_PATH = os.path.join('models', 'mnist_compiled_model.keras')
model = None
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
else:
    print(f"⚠️ Warning: modelo no encontrado en {MODEL_PATH}. Rutas de predicción estarán inactivas.")

# --- Log de predicciones ---
PRED_LOG = os.path.join('app', 'predictions.json')
if not os.path.exists(PRED_LOG):
    with open(PRED_LOG, 'w', encoding='utf-8') as f:
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

def get_current_user():
    """Devuelve un ID único para cada usuario anónimo"""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]

# ---------------- Rutas principales ----------------
@app.route('/')
def index():
    return render_template('index.html')

# ---------------- Rutas de predicción ----------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    img_input = data['image']
    user_id = get_current_user()

    try:
        arr = preprocess_image(img_input, target_size=(28,28), flatten=True)
        pred = model.predict(arr.reshape(1, -1))
        label = int(np.argmax(pred))
        confidence = float(np.max(pred))

        record = {
            'time': datetime.utcnow().isoformat(),
            'user': user_id,
            'filename': None,  # canvas
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

    user_id = get_current_user()

    for f in files:
        try:
            img_bytes = f.read()
            arr = preprocess_image(img_bytes, target_size=(28,28), flatten=True)
            pred = model.predict(arr.reshape(1, -1))
            label = int(np.argmax(pred))
            confidence = float(np.max(pred))
            record = {
                'time': datetime.utcnow().isoformat(),
                'user': user_id,
                'filename': f.filename or None,
                'pred': label,
                'confidence': confidence,
            }
            save_prediction_local(record)
            results.append(record)
        except Exception as e:
            results.append({'filename': getattr(f, 'filename', None), 'error': str(e)})

    return jsonify(results)

# ---------------- Historial de predicciones ----------------
@app.route('/history', methods=['GET'])
def history():
    user_id = get_current_user()
    limit = int(request.args.get('limit', 50))

    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    # Filtrar solo por el usuario actual
    data = [d for d in data if d.get('user') == user_id]

    return jsonify(data[:limit])

# ---------------- Exportación CSV ----------------
@app.route('/export', methods=['GET'])
def export():
    user_id = get_current_user()
    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    # Exportar solo las predicciones del usuario actual
    data = [d for d in data if d.get('user') == user_id]

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
    """Renderiza stats.html con predicciones SOLO del usuario actual"""
    user_id = get_current_user()
    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    data = [d for d in data if d.get('user') == user_id]
    return render_template("stats.html", history=data)

# ---------------- QR de estadísticas ----------------
@app.route('/generate_qr', methods=['GET'])
def generate_qr():
    """Genera un QR con las predicciones SOLO del usuario actual"""
    user_id = get_current_user()
    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    # Filtrar predicciones de este usuario
    user_history = [d for d in data if d.get('user') == user_id]

    img_bytes = generate_qr_from_data(user_history)
    return send_file(io.BytesIO(img_bytes), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
