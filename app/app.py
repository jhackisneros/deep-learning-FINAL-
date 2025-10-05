# app.py
import sys
import os
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
import io
import json
from datetime import datetime
import numpy as np
from tensorflow import keras
from utils.preprocessing import preprocess_image
from utils.qr_utils import generate_qr_from_url
from utils.export_utils import export_predictions_to_csv

app = Flask(__name__)
app.secret_key = "super_secret_key"
app.config['UPLOAD_FOLDER'] = os.path.join('app', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------- Cargar modelos ----------------
MODEL_PATH = os.path.join('models', 'mnist_compiled_model.keras')
mlp_model = None
if os.path.exists(MODEL_PATH):
    mlp_model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    mlp_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
else:
    print(f"⚠️ Warning: modelo MLP no encontrado.")

CNN_MODEL_PATH = os.path.join('models', 'cnn_model.keras')
cnn_model = None
if os.path.exists(CNN_MODEL_PATH):
    cnn_model = keras.models.load_model(CNN_MODEL_PATH)
else:
    print(f"⚠️ Warning: modelo CNN no encontrado.")

# ---------------- Métricas CNN ----------------
CNN_METRICS_PATH = os.path.join('models', 'cnn_metrics.json')
cnn_metrics = {}
if os.path.exists(CNN_METRICS_PATH):
    with open(CNN_METRICS_PATH, 'r') as f:
        cnn_metrics = json.load(f)

# ---------------- Log de predicciones ----------------
PRED_LOG = os.path.join('app', 'predictions.json')
if not os.path.exists(PRED_LOG):
    with open(PRED_LOG, 'w', encoding='utf-8') as f:
        json.dump([], f)

# ---------------- Usuarios demo ----------------
USERS = {}

# ---------------- Helpers ----------------
def save_prediction_local(record: dict):
    """Guarda predicción en predictions.json"""
    with open(PRED_LOG, 'r+', encoding='utf-8') as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            data = []
        data.insert(0, record)
        fh.seek(0)
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.truncate()

def get_current_user():
    """Devuelve usuario logueado o ID anónimo"""
    if session.get('user'):
        return session['user']
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]

def sanitize_history(data):
    """Rellena valores faltantes para compatibilidad con JSON viejo"""
    for rec in data:
        if 'model' not in rec:
            rec['model'] = 'MLP'
        if 'confidence' not in rec or rec['confidence'] is None:
            rec['confidence'] = 0.0
    return data

# ---------------- Rutas principales ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if USERS.get(username) == password:
            session['user'] = username
            return redirect(url_for('index'))
        return render_template('login.html', error="Usuario o contraseña incorrectos")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in USERS:
            return render_template('register.html', error="Usuario ya existe")
        USERS[username] = password
        session['user'] = username
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/profile')
def profile():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template('profile.html', user=session['user'])

@app.route('/my_predictions')
def my_predictions():
    user_id = get_current_user()
    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    user_data = [d for d in data if d.get('user') == user_id]
    user_data = sanitize_history(user_data)
    return render_template('my_predictions.html', history=user_data)

# ---------------- Predicciones ----------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    user_id = get_current_user()

    results = []
    try:
        arr = preprocess_image(data['image'], target_size=(28,28), flatten=True)
        if mlp_model:
            pred_mlp = mlp_model.predict(arr.reshape(1, -1))
            label_mlp = int(np.argmax(pred_mlp))
            conf_mlp = float(np.max(pred_mlp))
            record_mlp = {
                'time': datetime.utcnow().isoformat(),
                'user': user_id,
                'filename': None,
                'pred': label_mlp,
                'confidence': conf_mlp,
                'model': 'MLP'
            }
            save_prediction_local(record_mlp)
            results.append(record_mlp)

        if cnn_model:
            pred_cnn = cnn_model.predict(arr.reshape(1,28,28,1))
            label_cnn = int(np.argmax(pred_cnn))
            conf_cnn = float(np.max(pred_cnn))
            record_cnn = {
                'time': datetime.utcnow().isoformat(),
                'user': user_id,
                'filename': None,
                'pred': label_cnn,
                'confidence': conf_cnn,
                'model': 'CNN'
            }
            save_prediction_local(record_cnn)
            results.append(record_cnn)

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files provided'}), 400
    user_id = get_current_user()
    results = []

    for f in files:
        try:
            arr = preprocess_image(f.read(), target_size=(28,28), flatten=True)
            # MLP
            if mlp_model:
                pred_mlp = mlp_model.predict(arr.reshape(1, -1))
                label_mlp = int(np.argmax(pred_mlp))
                conf_mlp = float(np.max(pred_mlp))
                record_mlp = {
                    'time': datetime.utcnow().isoformat(),
                    'user': user_id,
                    'filename': f.filename,
                    'pred': label_mlp,
                    'confidence': conf_mlp,
                    'model': 'MLP'
                }
                save_prediction_local(record_mlp)
                results.append(record_mlp)
            # CNN
            if cnn_model:
                pred_cnn = cnn_model.predict(arr.reshape(1,28,28,1))
                label_cnn = int(np.argmax(pred_cnn))
                conf_cnn = float(np.max(pred_cnn))
                record_cnn = {
                    'time': datetime.utcnow().isoformat(),
                    'user': user_id,
                    'filename': f.filename,
                    'pred': label_cnn,
                    'confidence': conf_cnn,
                    'model': 'CNN'
                }
                save_prediction_local(record_cnn)
                results.append(record_cnn)
        except Exception as e:
            results.append({'filename': getattr(f, 'filename', None), 'error': str(e)})
    return jsonify(results)

# ---------------- Historial y stats ----------------
@app.route('/history', methods=['GET'])
def history():
    user_id = get_current_user()
    limit = int(request.args.get('limit', 50))
    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    data = [d for d in data if d.get('user') == user_id]
    data = sanitize_history(data)
    return jsonify(data[:limit])

@app.route('/export', methods=['GET'])
def export():
    user_id = get_current_user()
    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    data = [d for d in data if d.get('user') == user_id]
    csv_bytes = export_predictions_to_csv(data)
    return send_file(
        io.BytesIO(csv_bytes),
        mimetype='text/csv',
        as_attachment=True,
        download_name='predictions_export.csv'
    )

@app.route('/stats')
def stats():
    user_id = get_current_user()
    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    data = [d for d in data if d.get('user') == user_id]
    data = sanitize_history(data)

    mlp_acc = "Ya calculada en Colab" if mlp_model else None
    cnn_acc = cnn_metrics.get("cnn_test_accuracy")

    return render_template("stats.html", history=data, mlp_accuracy=mlp_acc, cnn_accuracy=cnn_acc)

# ---------------- QR ----------------
@app.route('/generate_qr', methods=['GET'])
def generate_qr_route():
    user_id = get_current_user()
    url = url_for('qr_view', user_id=user_id, _external=True)
    img_bytes = generate_qr_from_url(url)
    return send_file(io.BytesIO(img_bytes), mimetype='image/png')

@app.route('/qr_view/<user_id>')
def qr_view(user_id):
    with open(PRED_LOG, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    user_history = [d for d in data if d.get('user') == user_id]
    user_history = sanitize_history(user_history)
    return render_template('qr_view.html', history=user_history)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
