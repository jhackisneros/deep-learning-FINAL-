# save_cnn_metrics.py
import os
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist

# --- Configurar rutas ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cnn_model.keras')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'cnn_metrics.json')

# --- Cargar modelo CNN ---
cnn_model = keras.models.load_model(MODEL_PATH)
print("✅ CNN cargada correctamente")

# --- Cargar datos MNIST ---
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, axis=-1)  # CNN espera shape (n,28,28,1)

# --- Evaluar precisión ---
y_test_cat = keras.utils.to_categorical(y_test, 10)
loss, acc = cnn_model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Precisión CNN sobre test set: {acc:.4f}")

# --- Guardar métricas ---
metrics = {"cnn_test_accuracy": float(acc)}
with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Métricas guardadas en {METRICS_PATH}")
