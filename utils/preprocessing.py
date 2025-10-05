# utils/preprocessing.py
import numpy as np
from PIL import Image
import base64
import io

# Compatibilidad con diferentes versiones de Pillow
try:
    resample_method = Image.Resampling.LANCZOS  # Pillow 10+
except AttributeError:
    resample_method = Image.LANCZOS  # Pillow <10

def preprocess_image(image_input, target_size=(28,28), flatten=True):
    """
    Convierte una imagen (bytes, PIL.Image o base64) a un array normalizado listo para Keras.

    Params:
    - image_input: bytes, PIL.Image o string base64 ('data:image/png;base64,...')
    - target_size: tupla (alto, ancho)
    - flatten: True para MLP (784,), False para CNN (28,28,1)

    Returns:
    - np.array listo para modelo MLP o CNN
    """
    # --- Detectar base64 ---
    if isinstance(image_input, str) and image_input.startswith('data:image'):
        header, base64_data = image_input.split(',', 1)
        image_input = base64.b64decode(base64_data)

    # --- Convertir bytes a PIL.Image ---
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert('L')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('L')
    else:
        raise ValueError("Tipo de entrada no soportado: debe ser bytes, PIL.Image o base64")

    # --- Redimensionar ---
    img = img.resize(target_size, resample_method)

    # --- Normalizar ---
    arr = np.array(img, dtype=np.float32) / 255.0

    if flatten:
        return arr.flatten()  # MLP
    else:
        return arr[..., np.newaxis]  # CNN (28,28,1)

def preprocess_canvas_data(canvas_data, target_size=(28,28), flatten=True):
    """
    Convierte datos de canvas (array 2D o 1D con valores 0-255) a formato listo para Keras.
    
    Params:
    - canvas_data: array 2D o 1D con valores 0-255
    - target_size: tupla (alto, ancho)
    - flatten: True para MLP, False para CNN

    Returns:
    - np.array listo para modelo MLP o CNN
    """
    arr = np.array(canvas_data, dtype=np.float32)
    if arr.max() > 1:
        arr /= 255.0  # normalizar

    img = Image.fromarray((arr*255).astype(np.uint8)).convert('L')
    img = img.resize(target_size, resample_method)
    
    img_array = np.array(img, dtype=np.float32)/255.0

    if flatten:
        return img_array.flatten()
    else:
        return img_array[..., np.newaxis]
