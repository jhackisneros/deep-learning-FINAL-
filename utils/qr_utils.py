# utils/qr_utils.py
import qrcode
import io
import json

def generate_qr_from_data(data: list) -> bytes:
    """
    Genera un QR que contiene el JSON de las predicciones.
    Devuelve los bytes de la imagen PNG.
    """
    # Convertir la lista de predicciones a JSON
    # Para que el QR no sea demasiado grande, limitamos a las últimas 50 predicciones
    limited_data = data[:50] if len(data) > 50 else data
    text = json.dumps(limited_data, ensure_ascii=False)

    qr = qrcode.QRCode(
        version=None,  # QR automático según tamaño del texto
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convertir a bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()
