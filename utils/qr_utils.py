import qrcode
import io
import json

def generate_qr_from_data(data: list) -> bytes:
    """
    Genera un QR que contiene un HTML con las predicciones.
    Devuelve los bytes de la imagen PNG.
    """
    # Crear HTML simple con tabla de predicciones
    rows = ""
    for rec in data:
        rows += f"<tr><td>{rec.get('time')}</td><td>{rec.get('filename','Canvas')}</td><td>{rec.get('pred')}</td><td>{rec.get('confidence')*100:.2f}%</td></tr>"

    html = f"""
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Mis Predicciones</title>
        </head>
        <body>
            <h2>Mis Predicciones</h2>
            <table border="1" cellpadding="5">
                <tr><th>Hora</th><th>Archivo/Canvas</th><th>Predicción</th><th>Confianza</th></tr>
                {rows}
            </table>
        </body>
    </html>
    """

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(html)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convertir a bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()
