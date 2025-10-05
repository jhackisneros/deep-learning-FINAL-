import qrcode
import io

def generate_qr_from_data(data: list, max_rows: int = 20) -> bytes:
    """
    Genera un QR que contiene un HTML con las últimas predicciones.
    Devuelve los bytes de la imagen PNG.
    """
    # Limitar filas para que no se vuelva demasiado largo el QR
    data = data[:max_rows]

    rows = ""
    for rec in data:
        time = rec.get('time', '-')
        filename = rec.get('filename') or "Canvas"
        pred = rec.get('pred', '-')
        conf = f"{(rec.get('confidence', 0)*100):.2f}%"
        rows += f"<tr><td>{time}</td><td>{filename}</td><td>{pred}</td><td>{conf}</td></tr>"

    html = f"""<html><head><meta charset="UTF-8"><title>Mis Predicciones</title></head>
    <body>
    <h3>Últimas Predicciones</h3>
    <table border="1" cellpadding="3" cellspacing="0">
    <tr><th>Hora</th><th>Archivo/Canvas</th><th>Predicción</th><th>Confianza</th></tr>
    {rows}
    </table>
    </body></html>"""

    qr = qrcode.QRCode(
        version=None,  # ⚡ autoajustable según la cantidad de datos
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=8,
        border=2,
    )
    qr.add_data(html)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convertir a bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()
