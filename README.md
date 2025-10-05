# deep-learning-FINAL-
https://github.com/jhackisneros/deep-learning-FINAL-.git
Este proyecto combina **modelos de Deep Learning** para clasificación de dígitos (dataset MNIST) con una **aplicación web** que permite realizar predicciones y visualizar métricas.  
Está pensado como un proyecto final que integra entrenamiento de modelos, despliegue en una app y conexión con Firebase para autenticación y almacenamiento.

---

## 📁 Estructura del Proyecto

deep-learning-FINAL--main/
├── README.md
├── requirements.txt
├── app/
│ ├── app.py # Aplicación principal
│ ├── auth.py # Autenticación Firebase
│ ├── predictions.json # Predicciones guardadas
│ └── save_cnn_metrics.py # Guarda métricas del modelo
├── firebase/
│ └── serviceAccountkey.json # Clave de servicio Firebase
├── models/
│ ├── CNN_MNIST.keras # Modelo CNN entrenado
│ └── mnist_compiled_model.keras
├── notebooks/
│ ├── (MLP)Ejercicio1_classify_numbers.ipynb
│ ├── CNN_MNIST.ipynb
│ └── CNN_MNIST.keras
└── utils/
├── analysis.py
├── export_utils.py
├── firebase_utils.py
├── generate_qr_from_url.py
├── interpreter.py
├── mlp_numpy.py
├── preprocessing.py
├── qr_utils.py
└── training_utils.py

yaml
Copiar código

---

## 🧰 Requisitos e Instalación

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/usuario/deep-learning-FINAL.git
cd deep-learning-FINAL--main
2️⃣ Crear un entorno virtual (recomendado)
bash
Copiar código
# Linux / Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
3️⃣ Instalar dependencias
Asegúrate de tener Python 3.8+ instalado.

bash
Copiar código
pip install --upgrade pip
pip install -r requirements.txt
🚀 Ejecución del Proyecto
🧠 1. Entrenamiento de modelos (opcional)
Puedes abrir los notebooks en notebooks/ y ejecutar el entrenamiento de la CNN o MLP sobre MNIST.
Por defecto, ya hay modelos guardados en models/, por lo que no es obligatorio volver a entrenarlos.

bash
Copiar código
jupyter notebook notebooks/CNN_MNIST.ipynb
🌐 2. Ejecutar la aplicación web
Dirígete a la carpeta app/ y ejecuta:

bash
Copiar código
cd app
python app.py
Por defecto, la aplicación se abrirá en:

arduino
Copiar código
http://localhost:5000
⚠️ Asegúrate de tener el archivo firebase/serviceAccountkey.json correctamente configurado para que funcione la autenticación.

🔐 Configuración de Firebase
Ve a Firebase Console.

Crea un nuevo proyecto y descarga el archivo de clave de servicio (serviceAccountkey.json).

Colócalo dentro de la carpeta firebase/.

Asegúrate de que las reglas de autenticación estén configuradas para permitir el login.

🐞 Problemas Comunes y Soluciones
Problema	Causa posible	Solución
❌ ModuleNotFoundError	Entorno virtual no activado o dependencias no instaladas	Activa el entorno virtual y ejecuta pip install -r requirements.txt
🔥 Error de TensorFlow / GPU	Falta de drivers o versión incompatible	Verifica que tienes la versión correcta de TensorFlow y, si usas GPU, instala CUDA/CuDNN
🔑 Error con Firebase	Archivo serviceAccountkey.json ausente o mal configurado	Descarga la clave correcta desde Firebase y colócala en firebase/
🧠 Error cargando modelo .keras	Ruta mal especificada o modelo no existente	Asegúrate de que el archivo esté en models/ y que el path coincida en el código

👨‍💻 Estructura Técnica
utils/ → contiene funciones auxiliares para preprocesamiento, entrenamiento, exportación y análisis.

models/ → modelos entrenados en formato .keras.

app/ → app web para probar los modelos, con autenticación y visualización de métricas.

firebase/ → credenciales de conexión a Firebase.
