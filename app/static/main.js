// static/main.js
document.addEventListener("DOMContentLoaded", () => {

    // --- Detectar darkmode ---
    const darkmodeLink = document.getElementById("darkmode-css");
    let isDark = darkmodeLink && !darkmodeLink.disabled;

    // --- Canvas MNIST ---
    const canvas = document.getElementById("canvas-mnist");
    const ctx = canvas.getContext("2d");

    function clearCanvas() {
        ctx.fillStyle = isDark ? "#1e1e1e" : "black"; // fondo según modo
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
    }

    clearCanvas();

    ctx.strokeStyle = "white"; // pincel siempre blanco
    ctx.lineWidth = 15;
    ctx.lineCap = "round";

    let drawing = false;

    canvas.addEventListener("mousedown", () => drawing = true);
    canvas.addEventListener("mouseup", () => drawing = false);
    canvas.addEventListener("mouseout", () => drawing = false);

    canvas.addEventListener("mousemove", (e) => {
        if (!drawing) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    });

    document.getElementById("clear-canvas").addEventListener("click", clearCanvas);

    function getCanvasData() {
        return canvas.toDataURL("image/png");
    }

    // --- Predicción imagen dibujada ---
    const predictCanvasBtn = document.getElementById("predict-canvas-btn");
    const canvasResults = document.getElementById("canvas-results");

    predictCanvasBtn.addEventListener("click", async () => {
        const imgData = getCanvasData();

        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({image: imgData})
            });
            const data = await res.json();
            canvasResults.innerHTML = "";

            // Mostrar predicciones de todos los modelos
            data.sort((a,b) => a.model.localeCompare(b.model)).forEach(d => {
                const p = document.createElement("p");
                const conf = (d.confidence ?? 0) * 100;
                p.textContent = `${d.model}: ${d.pred} (${conf.toFixed(2)}%)`;
                canvasResults.appendChild(p);
            });

        } catch (err) {
            alert("Error al predecir dibujo: " + err.message);
        }
    });

    // --- Subida de imágenes ---
    const fileInput = document.getElementById("file-input");
    const predictBtn = document.getElementById("predict-files-btn");
    const resultsList = document.getElementById("file-results");

    predictBtn.addEventListener("click", async () => {
        const files = fileInput.files;
        if (!files.length) return alert("Selecciona archivos primero");

        const formData = new FormData();
        for (let f of files) formData.append("files", f);

        try {
            const res = await fetch("/predict_batch", { method: "POST", body: formData });
            const data = await res.json();
            resultsList.innerHTML = "";

            // Agrupar resultados por archivo
            const grouped = {};
            data.forEach(d => {
                if (!grouped[d.filename]) grouped[d.filename] = [];
                grouped[d.filename].push(d);
            });

            for (let filename in grouped) {
                const entries = grouped[filename];
                const li = document.createElement("li");

                // Mostrar errores si existen
                if (entries.some(e => e.error)) {
                    li.textContent = `Error (${filename}): ${entries.find(e => e.error).error}`;
                } else {
                    // Mostrar predicciones de todos los modelos
                    const text = entries
                        .sort((a,b) => a.model.localeCompare(b.model))
                        .map(e => `${e.model}: ${e.pred} (${(e.confidence*100).toFixed(2)}%)`)
                        .join(" | ");
                    li.textContent = `${filename}: ${text}`;
                }

                resultsList.appendChild(li);
            }
        } catch (err) {
            alert("Error al predecir archivos: " + err.message);
        }
    });

    // --- Generar QR ---
    const qrBtn = document.getElementById("generate-qr-btn");
    const qrImg = document.getElementById("qr-img");

    qrBtn.addEventListener("click", () => {
        const url = qrBtn.dataset.url;
        qrImg.src = url + "?t=" + new Date().getTime();
    });

    // --- Actualizar colores si cambia el darkmode dinámicamente ---
    if (darkmodeLink) {
        const observer = new MutationObserver(() => {
            isDark = !darkmodeLink.disabled;
            ctx.fillStyle = isDark ? "#1e1e1e" : "black";
            ctx.strokeStyle = "white";
            clearCanvas();
        });
        observer.observe(darkmodeLink, { attributes: true, attributeFilter: ["disabled"] });
    }

});
