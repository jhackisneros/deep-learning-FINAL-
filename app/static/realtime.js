// static/realtime.js
document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("canvas-mnist");
    const ctx = canvas.getContext("2d");
    const feedbackEl = document.getElementById("prediction-realtime");

    // --- Detectar darkmode ---
    const darkmodeLink = document.getElementById("darkmode-css");
    let isDark = darkmodeLink && !darkmodeLink.disabled;

    function clearCanvas() {
        ctx.fillStyle = isDark ? "#1e1e1e" : "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
    }

    clearCanvas();

    ctx.strokeStyle = "white";
    ctx.lineWidth = 15;
    ctx.lineCap = "round";

    let drawing = false;
    let timeoutId = null;

    canvas.addEventListener("mousedown", () => drawing = true);
    canvas.addEventListener("mouseup", () => drawing = false);
    canvas.addEventListener("mouseout", () => drawing = false);
    canvas.addEventListener("mousemove", drawRealtime);

    function drawRealtime(e) {
        if (!drawing) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);

        // Debounce 500ms para no saturar el servidor
        if (timeoutId) clearTimeout(timeoutId);
        timeoutId = setTimeout(sendRealtimePrediction, 500);
    }

    async function sendRealtimePrediction() {
        const dataURL = canvas.toDataURL("image/png");
        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataURL })
            });
            const json = await res.json();

            if (Array.isArray(json) && json.length > 0) {
                // Ordenar por modelo (MLP primero, CNN después)
                const predictions = json
                    .sort((a, b) => a.model.localeCompare(b.model))
                    .map(p => {
                        const conf = (p.confidence ?? 0) * 100;
                        return `${p.model}: ${p.pred} (${conf.toFixed(2)}%)`;
                    })
                    .join(" | ");
                feedbackEl.innerText = `Predicción en tiempo real: ${predictions}`;
            } else if (json.error) {
                feedbackEl.innerText = `Error: ${json.error}`;
            } else {
                feedbackEl.innerText = "Predicción no disponible";
            }
        } catch (err) {
            feedbackEl.innerText = `Error conexión: ${err.message}`;
        }
    }

    // Botón para limpiar canvas y feedback
    document.getElementById("clear-canvas").addEventListener("click", () => {
        clearCanvas();
        feedbackEl.innerText = "Predicción en tiempo real: -";
    });

    // --- Actualizar colores si cambia el darkmode dinámicamente ---
    if (darkmodeLink) {
        const observer = new MutationObserver(() => {
            isDark = !darkmodeLink.disabled;
            clearCanvas();
        });
        observer.observe(darkmodeLink, { attributes: true, attributeFilter: ["disabled"] });
    }
});
