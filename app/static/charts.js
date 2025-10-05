// static/charts.js

document.addEventListener("DOMContentLoaded", () => {
    // --- Configuración de gráficas ---
    const ctxLoss = document.getElementById("loss-chart")?.getContext("2d");
    const ctxAcc = document.getElementById("accuracy-chart")?.getContext("2d");

    if (!ctxLoss || !ctxAcc) return; // Evitar errores si los canvas no existen

    const lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Pérdida (Loss)',
                data: [],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: true }
            },
            scales: { y: { beginAtZero: true } }
        }
    });

    const accChart = new Chart(ctxAcc, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Precisión (Accuracy)',
                data: [],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: true }
            },
            scales: { y: { min: 0, max: 1 } }
        }
    });

    // --- Función para actualizar las gráficas dinámicamente ---
    async function updateCharts() {
        try {
            const res = await fetch("/history?limit=50"); // Traer historial de predicciones
            if (!res.ok) return;

            const data = await res.json();
            if (!Array.isArray(data) || data.length === 0) return;

            // Etiquetas de eje X: número de predicción
            const labels = data.map((_, i) => i + 1);
            const accData = data.map(d => d.confidence ?? 0);
            const lossData = data.map(d => 1 - (d.confidence ?? 0));

            // Actualizar precisión
            accChart.data.labels = labels;
            accChart.data.datasets[0].data = accData;
            accChart.update();

            // Actualizar pérdida
            lossChart.data.labels = labels;
            lossChart.data.datasets[0].data = lossData;
            lossChart.update();
        } catch (err) {
            console.error("Error actualizando gráficas:", err);
        }
    }

    updateCharts();
    setInterval(updateCharts, 5000); // Refrescar cada 5 segundos
});
