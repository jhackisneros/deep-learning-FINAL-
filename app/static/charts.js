// static/charts.js

document.addEventListener("DOMContentLoaded", () => {
    // --- Configuración de gráficas ---
    const ctxLoss = document.getElementById("loss-chart")?.getContext("2d");
    const ctxAcc = document.getElementById("accuracy-chart")?.getContext("2d");

    if (!ctxLoss || !ctxAcc) {
        console.warn("Los canvas de estadísticas no existen en esta página.");
        return;
    }

    const lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'MLP Loss',
                    data: [],
                    borderColor: 'rgba(54,99,235,1)',
                    backgroundColor: 'rgba(54,99,235,0.2)',
                    fill: true,
                    tension: 0.3,
                    spanGaps: true
                },
                {
                    label: 'CNN Loss',
                    data: [],
                    borderColor: 'rgba(255,99,132,1)',
                    backgroundColor: 'rgba(255,99,132,0.2)',
                    fill: true,
                    tension: 0.3,
                    spanGaps: true
                }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: true } },
            scales: { y: { beginAtZero: true } }
        }
    });

    const accChart = new Chart(ctxAcc, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'MLP Accuracy',
                    data: [],
                    borderColor: 'rgba(54,162,235,1)',
                    backgroundColor: 'rgba(54,162,235,0.2)',
                    fill: true,
                    tension: 0.3,
                    spanGaps: true
                },
                {
                    label: 'CNN Accuracy',
                    data: [],
                    borderColor: 'rgba(255,159,64,1)',
                    backgroundColor: 'rgba(255,159,64,0.2)',
                    fill: true,
                    tension: 0.3,
                    spanGaps: true
                }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: true } },
            scales: { y: { min: 0, max: 1 } }
        }
    });

    // --- Función para actualizar gráficas dinámicamente ---
    async function updateCharts() {
        try {
            const res = await fetch("/history?limit=50");
            if (!res.ok) {
                console.warn("No se pudo obtener historial de predicciones:", res.status);
                return;
            }

            const data = await res.json();
            if (!Array.isArray(data) || data.length === 0) {
                console.info("No hay datos de historial disponibles para las gráficas.");
                return;
            }

            const labels = data.map((_, i) => i + 1);

            const mlpAcc = data.map(d => d.model === 'MLP' ? (d.confidence ?? 0) : null);
            const cnnAcc = data.map(d => d.model === 'CNN' ? (d.confidence ?? 0) : null);

            const mlpLoss = mlpAcc.map(v => v !== null ? 1 - v : null);
            const cnnLoss = cnnAcc.map(v => v !== null ? 1 - v : null);

            accChart.data.labels = labels;
            accChart.data.datasets[0].data = mlpAcc;
            accChart.data.datasets[1].data = cnnAcc;
            accChart.update();

            lossChart.data.labels = labels;
            lossChart.data.datasets[0].data = mlpLoss;
            lossChart.data.datasets[1].data = cnnLoss;
            lossChart.update();
        } catch (err) {
            console.error("Error actualizando gráficas:", err);
        }
    }

    updateCharts();
    setInterval(updateCharts, 5000); // Refrescar cada 5 segundos
});
