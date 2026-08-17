let map;
let marker;

document.addEventListener('DOMContentLoaded', () => {
    // Initialize map centered around US as default
    map = L.map('map').setView([39.8283, -98.5795], 4); 

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    const latInput = document.getElementById('latitude');
    const lngInput = document.getElementById('longitude');

    map.on('click', function(e) {
        const lat = e.latlng.lat;
        const lng = e.latlng.lng;
        
        if (marker) {
            marker.setLatLng(e.latlng);
        } else {
            marker = L.marker(e.latlng).addTo(map);
        }
        
        latInput.value = lat.toFixed(6);
        lngInput.value = lng.toFixed(6);
    });
});

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('loadingSpinner');
    const resultsPanel = document.getElementById('resultsPanel');

    // UI Loading State
    btnText.textContent = 'Processing...';
    spinner.classList.remove('hidden');
    submitBtn.disabled = true;
    resultsPanel.classList.add('hidden');

    const payload = {
        latitude: document.getElementById('latitude').value,
        longitude: document.getElementById('longitude').value,
        date_start: document.getElementById('date_start').value,
        date_end: document.getElementById('date_end').value,
        model_type: document.getElementById('model_type').value
    };

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'An error occurred during prediction.');
        }

        // Populate Results
        document.getElementById('resPrediction').textContent = data.prediction;
        
        // Change color based on grass
        if (data.is_grass) {
            document.getElementById('resPrediction').style.color = '#00f2fe';
        } else {
            document.getElementById('resPrediction').style.color = '#ff6b6b';
        }

        document.getElementById('resConfidence').textContent = `${data.confidence.toFixed(1)}%`;
        document.getElementById('resPercentage').textContent = `${data.grass_percentage.toFixed(1)}%`;
        document.getElementById('resNdvi').textContent = data.ndvi_mean.toFixed(3);
        
        // Append a timestamp query parameter to bypass browser image caching for new predictions
        const timestamp = new Date().getTime();
        document.getElementById('resMap').src = `${data.visualization_path}?t=${timestamp}`;
        document.getElementById('resNdviMap').src = data.ndvi_thumbnail_url;

        // Show Results
        resultsPanel.classList.remove('hidden');

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        // Reset UI
        btnText.textContent = 'Analyze Terrain';
        spinner.classList.add('hidden');
        submitBtn.disabled = false;
    }
});
