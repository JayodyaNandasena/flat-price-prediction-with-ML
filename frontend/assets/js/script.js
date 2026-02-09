const API_URL = 'http://localhost:8000/predict';

document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    // Hide previous results/errors
    document.getElementById('resultContainer').classList.remove('show');
    document.getElementById('resultContainer').classList.add('empty');
    document.getElementById('errorContainer').classList.remove('show');

    // Show loading state
    const predictBtn = document.getElementById('predictBtn');
    const spinner = document.getElementById('loadingSpinner');
    const btnText = predictBtn.querySelector('span:first-child');
    predictBtn.disabled = true;
    spinner.style.display = 'inline-block';
    btnText.textContent = 'Calculating...';

    // Collect form data
    const formData = new FormData(e.target);
    const data = {};

    for (let [key, value] of formData.entries()) {
        if (key === 'extra_area_type_name' || key === 'district_name' ||
            key === 'gas' || key === 'hot_water' || key === 'central_heating') {
            data[key] = value;
        } else {
            const numValue = parseFloat(value);
            data[key] = Number.isInteger(numValue) ? parseInt(value) : numValue;
        }
    }

    console.log('Sending data:', data);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `API Error: ${response.status}`);
        }

        const result = await response.json();
        console.log('API Response:', result);

        // Display result
        const predictedPrice = result.predicted_price;
        const MAE = 130038.20; // validated mean absolute error (RUB)

        const lowerBound = predictedPrice - MAE;
        const upperBound = predictedPrice + MAE;

        const formatter = new Intl.NumberFormat('ru-RU', {
            style: 'currency',
            currency: 'RUB',
            minimumFractionDigits: 0
        });

        const formattedPrice = formatter.format(Math.round(predictedPrice));
        const formattedPriceRange = `${formatter.format(Math.round(lowerBound))} - ${formatter.format(Math.round(upperBound))}`;

        document.getElementById('predictedPrice').textContent = formattedPrice;
        document.getElementById('predictedPriceRange').textContent = formattedPriceRange;

        // Generate details
        const pricePerSqM = Math.round(predictedPrice / data.total_area);
        const age = 2026 - data.year;
        const details = `
                    <strong>Price per m² : </strong> ${pricePerSqM.toLocaleString('ru-RU')} ₽<br>
                    <strong>Property Age : </strong> ${age} years<br>
                    <strong>District : </strong> ${data.district_name}<br>
                    <strong>Total Living Area : </strong> ${data.total_area} m²<br>
                    <strong>Rooms : </strong> ${data.rooms_count} <br> 
                    <strong>Floor : </strong> ${data.floor} / ${data.floor_max}
                `;
        document.getElementById('resultDetails').innerHTML = details;

        // Show result
        document.getElementById('resultContainer').classList.remove('empty');
        document.getElementById('resultContainer').classList.add('show');

    } catch (error) {
        console.error('Prediction error:', error);

        let errorMessage = 'Failed to get prediction. ';
        if (error.message.includes('Failed to fetch')) {
            errorMessage += 'Please make sure the API server is running at ' + API_URL;
        } else {
            errorMessage += error.message;
        }

        document.getElementById('errorMessage').textContent = errorMessage;
        document.getElementById('errorContainer').classList.add('show');
    } finally {
        predictBtn.disabled = false;
        spinner.style.display = 'none';
        btnText.textContent = 'Get Price Prediction';
    }
});

// Auto-calculate total area
const areaInputs = ['kitchen_area', 'bath_area', 'other_area'];
areaInputs.forEach(id => {
    document.getElementById(id).addEventListener('input', function () {
        const kitchen = parseFloat(document.getElementById('kitchen_area').value) || 0;
        const bath = parseFloat(document.getElementById('bath_area').value) || 0;
        const other = parseFloat(document.getElementById('other_area').value) || 0;
        const total = kitchen + bath + other;
        if (total > 0) {
            document.getElementById('total_area').value = total.toFixed(1);
        }
    });
});

// Floor validation
document.getElementById('floor').addEventListener('input', function () {
    const floor = parseInt(this.value);
    const maxFloor = parseInt(document.getElementById('floor_max').value);
    if (floor && maxFloor && floor > maxFloor) {
        this.setCustomValidity('Floor cannot exceed total floors');
    } else {
        this.setCustomValidity('');
    }
});

document.getElementById('floor_max').addEventListener('input', function () {
    const floor = parseInt(document.getElementById('floor').value);
    const maxFloor = parseInt(this.value);
    const floorInput = document.getElementById('floor');
    if (floor && maxFloor && floor > maxFloor) {
        floorInput.setCustomValidity('Floor cannot exceed total floors');
    } else {
        floorInput.setCustomValidity('');
    }
});