document.addEventListener('DOMContentLoaded', () => {
    initDashboard();
    initChatbot();
});

const GEOJSON_URL = 'https://raw.githubusercontent.com/inosaint/StatesOfIndia/master/karnataka.geojson';

async function initDashboard() {
    const mapContainer = document.getElementById('karnataka-map');
    let width = mapContainer.clientWidth;
    let height = mapContainer.clientHeight;

    console.log(`Map container size: ${width}x${height}`);

    if (width === 0 || height === 0) {
        console.warn('Map container has 0 width or height. Using fallback values.');
        width = 800;
        height = 600;
    }

    const svg = d3.select('#karnataka-map')
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

    try {
        console.log('Fetching GeoJSON from:', GEOJSON_URL);
        const karnataka = await d3.json(GEOJSON_URL);
        console.log('GeoJSON loaded successfully:', karnataka);

        document.getElementById('map-loader').style.display = 'none';

        const projection = d3.geoMercator();
        const path = d3.geoPath().projection(projection);

        // Automatically scale and center the map
        projection.fitSize([width, height], karnataka);

        const tooltip = d3.select('#tooltip');

        // Bind data and render districts
        svg.selectAll('.district')
            .data(karnataka.features)
            .enter()
            .append('path')
            .attr('class', 'district')
            .attr('d', path)
            .attr('fill', d => {
                const geoName = d.properties.district || d.properties.NAME_2 || d.properties.name;
                const name = normalizeName(geoName);
                const data = districtData[name];

                if (!data) {
                    console.warn(`No data found for district: ${geoName} (normalized: ${name})`);
                    return '#334155';
                }

                // Color scale: Green -> Yellow -> Red
                if (data.risk_score > 70) return '#ef4444';
                if (data.risk_score > 50) return '#eab308';
                return '#22c55e';
            })
            .on('mouseover', function (event, d) {
                const geoName = d.properties.district || d.properties.NAME_2 || d.properties.name;
                const name = normalizeName(geoName);
                const data = districtData[name];

                d3.select(this).style('stroke', 'white').style('stroke-width', '2px');

                tooltip.classList ? tooltip.classList.remove('hidden') : d3.select('#tooltip').classed('hidden', false);
                const tt = document.getElementById('tooltip');
                tt.classList.remove('hidden');
                tt.innerHTML = `
                    <div style="font-weight: 700; color: #ef4444; margin-bottom: 4px;">${name}</div>
                    <div style="font-size: 0.8rem;">Predicted Cases: <strong>${data ? data.predicted_cases : 'N/A'}</strong></div>
                    <div style="font-size: 0.8rem;">Risk Score: <strong>${data ? data.risk_score + '%' : 'N/A'}</strong></div>
                `;
            })
            .on('mousemove', function (event) {
                const tt = d3.select('#tooltip');
                tt.style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 20) + 'px');
            })
            .on('mouseout', function () {
                d3.select(this).style('stroke', 'var(--bg-dark)').style('stroke-width', '0.5px');
                document.getElementById('tooltip').classList.add('hidden');
            })
            .on('click', function (event, d) {
                const geoName = d.properties.district || d.properties.NAME_2 || d.properties.name;
                const name = normalizeName(geoName);
                showDistrictDetails(name);
            });

        // Update stats
        updateGlobalStats();

    } catch (error) {
        console.error('Error loading map:', error);
        document.getElementById('map-loader').style.display = 'none';
        const errorDiv = document.getElementById('map-error');
        errorDiv.classList.remove('hidden');
        errorDiv.querySelector('p').innerHTML = `Failed to load the map: ${error.message}. <br><br><strong>Important:</strong> If you are opening the file directly, browsers block external data. Please run: <code>python -m http.server 8000</code> in this folder.`;
        lucide.createIcons();
    }
}

function showDistrictDetails(name) {
    const infoContainer = document.getElementById('district-info');
    const data = districtData[name];

    if (!data) {
        infoContainer.innerHTML = `<div class="empty-state"><p>No detailed data available for ${name}</p></div>`;
        return;
    }

    infoContainer.innerHTML = `
        <div class="district-header">
            <div>
                <h2>${name}</h2>
                <p style="color: var(--text-muted)">Epidemic Prediction Report</p>
            </div>
            <div class="badge" style="background: ${data.risk_score > 50 ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)'}; color: ${data.risk_score > 50 ? '#ef4444' : '#22c55e'}; padding: 0.5rem 1rem; border-radius: 2rem; font-weight: 700;">
                ${data.status.toUpperCase()} RISK
            </div>
        </div>
        <div class="district-stats">
            <div class="stat-item">
                <p class="label">Predicted Cases</p>
                <p class="value">${data.predicted_cases}</p>
                <p class="trend up">+5% vs avg</p>
            </div>
            <div class="stat-item">
                <p class="label">Risk Score</p>
                <p class="value">${data.risk_score}%</p>
                <p class="trend">Based on RF Model</p>
            </div>
            <div class="stat-item">
                <p class="label">Outbreak Threshold</p>
                <p class="value">${data.threshold}</p>
                <p class="trend">Weekly Q75</p>
            </div>
            <div class="stat-item">
                <p class="label">Recommended Action</p>
                <p class="value" style="font-size: 1.1rem; color: var(--accent-blue)">${data.risk_score > 60 ? 'Increase Surveillance' : 'Standard Monitoring'}</p>
            </div>
        </div>
    `;
}

function updateGlobalStats() {
    let totalCases = 0;
    let totalRisk = 0;
    let highRiskCount = 0;
    const districts = Object.values(districtData);

    districts.forEach(d => {
        totalCases += d.predicted_cases;
        totalRisk += d.risk_score;
        if (d.risk_score > 65) highRiskCount++;
    });

    document.getElementById('total-cases').textContent = totalCases;
    document.getElementById('avg-risk').textContent = (totalRisk / districts.length).toFixed(1) + '%';
    document.getElementById('high-risk-count').textContent = highRiskCount;
}

function initChatbot() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');

    const addMessage = (text, sender) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        msgDiv.innerHTML = `<p>${text}</p>`;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    const handleSend = () => {
        const text = input.value.trim();
        if (!text) return;

        addMessage(text, 'user');
        input.value = '';

        // Mock response
        setTimeout(() => {
            let response = "I'm analyzing the data for you. District Kolar currently shows the highest risk score at 80.2%. Would you like to see a detailed report for Kolar?";
            if (text.toLowerCase().includes('bangalore') || text.toLowerCase().includes('bengaluru')) {
                response = "Bengaluru Urban is currently in the Low Risk category with a risk score of 30.7%. Predicted cases for next week: 9.";
            }
            addMessage(response, 'system');
        }, 1000);
    };

    sendBtn.addEventListener('click', handleSend);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSend();
    });
}
