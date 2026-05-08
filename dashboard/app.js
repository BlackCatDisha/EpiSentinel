document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
    initDashboard();
    initChatbot();
});

function initSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle');

    if (!sidebar || !toggleBtn) {
        console.warn('Sidebar or toggle button not found');
        return;
    }

    toggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });
}

const GEOJSON_URL = 'https://raw.githubusercontent.com/inosaint/StatesOfIndia/master/karnataka.geojson';

let cachedGeoJSON = null;

async function initDashboard() {
    const mapContainer = document.getElementById('karnataka-map');
    
    // Add resize listener
    window.addEventListener('resize', debounce(() => {
        if (cachedGeoJSON) renderMap(cachedGeoJSON);
    }, 250));

    try {
        if (!cachedGeoJSON) {
            console.log('Fetching GeoJSON from:', GEOJSON_URL);
            cachedGeoJSON = await d3.json(GEOJSON_URL);
            console.log('GeoJSON loaded successfully');
        }
        
        renderMap(cachedGeoJSON);
        updateGlobalStats();

    } catch (error) {
        console.error('Error loading map:', error);
        document.getElementById('map-loader').style.display = 'none';
        const errorDiv = document.getElementById('map-error');
        errorDiv.classList.remove('hidden');
        errorDiv.querySelector('p').innerHTML = `Failed to load the map: ${error.message}.`;
        lucide.createIcons(); 
    }

    // Modal Close Logic
    const closeBtn = document.getElementById('close-modal');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            document.getElementById('district-modal').classList.add('hidden');
        });
    }

    window.addEventListener('click', (event) => {
        const modal = document.getElementById('district-modal');
        if (event.target === modal) {
            modal.classList.add('hidden');
        }
    });
}

function renderMap(karnataka) {
    const mapContainer = document.getElementById('karnataka-map');
    if (!mapContainer) return;

    // Clear previous SVG
    d3.select('#karnataka-map svg').remove();

    let width = mapContainer.clientWidth;
    let height = mapContainer.clientHeight;

    if (width === 0 || height === 0) {
        width = mapContainer.parentElement.clientWidth - 40;
        height = 500;
    }

    const svg = d3.select('#karnataka-map')
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

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

            if (!data) return '#334155';
            if (data.risk_score > 70) return '#ef4444';
            if (data.risk_score > 50) return '#eab308';
            return '#22c55e';
        })
        .on('mouseover', function (event, d) {
            const geoName = d.properties.district || d.properties.NAME_2 || d.properties.name;
            const name = normalizeName(geoName);
            const data = districtData[name];

            d3.select(this).style('stroke', 'white').style('stroke-width', '2px');

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
            tt.style('left', (event.pageX + 5) + 'px')
              .style('top', (event.pageY + 5) + 'px');
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
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function showDistrictDetails(name) {
    const infoContainer = document.getElementById('district-info');
    const modal = document.getElementById('district-modal');
    const data = districtData[name];

    if (!data) {
        infoContainer.innerHTML = `<div class="empty-state"><p>No detailed data available for ${name}</p></div>`;
        modal.classList.remove('hidden');
        return;
    }

    // Determine Badge Color based on risk
    let badgeColor = 'rgba(34, 197, 94, 0.2)'; // Green
    let textColor = '#22c55e';
    if (data.risk_score > 70) {
        badgeColor = 'rgba(239, 68, 68, 0.2)'; // Red
        textColor = '#ef4444';
    } else if (data.risk_score > 50) {
        badgeColor = 'rgba(234, 179, 8, 0.2)'; // Yellow
        textColor = '#eab308';
    }

    infoContainer.innerHTML = `
        <div class="district-header">
            <div>
                <h2>${name}</h2>
                <p style="color: var(--text-muted)">Epidemic Prediction Report</p>
            </div>
            <div class="badge" style="background: ${badgeColor}; color: ${textColor}; padding: 0.5rem 1.2rem; border-radius: 2rem; font-weight: 700; font-size: 0.9rem; border: 1px solid ${textColor}44;">
                ${data.status.toUpperCase()} RISK
            </div>
        </div>
        <div class="district-stats">
            <div class="stat-item">
                <p class="label">Predicted Cases</p>
                <p class="value">${data.predicted_cases}</p>
                <p class="trend ${data.predicted_cases > 5 ? 'up' : 'down'}">
                    ${data.predicted_cases > 5 ? '<i data-lucide="trending-up"></i> +5% vs avg' : '<i data-lucide="trending-down"></i> -2% vs avg'}
                </p>
            </div>
            <div class="stat-item">
                <p class="label">Risk Score</p>
                <p class="value">${data.risk_score}%</p>
                <p class="trend">Random Forest Prediction</p>
            </div>
            <div class="stat-item">
                <p class="label">Outbreak Threshold</p>
                <p class="value">${data.threshold}</p>
                <p class="trend">Weekly Q75 Baseline</p>
            </div>
            <div class="stat-item">
                <p class="label">Recommended Action</p>
                <p class="value" style="font-size: 1.1rem; color: var(--accent-blue)">
                    ${data.risk_score > 60 ? 'Intensive Surveillance' : 'Routine Monitoring'}
                </p>
            </div>
        </div>
    `;

    modal.classList.remove('hidden');
    lucide.createIcons();
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
