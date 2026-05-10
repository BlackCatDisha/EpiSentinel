document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
    initDashboard();
    initChatbot();
});

function initSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle');

    if (!sidebar || !toggleBtn) return;

    toggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });
}

const INDIA_GEOJSON_URL = 'https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson';
const STATE_GEOJSON_BASE_URL = 'https://raw.githubusercontent.com/inosaint/StatesOfIndia/master/';

let currentView = 'india'; // 'india' or 'state'
let currentGeoJSON = null;
let currentLevelData = stateData;
let selectedState = null;
let stateDistrictData = {}; // Cache for dummy district data

async function initDashboard() {
    // Add resize listener
    window.addEventListener('resize', debounce(() => {
        if (currentGeoJSON) renderMap(currentGeoJSON, currentView === 'india' ? 'state' : 'district');
    }, 250));

    // Back button logic
    const backBtn = document.getElementById('back-to-india');
    if (backBtn) {
        backBtn.addEventListener('click', () => {
            loadIndiaView();
        });
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

    // Start with India view
    loadIndiaView();
}

async function loadIndiaView() {
    currentView = 'india';
    currentLevelData = stateData;
    selectedState = null;

    document.getElementById('map-title').textContent = 'India Overview';
    document.getElementById('back-to-india').classList.add('hidden');
    document.getElementById('map-loader').style.display = 'block';

    try {
        const response = await fetch(INDIA_GEOJSON_URL);
        currentGeoJSON = await response.json();
        renderMap(currentGeoJSON, 'state');
        updateGlobalStats(stateData);
    } catch (error) {
        console.error('Error loading India map:', error);
        showMapError(error.message);
    }
}

async function loadStateView(stateName) {
    currentView = 'state';
    selectedState = stateName;

    document.getElementById('map-title').textContent = `${stateName} Analysis`;
    document.getElementById('back-to-india').classList.remove('hidden');
    document.getElementById('map-loader').style.display = 'block';
    document.getElementById('map-error').classList.add('hidden');

    try {
        let geojson = null;

        // 1. Try local Karnataka first
        if (stateName === 'Karnataka') {
            try {
                const response = await fetch('karnataka_districts.json');
                if (response.ok) geojson = await response.json();
            } catch (e) { console.warn("Local Karnataka GeoJSON failed, trying fallbacks."); }
        }

        // 2. Try online sources if not loaded yet
        if (!geojson) {
            const stateSlug = stateName.toLowerCase().replace(/ /g, '_');
            const stateSlugSimple = stateName.toLowerCase().replace(/ /g, '');

            const urls = [
                `${STATE_GEOJSON_BASE_URL}${stateSlug}.geojson`,
                `https://raw.githubusercontent.com/geohacker/india/master/district/${stateSlug}.geojson`,
                `https://raw.githubusercontent.com/geohacker/india/master/district/${stateSlugSimple}.geojson`,
                `https://raw.githubusercontent.com/HindustanTimesLabs/shapefiles/master/india/${stateSlug}.json`
            ];

            for (const url of urls) {
                try {
                    const response = await fetch(url);
                    if (response.ok) {
                        geojson = await response.json();
                        console.log(`Loaded ${stateName} map from: ${url}`);
                        break;
                    }
                } catch (e) { continue; }
            }
        }

        if (!geojson) {
            throw new Error(`District boundaries for ${stateName} could not be retrieved from online repositories.`);
        }

        currentGeoJSON = geojson;

        // 3. Use real data for Karnataka, generate dummy for others
        if (stateName === 'Karnataka') {
            currentLevelData = districtData;
        } else {
            if (!stateDistrictData[stateName]) {
                stateDistrictData[stateName] = generateDummyDistricts(currentGeoJSON.features);
            }
            currentLevelData = stateDistrictData[stateName];
        }

        renderMap(currentGeoJSON, 'district');
        updateGlobalStats(currentLevelData);
    } catch (error) {
        console.error(`Error loading ${stateName} map:`, error);
        showMapError(`Failed to load district map for ${stateName}. ${error.message}`);
        setTimeout(loadIndiaView, 8000); // Give user more time to read the error
    }
}

function renderMap(geojson, type) {
    const mapContainer = document.getElementById('map-canvas');
    if (!mapContainer) return;

    // Clear previous SVG
    d3.select('#map-canvas svg').remove();

    // Get container dimensions accurately
    const rect = mapContainer.getBoundingClientRect();
    let width = rect.width;
    let height = rect.height;

    // Fallback if dimensions are still 0
    if (width === 0 || height === 0) {
        width = mapContainer.clientWidth || 800;
        height = mapContainer.clientHeight || 500;
    }

    const svg = d3.select('#map-canvas')
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

    document.getElementById('map-loader').style.display = 'none';

    const projection = d3.geoMercator();

    // fitSize requires an array of [width, height]
    projection.fitSize([width, height], geojson);

    const path = d3.geoPath().projection(projection);

    const tooltip = d3.select('#tooltip');

    svg.selectAll('.region')
        .data(geojson.features)
        .enter()
        .append('path')
        .attr('class', 'region')
        .attr('d', path)
        .attr('fill', d => {
            const name = d.properties.ST_NM || d.properties.NAME_1 || d.properties.district || d.properties.NAME_2 || d.properties.name;
            const normName = normalizeName(name);
            const data = currentLevelData[normName];

            if (!data) return '#334155';
            if (data.risk_score > 70) return '#ef4444';
            if (data.risk_score > 50) return '#eab308';
            return '#22c55e';
        })
        .style('stroke', 'var(--bg-dark)')
        .style('stroke-width', '0.5px')
        .style('cursor', 'pointer')
        .on('mouseover', function (event, d) {
            const name = d.properties.ST_NM || d.properties.NAME_1 || d.properties.district || d.properties.NAME_2 || d.properties.name;
            const normName = normalizeName(name);
            const data = currentLevelData[normName];

            d3.select(this).style('stroke', 'white').style('stroke-width', '2px');

            const tt = document.getElementById('tooltip');
            tt.classList.remove('hidden');
            tt.innerHTML = `
                <div style="font-weight: 700; color: #ef4444; margin-bottom: 4px;">${normName}</div>
                <div style="font-size: 0.8rem;">Predicted Cases: <strong>${data ? data.predicted_cases : 'N/A'}</strong></div>
                <div style="font-size: 0.8rem;">Risk Score: <strong>${data ? data.risk_score + '%' : 'N/A'}</strong></div>
                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 4px;">Click to ${type === 'state' ? 'drill down' : 'view details'}</div>
            `;
        })
        .on('mousemove', function (event) {
            const tt = d3.select('#tooltip');
            tt.style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY + 10) + 'px');
        })
        .on('mouseout', function () {
            d3.select(this).style('stroke', 'var(--bg-dark)').style('stroke-width', '0.5px');
            document.getElementById('tooltip').classList.add('hidden');
        })
        .on('click', function (event, d) {
            const name = d.properties.ST_NM || d.properties.NAME_1 || d.properties.district || d.properties.NAME_2 || d.properties.name;
            const normName = normalizeName(name);

            if (type === 'state') {
                loadStateView(normName);
            } else {
                showDistrictDetails(normName, currentLevelData);
            }
        });
}

function showDistrictDetails(name, dataSet) {
    const infoContainer = document.getElementById('district-info');
    const modal = document.getElementById('district-modal');
    const data = dataSet[name];

    if (!data) {
        infoContainer.innerHTML = `<div class="empty-state"><p>No detailed data available for ${name}</p></div>`;
        modal.classList.remove('hidden');
        return;
    }

    let badgeColor = 'rgba(34, 197, 94, 0.2)';
    let textColor = '#22c55e';
    if (data.risk_score > 70) {
        badgeColor = 'rgba(239, 68, 68, 0.2)';
        textColor = '#ef4444';
    } else if (data.risk_score > 50) {
        badgeColor = 'rgba(234, 179, 8, 0.2)';
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
                <p class="trend ${data.predicted_cases > 50 ? 'up' : 'down'}">
                    ${data.predicted_cases > 50 ? '<i data-lucide="trending-up"></i> High Volume' : '<i data-lucide="trending-down"></i> Within Normal Range'}
                </p>
            </div>
            <div class="stat-item">
                <p class="label">Risk Score</p>
                <p class="value">${data.risk_score}%</p>
                <p class="trend">Machine Learning Prediction</p>
            </div>
            <div class="stat-item">
                <p class="label">Primary Risk Driver</p>
                <p class="value" style="font-size: 1.1rem; color: var(--primary-red)">
                    ${data.top_driver || 'Environmental Conditions'}
                </p>
                <p class="trend">Explainability Layer (XAI)</p>
            </div>
            <div class="stat-item">
                <p class="label">Recommended Action</p>
                <p class="value" style="font-size: 1.1rem; color: var(--accent-blue)">
                    ${data.risk_score > 60 ? 'Intensive Surveillance' : 'Routine Monitoring'}
                </p>
            </div>
        </div>
        <div class="explanation-box" style="margin-top: 1.5rem; padding: 1.5rem; background: rgba(255,255,255,0.03); border-radius: 1rem; border: 1px solid rgba(255,255,255,0.1);">
            <h3 style="font-size: 1rem; color: var(--text-muted); margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
                <i data-lucide="info" style="width: 18px; height: 18px;"></i>
                Machine Learning Explanation (XAI)
            </h3>
            <p style="font-size: 0.95rem; line-height: 1.6; color: var(--text-bright);">
                ${data.detailed_explanation || 'Detailed environmental and historical factors are being analyzed for this region.'}
            </p>
        </div>
    `;

    modal.classList.remove('hidden');
    lucide.createIcons();
}

function updateGlobalStats(dataSet) {
    let totalCases = 0;
    let totalRisk = 0;
    let highRiskCount = 0;
    const items = Object.values(dataSet);

    items.forEach(d => {
        totalCases += d.predicted_cases;
        totalRisk += d.risk_score;
        if (d.risk_score > 65) highRiskCount++;
    });

    document.getElementById('total-cases').textContent = totalCases.toLocaleString();
    document.getElementById('avg-risk').textContent = (totalRisk / items.length).toFixed(1) + '%';
    document.getElementById('high-risk-count').textContent = highRiskCount;
}

function showMapError(message) {
    document.getElementById('map-loader').style.display = 'none';
    const errorDiv = document.getElementById('map-error');
    errorDiv.classList.remove('hidden');
    errorDiv.querySelector('p').textContent = message;
    lucide.createIcons();
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

        setTimeout(() => {
            let response = "I'm analyzing the data for you. Karnataka currently shows the highest aggregated risk score. Would you like to see a state-wise breakdown?";
            if (text.toLowerCase().includes('karnataka')) {
                response = "In Karnataka, Kolar and Mysuru are currently high-risk districts. Would you like to see the detailed prediction for these areas?";
            }
            addMessage(response, 'system');
        }, 1000);
    };

    sendBtn.addEventListener('click', handleSend);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSend();
    });
}
