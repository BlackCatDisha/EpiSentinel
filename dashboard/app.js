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

// Use local GeoJSON file for reliability (no external dependency)
const INDIA_GEOJSON_URL = 'india_states.geojson';

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

    // Only Karnataka has full district data — show preview for others
    if (stateName !== 'Karnataka') {
        document.getElementById('map-loader').style.display = 'none';
        if (currentGeoJSON && currentView === 'state') {
            renderMap(currentGeoJSON, 'state');
        }
        currentLevelData = stateData;
        updateGlobalStats(stateData);
        document.getElementById('back-to-india').classList.remove('hidden');
        showMapMessage(
            `Coming soon — detailed district risk data is currently available only for Karnataka. This preview shows the district boundaries for ${stateName} while we expand our coverage.`,
            'info'
        );
        return;
    }

    try {
        let geojson = null;

        // Load local Karnataka GeoJSON
        try {
            const response = await fetch('karnataka_districts.json');
            if (response.ok) geojson = await response.json();
        } catch (e) { console.warn('Local Karnataka GeoJSON failed:', e); }

        if (!geojson) {
            throw new Error('Could not load Karnataka district boundaries. Make sure karnataka_districts.json is present.');
        }

        currentGeoJSON = geojson;
        currentLevelData = districtData;

        renderMap(currentGeoJSON, 'district');
        updateGlobalStats(currentLevelData);
    } catch (error) {
        console.error(`Error loading ${stateName} map:`, error);
        showMapError(`Failed to load Karnataka district map. ${error.message}`);
        setTimeout(loadIndiaView, 8000);
    }
}

function renderMap(geojson, type) {
    const mapContainer = document.getElementById('map-canvas');
    if (!mapContainer) return;

    // Clear all previous SVGs
    d3.selectAll('#map-canvas svg').remove();

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

            // On India overview: Karnataka glows, all other states are greyed
            if (type === 'state') {
                return normName === 'Karnataka' ? '#10b981' : '#6b7280';
            }

            // District view: colour by risk score
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
            const isPreviewState = type === 'state' && normName !== 'Karnataka';

            d3.select(this).style('stroke', 'white').style('stroke-width', '2px');

            const tt = document.getElementById('tooltip');
            tt.classList.remove('hidden');
            tt.innerHTML = `
                <div style="font-weight: 700; color: #ef4444; margin-bottom: 4px;">${normName}</div>
                <div style="font-size: 0.8rem;">Predicted Cases: <strong>${data ? data.predicted_cases : 'N/A'}</strong></div>
                <div style="font-size: 0.8rem;">Risk Score: <strong>${data ? data.risk_score + '%' : 'N/A'}</strong></div>
                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 4px;">
                    ${isPreviewState ? 'Preview only — detailed Karnataka data is live now.' : `Click to ${type === 'state' ? 'drill down' : 'view details'}`}
                </div>
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

function showMapMessage(message, type = 'error') {
    document.getElementById('map-loader').style.display = 'none';
    const errorDiv = document.getElementById('map-error');
    errorDiv.classList.remove('hidden');
    errorDiv.classList.toggle('info', type === 'info');
    errorDiv.innerHTML = `
        <i data-lucide="${type === 'info' ? 'info' : 'alert-circle'}"></i>
        <p>${message}</p>
    `;
    lucide.createIcons();
}

function hideMapMessage() {
    const errorDiv = document.getElementById('map-error');
    errorDiv.classList.add('hidden');
    errorDiv.classList.remove('info');
}

function showMapError(message) {
    showMapMessage(message, 'error');
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
    const input   = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');

    // ── Helpers ──────────────────────────────────────────────────────────────
    function addMessage(html, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        msgDiv.innerHTML = `<p>${html}</p>`;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTyping() {
        const el = document.createElement('div');
        el.className = 'message system typing-indicator';
        el.id = 'typing-bubble';
        el.innerHTML = '<span></span><span></span><span></span>';
        chatMessages.appendChild(el);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return el;
    }

    function removeTyping() {
        const el = document.getElementById('typing-bubble');
        if (el) el.remove();
    }

    // ── Knowledge Base ────────────────────────────────────────────────────────
    const PREVENTION = [
        '🦟 <b>Eliminate standing water</b> — change cooler water weekly, clear clogged drains, cover water tanks.',
        '🛏️ <b>Use mosquito nets</b> while sleeping, especially during peak transmission (dawn/dusk).',
        '👕 <b>Wear full-sleeve clothing</b> in high-risk areas. Light-coloured fabric repels mosquitoes.',
        '🧴 <b>Apply repellents</b> (DEET or Picaridin-based) when outdoors in high-risk districts.',
        '🏥 <b>Seek immediate care</b> if you experience sudden high fever, severe headache, or rash.',
        '🌿 <b>Use neem oil or citronella</b> as a natural repellent in homes.',
        '🚿 <b>Fumigate high-risk zones</b> — coordinate with local BBMP/PHC for fogging drives.',
        '🧹 <b>Community clean-up drives</b> — clear garbage, tyres, and flower pots that collect water.',
    ];

    const SYMPTOMS = `<b>Dengue Symptoms:</b><br>
• Sudden high fever (104°F / 40°C)<br>
• Severe headache behind eyes<br>
• Muscle/joint pain ("breakbone fever")<br>
• Skin rash (appears 2–5 days after fever)<br>
• Mild bleeding (nose, gums)<br>
• Fatigue and nausea<br><br>
⚠️ <b>Warning signs (go to hospital immediately):</b> severe abdominal pain, vomiting blood, rapid breathing, bleeding under skin.`;

    const TREATMENT = `<b>Dengue Treatment:</b><br>
• <b>No antiviral drug exists</b> — treatment is supportive.<br>
• Drink plenty of fluids (ORS, coconut water, fruit juice).<br>
• Take <b>Paracetamol</b> for fever — <b>avoid Aspirin & Ibuprofen</b> (increase bleeding risk).<br>
• Rest and monitor platelet count daily.<br>
• Hospitalize if platelets drop below 100,000/µL.<br>
• Papaya leaf extract may help improve platelet count (consult doctor first).`;

    const MODEL_INFO = `<b>EpiSentinel Ensemble Model:</b><br>
• <b>ROC-AUC: 0.821</b> | Recall: <b>85.6%</b> | F1: <b>0.720</b><br>
• Weights: XGBoost 49% + CatBoost 30% + RF 9% + Poisson 11% + LightGBM 1%<br>
• Data: 2022–2023 Karnataka district-week records (out-of-sample test)<br>
• Key features: 4-week rolling case avg, lagged rainfall, temperature, humidity, NDWI, seasonal cycle<br>
• Threshold: 0.15 probability → outbreak predicted`;

    // ── Response Engine ───────────────────────────────────────────────────────
    function getResponse(text) {
        const t = text.toLowerCase().trim();

        // ── Greetings ──
        if (/^(hi|hello|hey|namaste|good\s?(morning|evening|afternoon))/.test(t)) {
            return `Hello! 👋 I'm <b>Sentinel AI</b>, your epidemic intelligence assistant.<br>
I can help you with:<br>
• District risk scores & predictions<br>
• Dengue prevention & symptoms<br>
• Model performance details<br>
• High-risk area alerts<br><br>
Try asking: <i>"Which districts are high risk?"</i> or <i>"How to prevent dengue?"</i>`;
        }

        // ── Prevention / precautions ──
        if (/prevent|precaution|avoid|protect|safe|repel|mosquito control|fogging|vector/.test(t)) {
            const tips = PREVENTION.slice(0, 5).join('<br><br>');
            return `🛡️ <b>Dengue Prevention Measures:</b><br><br>${tips}<br><br>
💡 Tip: Click on any district on the map to see its specific risk level and recommended action.`;
        }

        // ── Symptoms ──
        if (/symptom|sign|fever|rash|headache|joint pain|breakbone|platelet|bleed/.test(t)) {
            return SYMPTOMS;
        }

        // ── Treatment / remedy ──
        if (/treat|cure|remedy|medicine|tablet|drug|hospital|doctor|paracetamol|papaya|platelet/.test(t)) {
            return TREATMENT;
        }

        // ── Model / accuracy / how does it work ──
        if (/model|accuracy|recall|auc|roc|f1|ensemble|xgboost|catboost|how.*work|predict|shap|explain/.test(t)) {
            return MODEL_INFO;
        }

        // ── What is dengue / about dengue ──
        if (/what is dengue|about dengue|dengue fever|aedes/.test(t)) {
            return `🦟 <b>What is Dengue?</b><br>
Dengue is a mosquito-borne viral infection transmitted by the <i>Aedes aegypti</i> mosquito.<br><br>
• Affects 400 million people globally per year<br>
• 4 serotypes: DENV-1, 2, 3, 4 (second infection with different type = higher severity)<br>
• Peak season in India: <b>post-monsoon (August–November)</b><br>
• Karnataka hotspots: Bengaluru, Chitradurga, Kolar, Mandya<br><br>
EpiSentinel predicts outbreak probability one week in advance using weather, lagged case counts and satellite data.`;
        }

        // ── Highest risk districts ──
        if (/high.*risk|worst|most danger|critical|top.*district|highest/.test(t)) {
            const sorted = Object.entries(districtData)
                .sort((a, b) => b[1].risk_score - a[1].risk_score)
                .slice(0, 5);
            const list = sorted.map(([d, v]) =>
                `• <b>${d}</b> — Risk: <b>${v.risk_score}%</b> | ${v.status}`
            ).join('<br>');
            return `🔴 <b>Top 5 Highest-Risk Districts:</b><br><br>${list}<br><br>
Click any district on the map for a full SHAP-based explanation.`;
        }

        // ── Lowest risk ──
        if (/low.*risk|safe|least.*danger|lowest/.test(t)) {
            const sorted = Object.entries(districtData)
                .filter(([,v]) => v.risk_score > 0)
                .sort((a, b) => a[1].risk_score - b[1].risk_score)
                .slice(0, 5);
            const list = sorted.map(([d, v]) =>
                `• <b>${d}</b> — Risk: <b>${v.risk_score}%</b>`
            ).join('<br>');
            return `🟢 <b>5 Lowest-Risk Districts:</b><br><br>${list}`;
        }

        // ── Total cases / summary ──
        if (/total|summary|overview|how many case|all district/.test(t)) {
            const items = Object.values(districtData);
            const totalCases = items.reduce((s, d) => s + d.predicted_cases, 0);
            const avgRisk = (items.reduce((s, d) => s + d.risk_score, 0) / items.length).toFixed(1);
            const critical = items.filter(d => d.risk_score > 70).length;
            const high = items.filter(d => d.risk_score > 50 && d.risk_score <= 70).length;
            return `📊 <b>Karnataka Overview (Week 50, 2023):</b><br><br>
• Total predicted cases: <b>${totalCases}</b><br>
• Average risk score: <b>${avgRisk}%</b><br>
• Critical districts (>70%): <b>${critical}</b><br>
• High-risk districts (50–70%): <b>${high}</b><br><br>
Data sourced from EpiSentinel Ensemble model (ROC-AUC: 0.821).`;
        }

        // ── Specific district lookup ──
        const districtNames = Object.keys(districtData);
        const matched = districtNames.find(d => t.includes(d.toLowerCase()));
        if (matched) {
            const d = districtData[matched];
            const riskColor = d.risk_score > 70 ? '🔴' : d.risk_score > 50 ? '🟡' : '🟢';
            const action = d.risk_score > 60 ? 'Intensive Surveillance + Vector Control' : 'Routine Monitoring';
            return `${riskColor} <b>${matched} District Report:</b><br><br>
• Risk Score: <b>${d.risk_score}%</b> (${d.status})<br>
• Predicted Cases: <b>${d.predicted_cases}</b><br>
• Primary Driver: <b>${d.top_driver}</b><br>
• Recommended Action: <b>${action}</b><br><br>
<i>${d.detailed_explanation}</i>`;
        }

        // ── Rainfall / weather / seasonal ──
        if (/rain|weather|temperature|humidity|season|monsoon|climate/.test(t)) {
            return `🌧️ <b>Weather & Dengue Risk:</b><br><br>
The EpiSentinel model uses these key weather signals:<br>
• <b>Rainfall (lag-1 week)</b> — creates breeding sites, top driver in many districts<br>
• <b>Temperature (lag-1 & lag-2)</b> — optimal mosquito breeding: 25–35°C<br>
• <b>Humidity</b> — high humidity (>70%) accelerates larval development<br>
• <b>NDWI</b> — satellite water body index (detects stagnant water pockets)<br><br>
Post-monsoon weeks (Aug–Nov) carry the highest transmission risk in Karnataka.`;
        }

        // ── Recommendations ──
        if (/recommend|action|what.*do|should.*do|advice|suggest/.test(t)) {
            const critical = Object.entries(districtData)
                .filter(([,v]) => v.risk_score > 70)
                .map(([d]) => d);
            return `📋 <b>Recommended Actions:</b><br><br>
🔴 <b>Critical districts</b> (${critical.join(', ')}):<br>
• Activate district rapid response teams<br>
• Deploy emergency fogging & larviciding<br>
• Increase IDSP weekly reporting frequency<br>
• Set up additional OPD dengue-testing beds<br><br>
🟡 <b>High-risk districts:</b> Enhance routine surveillance, awareness drives<br>
🟢 <b>Low-risk districts:</b> Maintain passive reporting, vector index monitoring`;
        }

        // ── Default fallback with suggestions ──
        const topDistrict = Object.entries(districtData)
            .sort((a, b) => b[1].risk_score - a[1].risk_score)[0];
        return `I can help with that! Try asking me:<br><br>
• <i>"Which districts are highest risk?"</i><br>
• <i>"How to prevent dengue?"</i><br>
• <i>"What are dengue symptoms?"</i><br>
• <i>"Tell me about ${topDistrict[0]}"</i><br>
• <i>"What does the model predict?"</i><br>
• <i>"What actions should be taken?"</i>`;
    }

    // ── Backend URL ───────────────────────────────────────────────────────────
    const BACKEND_URL = 'http://127.0.0.1:8000/chat/general';
    let backendAvailable = null; // null = untested, true/false after first attempt

    async function tryBackend(userMessage) {
        // Build a compact district context summary to give Gemini live data
        const topDistricts = Object.entries(districtData)
            .sort((a, b) => b[1].risk_score - a[1].risk_score)
            .slice(0, 8)
            .map(([d, v]) => `${d}: ${v.risk_score}% (${v.status})`)
            .join(', ');
        const context = `Karnataka dengue dashboard. Top districts by risk — ${topDistricts}`;

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000); // 5s timeout

        try {
            const res = await fetch(BACKEND_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMessage, district_context: context }),
                signal: controller.signal,
            });
            clearTimeout(timeout);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            backendAvailable = true;
            // Convert markdown-style **bold** to <b> for display
            return data.response
                .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
                .replace(/\n/g, '<br>');
        } catch (e) {
            clearTimeout(timeout);
            backendAvailable = false;
            return null; // signal to use fallback
        }
    }

    // ── Event Handlers ────────────────────────────────────────────────────────
    const handleSend = async () => {
        const text = input.value.trim();
        if (!text) return;
        addMessage(text, 'user');
        input.value = '';
        input.disabled = true;
        sendBtn.disabled = true;

        const typingEl = showTyping();

        // Try Gemini backend first; fall back to local engine
        let response = null;
        if (backendAvailable !== false) {
            response = await tryBackend(text);
        }
        if (response === null) {
            // Small extra delay so fallback doesn't feel instant-jarring
            await new Promise(r => setTimeout(r, 400 + Math.random() * 400));
            response = getResponse(text);
        }

        removeTyping();
        addMessage(response, 'system');
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    };

    sendBtn.addEventListener('click', handleSend);
    input.addEventListener('keypress', e => { if (e.key === 'Enter') handleSend(); });
}
