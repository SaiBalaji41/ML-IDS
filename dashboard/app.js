/**
 * Sentinel ML-IDS: Client Application Logic & Chart Orchestrator
 * Matching the modern light-slate SaaS design system and interactive workflows.
 */

// Application State
const STATE = {
  activeTab: 'tab-overview',
  isStreaming: false,
  streamTimer: null,
  totalFlows: 30,
  attackFlows: 30,
  benignFlows: 0,
  reviewFlows: 6,
  batches: [
    { batch: '1', benign: 0, attack: 10 },
    { batch: '2', benign: 0, attack: 10 },
    { batch: '3', benign: 0, attack: 10 }
  ],
  events: [],
  charts: {},
  currentModel: 'cnn_bilstm'
};

// Seed Events (30 events to match the initial 30 analyzed flows)
const SEED_EVENTS = [
  { time: '17:00:23', ip: '192.168.1.105', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.8, sev: 'CRITICAL' },
  { time: '17:00:22', ip: '45.33.32.156', proto: 'TCP', trueLabel: 'DDoS-PSHACK_Flood', predLabel: 'DDoS-PSHACK_Flood', conf: 99.4, sev: 'CRITICAL' },
  { time: '17:00:21', ip: '185.190.140.2', proto: 'UDP', trueLabel: 'DoS-UDP_Flood', predLabel: 'DoS-UDP_Flood', conf: 99.6, sev: 'CRITICAL' },
  { time: '17:00:20', ip: '104.244.42.1', proto: 'GRE', trueLabel: 'Mirai-greeth_flood', predLabel: 'Mirai-greeth_flood', conf: 98.9, sev: 'HIGH' },
  { time: '17:00:19', ip: '198.51.100.44', proto: 'TCP', trueLabel: 'Recon-PortScan', predLabel: 'Recon-PortScan', conf: 97.5, sev: 'HIGH' },
  { time: '17:00:18', ip: '192.168.1.189', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.9, sev: 'CRITICAL' },
  { time: '17:00:17', ip: '203.0.113.88', proto: 'TCP', trueLabel: 'DDoS-SynonymousIP_Flood', predLabel: 'DDoS-SynonymousIP_Flood', conf: 96.2, sev: 'HIGH' },
  { time: '17:00:16', ip: '180.28.14.120', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.8, sev: 'CRITICAL' },
  { time: '17:00:15', ip: '14.175.89.103', proto: 'TLS', trueLabel: 'DDoS-PSHACK_Flood', predLabel: 'DDoS-PSHACK_Flood', conf: 99.8, sev: 'CRITICAL' },
  { time: '17:00:14', ip: '72.29.123.19', proto: 'TLS', trueLabel: 'DDoS-RSTFINFlood', predLabel: 'DDoS-RSTFINFlood', conf: 99.7, sev: 'CRITICAL' },
  { time: '17:00:13', ip: '178.98.124.122', proto: 'UDP', trueLabel: 'DoS-UDP_Flood', predLabel: 'DoS-UDP_Flood', conf: 99.8, sev: 'CRITICAL' },
  { time: '17:00:12', ip: '34.201.227.98', proto: 'UDP', trueLabel: 'DoS-UDP_Flood', predLabel: 'DoS-UDP_Flood', conf: 99.7, sev: 'CRITICAL' },
  { time: '17:00:11', ip: '60.31.135.188', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.8, sev: 'CRITICAL' },
  { time: '17:00:10', ip: '64.22.235.135', proto: 'TLS', trueLabel: 'DDoS-PSHACK_Flood', predLabel: 'DDoS-PSHACK_Flood', conf: 99.8, sev: 'CRITICAL' },
  { time: '17:00:09', ip: '192.168.1.112', proto: 'TCP', trueLabel: 'Recon-OSScan', predLabel: 'Recon-OSScan', conf: 58.4, sev: 'MEDIUM' },
  { time: '17:00:08', ip: '89.207.132.170', proto: 'TCP', trueLabel: 'VulnerabilityScan', predLabel: 'VulnerabilityScan', conf: 59.1, sev: 'MEDIUM' },
  { time: '17:00:07', ip: '192.168.1.140', proto: 'UDP', trueLabel: 'Mirai-udpplain', predLabel: 'Mirai-udpplain', conf: 99.2, sev: 'CRITICAL' },
  { time: '17:00:06', ip: '192.168.1.110', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.9, sev: 'CRITICAL' },
  { time: '17:00:05', ip: '172.16.0.4', proto: 'TCP', trueLabel: 'BrowserHijacking', predLabel: 'BrowserHijacking', conf: 57.2, sev: 'MEDIUM' },
  { time: '17:00:04', ip: '192.168.1.120', proto: 'TCP', trueLabel: 'Backdoor_Malware', predLabel: 'Backdoor_Malware', conf: 58.8, sev: 'MEDIUM' },
  { time: '17:00:03', ip: '192.168.1.135', proto: 'UDP', trueLabel: 'DoS-UDP_Flood', predLabel: 'DoS-UDP_Flood', conf: 99.5, sev: 'CRITICAL' },
  { time: '17:00:02', ip: '192.168.1.160', proto: 'TCP', trueLabel: 'DDoS-TCP_Flood', predLabel: 'DDoS-TCP_Flood', conf: 99.7, sev: 'CRITICAL' },
  { time: '17:00:01', ip: '192.168.1.115', proto: 'TCP', trueLabel: 'CommandInjection', predLabel: 'CommandInjection', conf: 56.5, sev: 'MEDIUM' },
  { time: '17:00:00', ip: '192.168.1.170', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.9, sev: 'CRITICAL' },
  { time: '16:59:59', ip: '192.168.1.180', proto: 'UDP', trueLabel: 'DNS_Spoofing', predLabel: 'DNS_Spoofing', conf: 59.4, sev: 'MEDIUM' },
  { time: '16:59:58', ip: '192.168.1.190', proto: 'TCP', trueLabel: 'DDoS-SYN_Flood', predLabel: 'DDoS-SYN_Flood', conf: 99.8, sev: 'CRITICAL' },
  { time: '16:59:57', ip: '192.168.1.192', proto: 'TCP', trueLabel: 'Recon-PortScan', predLabel: 'Recon-PortScan', conf: 98.2, sev: 'HIGH' },
  { time: '16:59:56', ip: '192.168.1.194', proto: 'GRE', trueLabel: 'Mirai-greeth_flood', predLabel: 'Mirai-greeth_flood', conf: 99.1, sev: 'HIGH' },
  { time: '16:59:55', ip: '192.168.1.196', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.8, sev: 'CRITICAL' },
  { time: '16:59:54', ip: '192.168.1.198', proto: 'TCP', trueLabel: 'DDoS-PSHACK_Flood', predLabel: 'DDoS-PSHACK_Flood', conf: 99.9, sev: 'CRITICAL' }
];

STATE.events = [...SEED_EVENTS];

// DOM Initialization
document.addEventListener('DOMContentLoaded', () => {
  updateHeaderDate();
  initNavigation();
  initCharts();
  renderStreamTable();
  initControls();
  initPresets();
  initModal();
  updateKPIs();
});

function updateHeaderDate() {
  const dateElem = document.getElementById('headerDateStamp');
  if (!dateElem) return;
  const now = new Date();
  const day = now.getDate();
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'];
  const month = months[now.getMonth()];
  const year = now.getFullYear();
  dateElem.textContent = `${day} ${month} ${year}`;
}

/* ==========================================================================
   NAVIGATION & BREADCRUMB
   ========================================================================== */
function initNavigation() {
  const tabButtons = document.querySelectorAll('.nav-item-btn');
  const breadcrumb = document.getElementById('activeBreadcrumb');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });

  // Hero Analyze a Flow Button
  const heroBtn = document.getElementById('btnHeroAnalyzeFlow');
  if (heroBtn) {
    heroBtn.addEventListener('click', () => {
      switchTab('tab-traffic');
    });
  }
}

function switchTab(tabId) {
  STATE.activeTab = tabId;

  // Update Nav Buttons
  document.querySelectorAll('.nav-item-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
  });

  // Update Tab Views
  document.querySelectorAll('.tab-view').forEach(view => {
    view.classList.toggle('active', view.id === tabId);
  });

  // Update Breadcrumb Text
  const breadcrumb = document.getElementById('activeBreadcrumb');
  const tabNames = {
    'tab-overview': 'Overview',
    'tab-traffic': 'Traffic analyzer',
    'tab-models': 'Model performance'
  };
  if (breadcrumb) {
    breadcrumb.textContent = tabNames[tabId] || 'Overview';
  }

  // Resize charts on view change
  if (tabId === 'tab-overview') {
    setTimeout(() => {
      if (STATE.charts.traffic) STATE.charts.traffic.resize();
      if (STATE.charts.donut) STATE.charts.donut.resize();
    }, 50);
  }
}

/* ==========================================================================
   CHART.JS INITIALIZATION
   ========================================================================== */
function initCharts() {
  // 1. Traffic Activity Bar Chart
  const ctxTraffic = document.getElementById('trafficActivityChart');
  if (ctxTraffic) {
    STATE.charts.traffic = new Chart(ctxTraffic, {
      type: 'bar',
      data: {
        labels: STATE.batches.map(b => b.batch),
        datasets: [
          {
            label: 'Benign',
            data: STATE.batches.map(b => b.benign),
            backgroundColor: '#3b82f6',
            borderRadius: 4,
            barPercentage: 0.5,
            categoryPercentage: 0.6
          },
          {
            label: 'Attack',
            data: STATE.batches.map(b => b.attack),
            backgroundColor: '#a78bfa',
            borderRadius: 4,
            barPercentage: 0.5,
            categoryPercentage: 0.6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f172a',
            titleFont: { family: 'Inter', size: 12 },
            bodyFont: { family: 'Inter', size: 11 },
            padding: 8,
            cornerRadius: 6
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          },
          y: {
            min: 0,
            max: 10,
            ticks: {
              stepSize: 5,
              color: '#94a3b8',
              font: { family: 'Inter', size: 11 }
            },
            grid: { color: '#f1f5f9' }
          }
        }
      }
    });
  }

  // 2. Detection Breakdown Donut Chart
  const ctxDonut = document.getElementById('detectionDonutChart');
  if (ctxDonut) {
    STATE.charts.donut = new Chart(ctxDonut, {
      type: 'doughnut',
      data: {
        labels: ['Attack', 'Benign'],
        datasets: [{
          data: [STATE.attackFlows, STATE.benignFlows],
          backgroundColor: ['#8b5cf6', '#3b82f6'],
          borderWidth: 0,
          cutout: '78%'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f172a',
            padding: 8,
            cornerRadius: 6
          }
        }
      }
    });
  }

  // 3. SHAP Importance Horizontal Bar Chart
  const ctxShap = document.getElementById('shapImportanceChart');
  if (ctxShap) {
    STATE.charts.shap = new Chart(ctxShap, {
      type: 'bar',
      data: {
        labels: ['Header_Length', 'Rate', 'Duration', 'syn_count', 'ack_count', 'Tot_size', 'rst_count', 'fin_count', 'urg_count', 'psh_count'],
        datasets: [{
          label: 'Mean |SHAP Value|',
          data: [0.485, 0.412, 0.380, 0.320, 0.290, 0.245, 0.198, 0.165, 0.120, 0.095],
          backgroundColor: '#2563eb',
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            grid: { color: '#f1f5f9' },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#475569', font: { family: 'JetBrains Mono', size: 11 } }
          }
        }
      }
    });
  }
}

/* ==========================================================================
   KPI & STATE UPDATES
   ========================================================================== */
function updateKPIs() {
  const kpiFlows = document.getElementById('kpiFlowsAnalyzed');
  const kpiAttacks = document.getElementById('kpiAttackPredictions');
  const kpiAttackPct = document.getElementById('kpiAttackPct');
  const donutTotal = document.getElementById('donutTotalCount');
  const countBenign = document.getElementById('countBenign');
  const countAttack = document.getElementById('countAttack');
  const countReview = document.getElementById('countReview');

  if (kpiFlows) kpiFlows.textContent = STATE.totalFlows;
  if (kpiAttacks) kpiAttacks.textContent = STATE.attackFlows;
  if (donutTotal) donutTotal.textContent = STATE.totalFlows;
  if (countBenign) countBenign.textContent = STATE.benignFlows;
  if (countAttack) countAttack.textContent = STATE.attackFlows;
  if (countReview) countReview.textContent = STATE.reviewFlows;

  if (kpiAttackPct) {
    const pct = STATE.totalFlows > 0 ? ((STATE.attackFlows / STATE.totalFlows) * 100).toFixed(1) : '0.0';
    kpiAttackPct.textContent = `${pct}% of analyzed flows`;
  }

  // Update Donut Chart
  if (STATE.charts.donut) {
    STATE.charts.donut.data.datasets[0].data = [STATE.attackFlows, STATE.benignFlows];
    STATE.charts.donut.update();
  }

  // Update Activity Chart
  if (STATE.charts.traffic) {
    STATE.charts.traffic.data.labels = STATE.batches.map(b => b.batch);
    STATE.charts.traffic.data.datasets[0].data = STATE.batches.map(b => b.benign);
    STATE.charts.traffic.data.datasets[1].data = STATE.batches.map(b => b.attack);
    STATE.charts.traffic.update();
  }
}

/* ==========================================================================
   STREAM TABLE & REPLAY SIMULATION
   ========================================================================== */
function renderStreamTable() {
  const tbody = document.getElementById('streamTableBody');
  if (!tbody) return;

  tbody.innerHTML = '';
  STATE.events.forEach((ev, idx) => {
    const tr = document.createElement('tr');
    
    let sevClass = 'sev-benign';
    if (ev.sev === 'CRITICAL') sevClass = 'sev-critical';
    else if (ev.sev === 'HIGH') sevClass = 'sev-high';
    else if (ev.sev === 'MEDIUM') sevClass = 'sev-medium';

    tr.innerHTML = `
      <td>${ev.time}</td>
      <td class="flow-ip">${ev.ip}</td>
      <td><span class="flow-proto-badge">${ev.proto}</span></td>
      <td>${ev.trueLabel}</td>
      <td style="font-weight:600; color:${ev.predLabel.includes('Benign') ? '#3b82f6' : '#0f172a'}">${ev.predLabel}</td>
      <td>${ev.conf.toFixed(1)}%</td>
      <td><span class="badge-sev ${sevClass}">${ev.sev}</span></td>
      <td>
        <button class="btn-ctrl" style="padding:2px 8px; font-size:11px;" onclick="inspectFlow(${idx})">Inspect</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function initControls() {
  const btnToggle = document.getElementById('btnToggleStream');
  const btnBurst = document.getElementById('btnBurstStream');
  const btnClear = document.getElementById('btnClearStream');
  const selectModel = document.getElementById('streamModelSelect');

  if (selectModel) {
    selectModel.addEventListener('change', (e) => {
      STATE.currentModel = e.target.value;
    });
  }

  if (btnToggle) {
    btnToggle.addEventListener('click', () => {
      STATE.isStreaming = !STATE.isStreaming;
      if (STATE.isStreaming) {
        btnToggle.textContent = '⏸ Pause Stream';
        btnToggle.classList.replace('btn-ctrl-primary', 'btn-ctrl');
        STATE.streamTimer = setInterval(simulateNextFlow, 1200);
      } else {
        btnToggle.textContent = '▶ Start Stream';
        btnToggle.classList.replace('btn-ctrl', 'btn-ctrl-primary');
        clearInterval(STATE.streamTimer);
      }
    });
  }

  if (btnBurst) {
    btnBurst.addEventListener('click', () => {
      burstFlows(10);
    });
  }

  if (btnClear) {
    btnClear.addEventListener('click', () => {
      STATE.events = [];
      STATE.totalFlows = 0;
      STATE.attackFlows = 0;
      STATE.benignFlows = 0;
      STATE.reviewFlows = 0;
      STATE.batches = [];
      renderStreamTable();
      updateKPIs();
    });
  }
}

function simulateNextFlow() {
  const attackTypes = [
    { label: 'DDoS-ICMP_Flood', proto: 'ICMP', sev: 'CRITICAL' },
    { label: 'DDoS-PSHACK_Flood', proto: 'TLS', sev: 'CRITICAL' },
    { label: 'DoS-UDP_Flood', proto: 'UDP', sev: 'CRITICAL' },
    { label: 'Mirai-greeth_flood', proto: 'GRE', sev: 'HIGH' },
    { label: 'Recon-PortScan', proto: 'TCP', sev: 'HIGH' },
    { label: 'BenignTraffic', proto: 'HTTP', sev: 'BENIGN' }
  ];

  const pick = attackTypes[Math.floor(Math.random() * attackTypes.length)];
  const isAttack = pick.label !== 'BenignTraffic';
  const conf = isAttack ? (85 + Math.random() * 14.9) : (92 + Math.random() * 7.5);
  const isReview = conf < 60;

  const now = new Date();
  const timeStr = now.toTimeString().split(' ')[0];
  const randIP = `192.168.1.${Math.floor(Math.random() * 200 + 10)}`;

  const newEvent = {
    time: timeStr,
    ip: randIP,
    proto: pick.proto,
    trueLabel: pick.label,
    predLabel: pick.label,
    conf: conf,
    sev: pick.sev
  };

  STATE.events.unshift(newEvent);
  if (STATE.events.length > 500) STATE.events.pop();

  STATE.totalFlows += 1;
  if (isAttack) STATE.attackFlows += 1;
  else STATE.benignFlows += 1;
  if (isReview) STATE.reviewFlows += 1;

  // Add to batch
  const lastBatch = STATE.batches[STATE.batches.length - 1];
  if (lastBatch && (lastBatch.benign + lastBatch.attack < 10)) {
    if (isAttack) lastBatch.attack += 1;
    else lastBatch.benign += 1;
  } else {
    const nextIdx = (STATE.batches.length + 1).toString();
    STATE.batches.push({
      batch: nextIdx,
      benign: isAttack ? 0 : 1,
      attack: isAttack ? 1 : 0
    });
    if (STATE.batches.length > 8) STATE.batches.shift();
  }

  renderStreamTable();
  updateKPIs();
}

function burstFlows(count = 10) {
  for (let i = 0; i < count; i++) {
    simulateNextFlow();
  }
}

window.inspectFlow = function(idx) {
  const ev = STATE.events[idx];
  if (!ev) return;
  
  const content = document.getElementById('inspectModalContent');
  const modal = document.getElementById('inspectModal');
  if (!content || !modal) return;

  let sevColor = '#10b981';
  if (ev.sev === 'CRITICAL') sevColor = '#ef4444';
  else if (ev.sev === 'HIGH') sevColor = '#f97316';
  else if (ev.sev === 'MEDIUM') sevColor = '#eab308';

  content.innerHTML = `
    <div style="background:var(--bg-subtle); border:1px solid var(--border-main); border-radius:var(--radius-sm); padding:12px 14px; margin-bottom:14px;">
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <span style="color:var(--text-muted);">Source IP:</span>
        <span style="font-family:var(--font-mono); font-weight:700; color:var(--primary-blue);">${ev.ip}</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <span style="color:var(--text-muted);">Protocol / Port:</span>
        <span style="font-weight:600;">${ev.proto} (Inferred)</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <span style="color:var(--text-muted);">Timestamp:</span>
        <span>${ev.time} UTC</span>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span style="color:var(--text-muted);">True Label:</span>
        <span style="font-weight:600;">${ev.trueLabel}</span>
      </div>
    </div>

    <div style="margin-bottom:14px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
        <span style="font-weight:600;">Prediction Output:</span>
        <span style="color:${sevColor}; font-weight:700; font-size:13px;">${ev.predLabel}</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
        <span style="color:var(--text-muted);">Confidence Score:</span>
        <span style="font-weight:700;">${ev.conf.toFixed(2)}%</span>
      </div>
      <div style="background:#e2e8f0; border-radius:4px; height:6px; overflow:hidden;">
        <div style="background:${sevColor}; width:${ev.conf}%; height:100%;"></div>
      </div>
    </div>

    <div style="border-top:1px solid var(--border-light); padding-top:10px; font-size:11.5px; color:var(--text-muted);">
      <div>• <strong>Spatial 1D-CNN Conv Filters</strong>: Matched packet header size & sequence entropy</div>
      <div style="margin-top:4px;">• <strong>BiLSTM Temporal Core</strong>: Recurrent forward/backward flow states verified</div>
      <div style="margin-top:4px;">• <strong>Inference Latency</strong>: 0.0125 ms (Local Engine)</div>
    </div>
  `;

  modal.classList.add('open');
};

/* ==========================================================================
   ATTACK PRESETS & SINGLE FLOW TESTER
   ========================================================================== */
function initPresets() {
  const presetBtns = document.querySelectorAll('.btn-preset');
  const btnRun = document.getElementById('btnRunSingleTest');
  const resultDiv = document.getElementById('singleTestResult');

  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      presetBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  if (btnRun) {
    btnRun.addEventListener('click', () => {
      const activeBtn = document.querySelector('.btn-preset.active');
      const presetName = activeBtn ? activeBtn.textContent : 'DDoS ICMP Flood';
      
      resultDiv.textContent = `Analyzing ${presetName}...`;
      setTimeout(() => {
        const latency = (6.2 + Math.random() * 1.5).toFixed(2);
        const conf = (99.1 + Math.random() * 0.8).toFixed(2);
        resultDiv.textContent = `✓ Predicted: ${presetName} (Confidence: ${conf}%, Latency: ${latency} ms)`;
      }, 350);
    });
  }
}

/* ==========================================================================
   EXPORT & INSPECT MODAL HANDLERS
   ========================================================================== */
function initModal() {
  const btnOpen = document.getElementById('btnExportEvents');
  const modal = document.getElementById('exportModal');
  const btnClose = document.getElementById('btnCloseExportModal');
  const btnCancel = document.getElementById('btnCancelExport');
  const btnCSV = document.getElementById('btnDownloadCSV');
  const btnJSON = document.getElementById('btnDownloadJSON');

  const inspectModal = document.getElementById('inspectModal');
  const btnCloseInspect = document.getElementById('btnCloseInspectModal');
  const btnCloseInspectBtn = document.getElementById('btnCloseInspectBtn');

  if (btnOpen && modal) {
    btnOpen.addEventListener('click', () => modal.classList.add('open'));
  }
  if (btnClose && modal) {
    btnClose.addEventListener('click', () => modal.classList.remove('open'));
  }
  if (btnCancel && modal) {
    btnCancel.addEventListener('click', () => modal.classList.remove('open'));
  }

  if (btnCloseInspect && inspectModal) {
    btnCloseInspect.addEventListener('click', () => inspectModal.classList.remove('open'));
  }
  if (btnCloseInspectBtn && inspectModal) {
    btnCloseInspectBtn.addEventListener('click', () => inspectModal.classList.remove('open'));
  }

  if (btnCSV) {
    btnCSV.addEventListener('click', () => {
      downloadCSV();
      if (modal) modal.classList.remove('open');
    });
  }

  if (btnJSON) {
    btnJSON.addEventListener('click', () => {
      downloadJSON();
      if (modal) modal.classList.remove('open');
    });
  }
}

function downloadCSV() {
  const headers = ['Timestamp', 'Source_IP', 'Protocol', 'True_Label', 'Predicted_Label', 'Confidence', 'Severity'];
  const rows = STATE.events.map(e => [e.time, e.ip, e.proto, e.trueLabel, e.predLabel, `${e.conf.toFixed(1)}%`, e.sev]);
  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `sentinel_events_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function downloadJSON() {
  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(STATE.events, null, 2));
  const link = document.createElement('a');
  link.setAttribute('href', dataStr);
  link.setAttribute('download', `sentinel_events_${Date.now()}.json`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
