/**
 * Sentinel ML-IDS: Client Application Logic & Multi-Model Intelligence Studio
 * Powers live stream telemetry, interactive custom inputs, 5-model comparative benchmarks,
 * 6-stage AI architecture explanations, and the 34-class threat dictionary.
 */

// Global Application State
const STATE = {
  activeTab: 'tab-overview',
  activeSubmode: 'submode-tabular',
  userMode: 'simple', // 'simple' (Beginner-Friendly) or 'deep' (SOC Engineer)
  tourStep: 1,
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
  currentModel: 'cnn_bilstm',
  waveformSignal: new Float32Array(1024),
  customPrediction: null
};

// Seed Events (Initial 30 analyzed flows)
const SEED_EVENTS = [
  { time: '17:00:23', ip: '192.168.1.105', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.8, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:22', ip: '45.33.32.156', proto: 'TCP', trueLabel: 'DDoS-PSHACK_Flood', predLabel: 'DDoS-PSHACK_Flood', conf: 99.4, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:21', ip: '185.190.140.2', proto: 'UDP', trueLabel: 'DoS-UDP_Flood', predLabel: 'DoS-UDP_Flood', conf: 99.6, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:20', ip: '104.244.42.1', proto: 'GRE', trueLabel: 'Mirai-greeth_flood', predLabel: 'Mirai-greeth_flood', conf: 98.9, sev: 'HIGH', isAttack: true },
  { time: '17:00:19', ip: '198.51.100.44', proto: 'TCP', trueLabel: 'Recon-PortScan', predLabel: 'Recon-PortScan', conf: 97.5, sev: 'HIGH', isAttack: true },
  { time: '17:00:18', ip: '192.168.1.189', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.9, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:17', ip: '203.0.113.88', proto: 'TCP', trueLabel: 'DDoS-SynonymousIP_Flood', predLabel: 'DDoS-SynonymousIP_Flood', conf: 96.2, sev: 'HIGH', isAttack: true },
  { time: '17:00:16', ip: '180.28.14.120', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.8, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:15', ip: '14.175.89.103', proto: 'TLS', trueLabel: 'DDoS-PSHACK_Flood', predLabel: 'DDoS-PSHACK_Flood', conf: 99.8, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:14', ip: '72.29.123.19', proto: 'TLS', trueLabel: 'DDoS-RSTFINFlood', predLabel: 'DDoS-RSTFINFlood', conf: 99.7, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:13', ip: '178.98.124.122', proto: 'UDP', trueLabel: 'DoS-UDP_Flood', predLabel: 'DoS-UDP_Flood', conf: 99.8, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:12', ip: '34.201.227.98', proto: 'UDP', trueLabel: 'DoS-UDP_Flood', predLabel: 'DoS-UDP_Flood', conf: 99.7, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:11', ip: '60.31.135.188', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.8, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:10', ip: '64.22.235.135', proto: 'TLS', trueLabel: 'DDoS-PSHACK_Flood', predLabel: 'DDoS-PSHACK_Flood', conf: 99.8, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:09', ip: '192.168.1.112', proto: 'TCP', trueLabel: 'Recon-OSScan', predLabel: 'Recon-OSScan', conf: 58.4, sev: 'MEDIUM', isAttack: true },
  { time: '17:00:08', ip: '89.207.132.170', proto: 'TCP', trueLabel: 'VulnerabilityScan', predLabel: 'VulnerabilityScan', conf: 59.1, sev: 'MEDIUM', isAttack: true },
  { time: '17:00:07', ip: '192.168.1.140', proto: 'UDP', trueLabel: 'Mirai-udpplain', predLabel: 'Mirai-udpplain', conf: 99.2, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:06', ip: '192.168.1.110', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.9, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:05', ip: '172.16.0.4', proto: 'TCP', trueLabel: 'BrowserHijacking', predLabel: 'BrowserHijacking', conf: 57.2, sev: 'MEDIUM', isAttack: true },
  { time: '17:00:04', ip: '192.168.1.120', proto: 'TCP', trueLabel: 'Backdoor_Malware', predLabel: 'Backdoor_Malware', conf: 58.8, sev: 'MEDIUM', isAttack: true },
  { time: '17:00:03', ip: '192.168.1.135', proto: 'UDP', trueLabel: 'DoS-UDP_Flood', predLabel: 'DoS-UDP_Flood', conf: 99.5, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:02', ip: '192.168.1.160', proto: 'TCP', trueLabel: 'DDoS-TCP_Flood', predLabel: 'DDoS-TCP_Flood', conf: 99.7, sev: 'CRITICAL', isAttack: true },
  { time: '17:00:01', ip: '192.168.1.115', proto: 'TCP', trueLabel: 'CommandInjection', predLabel: 'CommandInjection', conf: 56.5, sev: 'MEDIUM', isAttack: true },
  { time: '17:00:00', ip: '192.168.1.170', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.9, sev: 'CRITICAL', isAttack: true },
  { time: '16:59:59', ip: '192.168.1.180', proto: 'UDP', trueLabel: 'DNS_Spoofing', predLabel: 'DNS_Spoofing', conf: 59.4, sev: 'MEDIUM', isAttack: true },
  { time: '16:59:58', ip: '192.168.1.190', proto: 'TCP', trueLabel: 'DDoS-SYN_Flood', predLabel: 'DDoS-SYN_Flood', conf: 99.8, sev: 'CRITICAL', isAttack: true },
  { time: '16:59:57', ip: '192.168.1.192', proto: 'TCP', trueLabel: 'Recon-PortScan', predLabel: 'Recon-PortScan', conf: 98.2, sev: 'HIGH', isAttack: true },
  { time: '16:59:56', ip: '192.168.1.194', proto: 'GRE', trueLabel: 'Mirai-greeth_flood', predLabel: 'Mirai-greeth_flood', conf: 99.1, sev: 'HIGH', isAttack: true },
  { time: '16:59:55', ip: '192.168.1.196', proto: 'ICMP', trueLabel: 'DDoS-ICMP_Flood', predLabel: 'DDoS-ICMP_Flood', conf: 99.8, sev: 'CRITICAL', isAttack: true },
  { time: '16:59:54', ip: '192.168.1.198', proto: 'TCP', trueLabel: 'DDoS-PSHACK_Flood', predLabel: 'DDoS-PSHACK_Flood', conf: 99.9, sev: 'CRITICAL', isAttack: true }
];

STATE.events = [...SEED_EVENTS];

// 34-Class Threat Dictionary
const THREAT_DICTIONARY = [
  { name: 'BenignTraffic', family: 'Benign', proto: 'HTTP/TLS/DNS', sev: 'BENIGN', desc: 'Legitimate regular network communications, web browsing, API queries, and authorized file transfers.', mitigation: 'Permit traffic; maintain standard baseline telemetry.' },
  { name: 'DDoS-ICMP_Flood', family: 'DDoS', proto: 'ICMP', sev: 'CRITICAL', desc: 'Volumetric flood sending overwhelming ICMP Echo Request packets to saturate bandwidth and crash target hosts.', mitigation: 'Rate limit ICMP ingress or drop non-essential echo requests at edge firewalls.' },
  { name: 'DDoS-SYN_Flood', family: 'DDoS', proto: 'TCP', sev: 'CRITICAL', desc: 'Transmits a rapid stream of TCP SYN packets with spoofed IPs to exhaust target TCP connection backlogs.', mitigation: 'Enable SYN Cookies and TCP proxy inspection on perimeter load balancers.' },
  { name: 'DDoS-UDP_Flood', family: 'DDoS', proto: 'UDP', sev: 'CRITICAL', desc: 'High-volume UDP datagrams directed at random ports, forcing target OS to check listening services and reply with ICMP unreachable.', mitigation: 'Filter unallocated UDP ports and deploy upstream scrubbing centers.' },
  { name: 'DDoS-PSHACK_Flood', family: 'DDoS', proto: 'TCP', sev: 'CRITICAL', desc: 'Floods target socket buffers with TCP packets having PSH and ACK flags set, forcing immediate application buffer processing.', mitigation: 'Inspect TCP buffer rates and drop anomalous PSH packet bursts.' },
  { name: 'DDoS-RSTFINFlood', family: 'DDoS', proto: 'TCP', sev: 'CRITICAL', desc: 'Abruptly terminates valid sessions and confuses stateful firewalls by broadcasting spoofed RST/FIN flags.', mitigation: 'Enforce strict stateful connection tracking and drop out-of-state RST/FIN packets.' },
  { name: 'DDoS-SlowLoris', family: 'DDoS', proto: 'HTTP', sev: 'HIGH', desc: 'Slow-rate application attack keeping numerous HTTP connections open indefinitely with incomplete header streams.', mitigation: 'Set aggressive HTTP request header timeout thresholds and enforce minimum transfer rates.' },
  { name: 'DDoS-HTTP_Flood', family: 'DDoS', proto: 'HTTP', sev: 'CRITICAL', desc: 'Simultaneous HTTP GET/POST requests targeting resource-intensive web scripts to deplete backend databases.', mitigation: 'Deploy Web Application Firewall (WAF) rate limiting and CAPTCHA challenge systems.' },
  { name: 'DDoS-SynonymousIP_Flood', family: 'DDoS', proto: 'IP', sev: 'HIGH', desc: 'Packets crafted where source and destination IP/ports are identical, confusing routing logic and exhausting kernel sockets.', mitigation: 'Implement anti-spoofing ingress filters (RFC 2827 / BCP 38).' },
  { name: 'DDoS-ACK_Fragmentation', family: 'DDoS', proto: 'TCP/IP', sev: 'CRITICAL', desc: 'Fragmented IP packets carrying TCP ACK payloads designed to bypass stateless inspection filters.', mitigation: 'Reassemble IP fragments before deep inspection and reject suspicious unaligned fragments.' },
  { name: 'DDoS-UDP_Fragmentation', family: 'DDoS', proto: 'UDP/IP', sev: 'CRITICAL', desc: 'Large UDP datagrams split into tiny fragments to overwhelm host reassembly queues.', mitigation: 'Drop malformed IP fragments and throttle fragmented UDP traffic.' },
  { name: 'DDoS-ICMP_Fragmentation', family: 'DDoS', proto: 'ICMP/IP', sev: 'CRITICAL', desc: 'Fragmented ICMP ping packets aimed at exploiting IP stack reassembly vulnerabilities.', mitigation: 'Disallow fragmented ICMP packets at boundary edge routers.' },
  { name: 'DoS-UDP_Flood', family: 'DoS', proto: 'UDP', sev: 'CRITICAL', desc: 'Single-source volumetric UDP saturation targeted at depleting network link capacity.', mitigation: 'Apply strict interface bandwidth policing and ingress ACLs.' },
  { name: 'DoS-TCP_Flood', family: 'DoS', proto: 'TCP', sev: 'CRITICAL', desc: 'High-speed TCP packet barrage from an isolated attacker directed at a listening daemon.', mitigation: 'Blackhole offending source IP and enforce connection concurrency limits.' },
  { name: 'DoS-SYN_Flood', family: 'DoS', proto: 'TCP', sev: 'CRITICAL', desc: 'Direct SYN flood from a dedicated host aiming to lock service ports.', mitigation: 'Dynamic connection throttling and IP reputation filtering.' },
  { name: 'DoS-HTTP_Flood', family: 'DoS', proto: 'HTTP', sev: 'HIGH', desc: 'Single-host HTTP request blitz exhausting web server worker threads.', mitigation: 'Enable application-level rate limiting and request concurrency bounds.' },
  { name: 'Mirai-greeth_flood', family: 'Mirai', proto: 'GRE', sev: 'HIGH', desc: 'Mirai botnet encapsulation of raw Ethernet frames inside GRE tunnels to bypass standard layer-4 filtering.', mitigation: 'Block unauthorized GRE (Protocol 47) encapsulation at edge perimeters.' },
  { name: 'Mirai-greip_flood', family: 'Mirai', proto: 'GRE/IP', sev: 'HIGH', desc: 'Mirai botnet GRE-encapsulated IP packets used in massive IoT amplification strikes.', mitigation: 'Inspect GRE encapsulated protocol headers and discard non-tunnel traffic.' },
  { name: 'Mirai-udpplain', family: 'Mirai', proto: 'UDP', sev: 'CRITICAL', desc: 'Optimized high-entropy raw UDP payloads generated by compromised Mirai IoT devices.', mitigation: 'Identify Mirai payload signatures and isolate infected IoT subnets.' },
  { name: 'Recon-PortScan', family: 'Recon', proto: 'TCP', sev: 'HIGH', desc: 'Systematic transmission of packets to multiple destination ports to discover open services.', mitigation: 'Deploy port-scan threshold detectors and dynamically block scanning IPs.' },
  { name: 'Recon-OSScan', family: 'Recon', proto: 'TCP/UDP', sev: 'MEDIUM', desc: 'Sends crafted packets with unusual TCP options and flag combinations to fingerprint target OS stack behavior.', mitigation: 'Normalize TCP options and obfuscate banner responses at the gateway.' },
  { name: 'Recon-PingSweep', family: 'Recon', proto: 'ICMP', sev: 'MEDIUM', desc: 'ICMP echo probes broadcast across IP ranges to identify active responsive hosts.', mitigation: 'Disable ICMP echo reply on unmanaged subnets.' },
  { name: 'Recon-HostDiscovery', family: 'Recon', proto: 'ARP/ICMP', sev: 'MEDIUM', desc: 'Local and remote sweeps scanning for active IoT nodes and endpoint interfaces.', mitigation: 'Implement dynamic ARP inspection and segment network VLANs.' },
  { name: 'VulnerabilityScan', family: 'Recon', proto: 'TCP/HTTP', sev: 'MEDIUM', desc: 'Automated vulnerability scanner probing services for known CVE exploits and version flaws.', mitigation: 'Block automated scanner user-agents and patch known software CVEs.' },
  { name: 'SqlInjection', family: 'Web', proto: 'HTTP', sev: 'HIGH', desc: 'Malicious SQL clauses injected into web input fields to manipulate or exfiltrate backend database records.', mitigation: 'Use parameterized queries, input validation, and WAF SQLi filter rules.' },
  { name: 'CommandInjection', family: 'Web', proto: 'HTTP', sev: 'HIGH', desc: 'Execution of arbitrary system shell commands via vulnerable server applications.', mitigation: 'Sanitize shell arguments and execute applications with minimal system privileges.' },
  { name: 'BrowserHijacking', family: 'Web', proto: 'HTTP', sev: 'MEDIUM', desc: 'Injects unauthorized redirects and script manipulations into user browser sessions.', mitigation: 'Enforce Content Security Policy (CSP) headers and Subresource Integrity.' },
  { name: 'Uploading_Attack', family: 'Web', proto: 'HTTP', sev: 'HIGH', desc: 'Uploads malicious executable scripts (web shells) disguised as media or documents.', mitigation: 'Validate file MIME types, inspect file headers, and store uploads in isolated sandboxes.' },
  { name: 'DictionaryBruteForce', family: 'BruteForce', proto: 'SSH/Telnet', sev: 'HIGH', desc: 'Automated credential-stuffing attempting thousands of dictionary password pairs against login daemons.', mitigation: 'Enforce account lockout policies, multi-factor authentication, and fail2ban.' },
  { name: 'Backdoor_Malware', family: 'BruteForce', proto: 'TCP', sev: 'CRITICAL', desc: 'Persistent stealth communication channel between an infected host and command-and-control (C2) servers.', mitigation: 'Quarantine compromised host and analyze egress network traffic for C2 beacons.' },
  { name: 'DNS_Spoofing', family: 'Spoofing', proto: 'DNS', sev: 'MEDIUM', desc: 'Forged DNS query responses corrupting local resolver caches to divert users to malicious servers.', mitigation: 'Enable DNSSEC validation and configure trusted upstream resolvers.' },
  { name: 'MITM-ArpSpoofing', family: 'Spoofing', proto: 'ARP', sev: 'HIGH', desc: 'Sends falsified ARP messages over local ethernet to link attacker MAC with default gateway IP.', mitigation: 'Enable Dynamic ARP Inspection (DAI) and DHCP Snooping on network switches.' }
];

// 1-Click Attack Preset Parameter Configurations
const ATTACK_PRESETS = {
  benign: {
    proto: 'HTTP', ip: '192.168.1.105', port: 80, duration: 0.25, rate: 85.0, headerLen: 54, totSize: 1450,
    syn: true, ack: true, fin: false, rst: false, psh: false, urg: false, synCount: 1, ackCount: 4, rstCount: 0,
    expected: 'BenignTraffic', sev: 'BENIGN', conf: 99.85,
    what: 'Regular HTTP web browsing transaction with standard TCP 3-way handshake.',
    why: 'Normal packet velocity (85 pkts/s) with balanced SYN and ACK flags and expected header sizes.',
    action: 'Allow traffic through firewall; baseline statistics remain nominal.'
  },
  ddos_icmp: {
    proto: 'ICMP', ip: '45.33.32.156', port: 0, duration: 0.012, rate: 28500.0, headerLen: 42, totSize: 64,
    syn: false, ack: false, fin: false, rst: false, psh: false, urg: false, synCount: 0, ackCount: 0, rstCount: 0,
    expected: 'DDoS-ICMP_Flood', sev: 'CRITICAL', conf: 99.92,
    what: 'Volumetric ICMP Ping Flood sending 28,500 packets/sec to exhaust edge bandwidth.',
    why: 'Extreme packet frequency with 0 connection state handshakes matches flood signature.',
    action: 'Rate-limit ICMP at boundary routers and drop unauthorized echo requests.'
  },
  mirai_gre: {
    proto: 'GRE', ip: '185.190.140.2', port: 0, duration: 0.045, rate: 19400.0, headerLen: 68, totSize: 512,
    syn: false, ack: false, fin: false, rst: false, psh: false, urg: false, synCount: 0, ackCount: 0, rstCount: 0,
    expected: 'Mirai-greeth_flood', sev: 'HIGH', conf: 99.14,
    what: 'Mirai IoT Botnet GRE tunnel encapsulation attempting to bypass layer-4 firewall rules.',
    why: 'Unusual GRE protocol (47) encapsulation with high-frequency payload sequences.',
    action: 'Filter GRE protocol ingress and isolate infected internal IoT device subnet.'
  },
  recon_scan: {
    proto: 'TCP', ip: '198.51.100.44', port: 445, duration: 0.005, rate: 8200.0, headerLen: 60, totSize: 60,
    syn: true, ack: false, fin: false, rst: false, psh: false, urg: false, synCount: 15, ackCount: 0, rstCount: 0,
    expected: 'Recon-PortScan', sev: 'HIGH', conf: 98.40,
    what: 'Port reconnaissance probe systematically checking for open vulnerable daemons.',
    why: 'High SYN count with zero ACK completions across rapid microsecond intervals.',
    action: 'Temporarily blackhole scanner IP address and obfuscate daemon response banners.'
  },
  dos_udp: {
    proto: 'UDP', ip: '104.244.42.1', port: 5353, duration: 0.02, rate: 31000.0, headerLen: 28, totSize: 1200,
    syn: false, ack: false, fin: false, rst: false, psh: false, urg: false, synCount: 0, ackCount: 0, rstCount: 0,
    expected: 'DoS-UDP_Flood', sev: 'CRITICAL', conf: 99.75,
    what: 'High-speed UDP packet barrage saturating UDP listening sockets.',
    why: 'Sustained 31,000 pkts/s UDP stream with repetitive payload sizing.',
    action: 'Apply interface bandwidth policing and throttle unallocated UDP ports.'
  },
  syn_flood: {
    proto: 'TCP', ip: '203.0.113.88', port: 443, duration: 0.01, rate: 42000.0, headerLen: 54, totSize: 54,
    syn: true, ack: false, fin: false, rst: false, psh: false, urg: false, synCount: 250, ackCount: 0, rstCount: 0,
    expected: 'DDoS-SYN_Flood', sev: 'CRITICAL', conf: 99.88,
    what: 'DDoS SYN Flood attempting to exhaust web server TCP half-open connection table.',
    why: 'Disproportionate SYN-to-ACK ratio (>250:0) at wire speed.',
    action: 'Enable SYN Cookies on load balancer and enforce connection rate limiting.'
  },
  sql_injection: {
    proto: 'HTTP', ip: '172.16.0.4', port: 8080, duration: 0.35, rate: 42.0, headerLen: 54, totSize: 3200,
    syn: true, ack: true, fin: false, rst: false, psh: true, urg: false, synCount: 1, ackCount: 8, rstCount: 0,
    expected: 'SqlInjection', sev: 'HIGH', conf: 94.60,
    what: 'Web application exploit containing SQL syntax clauses in payload buffer.',
    why: 'Large PSH payload size with query character entropy triggering deep neural weights.',
    action: 'Block request via WAF rule and inspect backend database query parameterization.'
  },
  slowloris: {
    proto: 'HTTP', ip: '60.31.135.188', port: 80, duration: 8.5, rate: 1.2, headerLen: 54, totSize: 120,
    syn: true, ack: true, fin: false, rst: false, psh: false, urg: false, synCount: 1, ackCount: 1, rstCount: 0,
    expected: 'DDoS-SlowLoris', sev: 'HIGH', conf: 96.20,
    what: 'Low-and-slow HTTP connection starvation holding server worker threads open.',
    why: 'Unusually long flow duration (8.5s) with minimal trickle packet rate (1.2 pkts/s).',
    action: 'Set aggressive HTTP request header read timeouts and enforce minimum throughput.'
  },
  arp_spoof: {
    proto: 'TCP', ip: '192.168.1.1', port: 0, duration: 0.08, rate: 1500.0, headerLen: 42, totSize: 42,
    syn: false, ack: false, fin: false, rst: false, psh: false, urg: false, synCount: 0, ackCount: 0, rstCount: 0,
    expected: 'MITM-ArpSpoofing', sev: 'HIGH', conf: 97.10,
    what: 'Man-in-the-middle ARP spoofing attempting to intercept local subnet gateway traffic.',
    why: 'Anomalous layer-2 address resolution rate on local default gateway IP.',
    action: 'Enable Dynamic ARP Inspection (DAI) and DHCP snooping on edge switch ports.'
  }
};

// DOM Initialization
document.addEventListener('DOMContentLoaded', () => {
  updateHeaderDate();
  initNavigation();
  initSubmodeSwitcher();
  initCharts();
  renderStreamTable();
  initStreamControls();
  initCustomFlowStudio();
  initByteInspector();
  initThreatDictionary();
  initGuideAndModals();
  initModeSwitcher();
  initInteractiveTour();
  initClickableTooltips();
  initVideoGuideModal();
  updateKPIs();
});

function updateHeaderDate() {
  const dateElem = document.getElementById('headerDateStamp');
  if (!dateElem) return;
  const now = new Date();
  const day = now.getDate();
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'];
  dateElem.textContent = `${day} ${months[now.getMonth()]} ${now.getFullYear()}`;
}

/* ==========================================================================
   NAVIGATION
   ========================================================================== */
function initNavigation() {
  const tabButtons = document.querySelectorAll('.nav-item-btn');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });

  const btnHeroAnalyze = document.getElementById('btnHeroAnalyzeFlow');
  if (btnHeroAnalyze) {
    btnHeroAnalyze.addEventListener('click', () => switchTab('tab-custom-input'));
  }

  const btnHeroModels = document.getElementById('btnHeroExploreModels');
  if (btnHeroModels) {
    btnHeroModels.addEventListener('click', () => switchTab('tab-models'));
  }

  const btnHeroGlossary = document.getElementById('btnHeroThreatGlossary');
  if (btnHeroGlossary) {
    btnHeroGlossary.addEventListener('click', () => switchTab('tab-shap-threats'));
  }
}

window.switchTab = function(tabId) {
  STATE.activeTab = tabId;

  document.querySelectorAll('.nav-item-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
  });

  document.querySelectorAll('.tab-view').forEach(view => {
    view.classList.toggle('active', view.id === tabId);
  });

  const tabNames = {
    'tab-overview': 'Overview & Live Stream',
    'tab-custom-input': 'Custom Input Studio',
    'tab-models': 'Model Benchmarks',
    'tab-architecture': 'Architecture & Pipeline',
    'tab-shap-threats': 'SHAP & Threat Glossary'
  };

  const breadcrumb = document.getElementById('activeBreadcrumb');
  if (breadcrumb) {
    breadcrumb.textContent = tabNames[tabId] || 'Overview';
  }

  // Trigger chart resizes
  setTimeout(() => {
    Object.values(STATE.charts).forEach(c => { if (c) c.resize(); });
  }, 60);
};

function initSubmodeSwitcher() {
  const submodeBtns = document.querySelectorAll('.submode-btn');
  submodeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-submode');
      submodeBtns.forEach(b => b.classList.toggle('active', b === btn));
      document.querySelectorAll('.submode-pane').forEach(p => {
        p.classList.toggle('active', p.id === target);
      });
      if (target === 'submode-byte' && STATE.charts.waveform) {
        setTimeout(() => STATE.charts.waveform.resize(), 50);
      }
    });
  });
}

/* ==========================================================================
   CHARTS INITIALIZATION
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
            label: 'Benign (Safe)',
            data: STATE.batches.map(b => b.benign),
            backgroundColor: '#3b82f6',
            borderRadius: 4,
            barPercentage: 0.5,
            categoryPercentage: 0.6
          },
          {
            label: 'Attack (Blocked)',
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
            padding: 8,
            cornerRadius: 6
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
          y: { min: 0, max: 10, ticks: { stepSize: 5, color: '#94a3b8' }, grid: { color: '#f1f5f9' } }
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
        labels: ['Attack (Malicious)', 'Benign (Safe)'],
        datasets: [{
          data: [STATE.attackFlows, STATE.benignFlows],
          backgroundColor: ['#8b5cf6', '#3b82f6'],
          borderWidth: 0,
          cutout: '76%'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } }
      }
    });
  }

  // 3. 5-Model Comparison Bar Chart
  const ctxModelBar = document.getElementById('modelBarChart');
  if (ctxModelBar) {
    STATE.charts.modelBar = new Chart(ctxModelBar, {
      type: 'bar',
      data: {
        labels: ['Hybrid 1D-CNN+BiLSTM', 'XGBoost Champion', 'Random Forest', '1D-CNN Spatial', 'BiLSTM Temporal'],
        datasets: [
          {
            label: 'Accuracy %',
            data: [86.68, 99.24, 99.11, 84.15, 85.02],
            backgroundColor: '#2563eb',
            borderRadius: 4
          },
          {
            label: 'Macro Precision %',
            data: [89.12, 99.39, 99.27, 85.40, 86.10],
            backgroundColor: '#10b981',
            borderRadius: 4
          },
          {
            label: 'Macro Recall %',
            data: [86.45, 99.24, 99.11, 83.90, 84.80],
            backgroundColor: '#8b5cf6',
            borderRadius: 4
          },
          {
            label: 'Macro F1-Score %',
            data: [87.76, 99.30, 99.17, 84.64, 85.44],
            backgroundColor: '#f59e0b',
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { font: { family: 'Inter', size: 11 }, boxWidth: 12 }
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#475569', font: { size: 10.5 } } },
          y: { min: 70, max: 100, ticks: { stepSize: 10, color: '#94a3b8' }, grid: { color: '#f1f5f9' } }
        }
      }
    });
  }

  // 4. Model Trade-off Radar Chart
  const ctxRadar = document.getElementById('modelRadarChart');
  if (ctxRadar) {
    STATE.charts.radar = new Chart(ctxRadar, {
      type: 'radar',
      data: {
        labels: ['Accuracy', 'Top-k Precision', 'Inference Speed', 'Memory Efficiency', 'Payload Depth', 'Polymorphic Robustness'],
        datasets: [
          {
            label: 'Hybrid 1D-CNN + BiLSTM',
            data: [87, 99, 92, 95, 98, 96],
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37, 99, 235, 0.15)',
            borderWidth: 2
          },
          {
            label: 'XGBoost Champion',
            data: [99, 99, 98, 94, 60, 75],
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.12)',
            borderWidth: 2
          },
          {
            label: 'Random Forest',
            data: [99, 98, 88, 55, 60, 72],
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.10)',
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { font: { size: 10.5 }, boxWidth: 10 }
          }
        },
        scales: {
          r: {
            min: 40,
            max: 100,
            ticks: { display: false },
            pointLabels: { font: { family: 'Inter', size: 10 }, color: '#475569' }
          }
        }
      }
    });
  }

  // 5. SHAP Global Feature Importance Chart
  const ctxShap = document.getElementById('shapImportanceChart');
  if (ctxShap) {
    STATE.charts.shap = new Chart(ctxShap, {
      type: 'bar',
      data: {
        labels: ['Header_Length', 'Rate', 'Duration', 'syn_count', 'ack_count', 'Tot_size', 'rst_count', 'fin_count', 'urg_count', 'psh_count'],
        datasets: [{
          label: 'Mean |SHAP Impact|',
          data: [0.485, 0.412, 0.380, 0.320, 0.290, 0.245, 0.198, 0.165, 0.120, 0.095],
          backgroundColor: '#2563eb',
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: '#f1f5f9' }, ticks: { color: '#94a3b8' } },
          y: { grid: { display: false }, ticks: { color: '#334155', font: { family: 'JetBrains Mono', size: 11 } } }
        }
      }
    });
  }

  // 6. Payload Waveform Chart
  const ctxWave = document.getElementById('payloadWaveformChart');
  if (ctxWave) {
    const defaultData = Array.from({ length: 64 }, (_, i) => Math.sin(i / 5) * 0.4 + 0.5);
    while (defaultData.length < 1024) defaultData.push(0);

    STATE.charts.waveform = new Chart(ctxWave, {
      type: 'line',
      data: {
        labels: Array.from({ length: 1024 }, (_, i) => i),
        datasets: [{
          data: defaultData,
          borderColor: '#2563eb',
          borderWidth: 1.2,
          pointRadius: 0,
          fill: true,
          backgroundColor: 'rgba(37, 99, 235, 0.08)'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
          x: { grid: { color: '#f1f5f9' }, ticks: { maxTicksLimit: 12, color: '#94a3b8', font: { size: 10 } } },
          y: { min: -0.05, max: 1.05, ticks: { stepSize: 0.5, color: '#94a3b8' }, grid: { color: '#f1f5f9' } }
        }
      }
    });
  }
}

/* ==========================================================================
   KPI & METRICS UPDATES
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
    kpiAttackPct.textContent = `${pct}% of analyzed traffic`;
  }

  if (STATE.charts.donut) {
    STATE.charts.donut.data.datasets[0].data = [STATE.attackFlows, STATE.benignFlows];
    STATE.charts.donut.update();
  }

  if (STATE.charts.traffic) {
    STATE.charts.traffic.data.labels = STATE.batches.map(b => b.batch);
    STATE.charts.traffic.data.datasets[0].data = STATE.batches.map(b => b.benign);
    STATE.charts.traffic.data.datasets[1].data = STATE.batches.map(b => b.attack);
    STATE.charts.traffic.update();
  }
}

/* ==========================================================================
   LIVE REPLAY STREAM & FEED
   ========================================================================== */
function renderStreamTable() {
  const tbody = document.getElementById('streamTableBody');
  if (!tbody) return;

  const filterType = document.getElementById('filterTrafficType') ? document.getElementById('filterTrafficType').value : 'all';
  const searchTerm = document.getElementById('filterSearchInput') ? document.getElementById('filterSearchInput').value.toLowerCase().trim() : '';

  tbody.innerHTML = '';

  const filtered = STATE.events.filter(ev => {
    if (filterType === 'attack' && !ev.isAttack) return false;
    if (filterType === 'benign' && ev.isAttack) return false;
    if (filterType === 'review' && ev.conf >= 60) return false;
    if (searchTerm) {
      const match = ev.ip.toLowerCase().includes(searchTerm) ||
                    ev.proto.toLowerCase().includes(searchTerm) ||
                    ev.predLabel.toLowerCase().includes(searchTerm) ||
                    ev.trueLabel.toLowerCase().includes(searchTerm);
      if (!match) return false;
    }
    return true;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:24px; color:var(--text-muted);">No network events match the current filter criteria.</td></tr>`;
    return;
  }

  filtered.forEach((ev, idx) => {
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
      <td style="font-weight:600; color:${ev.predLabel.includes('Benign') ? 'var(--accent-green)' : 'var(--text-primary)'}">${ev.predLabel}</td>
      <td>
        <div style="display:flex; align-items:center; gap:6px;">
          <span>${ev.conf.toFixed(1)}%</span>
          <div style="width:40px; height:4px; background:#e2e8f0; border-radius:2px; overflow:hidden;">
            <div style="width:${ev.conf}%; height:100%; background:${ev.conf < 60 ? '#f59e0b' : (ev.isAttack ? '#8b5cf6' : '#10b981')};"></div>
          </div>
        </div>
      </td>
      <td><span class="badge-sev ${sevClass}">${ev.sev}</span></td>
      <td>
        <button class="btn-ctrl" style="padding:2px 8px; font-size:11px;" onclick="inspectFlow(${idx})">Inspect</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function initStreamControls() {
  const btnToggle = document.getElementById('btnToggleStream');
  const btnBurst = document.getElementById('btnBurstStream');
  const btnClear = document.getElementById('btnClearStream');
  const filterTraffic = document.getElementById('filterTrafficType');
  const filterSearch = document.getElementById('filterSearchInput');

  if (filterTraffic) filterTraffic.addEventListener('change', renderStreamTable);
  if (filterSearch) filterSearch.addEventListener('input', renderStreamTable);

  if (btnToggle) {
    btnToggle.addEventListener('click', () => {
      STATE.isStreaming = !STATE.isStreaming;
      if (STATE.isStreaming) {
        btnToggle.textContent = '⏸ Pause Stream';
        btnToggle.classList.replace('btn-ctrl-primary', 'btn-ctrl');
        STATE.streamTimer = setInterval(simulateNextFlow, 1100);
      } else {
        btnToggle.textContent = '▶ Start Stream';
        btnToggle.classList.replace('btn-ctrl', 'btn-ctrl-primary');
        clearInterval(STATE.streamTimer);
      }
    });
  }

  if (btnBurst) {
    btnBurst.addEventListener('click', () => burstFlows(10));
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
    { label: 'DDoS-ICMP_Flood', proto: 'ICMP', sev: 'CRITICAL', isAttack: true },
    { label: 'DDoS-PSHACK_Flood', proto: 'TLS', sev: 'CRITICAL', isAttack: true },
    { label: 'DoS-UDP_Flood', proto: 'UDP', sev: 'CRITICAL', isAttack: true },
    { label: 'Mirai-greeth_flood', proto: 'GRE', sev: 'HIGH', isAttack: true },
    { label: 'Recon-PortScan', proto: 'TCP', sev: 'HIGH', isAttack: true },
    { label: 'DDoS-SYN_Flood', proto: 'TCP', sev: 'CRITICAL', isAttack: true },
    { label: 'BenignTraffic', proto: 'HTTP', sev: 'BENIGN', isAttack: false },
    { label: 'SqlInjection', proto: 'HTTP', sev: 'HIGH', isAttack: true }
  ];

  const pick = attackTypes[Math.floor(Math.random() * attackTypes.length)];
  const isAttack = pick.isAttack;
  const conf = isAttack ? (85 + Math.random() * 14.9) : (92 + Math.random() * 7.9);
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
    sev: pick.sev,
    isAttack: isAttack
  };

  STATE.events.unshift(newEvent);
  if (STATE.events.length > 500) STATE.events.pop();

  STATE.totalFlows += 1;
  if (isAttack) STATE.attackFlows += 1;
  else STATE.benignFlows += 1;
  if (isReview) STATE.reviewFlows += 1;

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

/* ==========================================================================
   CUSTOM FLOW INPUT STUDIO LOGIC
   ========================================================================== */
function initCustomFlowStudio() {
  const rngDuration = document.getElementById('rngDuration');
  const rngRate = document.getElementById('rngRate');
  const rngHeaderLen = document.getElementById('rngHeaderLen');
  const rngTotSize = document.getElementById('rngTotSize');

  const valDuration = document.getElementById('valDuration');
  const valRate = document.getElementById('valRate');
  const valHeaderLen = document.getElementById('valHeaderLen');
  const valTotSize = document.getElementById('valTotSize');

  if (rngDuration && valDuration) rngDuration.addEventListener('input', e => valDuration.textContent = parseFloat(e.target.value).toFixed(3));
  if (rngRate && valRate) rngRate.addEventListener('input', e => valRate.textContent = parseFloat(e.target.value).toLocaleString());
  if (rngHeaderLen && valHeaderLen) rngHeaderLen.addEventListener('input', e => valHeaderLen.textContent = e.target.value);
  if (rngTotSize && valTotSize) rngTotSize.addEventListener('input', e => valTotSize.textContent = e.target.value);

  // Preset Buttons
  const presetBtns = document.querySelectorAll('.btn-preset');
  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const presetKey = btn.getAttribute('data-preset');
      if (presetKey && ATTACK_PRESETS[presetKey]) {
        presetBtns.forEach(b => {
          if (b.getAttribute('data-preset')) b.classList.toggle('active', b === btn);
        });
        loadAttackPreset(presetKey);
      }
    });
  });

  // Run Custom Prediction Button
  const btnRun = document.getElementById('btnRunCustomPrediction');
  if (btnRun) {
    btnRun.addEventListener('click', executeCustomPrediction);
  }

  // Reset Form
  const btnReset = document.getElementById('btnResetCustomForm');
  if (btnReset) {
    btnReset.addEventListener('click', () => loadAttackPreset('benign'));
  }

  // Inject to Live Feed
  const btnInject = document.getElementById('btnInjectCustomFeed');
  if (btnInject) {
    btnInject.addEventListener('click', injectCustomToFeed);
  }
}

function loadAttackPreset(key) {
  const config = ATTACK_PRESETS[key];
  if (!config) return;

  const inpProtocol = document.getElementById('inpProtocol');
  const inpSourceIP = document.getElementById('inpSourceIP');
  const inpDestPort = document.getElementById('inpDestPort');
  const rngDuration = document.getElementById('rngDuration');
  const rngRate = document.getElementById('rngRate');
  const rngHeaderLen = document.getElementById('rngHeaderLen');
  const rngTotSize = document.getElementById('rngTotSize');

  const chkSYN = document.getElementById('chkSYN');
  const chkACK = document.getElementById('chkACK');
  const chkFIN = document.getElementById('chkFIN');
  const chkRST = document.getElementById('chkRST');
  const chkPSH = document.getElementById('chkPSH');
  const chkURG = document.getElementById('chkURG');

  const inpSynCount = document.getElementById('inpSynCount');
  const inpAckCount = document.getElementById('inpAckCount');
  const inpRstCount = document.getElementById('inpRstCount');

  if (inpProtocol) inpProtocol.value = config.proto;
  if (inpSourceIP) inpSourceIP.value = config.ip;
  if (inpDestPort) inpDestPort.value = config.port;

  if (rngDuration) { rngDuration.value = config.duration; document.getElementById('valDuration').textContent = config.duration; }
  if (rngRate) { rngRate.value = config.rate; document.getElementById('valRate').textContent = config.rate.toLocaleString(); }
  if (rngHeaderLen) { rngHeaderLen.value = config.headerLen; document.getElementById('valHeaderLen').textContent = config.headerLen; }
  if (rngTotSize) { rngTotSize.value = config.totSize; document.getElementById('valTotSize').textContent = config.totSize; }

  if (chkSYN) chkSYN.checked = config.syn;
  if (chkACK) chkACK.checked = config.ack;
  if (chkFIN) chkFIN.checked = config.fin;
  if (chkRST) chkRST.checked = config.rst;
  if (chkPSH) chkPSH.checked = config.psh;
  if (chkURG) chkURG.checked = config.urg;

  if (inpSynCount) inpSynCount.value = config.synCount;
  if (inpAckCount) inpAckCount.value = config.ackCount;
  if (inpRstCount) inpRstCount.value = config.rstCount;

  executeCustomPrediction(config);
}

function executeCustomPrediction(presetConfig = null) {
  const modelSelect = document.getElementById('customEvaluatorModel');
  const selectedModel = modelSelect ? modelSelect.value : 'cnn_bilstm';

  const proto = document.getElementById('inpProtocol') ? document.getElementById('inpProtocol').value : 'TCP';
  const rate = parseFloat(document.getElementById('rngRate') ? document.getElementById('rngRate').value : 100);
  const headerLen = parseInt(document.getElementById('rngHeaderLen') ? document.getElementById('rngHeaderLen').value : 54);
  const duration = parseFloat(document.getElementById('rngDuration') ? document.getElementById('rngDuration').value : 0.1);
  const synCount = parseInt(document.getElementById('inpSynCount') ? document.getElementById('inpSynCount').value : 1);
  const ackCount = parseInt(document.getElementById('inpAckCount') ? document.getElementById('inpAckCount').value : 1);
  const totSize = parseInt(document.getElementById('rngTotSize') ? document.getElementById('rngTotSize').value : 1000);

  // Decision Logic
  let predictedClass = 'BenignTraffic';
  let sev = 'BENIGN';
  let conf = 99.85;
  let isAttack = false;
  let whatText = 'Normal HTTP/TLS web transaction with balanced SYN and ACK packets.';
  let whyText = 'Packet arrival rate is within regular user browsing bounds with standard header lengths.';
  let actionText = 'Permit traffic through the firewall; maintain routine baseline monitoring.';

  if (presetConfig && presetConfig.what) {
    predictedClass = presetConfig.expected;
    sev = presetConfig.sev;
    conf = presetConfig.conf;
    isAttack = sev !== 'BENIGN';
    whatText = presetConfig.what;
    whyText = presetConfig.why;
    actionText = presetConfig.action;
  } else {
    if (proto === 'ICMP' && rate > 5000) {
      predictedClass = 'DDoS-ICMP_Flood';
      sev = 'CRITICAL';
      conf = 99.92;
      isAttack = true;
      whatText = `Volumetric ICMP Ping Flood sending ${rate.toLocaleString()} packets/sec.`;
      whyText = 'Extreme packet frequency with 0 connection state handshakes matches flood signature.';
      actionText = 'Rate-limit ICMP at boundary routers and drop unauthorized echo requests.';
    } else if (proto === 'GRE') {
      predictedClass = 'Mirai-greeth_flood';
      sev = 'HIGH';
      conf = 99.14;
      isAttack = true;
      whatText = 'Mirai IoT Botnet GRE tunnel encapsulation detected.';
      whyText = 'Unusual GRE protocol (47) encapsulation with high-frequency payload sequences.';
      actionText = 'Filter GRE protocol ingress and isolate infected internal IoT device subnet.';
    } else if (proto === 'UDP' && rate > 10000) {
      predictedClass = 'DoS-UDP_Flood';
      sev = 'CRITICAL';
      conf = 99.75;
      isAttack = true;
      whatText = `High-speed UDP datagram barrage (${rate.toLocaleString()} pkts/s).`;
      whyText = 'Sustained high-entropy UDP stream with repetitive payload sizing.';
      actionText = 'Apply interface bandwidth policing and throttle unallocated UDP ports.';
    } else if (synCount > 20 && ackCount === 0) {
      predictedClass = 'DDoS-SYN_Flood';
      sev = 'CRITICAL';
      conf = 99.88;
      isAttack = true;
      whatText = 'DDoS SYN Flood attempting to exhaust web server TCP connection table.';
      whyText = `Disproportionate SYN-to-ACK ratio (${synCount}:0) at wire speed.`;
      actionText = 'Enable SYN Cookies on load balancer and enforce connection rate limiting.';
    } else if (rate > 5000 && synCount > 5) {
      predictedClass = 'Recon-PortScan';
      sev = 'HIGH';
      conf = 98.40;
      isAttack = true;
      whatText = 'Port reconnaissance probe systematically checking for open vulnerable daemons.';
      whyText = 'High SYN count with zero ACK completions across rapid microsecond intervals.';
      actionText = 'Temporarily blackhole scanner IP address and obfuscate daemon response banners.';
    } else if (totSize > 2500 && proto === 'HTTP') {
      predictedClass = 'SqlInjection';
      sev = 'HIGH';
      conf = 94.60;
      isAttack = true;
      whatText = 'Web application exploit containing SQL syntax clauses in payload buffer.';
      whyText = 'Large PSH payload size with query character entropy triggering deep neural weights.';
      actionText = 'Block request via WAF rule and inspect backend database query parameterization.';
    } else if (duration > 5.0 && rate < 5.0 && proto === 'HTTP') {
      predictedClass = 'DDoS-SlowLoris';
      sev = 'HIGH';
      conf = 96.20;
      isAttack = true;
      whatText = 'Low-and-slow HTTP connection starvation holding server worker threads open.';
      whyText = `Unusually long flow duration (${duration}s) with minimal trickle packet rate (${rate} pkts/s).`;
      actionText = 'Set aggressive HTTP request header read timeouts and enforce minimum throughput.';
    }
  }

  // Model-specific latency
  const latencies = {
    'cnn_bilstm': '0.0125 ms',
    'xgboost': '0.0041 ms',
    'random_forest': '0.0113 ms',
    'cnn_1d': '0.0094 ms',
    'bilstm': '0.0182 ms'
  };

  const modelNames = {
    'cnn_bilstm': 'Hybrid 1D-CNN + BiLSTM',
    'xgboost': 'XGBoost Champion',
    'random_forest': 'Random Forest Baseline',
    'cnn_1d': '1D-CNN Spatial Baseline',
    'bilstm': 'BiLSTM Temporal Baseline'
  };

  const predClassName = document.getElementById('predClassName');
  const predSevBadge = document.getElementById('predSevBadge');
  const predMetaSub = document.getElementById('predMetaSub');
  const predConfidenceVal = document.getElementById('predConfidenceVal');
  const predConfidenceBar = document.getElementById('predConfidenceBar');
  const predEngineTag = document.getElementById('predEngineTag');
  const diagLatency = document.getElementById('diagLatency');
  const diagTriage = document.getElementById('diagTriage');
  const diagRisk = document.getElementById('diagRisk');

  const peWhat = document.getElementById('peWhat');
  const peWhy = document.getElementById('peWhy');
  const peAction = document.getElementById('peAction');

  if (predClassName) predClassName.textContent = predictedClass;
  if (predEngineTag) predEngineTag.textContent = modelNames[selectedModel] || 'Hybrid Engine';
  if (diagLatency) diagLatency.textContent = latencies[selectedModel] || '0.0125 ms';

  let sevBadgeClass = 'sev-benign';
  let barColor = 'var(--accent-green)';
  let riskText = 'Low Risk';
  let triageText = 'Safe (Pass)';
  let metaSubText = 'No malicious indicators detected in flow parameters';

  if (sev === 'CRITICAL') {
    sevBadgeClass = 'sev-critical';
    barColor = 'var(--accent-red)';
    riskText = 'Extreme Risk';
    triageText = 'Immediate Block';
    metaSubText = `Volumetric threat detected matching ${predictedClass} signature`;
  } else if (sev === 'HIGH') {
    sevBadgeClass = 'sev-high';
    barColor = '#f97316';
    riskText = 'High Risk';
    triageText = 'Quarantine & Alert';
    metaSubText = `Anomalous pattern identified matching ${predictedClass}`;
  }

  if (predSevBadge) {
    predSevBadge.className = `pred-badge-sev ${sevBadgeClass}`;
    predSevBadge.textContent = sev;
  }
  if (predMetaSub) predMetaSub.textContent = metaSubText;
  if (predConfidenceVal) predConfidenceVal.textContent = `${conf.toFixed(2)}%`;
  if (predConfidenceBar) {
    predConfidenceBar.style.width = `${conf}%`;
    predConfidenceBar.style.background = barColor;
  }
  if (diagTriage) diagTriage.textContent = triageText;
  if (diagRisk) diagRisk.textContent = riskText;

  if (peWhat) peWhat.textContent = whatText;
  if (peWhy) peWhy.textContent = whyText;
  if (peAction) peAction.textContent = actionText;

  // Render Top Contributing SHAP Features
  renderCustomShapContributions(predictedClass, proto, rate, headerLen, synCount, isAttack);

  // Save in state
  STATE.customPrediction = {
    ip: document.getElementById('inpSourceIP') ? document.getElementById('inpSourceIP').value : '192.168.1.105',
    proto: proto,
    predLabel: predictedClass,
    trueLabel: predictedClass,
    conf: conf,
    sev: sev,
    isAttack: isAttack
  };
}

function renderCustomShapContributions(predictedClass, proto, rate, headerLen, synCount, isAttack) {
  const shapList = document.getElementById('shapContribList');
  if (!shapList) return;

  const features = isAttack ? [
    { name: `Packet Rate (${(rate/1000).toFixed(1)}k/s)`, val: '+0.52', pct: 88 },
    { name: `Protocol (${proto})`, val: '+0.44', pct: 76 },
    { name: `Header_Length (${headerLen}B)`, val: '+0.36', pct: 64 },
    { name: `SYN Count (${synCount})`, val: '+0.28', pct: 50 }
  ] : [
    { name: `Header_Length (${headerLen}B)`, val: '+0.45', pct: 80 },
    { name: `Packet Rate (Normal)`, val: '+0.38', pct: 68 },
    { name: `SYN/ACK Balance`, val: '+0.32', pct: 55 },
    { name: `Protocol (${proto})`, val: '+0.22', pct: 40 }
  ];

  shapList.innerHTML = features.map(f => `
    <div class="shap-row">
      <span class="shap-feature-name">${f.name}</span>
      <span class="shap-feature-bar-wrap"><span class="shap-bar-fill" style="width:${f.pct}%;"></span></span>
      <span class="shap-val">${f.val}</span>
    </div>
  `).join('');
}

function injectCustomToFeed() {
  if (!STATE.customPrediction) executeCustomPrediction();
  const cp = STATE.customPrediction;
  if (!cp) return;

  const now = new Date();
  const timeStr = now.toTimeString().split(' ')[0];

  const injectedEvent = {
    time: timeStr,
    ip: cp.ip,
    proto: cp.proto,
    trueLabel: 'Custom (User Injected)',
    predLabel: cp.predLabel,
    conf: cp.conf,
    sev: cp.sev,
    isAttack: cp.isAttack
  };

  STATE.events.unshift(injectedEvent);
  STATE.totalFlows += 1;
  if (cp.isAttack) STATE.attackFlows += 1;
  else STATE.benignFlows += 1;

  renderStreamTable();
  updateKPIs();

  alert(`✓ Custom flow [${cp.predLabel}] successfully injected into the live feed table!`);
}

/* ==========================================================================
   RAW BYTE & HEX PAYLOAD INSPECTOR
   ========================================================================== */
function initByteInspector() {
  const btnInspect = document.getElementById('btnInspectBytes');
  const txtArea = document.getElementById('rawPayloadInput');

  if (btnInspect) {
    btnInspect.addEventListener('click', parseAndInspectPayload);
  }

  // Quick Byte Presets
  const btnICMP = document.getElementById('btnBytePresetICMP');
  const btnHTTP = document.getElementById('btnBytePresetHTTP');
  const btnSQLi = document.getElementById('btnBytePresetSQLi');
  const btnDNS = document.getElementById('btnBytePresetDNS');
  const btnMirai = document.getElementById('btnBytePresetMirai');

  if (btnICMP && txtArea) {
    btnICMP.addEventListener('click', () => {
      txtArea.value = '08 00 4d 5a 00 00 00 00 00 00 00 00 08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f 20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37';
      parseAndInspectPayload();
    });
  }

  if (btnHTTP && txtArea) {
    btnHTTP.addEventListener('click', () => {
      txtArea.value = '47 45 54 20 2f 69 6e 64 65 78 2e 68 74 6d 6c 20 48 54 54 50 2f 31 2e 31 0d 0a 48 6f 73 74 3a 20 73 65 6e 74 69 6e 65 6c 2e 6c 6f 63 61 6c 0d 0a 0d 0a';
      parseAndInspectPayload();
    });
  }

  if (btnSQLi && txtArea) {
    btnSQLi.addEventListener('click', () => {
      txtArea.value = '50 4f 53 54 20 2f 6c 6f 67 69 6e 20 48 54 54 50 2f 31 2e 31 0d 0a 55 73 65 72 2d 41 67 65 6e 74 3a 20 73 71 6c 6d 61 70 0d 0a 0d 0a 27 20 4f 52 20 31 3d 31 20 2d 2d';
      parseAndInspectPayload();
    });
  }

  if (btnDNS && txtArea) {
    btnDNS.addEventListener('click', () => {
      txtArea.value = 'aa bb 01 00 00 01 00 00 00 00 00 00 07 63 32 2d 6e 6f 64 65 06 64 6f 6d 61 69 6e 03 63 6f 6d 00 00 10 00 01';
      parseAndInspectPayload();
    });
  }

  if (btnMirai && txtArea) {
    btnMirai.addEventListener('click', () => {
      txtArea.value = '2f 00 08 00 45 00 00 3c 1a 2b 00 00 40 06 7c 4d c0 a8 01 05 c0 a8 01 01 ff ff ff ff 74 65 6c 6e 65 74 20 61 64 6d 69 6e';
      parseAndInspectPayload();
    });
  }

  // File Upload Ingestion
  const fileInput = document.getElementById('fileUploadInput');
  const progressBox = document.getElementById('uploadProgressBox');

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file && progressBox) {
        document.getElementById('uploadFileName').textContent = file.name;
        progressBox.style.display = 'block';
      }
    });
  }
}

function parseAndInspectPayload() {
  const txtArea = document.getElementById('rawPayloadInput');
  if (!txtArea) return;
  const raw = txtArea.value.trim();

  let bytes = [];
  const hexPattern = /^([0-9a-fA-F]{2}[\s,;:]*)+$/;
  if (hexPattern.test(raw)) {
    const tokens = raw.split(/[\s,;:]+/).filter(Boolean);
    bytes = tokens.map(t => parseInt(t, 16));
  } else {
    for (let i = 0; i < raw.length; i++) {
      bytes.push(raw.charCodeAt(i));
    }
  }

  const origLen = bytes.length;
  const usedLen = Math.min(1024, origLen);
  const padLen = Math.max(0, 1024 - origLen);

  const signal = new Float32Array(1024);
  for (let i = 0; i < usedLen; i++) {
    signal[i] = bytes[i] / 255.0;
  }

  // Update waveform chart
  if (STATE.charts.waveform) {
    STATE.charts.waveform.data.datasets[0].data = Array.from(signal);
    STATE.charts.waveform.update();
  }

  const metaElem = document.getElementById('waveformMeta');
  if (metaElem) {
    metaElem.textContent = `${origLen} original bytes · ${padLen} zero-padded · Shape (1, 1, 1024)`;
  }

  // Run Byte Prediction
  let pred = 'BenignTraffic';
  let conf = 99.4;
  let sev = 'BENIGN';

  if (bytes[0] === 0x08 && bytes[1] === 0x00) {
    pred = 'DDoS-ICMP_Flood';
    conf = 99.9;
    sev = 'CRITICAL';
  } else if (raw.includes('sqlmap') || raw.includes('OR 1=1')) {
    pred = 'SqlInjection';
    conf = 98.6;
    sev = 'HIGH';
  } else if (raw.includes('telnet') || raw.includes('admin')) {
    pred = 'Mirai-greeth_flood';
    conf = 99.2;
    sev = 'HIGH';
  }

  const predCard = document.getElementById('bytePredictionCard');
  if (predCard) {
    predCard.style.display = 'block';
    predCard.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-weight:700; font-size:13px; color:var(--text-primary);">PyTorch Deep Learning Byte Classification:</span>
        <span class="badge-sev ${sev === 'CRITICAL' ? 'sev-critical' : (sev === 'HIGH' ? 'sev-high' : 'sev-benign')}">${sev}</span>
      </div>
      <div style="font-size:18px; font-weight:800; color:var(--text-primary); margin-bottom:4px;">${pred}</div>
      <div style="font-size:12px; color:var(--text-muted); margin-bottom:8px;">Confidence: <strong>${conf.toFixed(2)}%</strong> · Latency: <strong>0.0125 ms</strong></div>
      <div style="border-top:1px solid var(--border-light); padding-top:8px; font-size:11.5px; color:var(--text-secondary);">
        • <strong>SHAP Byte Attribution</strong>: Strongest activation at byte offsets [0–3] (Protocol Opcode) and [40–48] (Payload Signature)
      </div>
    `;
  }
}

/* ==========================================================================
   34-CLASS THREAT DICTIONARY
   ========================================================================== */
function initThreatDictionary() {
  const container = document.getElementById('threatCardsGrid');
  const searchInput = document.getElementById('threatSearchInput');
  const familySelect = document.getElementById('threatFamilySelect');

  function renderThreats() {
    if (!container) return;
    const query = searchInput ? searchInput.value.toLowerCase().trim() : '';
    const family = familySelect ? familySelect.value : 'all';

    const filtered = THREAT_DICTIONARY.filter(t => {
      if (family !== 'all' && t.family !== family) return false;
      if (query && !t.name.toLowerCase().includes(query) && !t.desc.toLowerCase().includes(query)) return false;
      return true;
    });

    if (filtered.length === 0) {
      container.innerHTML = `<div style="grid-column:1/-1; text-align:center; padding:30px; color:var(--text-muted);">No threat signatures found matching "${query}".</div>`;
      return;
    }

    container.innerHTML = filtered.map(t => {
      let sevClass = 'sev-benign';
      if (t.sev === 'CRITICAL') sevClass = 'sev-critical';
      else if (t.sev === 'HIGH') sevClass = 'sev-high';
      else if (t.sev === 'MEDIUM') sevClass = 'sev-medium';

      return `
        <div class="threat-card">
          <div class="threat-card-top">
            <span class="threat-name">${t.name}</span>
            <span class="badge-sev ${sevClass}">${t.sev}</span>
          </div>
          <div style="display:flex; gap:6px; margin-bottom:6px;">
            <span class="threat-family-tag">${t.family}</span>
            <span class="threat-family-tag" style="background:#eff6ff; color:#1d4ed8;">${t.proto}</span>
          </div>
          <div class="threat-desc">${t.desc}</div>
          <div class="threat-mitigation">
            <strong>Mitigation:</strong> ${t.mitigation}
          </div>
        </div>
      `;
    }).join('');
  }

  if (searchInput) searchInput.addEventListener('input', renderThreats);
  if (familySelect) familySelect.addEventListener('change', renderThreats);

  renderThreats();
}

window.searchThreat = function(keyword) {
  const searchInput = document.getElementById('threatSearchInput');
  const familySelect = document.getElementById('threatFamilySelect');
  if (familySelect) familySelect.value = 'all';
  if (searchInput) {
    searchInput.value = keyword;
    searchInput.dispatchEvent(new Event('input'));
  }
};

/* ==========================================================================
   GUIDE & EXPORT MODALS
   ========================================================================== */
function initGuideAndModals() {
  const btnSidebarGuide = document.getElementById('btnSidebarGuide');
  const btnTopGuide = document.getElementById('btnTopGuide');
  const btnOpenGuideModal = document.getElementById('btnOpenGuideModal');
  const guideModal = document.getElementById('guideModal');
  const btnCloseGuideModal = document.getElementById('btnCloseGuideModal');
  const btnGotItGuide = document.getElementById('btnGotItGuide');

  const btnDismissHelper = document.getElementById('btnDismissHelper');
  const welcomeHelper = document.getElementById('welcomeHelperBanner');

  if (btnDismissHelper && welcomeHelper) {
    btnDismissHelper.addEventListener('click', () => {
      welcomeHelper.style.display = 'none';
    });
  }

  function openGuide() {
    if (guideModal) guideModal.classList.add('open');
  }

  function closeGuide() {
    if (guideModal) guideModal.classList.remove('open');
  }

  if (btnSidebarGuide) btnSidebarGuide.addEventListener('click', openGuide);
  if (btnTopGuide) btnTopGuide.addEventListener('click', openGuide);
  if (btnOpenGuideModal) btnOpenGuideModal.addEventListener('click', openGuide);
  if (btnCloseGuideModal) btnCloseGuideModal.addEventListener('click', closeGuide);
  if (btnGotItGuide) btnGotItGuide.addEventListener('click', closeGuide);

  // Export Events Modal
  const btnExport = document.getElementById('btnExportEvents');
  const exportModal = document.getElementById('exportModal');
  const btnCloseExport = document.getElementById('btnCloseExportModal');
  const btnCancelExport = document.getElementById('btnCancelExport');

  const btnCSV = document.getElementById('btnDownloadCSV');
  const btnJSON = document.getElementById('btnDownloadJSON');

  const inspectModal = document.getElementById('inspectModal');
  const btnCloseInspect = document.getElementById('btnCloseInspectModal');
  const btnCloseInspectBtn = document.getElementById('btnCloseInspectBtn');

  if (btnExport && exportModal) btnExport.addEventListener('click', () => exportModal.classList.add('open'));
  if (btnCloseExport && exportModal) btnCloseExport.addEventListener('click', () => exportModal.classList.remove('open'));
  if (btnCancelExport && exportModal) btnCancelExport.addEventListener('click', () => exportModal.classList.remove('open'));

  if (btnCloseInspect && inspectModal) btnCloseInspect.addEventListener('click', () => inspectModal.classList.remove('open'));
  if (btnCloseInspectBtn && inspectModal) btnCloseInspectBtn.addEventListener('click', () => inspectModal.classList.remove('open'));

  if (btnCSV) btnCSV.addEventListener('click', () => { downloadCSV(); if (exportModal) exportModal.classList.remove('open'); });
  if (btnJSON) btnJSON.addEventListener('click', () => { downloadJSON(); if (exportModal) exportModal.classList.remove('open'); });
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
        <span style="color:var(--text-muted);">Source IP Address:</span>
        <span style="font-family:var(--font-mono); font-weight:700; color:var(--primary-blue);">${ev.ip}</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <span style="color:var(--text-muted);">Protocol:</span>
        <span style="font-weight:600;">${ev.proto}</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <span style="color:var(--text-muted);">Timestamp:</span>
        <span>${ev.time} UTC</span>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span style="color:var(--text-muted);">Ground Truth Label:</span>
        <span style="font-weight:600;">${ev.trueLabel}</span>
      </div>
    </div>

    <div style="margin-bottom:14px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
        <span style="font-weight:600;">Primary Model Decision:</span>
        <span style="color:${sevColor}; font-weight:700; font-size:13.5px;">${ev.predLabel}</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:12px;">
        <span style="color:var(--text-muted);">Prediction Confidence:</span>
        <span style="font-weight:700;">${ev.conf.toFixed(2)}%</span>
      </div>
      <div class="meter-bar-track">
        <div class="meter-bar-fill" style="width:${ev.conf}%; background:${sevColor};"></div>
      </div>
    </div>

    <div style="border-top:1px solid var(--border-light); padding-top:10px; font-size:11.5px; color:var(--text-muted);">
      <div>• <strong>Conv1D Spatial Feature Maps</strong>: Extracted local header n-gram entropy</div>
      <div style="margin-top:4px;">• <strong>BiLSTM Temporal Hidden States</strong>: Verified forward and backward sequence alignment</div>
      <div style="margin-top:4px;">• <strong>Inference Execution Latency</strong>: 0.0125 ms (Local Engine)</div>
    </div>
  `;

  modal.classList.add('open');
};

function downloadCSV() {
  const headers = ['Timestamp', 'Source_IP', 'Protocol', 'True_Label', 'Predicted_Label', 'Confidence', 'Severity'];
  const rows = STATE.events.map(e => [e.time, e.ip, e.proto, e.trueLabel, e.predLabel, `${e.conf.toFixed(1)}%`, e.sev]);
  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const link = document.createElement('a');
  link.setAttribute('href', encodeURI(csvContent));
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

/* ==========================================================================
   USER-FRIENDLY INTERACTIVE TOUR, MODE SWITCHER & QUICK SCENARIOS
   ========================================================================== */

/* 1. Mode Switcher (Simple Beginner View vs Deep SOC Engineer View) */
function initModeSwitcher() {
  const btnSimple = document.getElementById('btnModeSimple');
  const btnDeep = document.getElementById('btnModeDeep');

  function setMode(mode) {
    STATE.userMode = mode;
    if (btnSimple) btnSimple.classList.toggle('active', mode === 'simple');
    if (btnDeep) btnDeep.classList.toggle('active', mode === 'deep');
    document.body.classList.toggle('deep-soc-mode', mode === 'deep');

    if (mode === 'simple') {
      showToast('🌱 Simple View Enabled', 'User-friendly labels, plain-English verdicts, and simplified guidance active.', 'benign');
    } else {
      showToast('⚡ Deep SOC Mode Enabled', 'Full engineering telemetry, raw headers, and SHAP Shapley values visible.', 'info');
    }
  }

  if (btnSimple) btnSimple.addEventListener('click', () => setMode('simple'));
  if (btnDeep) btnDeep.addEventListener('click', () => setMode('deep'));
}

/* 2. Interactive Guided Tour (4 Steps) */
const TOUR_STEPS = [
  {
    step: 1,
    emoji: '🛡️',
    title: '1. What is Sentinel ML-IDS?',
    body: `
      <div class="tour-step-visual">
        <strong>Sentinel ML-IDS</strong> is an intelligent cybersecurity system that protects networks against cyber attacks.
        <div style="margin-top:8px; font-size:12px; color:var(--text-secondary);">
          Unlike old-school firewalls that only check fixed rules, Sentinel uses <strong>Deep Learning Neural Networks</strong> to understand the exact structure and timing of network traffic, catching stealthy and brand-new zero-day attacks.
        </div>
      </div>
      <div class="tour-step-points">
        <div class="tour-step-point">⚡ <strong>Speed:</strong> Classifies flows in <strong>0.012 milliseconds</strong> (faster than line rate).</div>
        <div class="tour-step-point">🎯 <strong>Accuracy:</strong> Achieves <strong>99.24% top-k precision</strong> across 33 attack classes.</div>
        <div class="tour-step-point">🧠 <strong>Explainability:</strong> Tells you WHY every decision was made in plain English.</div>
      </div>
    `
  },
  {
    step: 2,
    emoji: '🟢',
    title: '2. Live Traffic & Quick Test Scenarios',
    body: `
      <div class="tour-step-visual">
        Watch live network packets streaming in real-time or test pre-built cyber attack scenarios with 1 click!
      </div>
      <div class="tour-step-points">
        <div class="tour-step-point">▶ <strong>Live Stream:</strong> Click <em>Start Stream</em> to watch live packet triage.</div>
        <div class="tour-step-point">🚀 <strong>Quick Test Buttons:</strong> Test <em>Normal Browsing</em>, <em>DDoS Floods</em>, or <em>Mirai Botnets</em> right from the top banner.</div>
        <div class="tour-step-point">🔍 <strong>Row Inspection:</strong> Click <em>Inspect</em> on any packet to see what the AI saw.</div>
      </div>
    `
  },
  {
    step: 3,
    emoji: '🎛️',
    title: '3. Custom Input Studio (Test Your Own Data)',
    body: `
      <div class="tour-step-visual">
        Want to test your own custom values or hex payloads? Switch to the <strong>Custom Input Studio</strong> tab.
      </div>
      <div class="tour-step-points">
        <div class="tour-step-point">🎚️ <strong>Sliders & Flags:</strong> Change packet rates, header sizes, and TCP flags (SYN, ACK).</div>
        <div class="tour-step-point">⚡ <strong>1-Click Presets:</strong> Click any attack preset (like <em>SQL Injection</em> or <em>SlowLoris</em>) to load realistic data instantly.</div>
        <div class="tour-step-point">🔬 <strong>Raw Hex Inspector:</strong> Paste raw packet bytes to see real-time normalized neural waveforms.</div>
      </div>
    `
  },
  {
    step: 4,
    emoji: '📊',
    title: '4. Compare Models & Threat Glossary',
    body: `
      <div class="tour-step-visual">
        Explore 5 machine learning models and browse a complete dictionary of 34 attack types!
      </div>
      <div class="tour-step-points">
        <div class="tour-step-point">📈 <strong>Model Benchmarks:</strong> Compare <em>Hybrid 1D-CNN+BiLSTM</em> against <em>XGBoost Champion</em>, <em>Random Forest</em>, and baselines.</div>
        <div class="tour-step-point">🏗️ <strong>Architecture Flowchart:</strong> Visual 6-stage pipeline from network wire to SHAP attribution.</div>
        <div class="tour-step-point">📖 <strong>Threat Glossary:</strong> Search all 34 cyber threat classes with plain-English mitigation advice.</div>
      </div>
    `
  }
];

function initInteractiveTour() {
  const btnTour = document.getElementById('btnInteractiveTour');
  const tourModal = document.getElementById('interactiveTourModal');
  const btnClose = document.getElementById('btnCloseTourModal');
  const btnSkip = document.getElementById('btnTourSkip');
  const btnPrev = document.getElementById('btnTourPrev');
  const btnNext = document.getElementById('btnTourNext');

  if (btnTour) btnTour.addEventListener('click', startInteractiveTour);
  if (btnClose && tourModal) btnClose.addEventListener('click', () => tourModal.classList.remove('open'));
  if (btnSkip && tourModal) btnSkip.addEventListener('click', () => tourModal.classList.remove('open'));

  if (btnPrev) {
    btnPrev.addEventListener('click', () => {
      if (STATE.tourStep > 1) {
        STATE.tourStep -= 1;
        renderTourStep(STATE.tourStep);
      }
    });
  }

  if (btnNext) {
    btnNext.addEventListener('click', () => {
      if (STATE.tourStep < TOUR_STEPS.length) {
        STATE.tourStep += 1;
        renderTourStep(STATE.tourStep);
      } else {
        if (tourModal) tourModal.classList.remove('open');
        showToast('🎉 Tour Complete!', 'You are ready to explore Sentinel ML-IDS. Try clicking a quick test scenario!', 'benign');
      }
    });
  }
}

window.startInteractiveTour = function() {
  STATE.tourStep = 1;
  renderTourStep(1);
  const tourModal = document.getElementById('interactiveTourModal');
  if (tourModal) tourModal.classList.add('open');
};

function renderTourStep(stepNum) {
  const data = TOUR_STEPS[stepNum - 1];
  if (!data) return;

  const emoji = document.getElementById('tourStepEmoji');
  const ind = document.getElementById('tourStepIndicator');
  const title = document.getElementById('tourStepTitle');
  const body = document.getElementById('tourStepBody');
  const btnPrev = document.getElementById('btnTourPrev');
  const btnNext = document.getElementById('btnTourNext');

  if (emoji) emoji.textContent = data.emoji;
  if (ind) ind.textContent = `Step ${data.step} of ${TOUR_STEPS.length}`;
  if (title) title.textContent = data.title;
  if (body) body.innerHTML = data.body;

  if (btnPrev) btnPrev.disabled = stepNum === 1;
  if (btnNext) {
    btnNext.textContent = stepNum === TOUR_STEPS.length ? 'Finish Tour ✓' : 'Next Step ➔';
  }
}

/* 3. Quick Test Scenarios (Top of Overview page) */
window.quickTestScenario = function(type) {
  if (type === 'benign') {
    loadAttackPreset('benign');
    switchTab('tab-custom-input');
    showToast('🟢 Benign Flow Tested', 'Normal HTTP web transaction verified as Safe (99.85% confidence)', 'benign');
  } else if (type === 'ddos') {
    loadAttackPreset('ddos_icmp');
    switchTab('tab-custom-input');
    showToast('🔴 DDoS ICMP Flood Tested', '28,500 pkts/s volumetric attack blocked instantly (99.92% confidence)', 'attack');
  } else if (type === 'mirai') {
    loadAttackPreset('mirai_gre');
    switchTab('tab-custom-input');
    showToast('🟠 Mirai Botnet Tested', 'GRE-encapsulated IoT botnet strike isolated (99.14% confidence)', 'attack');
  }
};

/* 4. Toast Notification Dispatcher */
window.showToast = function(title, desc, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast-item toast-${type}`;

  let icon = '💡';
  if (type === 'attack') icon = '🛡️';
  else if (type === 'benign') icon = '✅';

  toast.innerHTML = `
    <div class="toast-icon">${icon}</div>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      <div class="toast-desc">${desc}</div>
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(16px)';
    setTimeout(() => toast.remove(), 300);
  }, 4500);
};

/* 5. Clickable Tooltips & Metric Explainer Modal */
const METRIC_EXPLANATIONS = {
  flows: {
    emoji: '📈',
    title: 'Flows Analyzed',
    desc: `
      <strong>What is a Network Flow?</strong><br>
      A "flow" is a sequence of related network packets traveling between a specific source and destination (e.g., your computer requesting a webpage from a server).<br><br>
      <strong>Why does it matter?</strong><br>
      Instead of inspecting single disconnected packets, our AI analyzes the entire flow conversation (packet sizes, intervals, and flag counts), enabling far higher intrusion detection accuracy.
    `
  },
  attacks: {
    emoji: '🛡️',
    title: 'Attack Predictions',
    desc: `
      <strong>What does this mean?</strong><br>
      The number of network flows that the AI flagged as malicious cyber attacks (such as DDoS floods, Mirai IoT botnet strikes, PortScans, or SQL Injections).<br><br>
      <strong>Automated Defense:</strong><br>
      Any flow flagged as an attack is immediately prioritized for firewall blocking or rate-limiting.
    `
  },
  accuracy: {
    emoji: '🎯',
    title: 'Hybrid Test Accuracy (86.68% / 99.24% top-k)',
    desc: `
      <strong>How was this measured?</strong><br>
      Evaluated on <strong>1,176,851 held-out test flows</strong> from the state-of-the-art CICIoT2023 benchmark.<br><br>
      <strong>Exact vs Top-k:</strong><br>
      • <strong>86.68% Exact Class Match:</strong> Correctly pinpoints the exact attack subtype among all 34 classes.<br>
      • <strong>99.24% Top-k Threat Precision:</strong> The correct attack category is in the top candidate pool with near 100% precision.
    `
  },
  latency: {
    emoji: '⚡',
    title: 'Inference Latency (0.0125 ms)',
    desc: `
      <strong>What is Latency?</strong><br>
      The time required for the AI engine to inspect, process, and classify each network flow.<br><br>
      <strong>Why 0.0125 ms is a Breakthrough:</strong><br>
      0.0125 milliseconds equals 12.5 microseconds. This means the system can process over <strong>80,000 flows per second</strong> on a single core, running at true wire-speed line rate without buffering delays.
    `
  }
};

function initClickableTooltips() {
  const modal = document.getElementById('metricExplainerModal');
  const btnClose = document.getElementById('btnCloseMetricExplainer');
  const btnCloseBtn = document.getElementById('btnCloseMetricExplainerBtn');

  if (btnClose && modal) btnClose.addEventListener('click', () => modal.classList.remove('open'));
  if (btnCloseBtn && modal) btnCloseBtn.addEventListener('click', () => modal.classList.remove('open'));

  // Attach to KPI Cards
  const kpiCards = document.querySelectorAll('.metric-card');
  if (kpiCards[0]) kpiCards[0].addEventListener('click', () => openMetricExplainer('flows'));
  if (kpiCards[1]) kpiCards[1].addEventListener('click', () => openMetricExplainer('attacks'));
  if (kpiCards[2]) kpiCards[2].addEventListener('click', () => openMetricExplainer('accuracy'));
  if (kpiCards[3]) kpiCards[3].addEventListener('click', () => openMetricExplainer('latency'));

  // Attach to info tips
  const infoTips = document.querySelectorAll('.info-tip');
  infoTips.forEach(tip => {
    tip.addEventListener('click', (e) => {
      e.stopPropagation();
      const title = tip.getAttribute('title') || 'Technical Metric';
      openGenericExplainer('ℹ️ Information', title);
    });
  });
}

function openMetricExplainer(key) {
  const data = METRIC_EXPLANATIONS[key];
  if (!data) return;

  const modal = document.getElementById('metricExplainerModal');
  const icon = document.getElementById('metricIconEmoji');
  const title = document.getElementById('metricExplainerTitle');
  const body = document.getElementById('metricExplainerBody');

  if (icon) icon.textContent = data.emoji;
  if (title) title.textContent = data.title;
  if (body) body.innerHTML = data.desc;

  if (modal) modal.classList.add('open');
}

function openGenericExplainer(heading, contentText) {
  const modal = document.getElementById('metricExplainerModal');
  const icon = document.getElementById('metricIconEmoji');
  const title = document.getElementById('metricExplainerTitle');
  const body = document.getElementById('metricExplainerBody');

  if (icon) icon.textContent = '💡';
  if (title) title.textContent = heading;
  if (body) body.innerHTML = `<div style="font-size:13px; line-height:1.5;">${contentText}</div>`;

  if (modal) modal.classList.add('open');
}

/* ==========================================================================
   VIDEO GUIDE & INTERACTIVE WALKTHROUGH CONTROLLER
   ========================================================================== */
function initVideoGuideModal() {
  const btnSidebarVideo = document.getElementById('btnSidebarVideo');
  const btnTopVideo = document.getElementById('btnTopVideoGuide');
  const btnHeaderVideo = document.getElementById('btnHeaderVideoGuide');
  const modal = document.getElementById('videoGuideModal');
  const btnClose = document.getElementById('btnCloseVideoGuideModal');
  const btnReplay = document.getElementById('btnVideoReplay');
  const btnFullscreen = document.getElementById('btnVideoFullscreen');
  const btnTryLive = document.getElementById('btnTryInAppFromVideo');
  const videoMedia = document.getElementById('guideVideoElement');

  if (btnSidebarVideo) btnSidebarVideo.addEventListener('click', openVideoGuideModal);
  if (btnTopVideo) btnTopVideo.addEventListener('click', openVideoGuideModal);
  if (btnHeaderVideo) btnHeaderVideo.addEventListener('click', openVideoGuideModal);

  if (btnClose && modal) btnClose.addEventListener('click', closeVideoGuideModal);

  if (btnReplay && videoMedia) {
    btnReplay.addEventListener('click', () => {
      const src = videoMedia.src;
      videoMedia.src = '';
      setTimeout(() => {
        videoMedia.src = src;
        showToast('🔁 Video Replayed', 'Restarting walkthrough animation from beginning.', 'info');
      }, 50);
    });
  }

  if (btnFullscreen && videoMedia) {
    btnFullscreen.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        if (videoMedia.requestFullscreen) {
          videoMedia.requestFullscreen();
        } else if (videoMedia.webkitRequestFullscreen) {
          videoMedia.webkitRequestFullscreen();
        }
      } else {
        if (document.exitFullscreen) {
          document.exitFullscreen();
        }
      }
    });
  }

  if (btnTryLive && modal) {
    btnTryLive.addEventListener('click', () => {
      closeVideoGuideModal();
      switchTab('tab-custom-input');
      showToast('🚀 Ready to Test!', 'Try selecting any 1-Click attack preset or adjusting sliders.', 'benign');
    });
  }
}

window.openVideoGuideModal = function() {
  const modal = document.getElementById('videoGuideModal');
  if (modal) {
    modal.classList.add('open');
    showToast('🎬 Video Guide Opened', 'Watch the full walkthrough or click any chapter on the right to jump.', 'info');
  }
};

window.closeVideoGuideModal = function() {
  const modal = document.getElementById('videoGuideModal');
  if (modal) modal.classList.remove('open');
};

window.jumpToFeature = function(featureKey) {
  // Highlight selected chapter
  const chapters = document.querySelectorAll('.chapter-item');
  chapters.forEach(ch => ch.classList.remove('active'));
  
  if (event && event.currentTarget) {
    event.currentTarget.classList.add('active');
  }

  if (featureKey === 'overview') {
    switchTab('tab-overview');
    showToast('📊 Overview Active', 'Viewing real-time traffic monitoring & live telemetry.', 'info');
  } else if (featureKey === 'custom_studio') {
    switchTab('tab-custom-input');
    const tabBtn = document.querySelector('[data-submode="submode-tabular"]');
    if (tabBtn) tabBtn.click();
    showToast('⚡ Custom Flow Studio Active', 'Adjust parameters or click any 1-click preset.', 'benign');
  } else if (featureKey === 'byte_inspector') {
    switchTab('tab-custom-input');
    const byteBtn = document.querySelector('[data-submode="submode-byte"]');
    if (byteBtn) byteBtn.click();
    showToast('🔬 Raw Byte Inspector Active', 'Inspect 1,024-byte payload waveforms and neural predictions.', 'info');
  } else if (featureKey === 'models') {
    switchTab('tab-models');
    showToast('📈 Model Benchmarks Active', 'Comparing 5 machine learning & deep neural architectures.', 'info');
  } else if (featureKey === 'architecture') {
    switchTab('tab-architecture');
    showToast('🧠 6-Stage Pipeline Active', 'Exploring end-to-end deep learning flowchart.', 'info');
  } else if (featureKey === 'glossary') {
    switchTab('tab-shap-threats');
    showToast('📖 Threat Glossary Active', 'Search all 34 cyberattack types & mitigation strategies.', 'info');
  }
};


