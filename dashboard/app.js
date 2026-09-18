'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = n => Number(n ?? 0).toLocaleString();
const pct = n => Number.isFinite(n) ? `${(n*100).toFixed(2)}%` : '—';
const STATE = {events:[], batches:[], total:0, attacks:0, reviews:0, filter:'all', streaming:false, busy:false, timer:null, generation:0, presets:[], csv:null, csvResults:null, stats:null, training:null};
const pageNames = {overview:'Overview',analyzer:'Traffic analyzer',performance:'Model performance',explain:'Explainability',deployment:'Deployment'};

async function api(path, data, timeout=120000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, {signal:controller.signal,...(data ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)} : {})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
    return result;
  } catch (error) { if (error.name === 'AbortError') throw new Error('The request timed out. Try again after the current operation completes.'); throw error; }
  finally {clearTimeout(timer);}
}
function showError(error) { $('notice').textContent=error.message || String(error);$('notice').hidden=false; }
function clearError() { $('notice').hidden=true; }
function toast(message) { $('toast').textContent=message;$('toast').hidden=false;clearTimeout(STATE.toastTimer);STATE.toastTimer=setTimeout(()=>$('toast').hidden=true,4500); }
async function busy(button, label, work) {
  if (button.disabled) return;
  const original=button.innerHTML;button.disabled=true;button.textContent=label;clearError();
  try {await work();} catch(e){showError(e);} finally {button.disabled=false;button.innerHTML=original;}
}
function go(page) {
  if (!pageNames[page]) page='overview';
  document.querySelectorAll('.page').forEach(el=>el.classList.toggle('active',el.id===page));
  document.querySelectorAll('.nav-link').forEach(el=>{el.classList.toggle('active',el.dataset.page===page);el.setAttribute('aria-current',el.dataset.page===page?'page':'false');el.title=pageNames[el.dataset.page];el.setAttribute('aria-label',pageNames[el.dataset.page]);});
  $('breadcrumb').textContent=pageNames[page];history.replaceState(null,'','#'+page);window.scrollTo({top:0,behavior:'instant'});
  if(page==='performance') loadPerformance().catch(showError);
  if(page==='deployment') loadCapture().catch(showError);
}
function download(name, data, type='application/json') {
  const url=URL.createObjectURL(new Blob([data],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function loadStats() {
  try {
    const s=await api('/api/stats');STATE.stats=s;
    $('connectionText').textContent=s.model_ready?'Model connected':'Model unavailable';
    $('connectionDot').style.background=s.model_ready?'#4d80dc':'#c39364';
    $('heroStatus').textContent=s.model_ready?'Inference engine ready':'Inference engine unavailable';
    $('accuracy').textContent=pct(s.accuracy);$('testSampleCount').textContent=`${fmt(s.test_samples)} held-out test flows`;
    renderReadiness(s);
    if(!s.model_ready)showError(new Error(s.errors?.cnn_bilstm || 'The hybrid checkpoint is unavailable.'));
  }catch(e){$('connectionText').textContent='Disconnected';$('heroStatus').textContent='Connection unavailable';$('connectionDot').style.background='#c39364';showError(e);}
}
async function runBatch() {
  if(STATE.busy)return;
  const generation=STATE.generation;STATE.busy=true;$('burst').disabled=true;
  try {
    const data=await api(`/api/replay?count=10&model=${encodeURIComponent($('streamModel').value)}`);
    if(generation!==STATE.generation)return;
    const attacks=data.events.filter(e=>e.is_attack).length;
    STATE.total+=data.events.length;STATE.attacks+=attacks;STATE.reviews+=data.events.filter(e=>e.severity==='REVIEW').length;
    STATE.events=[...data.events.reverse(),...STATE.events].slice(0,500);
    STATE.batches.push({benign:data.events.length-attacks,attack:attacks});STATE.batches=STATE.batches.slice(-20);
    $('flowCount').textContent=fmt(STATE.total);$('attackCount').textContent=fmt(STATE.attacks);
    $('attackRatio').textContent=`${(STATE.attacks/STATE.total*100).toFixed(1)}% of analyzed flows`;
    const mean=data.events.reduce((n,e)=>n+e.latency_ms,0)/data.events.length;
    $('latency').innerHTML=`${mean.toFixed(1)}<small>ms</small>`;
    renderEvents();renderTraffic();
  }catch(e){stopReplay();showError(e);}finally{STATE.busy=false;$('burst').disabled=false;}
}
function stopReplay() {STATE.streaming=false;clearTimeout(STATE.timer);$('streamToggle').innerHTML='<svg><use href="#i-play"/></svg>Start replay';}
async function replayLoop() {await runBatch();if(STATE.streaming)STATE.timer=setTimeout(replayLoop,2000);}
function renderEvents() {
  const search=$('eventSearch').value.trim().toLowerCase();
  const rows=STATE.events.filter(e=>(STATE.filter==='all'||(STATE.filter==='attack'?e.is_attack:!e.is_attack))&&`${e.predicted_label} ${e.true_label} ${e.event_id}`.toLowerCase().includes(search));
  $('eventsBody').innerHTML=rows.slice(0,30).map(e=>`<tr><td><strong>#${esc(e.event_id)}</strong><small>${esc(e.timestamp)} · ${esc(e.model==='cnn_bilstm'?'Hybrid':'XGBoost')}</small></td><td class="label-name"><strong>${esc(e.predicted_label)}</strong></td><td class="label-name">${esc(e.true_label)}</td><td><div class="confidence"><span>${e.confidence.toFixed(1)}%</span><span class="confidence-track"><span style="width:${e.confidence}%"></span></span></div></td><td><span class="status ${e.severity==='REVIEW'?'review':e.is_attack?'alert':''}">${esc(e.severity==='REVIEW'?'Review':e.is_attack?'Attack':'Benign')}</span></td><td><span class="${e.is_correct?'match':'mismatch'}">${e.is_correct?'✓ Match':'≠ Mismatch'}</span></td></tr>`).join('')||'<tr><td colspan="6" class="empty-cell">No matching flows. Run a replay or adjust your filter.</td></tr>';
  $('eventCount').textContent=fmt(STATE.total);$('eventsSummary').textContent=`Showing ${Math.min(30,rows.length)} of ${fmt(rows.length)} matching flows · latest 500 retained`;
}
function renderTraffic() {
  $('donutCount').textContent=fmt(STATE.total);$('benignTotal').textContent=fmt(STATE.total-STATE.attacks);$('attackTotal').textContent=fmt(STATE.attacks);$('reviewTotal').textContent=fmt(STATE.reviews);
  const ratio=STATE.total? (STATE.total-STATE.attacks)/STATE.total*100:0;
  $('donut').style.background=STATE.total?`conic-gradient(#7297e4 0 ${ratio}%, #b59ad5 ${ratio}% 100%)`:'#edf1f7';
  $('batchCount').textContent=`${STATE.batches.length} recent batches`;
  if(!STATE.batches.length){$('activityChart').innerHTML='<div class="chart-empty">Start replay to see traffic activity</div>';return;}
  const W=580,H=155,left=25,bottom=128;const n=Math.max(STATE.batches.length,10);const step=(W-left-15)/n;
  let svg='';for(let t=0;t<=10;t+=5){const y=bottom-t*10;svg+=`<line x1="25" y1="${y}" x2="570" y2="${y}" stroke="#e8edf5" stroke-width="1"/><text x="2" y="${y+3}" fill="#889bb4" font-size="8">${t}</text>`;}
  STATE.batches.forEach((b,i)=>{const x=left+i*step+step*.25,w=Math.min(18,step*.35);svg+=`<rect x="${x}" y="${bottom-b.benign*10}" width="${w}" height="${b.benign*10}" rx="2" fill="#7399e2"/><rect x="${x+w+3}" y="${bottom-b.attack*10}" width="${w}" height="${b.attack*10}" rx="2" fill="#b19bd7"/><text x="${x+w}" y="148" fill="#8196b3" text-anchor="middle" font-size="8">${i+1}</text>`;});
  $('activityChart').innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Benign and attack prediction counts by batch">${svg}</svg>`;
}
function resetSession() {stopReplay();STATE.generation++;STATE.events=[];STATE.batches=[];STATE.total=STATE.attacks=STATE.reviews=0;$('flowCount').textContent=$('attackCount').textContent='0';$('attackRatio').textContent='Awaiting traffic';$('latency').textContent='—';renderEvents();renderTraffic();}
async function loadPresets() {
  STATE.presets=await api('/api/presets');
  for(const [id,first] of [['presetSelect','Choose an example'],['shapPreset','Current analyzer input']]){
    $(id).innerHTML=`<option value="">${first}</option>`+STATE.presets.map((p,i)=>`<option value="${i}">${esc(p.name)}</option>`).join('');
  }
}
function flowInput() {let data;try{data=JSON.parse($('flowJson').value);}catch{throw new Error('Choose an example or enter valid feature JSON.');}if(!data||Array.isArray(data)||typeof data!=='object')throw new Error('Features must be a JSON object.');return data;}
function renderVerdict(r) {
  $('verdict').className='verdict-result';$('verdict').innerHTML=`<span class="status ${r.is_attack?'alert':''}">${r.is_attack?'Attack predicted':'Benign predicted'}</span><h3>${esc(r.predicted_label)}</h3><p>${r.confidence.toFixed(2)}% confidence · ${r.latency_ms.toFixed(1)} ms · ${esc(r.model)}</p>${r.top_classes.map(c=>`<div class="prob-row"><span>${esc(c.label)}</span><strong>${c.prob.toFixed(2)}%</strong></div>`).join('')}<p class="helper">${r.severity==='REVIEW'?'Confidence is below 60%. Review the input and prediction.':'Model probabilities are not calibrated security risk scores.'}</p>`;
}
async function loadPerformance() {
  const [models,training]=await Promise.all([api('/api/models'),api('/api/training')]);STATE.training=training;
  $('modelRows').innerHTML=models.comparison.map(r=>`<tr><td><strong>${esc(r.name)}</strong></td><td>${pct(r.metrics.accuracy)}</td><td>${pct(r.metrics.macro_precision)}</td><td>${pct(r.metrics.macro_recall)}</td><td>${pct(r.metrics.macro_f1)}</td><td>${fmt(r.metrics.sample_count ?? r.metrics.samples_evaluated ?? r.metrics.total_samples)}</td><td><span class="status">${r.source==='verified_delivery'?'New training run':'Saved research'}</span></td></tr>`).join('');
  const s=training.status;const complete=s.state==='complete';const epoch=s.epochs_completed||s.epoch||0;const max=s.max_epochs||epoch;
  $('trainingState').textContent=s.state||'No new run';$('trainingState').className='pill'+(complete?' green':'');
  $('trainingDetail').textContent=complete?`Completed ${epoch} epochs · best checkpoint: epoch ${s.best_epoch}`:`${s.state||'No new training run'}${epoch?` · epoch ${epoch} of ${max}`:''}`;
  $('trainingProgress').style.width=`${complete?100:max?epoch/max*100:0}%`;
  $('trainingFacts').innerHTML=`<span><strong>${fmt(s.training_samples||s.train_samples)}</strong> training flows</span><span><strong>${fmt(s.validation_samples)}</strong> validation flows</span><span><strong>${s.parameter_count?fmt(s.parameter_count):'—'}</strong> parameters</span><span>Representation: <strong>46 flow features</strong></span>`;
  if(s.error)$('trainingDetail').textContent=s.error;
  drawLearning(training.history);
  const hybrid=models.comparison.find(r=>r.key==='cnn_bilstm');const m=hybrid?.metrics||{};
  if(m.binary_confusion_matrix){const cm=m.binary_confusion_matrix;$('binaryMatrix').innerHTML=`<div class="matrix"><span></span><span>Pred. benign</span><span>Pred. attack</span><span>Actual benign</span><div class="matrix-cell">${fmt(cm[0][0])}</div><div class="matrix-cell error">${fmt(cm[0][1])}</div><span>Actual attack</span><div class="matrix-cell error">${fmt(cm[1][0])}</div><div class="matrix-cell">${fmt(cm[1][1])}</div></div><p class="helper">Attack precision ${pct(m.binary_precision)} · recall ${pct(m.binary_recall)}</p>`;}
  else $('binaryMatrix').innerHTML='<p class="helper">The confusion matrix will appear when the full test evaluation completes.</p>';
  $('classRows').innerHTML=Object.entries(m.classwise||{}).filter(([,v])=>typeof v==='object'&&'precision'in v).map(([name,v])=>`<tr><td><strong>${esc(name)}</strong></td><td>${pct(v.precision)}</td><td>${pct(v.recall)}</td><td>${pct(v['f1-score'])}</td><td>${fmt(v.support)}</td></tr>`).join('')||'<tr><td colspan="5" class="empty-cell">Classwise evaluation is not yet available for this training run.</td></tr>';
}
function drawLearning(rows) {
  if(!rows.length){$('learningChart').innerHTML='<p class="helper">Training curves will appear after the first epoch.</p>';return;}
  const x=i=>35+i*440/Math.max(1,rows.length-1),y=v=>185-v*155;
  const path=key=>rows.map((r,i)=>`${i?'L':'M'}${x(i)},${y(r[key])}`).join(' ');
  let grid='';for(let i=0;i<=4;i++){const v=i/4;grid+=`<line x1="35" y1="${y(v)}" x2="480" y2="${y(v)}" stroke="#e8edf6"/><text x="0" y="${y(v)+3}" font-size="9" fill="#7d92af">${v*100}%</text>`;}
  $('learningChart').innerHTML=`<svg viewBox="0 0 500 220" role="img" aria-label="Training and validation accuracy by epoch">${grid}<path d="${path('accuracy')}" stroke="#5987d9" stroke-width="2" fill="none"/><path d="${path('val_accuracy')}" stroke="#b393ce" stroke-width="2" fill="none"/><text x="35" y="210" font-size="9" fill="#7d92af">Epoch 1</text><text x="430" y="210" font-size="9" fill="#7d92af">Epoch ${rows.length}</text></svg><div class="legend"><span><i class="green-bg"></i>Training accuracy</span><span><i class="orange-bg"></i>Validation accuracy</span></div><p class="helper">Loss and accuracy values are included in the downloadable training history.</p>`;
}
async function explain() {
  const v=$('shapPreset').value;const features=v!==''?STATE.presets[Number(v)].features:flowInput();
  const r=await api('/api/explain',{model:$('shapModel').value,features},180000);
  $('shapSubtitle').textContent=`Predicted class: ${r.label}`;$('shapResult').className='';
  const max=Math.max(...r.features.map(f=>Math.abs(f.contribution)),.00001);
  $('shapResult').innerHTML=`<div class="shap-summary"><span>Background probability<strong>${pct(r.base_value)}</strong></span><span>Prediction probability<strong>${pct(r.probability)}</strong></span></div>${r.features.slice(0,12).map(f=>`<div class="shap-feature"><div class="shap-feature-label"><span>${esc(f.feature)}</span><strong>${f.contribution>=0?'+':''}${(f.contribution*100).toFixed(2)} pp</strong></div><div class="shap-bar ${f.contribution<0?'negative':''}" style="width:${Math.abs(f.contribution)/max*100}%"></div></div>`).join('')}<p class="helper">${esc(r.method)}. Top 12 of ${r.features.length} features. Additivity residual: ${r.additivity_residual.toExponential(2)}.</p>`;
}
function renderReadiness(s) {
  const ready=[['Flow inference',s.model_ready,'Loaded CNN-BiLSTM checkpoint'],['Dataset',true,`${fmt(s.test_samples)} held-out flows · ${s.class_count} classes`],['Packet-byte model',s.trained_packet_model,s.trained_packet_model?'Trained payload checkpoint available':'Labeled PCAP training required'],['Kafka streaming',false,'Adapter available; broker connection is configured in the CLI']];
  $('readiness').innerHTML=ready.map(([name,ok,desc])=>`<div class="readiness-item"><strong>${name}</strong><span class="status ${ok?'':'review'}">${ok?'Ready':name==='Kafka streaming'?'Not connected':'Needs data'}</span><p>${esc(desc)}</p></div>`).join('');
}
async function loadCapture() {
  const r=await api('/api/capture');
  const previous=$('interface').value;$('interface').innerHTML='<option value="">Default interface</option>'+r.interfaces.map(i=>`<option value="${esc(i)}">${esc(i)}</option>`).join('');$('interface').value=previous;
  $('captureState').textContent=r.running?'Capturing':r.error?'Capture error':'Idle';$('packetCount').textContent=`${fmt(r.packet_count)} packets`;
  $('captureMessage').textContent=r.error||`${r.detection_ready?'Payload model available.':'Observation mode: a trained byte model is required for packet predictions.'} ${r.limit}.`;
  $('startCapture').disabled=r.running;$('stopCapture').disabled=!r.running;
  $('packetRows').innerHTML=r.events.slice(0,30).map(e=>`<tr><td>${esc(e.timestamp)}</td><td>${esc(e.src)}</td><td>${esc(e.dst)}</td><td>${esc(e.protocol)}</td><td>${fmt(e.length)}</td><td>${esc(e.predicted_label||e.error||'Observed only')}</td></tr>`).join('')||'<tr><td colspan="6" class="empty-cell">No packets captured in this session.</td></tr>';
}
function bind() {
  document.querySelectorAll('[data-page]').forEach(el=>el.addEventListener('click',()=>go(el.dataset.page)));
  document.querySelectorAll('[data-go]').forEach(el=>el.addEventListener('click',()=>go(el.dataset.go)));
  document.querySelector('.brand').addEventListener('click',e=>{e.preventDefault();go('overview');});
  $('burst').addEventListener('click',()=>{clearError();runBatch();});
  $('streamToggle').addEventListener('click',()=>{if(STATE.streaming){stopReplay();return;}clearError();STATE.streaming=true;$('streamToggle').textContent='Pause replay';replayLoop();});
  $('streamModel').addEventListener('change',resetSession);$('clearEvents').addEventListener('click',resetSession);
  $('eventSearch').addEventListener('input',renderEvents);
  document.querySelectorAll('[data-filter]').forEach(el=>el.addEventListener('click',()=>{STATE.filter=el.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>{b.classList.toggle('selected',b===el);b.setAttribute('aria-pressed',b===el?'true':'false');});renderEvents();}));
  $('exportEvents').addEventListener('click',()=>{if(!STATE.events.length){toast('Run a replay to create events for export.');return;}download('ids-detection-events.json',JSON.stringify({source:'held_out_replay',session_total:STATE.total,retained_events:STATE.events},null,2));});
  $('presetSelect').addEventListener('change',e=>{if(e.target.value!=='')$('flowJson').value=JSON.stringify(STATE.presets[Number(e.target.value)].features,null,2);});
  $('classify').addEventListener('click',()=>busy($('classify'),'Analyzing…',async()=>renderVerdict(await api('/api/classify',{model:$('inspectModel').value,features:flowInput()}))));
  $('explainCurrent').addEventListener('click',()=>{try{flowInput();$('shapPreset').value='';$('shapModel').value=['cnn_bilstm','xgboost'].includes($('inspectModel').value)?$('inspectModel').value:'cnn_bilstm';go('explain');}catch(e){showError(e);}});
  $('runShap').addEventListener('click',()=>busy($('runShap'),'Computing SHAP…',explain));
  $('csvUpload').addEventListener('change',async e=>{STATE.csv=null;STATE.csvResults=null;$('analyzeCsv').disabled=true;$('exportCsv').hidden=true;$('csvResult').textContent='';const file=e.target.files[0];if(!file)return;if(file.size>3.8*1024*1024){showError(new Error('Choose a CSV smaller than 3.8 MB.'));return;}STATE.csv=await file.text();$('csvFilename').textContent=file.name;$('analyzeCsv').disabled=false;});
  $('analyzeCsv').addEventListener('click',()=>busy($('analyzeCsv'),'Analyzing…',async()=>{const r=await api('/api/analyze-csv',{csv:STATE.csv,model:$('inspectModel').value});STATE.csvResults=r;$('csvResult').textContent=`${r.rows} flows analyzed · ${r.attacks} predicted attacks · ${r.rows-r.attacks} benign`;$('exportCsv').hidden=false;}));
  $('exportCsv').addEventListener('click',()=>download('ids-batch-predictions.json',JSON.stringify(STATE.csvResults,null,2)));
  $('refreshDeployment').addEventListener('click',()=>busy($('refreshDeployment'),'Refreshing…',async()=>{await loadStats();await loadCapture();}));
  $('startCapture').addEventListener('click',()=>busy($('startCapture'),'Starting…',async()=>{await api('/api/capture/start',{interface:$('interface').value});await loadCapture();}));
  $('stopCapture').addEventListener('click',()=>busy($('stopCapture'),'Stopping…',async()=>{await api('/api/capture/stop',{});await loadCapture();}));
}
async function init() {
  bind();go(location.hash.slice(1)||'overview');
  const clock=()=>{$('clock').textContent=new Intl.DateTimeFormat(undefined,{day:'2-digit',month:'short',year:'numeric'}).format(new Date());};clock();
  await Promise.allSettled([loadStats(),loadPresets().catch(showError)]);
  setInterval(()=>{if(!document.hidden&&$('deployment').classList.contains('active'))loadCapture().catch(showError);},3000);
  setInterval(()=>{if(!document.hidden){loadStats();if($('performance').classList.contains('active'))loadPerformance().catch(showError);}},15000);
}
init();
