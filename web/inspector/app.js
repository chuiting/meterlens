// Meter Lens Inspector: four hash-routed tabs backed by /api/inspector/*.

const API = '/api/inspector';
const $ = (sel, root = document) => root.querySelector(sel);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = x => Math.round(x * 100);
const tier = p => p >= 90 ? 'green' : p >= 75 ? 'amber' : 'red';

async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  return res.json();
}

// routing

const loaded = {};
function route() {
  const [tab, query] = (location.hash.slice(1) || 'capture').split('?');
  const params = new URLSearchParams(query);
  document.querySelectorAll('main[data-tab]').forEach(m => { m.hidden = m.dataset.tab !== tab; });
  document.querySelectorAll('.nav').forEach(a => a.classList.toggle('active', a.dataset.tab === tab));
  if (tab === 'capture' && params.get('image')) return loadStoredTrace(params.get('image'));
  if (tab === 'capture' && params.get('dataset')) return loadDatasetTrace(params.get('dataset'));
  if (tab === 'history') return loadHistory();
  if (!loaded[tab] && tabs[tab]) { loaded[tab] = true; tabs[tab](); }
}
window.addEventListener('hashchange', route);

// capture

$('#reset').onclick = () => { location.hash = '#capture'; location.reload(); };
const fileInput = $('#file');
fileInput.addEventListener('change', () => { if (fileInput.files[0]) traceUpload(fileInput.files[0]); fileInput.value = ''; });

const dropZone = $('#capture');
dropZone.addEventListener('dragover', e => { e.preventDefault(); $('#capture-empty').classList.add('drag'); });
dropZone.addEventListener('dragleave', () => $('#capture-empty').classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); $('#capture-empty').classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) traceUpload(f);
});

// Sample photos (web/samples/).
async function loadSamples() {
  const box = $('#samples');
  let list;
  try { list = await (await fetch('/samples/samples.json')).json(); } catch (e) { return; }
  box.innerHTML = list.map(s => `<button class="sample ${s.kind}" data-file="${s.file}" title="${esc(s.expect)}">
    <img src="/samples/${s.thumb}" alt=""><span class="lbl">${esc(s.label)}</span><span class="badge ${s.kind}">${s.kind === 'good' ? 'good photo' : 'tricky photo'}</span></button>`).join('');
  box.querySelectorAll('.sample').forEach(b => b.onclick = async () => {
    const blob = await (await fetch('/samples/' + b.dataset.file)).blob();
    traceUpload(new File([blob], b.dataset.file, { type: 'image/jpeg' }));
  });
}
loadSamples();

async function traceUpload(file) {
  const fd = new FormData(); fd.append('file', file, file.name || 'photo.jpg');
  await runTrace(() => api('/trace', { method: 'POST', body: fd }), file.name);
}
async function loadStoredTrace(imageId) {
  await runTrace(() => api('/trace/' + imageId), imageId + '.jpg');
}
async function loadDatasetTrace(key) {
  await runTrace(() => api('/trace/dataset/' + key), key.split('/')[1] + '.jpg');
}
async function runTrace(fetcher, name) {
  const btn = $('label.btn'), body = $('#capture-body'), empty = $('#capture-empty');
  btn.classList.add('busy'); $('#loading').hidden = false;
  try {
    const t = await fetcher();
    empty.hidden = true; body.hidden = false; $('#reset').hidden = false;
    body.innerHTML = renderTrace(t, name);
  } catch (err) {
    empty.hidden = false; body.hidden = true;
    empty.innerHTML = `<div class="error">${esc(err.message)}</div>`;
  } finally { btn.classList.remove('busy'); $('#loading').hidden = true; }
}

function renderTrace(t, name) {
  if (!t.detected) {
    return `<div class="capture-left">${photoPanel(t)}</div>
      <div class="capture-right">${stage(1, 'Counter Detection - YOLOv8', t.timings_ms.detect, `<div class="error">No counter detected - nothing else can run.</div>`, true)}</div>`;
  }
  const f = t.fusion, d = t.decision, s = t.sensing, T = t.timings_ms;
  const flagged = d.status !== 'APPROVED';
  const minConf = pct(f.confidence);
  const left = `
    ${photoPanel(t)}
    <div class="card">
      <div class="card-head"><span class="card-label">Fused Reading · ${esc(name)}</span>
        <span class="decision ${flagged ? 'flagged' : 'approved'}">${flagged ? '⚑ Flagged' : '✓ Approved'}</span></div>
      <div style="display:flex; align-items:baseline; gap:10px"><div class="reading num">${esc(f.reading)}</div><span class="note">kWh</span>
        <span class="pill ${tier(minConf)}" style="margin-left:auto">min confidence ${(f.confidence * 100).toFixed(1)}%</span></div>
      <div class="chips">${f.reading.split('').map((ch, i) => { const p = pct(f.per_digit_confidence[i]); return `<div class="chip ${tier(p)}"><div class="ch">${ch}</div><div class="conf">${p}%</div></div>`; }).join('')}</div>
    </div>`;

  const det = t.detection, px = t.image_size;
  const bboxPx = [det.bbox.x * px.w, det.bbox.y * px.h, det.bbox.w * px.w, det.bbox.h * px.h].map(Math.round).join(' ');
  const cnnMembers = Object.entries(t.members).filter(([n]) => n.startsWith('cnn'));
  const hogMembers = Object.entries(t.members).filter(([n]) => !n.startsWith('cnn'));
  let n = 0;
  const right = [
    stage(++n, 'Counter Detection - YOLOv8', T.detect, `
      <div style="display:flex; gap:18px; align-items:center; flex-wrap:wrap">
        <div class="kv" style="flex:1; min-width:200px"><span class="k">confidence</span><span class="v mono">${det.confidence.toFixed(2)}</span><span class="k">bbox</span><span class="v mono">${bboxPx} <span>px</span></span></div>
        <div class="crops">
          <div><div class="card-label" style="margin-bottom:4px">Color Crop</div><img src="${t.crop.rgb}" alt=""></div>
          <div><div class="card-label" style="margin-bottom:4px">Grayscale Crop</div><img src="${t.crop.gray}" alt=""></div>
        </div>
      </div>`),
    stage(++n, 'Quality Gate - PCA + KMeans', T.sensing, gauge(s) + (s.histogram ? histogram(s) : ''), s.flag),
    stage(++n, 'Digit Segmentation', T.segment, `
      <div class="seg">
        <span class="card-label">Strip 80×106</span>${t.segmentation.strips.map(u => `<img src="${u}" alt="" style="height:96px">`).join('')}
        <span class="card-label">HOG Features</span>${t.segmentation.hog.map(u => `<img src="${u}" alt="" style="height:72px">`).join('')}
        <span></span>${[1, 2, 3, 4, 5].map(i => `<span class="card-label c">d${i}</span>`).join('')}
      </div>`),
    stage(++n, 'CNN Heads - 5 × Softmax', cnnMembers.reduce((a, [k]) => a + (T[k] || 0), 0),
      cnnMembers.map(([k, m]) => memberRow(k, m)).join('')),
    stage(++n, 'SVM + HOG', hogMembers.reduce((a, [k]) => a + (T[k] || 0), 0),
      hogMembers.map(([k, m]) => memberRow(k, m)).join('')),
    stage(++n, 'Confidence-Weighted Fusion', T.fusion, fusionTable(t)),
    stage(++n, 'Decision', T.decision, `
      <div class="checks">
        <div class="check"><div class="card-label">Sensing</div><div class="row"><span class="mono" style="font-weight:700">${s.distance == null ? 'n/a' : Math.round(s.distance) + (s.flag ? ' > ' : ' < ') + Math.round(s.threshold)}</span><span class="pill ${s.flag ? 'red' : 'green'}">${s.flag ? 'flagged' : 'ok'}</span></div></div>
        <div class="check"><div class="card-label">Confidence</div><div class="row"><span class="mono" style="font-weight:700">${f.confidence.toFixed(3)} ${f.confidence >= d.threshold ? '≥' : '<'} ${d.threshold.toFixed(3)}</span><span class="pill ${f.confidence >= d.threshold ? 'green' : 'red'}">${f.confidence >= d.threshold ? 'ok' : 'low'}</span></div></div>
        <div class="check ${flagged ? 'bad' : 'ok'}"><div class="card-label">Result</div><div class="row"><span style="font-weight:800">${esc(d.status)}</span>${d.reasons.length ? `<span class="mono" style="font-size:11px">${d.reasons.map(r => esc(r.kind[0].toUpperCase() + r.kind.slice(1))).join(' + ')}</span>` : ''}</div></div>
      </div>`, flagged),
  ].join('');
  return `<div class="capture-left">${left}</div><div class="capture-right">${right}</div>`;
}

function stage(n, title, ms, body, bad = false) {
  return `<div class="stage"><div class="n ${bad ? 'bad' : ''}">${n}</div><div class="card">
    <div class="card-head"><div class="t">${title}</div></div>${body}</div></div>`;
}

function photoPanel(t) {
  const b = t.detection && t.detection.bbox;
  const box = (o, cls) => `<div class="box ${cls}" style="left:${o.x * 100}%; top:${o.y * 100}%; width:${o.w * 100}%; height:${o.h * 100}%"></div>`;
  return `<div class="photo"><div class="frame"><img src="${t.image_url}" alt="">
    ${b ? box(b, '') + `<div class="tag" style="left:${b.x * 100}%; top:calc(${b.y * 100}% - 22px)">counter ${t.detection.confidence.toFixed(2)}</div>` : ''}
    ${b ? t.detection.digit_bboxes.map(o => box(o, 'digit')).join('') : ''}
    </div>${b ? `<div class="mini"><span class="tag">crop ${t.crop.size[0]}×${t.crop.size[1]}</span><img src="${t.crop.rgb}" alt=""></div>` : ''}</div>`;
}

function gauge(s) {
  if (s.distance == null) return `<div class="note">sensing gate not fitted</div>`;
  const pos = Math.min(100, s.distance / s.threshold * 100);
  return `<div class="gauge">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px"><span class="card-label">Distance from Typical</span><span class="pill ${s.flag ? 'red' : 'green'}">${s.flag ? 'Flagged' : 'Passed'}</span></div>
    <div class="track"><div class="dot ${s.flag ? 'bad' : ''}" style="left:${pos}%"></div><div class="thr"></div></div>
    <div style="display:flex; justify-content:space-between; margin-top:8px" class="mono note"><span>typical</span><span style="color:var(--red)">threshold</span></div>
    <div class="mono note" style="margin-top:6px">${Math.round(s.distance)} / ${Math.round(s.threshold)}</div>
  </div>`;
}

// Validation-split distance histogram with this photo and the threshold marked.
function histogram(s) {
  const h = s.histogram, W = 420, H = 90, top = 6, base = 72, bw = 17, gap = 3;
  const max = Math.max(...h.counts), n = h.counts.length;
  const x = v => 8 + (v - h.min) / (h.max - h.min) * (n * (bw + gap));
  const bars = h.counts.map((c, i) => {
    const hh = Math.max(1, c / max * (base - top - 6)), left = 8 + i * (bw + gap);
    const edge = h.min + (i + 1) * (h.max - h.min) / n;
    return `<rect x="${left}" y="${(base - hh).toFixed(1)}" width="${bw}" height="${hh.toFixed(1)}" rx="2" fill="${edge > s.threshold ? '#F4B98A' : '#E9D9C8'}"/>`;
  }).join('');
  const tx = Math.min(x(s.threshold), W - 8), px = Math.min(x(s.distance), W - 8);
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="display:block; stroke:none; max-width:100%">${bars}
    <line x1="${tx}" y1="${top}" x2="${tx}" y2="${base + 2}" stroke="#B3452F" stroke-width="1.5" stroke-dasharray="3 3"/><text x="${tx + 4}" y="14" font-size="10" font-family="JetBrains Mono, monospace" font-weight="700" fill="#B3452F">threshold</text>
    <line x1="${px}" y1="${top}" x2="${px}" y2="${base + 2}" stroke="#D9661E" stroke-width="2"/><text x="${px > W - 70 ? px - 5 : px + 5}" y="30" text-anchor="${px > W - 70 ? 'end' : 'start'}" font-size="10" font-family="JetBrains Mono, monospace" font-weight="700" fill="#D9661E">this photo</text>
    <text x="8" y="86" font-size="9" fill="#8A7362" font-weight="600">Validation photos by distance from typical; past the threshold = flagged</text></svg>`;
}

function memberRow(name, m) {
  const boxes = m.probs.map((row, i) => {
    const top = row.indexOf(Math.max(...row));
    const second = row.map((v, j) => [v, j]).filter(([, j]) => j !== top).sort((a, b) => b[0] - a[0])[0][1];
    const bars = row.map((v, j) => {
      const cls = j === top ? 'top' + (v < 0.75 ? ' low' : '') : j === second ? 'second' : '';
      const label = j === top ? `<span class="pct mono">${pct(v)}%</span><span class="dg mono">${j}</span>` : '';
      return `<div class="${cls}" style="height:${Math.max(2, v * 56).toFixed(1)}px">${label}</div>`;
    }).join('');
    return `<div class="head-box"><span class="card-label">d${i + 1}</span><div class="bars">${bars}</div></div>`;
  }).join('');
  return `<div class="member-row"><div class="card-label" style="margin-bottom:6px">${esc(modelName(name))} → <span class="mono" style="color:var(--ink)">${esc(m.reading)}</span></div><div class="heads">${boxes}</div></div>`;
}

function fusionTable(t) {
  const names = Object.keys(t.members), f = t.fusion;
  const minIdx = f.per_digit_confidence.indexOf(Math.min(...f.per_digit_confidence));
  const head = names.map(n => `<th>${esc(modelName(n))} <span class="mono" style="text-transform:none; letter-spacing:0">w=${f.weights[n].toFixed(3)}</span></th>`).join('');
  const rows = [0, 1, 2, 3, 4].map(i => {
    const cells = names.map(n => {
      const m = t.members[n], dgt = m.reading[i], dis = dgt !== f.reading[i];
      return `<td><span class="mono dgt ${dis ? 'dis' : ''}">${dgt}</span> <span class="mono p">${m.per_digit_confidence[i].toFixed(2)}</span></td>`;
    }).join('');
    const c = f.per_digit_confidence[i];
    return `<tr class="fuse-row" data-d="${i}"><td class="card-label" style="font-weight:700"><span class="chev">▸</span>d${i + 1}</td>${cells}<td class="fused"><span class="mono dgt">${f.reading[i]}</span> <span class="mono p ${i === minIdx ? 'min' : ''}">${c.toFixed(3)}${i === minIdx ? ' ← min' : ''}</span></td></tr>`
      + `<tr class="fuse-probs" hidden><td colspan="${names.length + 2}">${probGrid(t, i)}</td></tr>`;
  }).join('');
  return `<table class="fuse"><tr><th>Position</th>${head}<th style="color:var(--accent-dark)">Fused</th></tr>${rows}</table>
    <div class="note" style="margin-top:8px"><span class="dgt dis" style="font-size:11px; padding:1px 5px">Red</span> = Differs from the fused digit</div>`;
}

// Every member's probability for all 10 digits at one position.
function probGrid(t, i) {
  const names = Object.keys(t.members), f = t.fusion, col = +f.reading[i];
  const line = (label, p, dp, cls = '', w = null) => {
    const top = p.indexOf(Math.max(...p));
    return `<tr class="${cls}"><td>${esc(label)}${w === null ? '' : ` <span class="mono pw">w=${w.toFixed(3)}</span>`}</td>${p.map((v, j) => `<td class="${j === col ? 'col' : ''}"><span class="${j === top ? 'pick' : ''}">${v.toFixed(dp)}</span></td>`).join('')}</tr>`;
  };
  const digits = [...Array(10).keys()].map(j => `<th class="${j === col ? 'col' : ''}">${j}</th>`).join('');
  const sum = names.map(n => `${f.weights[n].toFixed(3)}×${t.members[n].probs[i][col].toFixed(2)}`).join(' + ');
  return `<table class="pgrid"><tr><th>d${i + 1} probability</th>${digits}</tr>
    ${names.map(n => line(modelName(n), t.members[n].probs[i], 2, '', f.weights[n])).join('')}${line('Fused', f.probs[i], 3, 'fz')}</table>
    <div class="psum mono">Fused p(${col}) = ${sum} = <b>${f.probs[i][col].toFixed(3)}</b></div>`;
}

document.addEventListener('click', e => {
  const row = e.target.closest('.fuse tr.fuse-row');
  if (!row) return;
  const grid = row.nextElementSibling, open = grid.hidden;
  grid.hidden = !open;
  row.classList.toggle('open', open);
  row.querySelector('.chev').textContent = open ? '▾' : '▸';
});

// history

let historyRows = [], historyFilter = 'all';
async function loadHistory() {
  const body = $('#history-body');
  try {
    historyRows = await (await fetch('/history')).json();
  } catch (err) { body.innerHTML = `<div class="error">${esc(err.message)}</div>`; return; }
  renderHistory();
}
function renderHistory() {
  const body = $('#history-body');
  const rows = historyRows.filter(r => historyFilter === 'all' || (historyFilter === 'flagged' ? r.decision === 'FLAGGED' : r.feedback === 'corrected'));
  const filters = ['all', 'flagged', 'corrected'].map(f => `<span class="tab ${f === historyFilter ? 'on' : ''}" data-filter="${f}">${f[0].toUpperCase() + f.slice(1)}</span>`).join('');
  const tr = rows.map(r => {
    const conf = pct(r.confidence || 0), flagged = r.decision === 'FLAGGED';
    const reasons = r.reasons.map(x => `<span class="ctext amber">${esc(x.kind[0].toUpperCase() + x.kind.slice(1))}</span>`).join(', ') || '<span class="note">-</span>';
    const feedback = r.feedback === 'corrected' ? `<span style="color:var(--red)">Corrected → <b class="mono">${esc(r.corrected_reading)}</b></span>`
      : r.feedback === 'confirmed' ? '<span style="color:var(--green)">Confirmed correct</span>' : '<span class="note">-</span>';
    return `<tr>
      <td><div class="thumbs"><img src="${r.image_url}" alt="" class="th-photo" loading="lazy"><canvas class="th-crop" width="220" height="72" data-src="${r.image_url}" data-box='${JSON.stringify(r.counter_bbox)}'></canvas></div></td>
      <td class="mono" style="font-size:15px; font-weight:800; letter-spacing:1px">${esc(r.ai_reading || '-')}</td>
      <td><span class="ctext ${tier(conf)}">${(r.confidence * 100).toFixed(1)}%</span></td>
      <td><span class="decision ${flagged ? 'flagged' : 'approved'}">${flagged ? '⚑ Flagged' : '✓ Approved'}</span></td>
      <td>${reasons}</td><td>${feedback}</td>
      <td class="note">${fmtDate(r.created_at)}</td>
      <td><a href="#capture?image=${r.image_id}" style="font-weight:700; font-size:12px">Open Trace</a></td></tr>`;
  }).join('');
  body.innerHTML = `<div class="tabs">${filters}</div>
    <div class="card" style="padding:4px 6px"><table><tr><th>Uploaded Photo</th><th>Reading</th><th>Confidence</th><th>Decision</th><th>Reasons</th><th>Feedback</th><th>Saved At</th><th></th></tr>
    ${tr || '<tr><td colspan="8" class="note" style="text-align:center; padding:30px">No readings</td></tr>'}</table></div>`;
  body.querySelectorAll('.tab').forEach(t => t.onclick = () => { historyFilter = t.dataset.filter; renderHistory(); });
  body.querySelectorAll('canvas.th-crop').forEach(drawCrop);
}
// Counter crop drawn from the stored photo and its saved box (fractions of the image).
function drawCrop(canvas) {
  const b = JSON.parse(canvas.dataset.box); if (!b) return;
  const img = new Image();
  img.onload = () => {
    const w = img.naturalWidth, h = img.naturalHeight;
    canvas.getContext('2d').drawImage(img, b.x * w, b.y * h, b.w * w, b.h * h, 0, 0, canvas.width, canvas.height);
  };
  img.src = canvas.dataset.src;
}
function fmtDate(iso) {
  const d = new Date(iso), two = n => String(n).padStart(2, '0');
  return `${two(d.getDate())}-${two(d.getMonth() + 1)}-${d.getFullYear()} ${two(d.getHours())}:${two(d.getMinutes())}`;
}

// models

const f1 = x => (x * 100).toFixed(1);
// display names for model keys
const modelName = s => s.replace(/\b(cnn_rgb|cnn|svm|rf|fused)\b/g, k => ({ cnn_rgb: 'CNN-RGB', fused: 'Fused' })[k] || k.toUpperCase());
async function loadModels() {
  const body = $('#models-body');
  let m;
  try { m = await api('/models'); } catch (err) { body.innerHTML = `<div class="error">${esc(err.message)}</div>`; return; }
  const h = m.headline, delta = (h.full - h.baseline_full) * 100;
  const finding = t => `<div class="finding"><b>Finding:</b> ${t}</div>`;
  const fusedRow = m.rows.find(r => r.role === 'deployed');
  const e2eRow = m.rows[m.rows.length - 1];   // YOLO + fused, end to end
  const gtRows = m.rows.filter(r => r !== e2eRow);   // single models and fused, on ground-truth crops
  const bestSingle = gtRows.filter(r => r !== fusedRow).sort((a, b) => b.test_full - a.test_full)[0];
  // findings are derived from the current results
  const weakest = r => r.test_per_digit.indexOf(Math.min(...r.test_per_digit));
  const d5WeakestEverywhere = gtRows.every(r => weakest(r) === 4);
  const n = m.dataset, total = n.training + n.validation + n.testing;
  const stats = `<div><div class="stats-title">End to End Performance (YOLO to Recognition)</div><div class="stats">
    <div class="stat"><div class="card-label">Dataset</div><div class="big">${total.toLocaleString('en')} photos</div><div class="sub">${n.training} train / ${n.validation} validation / ${n.testing} test</div></div>
    <div class="stat"><div class="card-label">YOLO mAP@0.5</div><div class="big">${f1(m.detection.mAP50)}%</div><div class="sub">Counter box located</div></div>
    <div class="stat"><div class="card-label">Per-Digit Accuracy</div><div class="big">${f1(h.digits)}%</div><div class="sub">Single digits read correctly</div></div>
    <div class="stat"><div class="card-label">Full-Reading Accuracy</div><div class="big">${f1(h.full)}%</div><div class="sub">All 5 digits correct</div></div>
    <div class="stat"><div class="card-label">Auto-Approved Accuracy</div><div class="big" style="color:var(--green)">${f1(h.approved_accuracy)}%</div><div class="sub">Reading correct when system approves</div></div>
    <div class="stat"><div class="card-label">Gap to Published Baseline</div><div class="big" style="color:var(--red)">${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(1)} pt</div><div class="sub">Full reading vs Laroca et al.</div></div>
  </div></div>`;

  const rows = m.rows.map(r => {
    const fused = r.model.includes('fused');
    const role = r.role === 'deployed' ? '<span class="pill green">Deployed</span>'
      : r.role.startsWith('member') ? `<span class="pill soft">Weight=${esc(r.role.split(' w ')[1])}</span>`
      : r.role.startsWith('built') ? '<span class="pill amber">Built but not deployed</span>' : `<span class="note">${esc(r.role)}</span>`;
    return `<tr ${r.role === 'deployed' ? 'class="hl-green"' : fused ? 'class="hl"' : ''}><td style="font-weight:800">${esc(modelName(r.model))}</td><td class="r mono">${f1(r.test_digits)}%</td><td class="r mono">${f1(r.test_full)}%</td><td class="r mono">${f1(r.test_d5)}%</td><td>${role}</td></tr>`;
  }).join('');
  const table = `<div class="card"><div class="card-head"><div class="t">The Models Built</div></div>
    <table><tr><th>Model</th><th class="r">Test Per-Digit Accuracy</th><th class="r">Test Full-Reading Accuracy</th><th class="r">d5 (Last Digit)</th><th>Role</th></tr>${rows}</table>
    ${finding(`${fusedRow.test_full > bestSingle.test_full ? 'Fusion beats every single model' : 'Fusion does not beat the best single model'}: ${f1(fusedRow.test_full)}% full-reading accuracy vs ${f1(bestSingle.test_full)}% for ${modelName(bestSingle.model)}. Running it on YOLO's crops instead of ground-truth crops ${fusedRow.test_full >= h.full ? 'costs' : 'adds'} ${Math.abs((fusedRow.test_full - h.full) * 100).toFixed(1)} pt.`)}</div>`;

  const decision = `<div class="card"><div class="card-head"><div class="t">Decision Layer</div></div><img src="${m.figures.decision_flow}" alt="" class="fig">
    ${finding(`Sending ${f1(h.flag_rate)}% of readings to review catches ${f1(h.errors_caught)}% of the errors, so approved readings are ${f1(h.approved_accuracy)}% correct (vs ${f1(h.full)}% with no decision layer). ${h.errors_approved} wrong readings still get approved.`)}</div>`;

  const labels = { baseline: 'baseline (BN + dropout)', '+he_init': '+ He initialisation', '+l2_reg': '+ L2 on dense layers', '+lr_step_decay': '+ LR step decay',
    '+augmentation': '+ augmentation', '+digit_permutation': '+ digit permutation, gray, +2% crop', '+real_finetune': '+ real-data fine-tune' };
  const last = m.ablation.length - 1;
  const abIdx = step => m.ablation.findIndex(a => a.step === step);
  // val accuracy change across steps, in points
  const gain = (from, to = from) => (m.ablation[abIdx(to)].val_full_sequence_accuracy - m.ablation[abIdx(from) - 1].val_full_sequence_accuracy) * 100;
  const pts = d => (d < 0 ? '−' : '+') + Math.abs(d).toFixed(1);
  const steps = m.ablation.slice(1).map((a, i) => ({ step: a.step, d: (a.val_full_sequence_accuracy - m.ablation[i].val_full_sequence_accuracy) * 100 }));
  const biggest = steps.reduce((a, b) => (b.d > a.d ? b : a));
  const knownSteps = ['+he_init', '+lr_step_decay', '+augmentation', '+digit_permutation'].every(st => abIdx(st) > 0);
  const dataWins = knownSteps && gain('+augmentation') + gain('+digit_permutation') > gain('+he_init', '+lr_step_decay');
  const ablation = `<div class="card"><div class="card-head"><div class="t">CNN Ablation</div></div><img src="${m.figures.ablation}" alt="" class="fig" style="margin-bottom:12px">
    <div class="abl">${m.ablation.map((a, i) => `<span class="note">${esc(labels[a.step] || a.step)}</span><div class="track"><div style="width:${a.val_full_sequence_accuracy * 100}%; background:${i === last ? 'var(--accent)' : 'var(--bar2)'}"></div></div><span class="mono r">${f1(a.val_full_sequence_accuracy)}</span>`).join('')}</div>
    ${finding(dataWins
      ? `Data mattered more than training tweaks: augmentation (${pts(gain('+augmentation'))} pt) and digit-permutation synthesis (${pts(gain('+digit_permutation'))} pt) gave most of the gain, while He init, L2 and LR decay together moved it ${pts(gain('+he_init', '+lr_step_decay'))} pt.`
      : `Validation accuracy went from ${f1(m.ablation[0].val_full_sequence_accuracy)}% to ${f1(m.ablation[last].val_full_sequence_accuracy)}%; the biggest single gain came from ${esc(labels[biggest.step] || biggest.step)} (${pts(biggest.d)} pt).`)}</div>`;

  const baseline = `<div class="card"><div class="card-head"><div class="t">Comparison with Published Baseline</div></div>
    <table><tr><th>Approach</th><th class="r">Digit Accuracy</th><th class="r">Full Reading Accuracy</th></tr>
    ${m.baseline.rows.map((r, i) => `<tr ${i === 0 ? 'class="hl-green"' : ''}><td ${i === 0 ? 'style="font-weight:800"' : ''}>${esc(r.approach)}</td><td class="r mono">${r.digit_accuracy.toFixed(1)}%</td><td class="r mono">${r.full_reading_accuracy.toFixed(1)}%</td></tr>`).join('')}
    <tr class="hl"><td style="font-weight:800">Ours - fused, detector crops (end to end)</td><td class="r mono">${f1(h.digits)}</td><td class="r mono">${f1(h.full)}</td></tr></table>
    ${finding(`We compare with the highlighted ${esc(m.baseline.rows[0].approach.split(',')[0])}, the same kind of model as ours. Digit by digit we score ${f1(h.digits)}% vs ${m.baseline.rows[0].digit_accuracy.toFixed(1)}%, but a reading only counts if all five digits are right, so missed digits${weakest(e2eRow) === 4 ? ', mostly the last one,' : ''} add up: ${f1(h.full)}% vs ${m.baseline.rows[0].full_reading_accuracy.toFixed(1)}% of full readings correct. Their results are an average of 10 runs; ours are from one run.`)}
    <div class="note" style="margin-top:6px">Source: Laroca et al. (2019), <a href="https://arxiv.org/pdf/1902.09600" target="_blank" rel="noopener"><i>Convolutional Neural Networks for Automatic Meter Reading</i></a>, Table 7.</div></div>`;

  const dt = m.detection;
  const yolo = `<div class="card"><div class="card-head"><div class="t">Counter Detection</div></div>
    <img src="${m.figures.iou}" alt="" class="fig" style="margin-bottom:8px">
    <div class="kv"><span class="k">mAP@0.5</span><span class="v mono">${f1(dt.mAP50)}%</span>
      <span class="k">precision / recall</span><span class="v mono">${f1(dt.precision)}% / ${f1(dt.recall)}%</span>
      <span class="k">mean IoU</span><span class="v mono">${dt.mean_iou.toFixed(3)} <span>(median ${dt.median_iou.toFixed(3)})</span></span>
      <span class="k">IoU below 0.5</span><span class="v mono" style="color:var(--red)">${dt['iou_below_0.5']} <span>(incl. ${dt.misses} missed counters)</span></span></div>
    ${finding(`YOLO finds the counter in ${dt.n_images - dt.misses} of ${dt.n_images} photos (median IoU ${dt.median_iou.toFixed(2)})${dt['iou_below_0.5'] / dt.n_images < 0.05 ? ', so detection is rarely what makes a reading wrong' : `, but ${dt['iou_below_0.5']} boxes are poor (IoU below 0.5)`}.`)}</div>`;

  const sg = m.sensing;
  const top3 = m.confusion_top.slice(0, 3);
  const neighbours = ([t, p]) => Math.abs(t - p) === 1 || Math.abs(t - p) === 9;   // 9 rolls over to 0
  const confusion = `<div class="card"><div class="card-head"><div class="t">Digit Confusion Matrix</div></div><img src="${m.figures.confusion}" alt="" class="fig">
    ${finding(`On hand-cut (ground-truth) crops, ${f1(m.confusion_accuracy)}% of the ${(n.testing * 5).toLocaleString('en')} test digits are read correctly. The ${f1(h.digits)}% per-digit accuracy at the top of the page ${h.digits < m.confusion_accuracy ? 'is a little lower because there' : 'differs because there'} the counter is cropped by YOLO, not by hand. The most common mix-ups are ${top3.map(([t, p, c]) => `${t}→${p} (${c})`).join(', ')}${top3.every(neighbours) ? ': digits next to each other on the counter wheel, typical of a digit caught mid-roll' : ''}.`)}</div>`;
  const gate = `<div class="card"><div class="card-head"><div class="t">Quality Gate</div></div>
    <img src="${m.figures.sensing_histogram}" alt="" class="fig" style="margin-bottom:8px">
    <div class="kv" style="grid-template-columns:210px 1fr"><span class="k">Test photos flagged as unusual</span><span class="v mono">${sg.flagged} of ${sg.n} <span>(${f1(sg.flag_rate)}%)</span></span>
      <span class="k">Flagged photos &amp; wrong readings</span><span class="v mono" style="color:var(--red)">${Math.round(sg.flagged_error_rate * sg.flagged)} of ${sg.flagged} <span>(${f1(sg.flagged_error_rate)}%)</span></span>
      <span class="k">Passed photos &amp; wrong readings</span><span class="v mono">${Math.round(sg.passed_error_rate * (sg.n - sg.flagged))} of ${sg.n - sg.flagged} <span>(${f1(sg.passed_error_rate)}%)</span></span></div>
    ${finding(sg.flagged_error_rate > sg.passed_error_rate
      ? `Only ${f1(sg.flag_rate)}% of photos are flagged, but ${f1(sg.flagged_error_rate)}% of those are misread vs ${f1(sg.passed_error_rate)}% of the rest: the gate picks out unusual photos (e.g. glare, LCD meters) that the recognisers struggle with.`
      : `${f1(sg.flag_rate)}% of photos are flagged, and they are not misread more often than the rest (${f1(sg.flagged_error_rate)}% vs ${f1(sg.passed_error_rate)}%), so on this split the flag is not a useful warning.`)}</div>`;

  const recognition = `<div class="card"><div class="card-head"><div class="t">Digit Recognition</div></div><img src="${m.figures.recognition}" alt="" class="fig">
    ${finding(`On hand-cut (ground-truth) crops, the fused model reads ${f1(fusedRow.test_full)}% of meters fully right, with all five digits correct. Per digit it does better, ${f1(fusedRow.test_digits)}% on average, but ${d5WeakestEverywhere ? `the last digit (d5) is the weakest position for every model, ${f1(fusedRow.test_d5)}% for fused` : `its weakest position is d${weakest(fusedRow) + 1}, ${f1(fusedRow.test_per_digit[weakest(fusedRow)])}%`}. One wrong digit spoils the whole reading, so the weakest position is what holds the full-reading accuracy down.`)}</div>`;

  body.innerHTML = stats + `<div class="models-flow">${table}${recognition}
    <div class="trio">${yolo}${confusion}${ablation}</div>
    <div class="pair">${gate}${decision}</div>
    ${baseline}</div>
    <div class="note">Generated by src/evaluate.py on ${esc(m.generated)}.</div>`;
}

// datasets

const ds = { split: 'testing', outcome: 'any', fusion: 'any', quality: 'any', q: '', page: 0, rows: [], sort: null, dir: 1 };
// sortable columns (missing values last)
const dsSortKeys = { image: r => r.basename, split: r => r.split, truth: r => r.truth, conf: r => r.confidence, decision: r => r.decision, iou: r => r.iou, sensing: r => r.sensing };
const PAGE = 20;
const dsFilters = {
  outcome: { any: 'Any', right: 'Read correctly', wrong: 'Read wrongly', wrong_approved: 'Wrong but approved', miss: 'Detector miss' },
  fusion: { any: 'Any', disagree: 'Members disagree', fixed: 'Fixed by fusion', all_wrong: 'Wrong in every member' },
  quality: { any: 'Any', flagged: 'Sensing flagged', passed: 'Sensing passed', low_iou: 'IoU < 0.5' },
};
const dsTests = {
  outcome: { right: r => r.correct, wrong: r => !r.correct, wrong_approved: r => !r.correct && r.decision === 'APPROVED', miss: r => !r.detected },
  fusion: {
    disagree: r => r.detected && new Set(Object.values(r.members)).size > 1,
    fixed: r => r.correct && Object.values(r.members).some(m => m !== r.truth),
    all_wrong: r => r.detected && Object.values(r.members).every(m => m !== r.truth),
  },
  quality: { flagged: r => r.sensing_flag, passed: r => r.detected && !r.sensing_flag, low_iou: r => r.iou < 0.5 },
};

async function loadDatasets() {
  const body = $('#datasets-body');
  body.innerHTML = '<div class="note">loading…</div>';
  const splits = ds.split === 'all' ? ['validation', 'testing'] : [ds.split];
  try {
    const parts = await Promise.all(splits.map(sp => api('/datasets/' + sp).then(d => d.rows.map(r => ({ ...r, split: sp, hasCrop: d.crops, hasPhoto: d.photos })))));
    ds.rows = parts.flat();
  } catch (err) { body.innerHTML = `<div class="error">${esc(err.message)}</div>`; return; }
  ds.page = 0;
  renderDatasets();
}
function traceLink(r) {
  return r.hasPhoto ? `<a href="#capture?dataset=${r.split}/${r.basename}" style="font-weight:700; font-size:12px">Trace</a>`
    : '<span class="note" title="The full-size dataset photos are not installed on this server">Trace</span>';
}
function renderDatasets() {
  const body = $('#datasets-body');
  const passes = (r, group) => ds[group] === 'any' || dsTests[group][ds[group]](r);
  const rows = ds.rows.filter(r => passes(r, 'outcome') && passes(r, 'fusion') && passes(r, 'quality') && (!ds.q || r.basename.includes(ds.q)));
  if (ds.sort) {
    const key = dsSortKeys[ds.sort];
    rows.sort((a, b) => {
      const x = key(a), y = key(b);
      if (x == null || y == null) return (x == null) - (y == null);
      return ds.dir * (typeof x === 'string' ? x.localeCompare(y) : x - y);
    });
  }
  const th = (label, key, style = '') => `<th data-sort="${key}" ${style}>${label}${ds.sort === key ? (ds.dir > 0 ? ' ▲' : ' ▼') : ''}</th>`;
  const count = (group, key) => key === 'any' ? ds.rows.length : ds.rows.filter(r => dsTests[group][key](r)).length;
  const select = (group, opts) => `<label class="filter"><span class="card-label">${group[0].toUpperCase() + group.slice(1)}</span><select data-group="${group}">
    ${Object.entries(opts).map(([k, l]) => `<option value="${k}" ${ds[group] === k ? 'selected' : ''}>${l} (${count(group, k)})</option>`).join('')}</select></label>`;
  const bar = `<div class="filters">
    <label class="filter"><span class="card-label">Split</span><select data-group="split">${['all', 'validation', 'testing'].map(x => `<option value="${x}" ${ds.split === x ? 'selected' : ''}>${x[0].toUpperCase() + x.slice(1)}</option>`).join('')}</select></label>
    ${select('outcome', dsFilters.outcome)}${select('fusion', dsFilters.fusion)}${select('quality', dsFilters.quality)}
    <label class="search-wrap"><svg viewBox="0 0 24 24" width="16" height="16"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4 4"/></svg><input class="search" placeholder="Search meter label" value="${esc(ds.q)}"></label></div>`;

  const start = ds.page * PAGE, page = rows.slice(start, start + PAGE);
  const digits = (reading, truth) => `<span class="mono" style="font-size:13px; font-weight:800">${reading.split('').map((c, i) => `<span class="d ${c !== truth[i] ? 'bad' : ''}">${c}</span>`).join('')}</span>`;
  const splitCell = r => ds.split === 'all' ? `<td class="note">${r.split[0].toUpperCase() + r.split.slice(1)}</td>` : '';
  const tr = page.map(r => {
    const conf = r.confidence == null ? 0 : pct(r.confidence);
    if (!r.detected) return `<tr><td><div class="thumb"></div></td><td class="mono">${r.basename}</td>${splitCell(r)}<td class="mono" style="font-weight:800">${r.truth}</td><td colspan="4" class="note">no counter detected</td><td></td><td><span class="decision flagged" style="font-size:11px; padding:4px 8px">Flagged</span></td><td class="mono">0.00</td><td></td><td>${traceLink(r)}</td></tr>`;
    return `<tr><td>${r.hasCrop ? `<img class="thumb" src="${r.crop_url}" alt="" loading="lazy">` : '<div class="thumb"></div>'}</td><td class="mono">${r.basename}</td>${splitCell(r)}<td class="mono" style="font-weight:800">${r.truth}</td>
      ${['cnn', 'cnn_rgb', 'svm'].map(m => `<td>${digits(r.members[m], r.truth)}</td>`).join('')}<td class="fusedcol">${digits(r.fused, r.truth)}</td>
      <td><span class="pill ${tier(conf)}">${(r.confidence * 100).toFixed(1)}%</span></td>
      <td><span class="decision ${r.decision === 'FLAGGED' ? 'flagged' : 'approved'}" style="font-size:11px; padding:4px 8px">${r.decision === 'FLAGGED' ? 'Flagged' : 'Approved'}</span></td>
      <td class="mono" style="${r.iou < 0.5 ? 'color:var(--red)' : ''}">${r.iou.toFixed(2)}</td>
      <td class="mono" style="${r.sensing_flag ? 'color:var(--red)' : ''}">${r.sensing == null ? '-' : Math.round(r.sensing).toLocaleString()}</td>
      <td>${traceLink(r)}</td></tr>`;
  }).join('');
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  body.innerHTML = bar + `<div class="card" style="padding:4px 6px"><table class="ds">
    <tr><th>Crop</th>${th('Image', 'image')}${ds.split === 'all' ? th('Split', 'split') : ''}${th('Ground Truth', 'truth')}<th>CNN Grayscale</th><th>CNN-RGB</th><th>SVM</th><th style="color:var(--accent-dark)">Fused</th>${th('Confidence', 'conf')}${th('Decision', 'decision')}${th('IoU', 'iou')}${th('Sensing', 'sensing')}<th></th></tr>
    ${tr || `<tr><td colspan="${ds.split === 'all' ? 13 : 12}" class="note" style="text-align:center; padding:30px">No images match</td></tr>`}</table>
    <div class="pager"><span class="note">${rows.length ? start + 1 : 0}–${Math.min(start + PAGE, rows.length)} of ${rows.length}</span>
      <button class="btn ghost" data-page="-1" ${ds.page === 0 ? 'disabled' : ''}>‹ Prev</button><button class="btn ghost" data-page="1" ${ds.page >= pages - 1 ? 'disabled' : ''}>Next ›</button></div></div>`;
  body.querySelectorAll('select').forEach(sel => sel.onchange = () => {
    ds[sel.dataset.group] = sel.value; ds.page = 0;
    sel.dataset.group === 'split' ? loadDatasets() : renderDatasets();
  });
  const search = body.querySelector('.search');
  search.oninput = () => {
    const caret = search.selectionStart;
    ds.q = search.value.trim(); ds.page = 0; renderDatasets();
    // restore focus after re-render
    const box = body.querySelector('.search'); box.focus(); box.setSelectionRange(caret, caret);
  };
  body.querySelectorAll('[data-page]').forEach(b => b.onclick = () => { ds.page += +b.dataset.page; renderDatasets(); window.scrollTo(0, 0); });
  body.querySelectorAll('th[data-sort]').forEach(h => h.onclick = () => {
    ds.dir = ds.sort === h.dataset.sort ? -ds.dir : 1; ds.sort = h.dataset.sort; ds.page = 0; renderDatasets();
  });
}

// figure zoom

// Click a figure or photo to see it full size; click or Esc to close.
document.addEventListener('click', e => {
  const img = e.target.closest('img.fig, img.th-photo, img.thumb');
  if (!img) return;
  const box = document.createElement('div');
  box.className = img.matches('.thumb') ? 'overlay zoom crop' : 'overlay zoom';
  box.innerHTML = `<img src="${img.src}" alt="">`;
  const close = () => { box.remove(); document.removeEventListener('keydown', onKey); };
  const onKey = k => { if (k.key === 'Escape') close(); };
  box.onclick = close;
  document.addEventListener('keydown', onKey);
  document.body.appendChild(box);
});

// tabs

const tabs = {
  history: loadHistory,
  models: loadModels,
  datasets: loadDatasets,
};

route();
