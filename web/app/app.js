// Meter Lens phone app: screen state and API calls.

const API = '';   // same origin: the API also serves this page

const state = {
  screen: 'splash',
  file: null,          // File chosen on the capture screen
  fileUrl: null,       // object URL for previews
  result: null,
  pendingCheck: null,  // /predict/check result awaiting the user's choice
  feedback: 'idle',    // idle | confirmed | editing | corrected  (result screen)
  corrected: '',
  history: [],
  selected: null,      // history item on the detail screen
};

const $ = (sel) => document.querySelector(sel);
const app = $('#app');

// helpers

function tierClass(pct) { return pct >= 90 ? 'green' : pct >= 75 ? 'amber' : 'red'; }
function pct(x) { return Math.round(x * 100); }
function fmtTs(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getDate())}-${p(d.getMonth() + 1)}-${d.getFullYear()} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
// Calendar day key and label ("Today", "Yesterday", "11 Sep 2026").
function dayKey(iso) { const d = new Date(iso); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`; }
function dayLabel(key) {
  const today = dayKey(new Date().toISOString());
  const yesterday = dayKey(new Date(Date.now() - 864e5).toISOString());
  if (key === today) return 'Today';
  if (key === yesterday) return 'Yesterday';
  const [y, m, d] = key.split('-').map(Number);
  return new Date(y, m - 1, d).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function toast(msg) {
  const el = document.createElement('div'); el.className = 'toast'; el.textContent = msg;
  document.body.appendChild(el); setTimeout(() => el.remove(), 2200);
}
const VALID_READING = /^\d{5}$/;

// Per-digit chips coloured by confidence tier.
function chipsHtml(reading, confs) {
  if (!reading) return '<div class="fb-note">No reading</div>';
  return reading.split('').map((ch, i) => {
    const p = pct(confs[i] ?? 0);
    return `<div class="chip ${tierClass(p)}"><div class="ch">${esc(ch)}</div><div class="conf">${p}%</div></div>`;
  }).join('');
}

function pillHtml(el, confidence, decision) {
  el.textContent = `${pct(confidence)}% confidence`;
  el.className = 'pill ' + (decision === 'APPROVED' ? 'green' : 'red');
}
function decisionHtml(el, decision, reasons) {
  if (!decision) { el.innerHTML = ''; el.className = 'decision'; return; }
  const ok = decision === 'APPROVED';
  el.className = 'decision ' + (ok ? 'approved' : 'flagged');
  el.innerHTML = ok ? '&#10003; Approved' : '&#9888; Flagged';
  el.title = ok ? '' : (reasons || []).map((r) => r.message).join(' - ');
}
function hasReason(reasons, kind) { return (reasons || []).some((r) => r.kind === kind); }

// Boxes are fractions of the image; find the drawn image rect first (object-fit: contain).
function drawnImageRect(img, container) {
  const cw = container.clientWidth, ch = container.clientHeight;
  const iw = img.naturalWidth || 1, ih = img.naturalHeight || 1;
  const scale = Math.min(cw / iw, ch / ih);
  const w = iw * scale, h = ih * scale;
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
}
function placeOverlays(container, img, result, { label = true, crop = true } = {}) {
  container.querySelectorAll('.box, .tag, .cropview').forEach((n) => n.remove());
  if (!result || !result.detected || !result.counter_bbox) return;
  const r = drawnImageRect(img, container);
  const toPx = (b) => ({ left: r.x + b.x * r.w, top: r.y + b.y * r.h, width: b.w * r.w, height: b.h * r.h });
  const mk = (cls, b) => {
    const d = document.createElement('div'); d.className = 'box ' + cls;
    const p = toPx(b); Object.assign(d.style, { left: p.left + 'px', top: p.top + 'px', width: p.width + 'px', height: p.height + 'px' });
    container.appendChild(d); return p;
  };
  const cp = mk('counter', result.counter_bbox);
  (result.digit_bboxes || []).forEach((b) => {
    // inset each strip slightly so the five boxes read as separate digits
    mk('digit', { x: b.x + b.w * 0.08, y: b.y + b.h * 0.08, w: b.w * 0.84, h: b.h * 0.84 });
  });
  if (label && result.reading) {
    const t = document.createElement('div'); t.className = 'tag'; t.textContent = 'Reading: ' + result.reading;
    t.style.left = Math.max(4, cp.left) + 'px'; t.style.top = Math.max(4, cp.top - 22) + 'px';
    container.appendChild(t);
  }
  if (crop) {
    const wrap = document.createElement('div'); wrap.className = 'cropview';
    wrap.innerHTML = '<div class="tag">Crop</div><canvas width="156" height="60"></canvas>';
    container.appendChild(wrap);
    const c = wrap.querySelector('canvas').getContext('2d');
    const b = result.counter_bbox, iw = img.naturalWidth, ih = img.naturalHeight;
    try { c.drawImage(img, b.x * iw, b.y * ih, b.w * iw, b.h * ih, 0, 0, 156, 60); } catch (e) { /* image not ready */ }
  }
}
function overlayWhenLoaded(container, img, src, result, opts) {
  const run = () => placeOverlays(container, img, result, opts);
  if (img.src !== src) { img.onload = run; img.src = src; } else if (img.complete) run(); else img.onload = run;
}

// API

async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}
async function predictCheck(file) {
  const fd = new FormData(); fd.append('file', file, file.name || 'photo.jpg');
  return api('/predict/check', { method: 'POST', body: fd });
}
async function predictRecognize(imageId, sensingFlag) {
  return api('/predict/recognize', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: imageId, sensing_flag: sensingFlag }),
  });
}
async function loadHistory() { state.history = await api('/history'); }
async function saveReading() {
  const r = state.result;
  const body = {
    image_id: r.image_id, ai_reading: r.reading, confidence: r.confidence,
    per_digit_confidence: r.per_digit_confidence, decision: r.decision, reasons: r.reasons || [],
    sensing_score: r.sensing_score ?? null, sensing_flag: !!r.sensing_flag,
    detected: r.detected, counter_bbox: r.counter_bbox, digit_bboxes: r.digit_bboxes,
    feedback: state.feedback === 'confirmed' ? 'confirmed' : state.feedback === 'corrected' ? 'corrected' : null,
    corrected_reading: state.feedback === 'corrected' ? state.corrected : null,
  };
  return api('/history', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}
async function deleteReading(id) { return api(`/history/${id}`, { method: 'DELETE' }); }

// feedback block (shared by result + detail)

function feedbackHtml(mode, aiReading, corrected, invalid) {
  if (!aiReading) return '<div class="fb-note">Nothing to confirm: no counter was detected.</div>';
  if (mode === 'confirmed') return `
    <div class="fb-row"><div class="fb-btn yes">&#10003; Correct</div><div class="fb-btn off">&#10005; Not correct</div></div>
    <div class="fb-note">No correction needed</div>`;
  if (mode === 'editing') return `
    <div class="fb-note" style="margin-top:0">Enter the correct reading (5 digits):</div>
    <div class="fb-edit"><input inputmode="numeric" maxlength="5" data-role="corrected" class="${invalid ? 'invalid' : ''}" value="${esc(corrected)}" placeholder="00000">
      <button class="fb-confirm" data-action="confirm-correction">Confirm</button></div>`;
  if (mode === 'corrected') return `
    <div class="fb-row"><div class="fb-btn off">&#10003; Correct</div><div class="fb-btn no">&#10005; Not correct</div></div>
    <div class="fb-corrected"><div class="fb-note" style="margin:0">User corrected reading: <b>${esc(corrected)}</b></div>
      <button class="fb-link" data-action="mark-incorrect">&#9998; Edit</button></div>`;
  return `<div class="fb-row">
    <button class="fb-btn yes" data-action="mark-correct">&#10003; Correct</button>
    <button class="fb-btn no" data-action="mark-incorrect">&#10005; Not correct</button></div>`;
}

// render

let shownScreen = null;
function show(screen) {
  state.screen = screen;
  document.querySelectorAll('section[data-screen]').forEach((s) => s.classList.toggle('active', s.dataset.screen === screen));
  app.classList.toggle('dark', screen === 'splash' || screen === 'scanning');
  if (screen !== shownScreen) window.scrollTo(0, 0);
  shownScreen = screen;
}

function renderCapture() {
  const img = $('#framePreview'), ph = $('#framePlaceholder');
  if (state.fileUrl) { img.src = state.fileUrl; img.hidden = false; ph.hidden = true; }
  else { img.hidden = true; ph.hidden = false; }
}

function renderResult() {
  const r = state.result;
  $('#resultTitle').textContent = r.detected ? 'Reading detected' : 'No counter detected';
  overlayWhenLoaded($('#resultPhoto'), $('#resultImg'), r.image_url || state.fileUrl, r);
  $('#resultReading').textContent = r.reading || '–';
  $('#resultTs').textContent = fmtTs(r.timestamp) || 'Just now';
  pillHtml($('#resultPill'), r.confidence || 0, r.decision);
  decisionHtml($('#resultDecision'), r.decision, r.reasons);
  $('#resultChips').innerHTML = chipsHtml(r.reading, r.per_digit_confidence || []);
  $('#resultFeedbackBody').innerHTML = feedbackHtml(state.feedback, r.reading, state.corrected, state.invalid);
  // saving requires a feedback answer
  const answered = state.feedback === 'confirmed' || state.feedback === 'corrected';
  $('#saveBtn').disabled = !r.detected || !answered;
  $('#saveBtn').textContent = !r.detected ? 'Save to history' : answered ? 'Save to history' : 'Confirm the reading to save';
  $('#resultFeedback').classList.toggle('needs-feedback', r.detected && !answered);
}

function renderHistory() {
  const list = $('#historyList');
  if (!state.history.length) { list.innerHTML = '<div class="empty">No saved readings yet.<br>Capture a meter photo to add one.</div>'; return; }
  // group by day, newest first
  const groups = new Map();
  for (const h of state.history) { const k = dayKey(h.created_at); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(h); }
  const item = (h) => `
    <div class="item" data-action="open" data-id="${h.id}">
      <div class="thumb" style="background-image:url('${esc(h.image_url)}')"></div>
      <div class="main">
        <div class="rd">${esc(h.reading || '–')} <span>kWh</span>${h.feedback === 'corrected' ? ' <span>(corrected)</span>' : ''}</div>
        <div class="ts">${fmtTs(h.created_at)}</div>
      </div>
      <div class="pill ${h.decision === 'APPROVED' ? 'green' : 'red'}">${pct(h.confidence)}%</div>
      <svg width="8" height="14" viewBox="0 0 8 14" fill="none"><path d="M1 1l6 6-6 6" stroke="#D9C4AE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </div>`;
  list.innerHTML = [...groups].map(([k, items]) =>
    `<div class="day-label">${dayLabel(k)}</div>${items.map(item).join('')}`).join('');
}

function renderDetail() {
  const h = state.selected; if (!h) return;
  $('#detailImgSmall').src = h.image_url;
  overlayWhenLoaded($('#detailPhoto'), $('#detailImg'), h.image_url, { ...h, reading: h.ai_reading }, { crop: true });
  $('#detailReading').textContent = h.reading || '–';
  $('#detailTs').textContent = fmtTs(h.created_at);
  pillHtml($('#detailPill'), h.confidence || 0, h.decision);
  decisionHtml($('#detailDecision'), h.decision, h.reasons);
  $('#detailChips').innerHTML = chipsHtml(h.ai_reading, h.per_digit_confidence || []);
  // only list reasons for flagged readings
  let flagNote = '';
  if (h.decision !== 'APPROVED') {
    const kinds = [];
    if (hasReason(h.reasons, 'sensing')) kinds.push('unusual photo');
    if (hasReason(h.reasons, 'confidence')) kinds.push('low confidence');
    flagNote = ` <span>(Flagged${kinds.length ? ': ' + kinds.join(', ') : ''})</span>`;
  }
  const rows = [`<div class="row"><div class="k">System detected:</div><div class="v">${esc(h.ai_reading || '–')} <span>kWh</span>${flagNote}</div></div>`];
  if (h.feedback === 'corrected') rows.push(`<div class="row"><div class="k">User corrected:</div><div class="v">${esc(h.corrected_reading)} <span>kWh</span> <span>&middot; ${fmtTs(h.feedback_at)}</span></div></div>`);
  if (h.feedback === 'confirmed') rows.push(`<div class="row"><div class="k">User confirmed:</div><div class="v">Reading correct - ${esc(h.ai_reading)} <span>kWh</span></div></div>`);
  if (!h.feedback) rows.push(`<div class="row"><div class="k">User:</div><div class="v"><span>not yet verified</span></div></div>`);
  $('#detailTrace').innerHTML = rows.join('');
}

function render() {
  show(state.screen);
  if (state.screen === 'capture') renderCapture();
  if (state.screen === 'result') renderResult();
  if (state.screen === 'history') renderHistory();
  if (state.screen === 'detail') renderDetail();
}
window.addEventListener('resize', () => {
  if (state.screen === 'result' && state.result) placeOverlays($('#resultPhoto'), $('#resultImg'), state.result);
  if (state.screen === 'detail' && state.selected) placeOverlays($('#detailPhoto'), $('#detailImg'), { ...state.selected, reading: state.selected.ai_reading });
});

// flows

const SPLASH_STATUS = ['Detecting meter…', 'Locating digits…', 'Reading numbers…', 'Capturing reading…'];
let splashTimer, statusTimer, statusIdx = 0;
function startSplash() {
  show('splash');
  statusTimer = setInterval(() => { statusIdx = (statusIdx + 1) % SPLASH_STATUS.length; $('#splashStatus').textContent = SPLASH_STATUS[statusIdx]; }, 900);
  splashTimer = setTimeout(leaveSplash, 2200);
}
function leaveSplash() {
  clearTimeout(splashTimer); clearInterval(statusTimer);
  state.screen = 'capture'; render();
}

// Forget the current photo and its result so the capture screen starts empty again.
function resetCapture() {
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.file = null; state.fileUrl = null;
  state.result = null; state.pendingCheck = null; state.feedback = 'idle'; state.corrected = ''; state.invalid = false;
}

// Check the photo first; if sensing-flagged, ask the user before running recognition.
async function scan(file) {
  resetCapture();
  state.file = file;
  state.fileUrl = URL.createObjectURL(file);
  show('scanning');
  const container = $('#scanPhoto'), img = $('#scanImg');
  container.querySelectorAll('.box, .tag, .cropview').forEach((n) => n.remove());
  img.src = state.fileUrl;
  $('#scanStatus').textContent = 'Checking photo…';
  try {
    const check = await predictCheck(file);
    if (!check.detected) {
      // no counter found
      state.result = {
        detected: false, reading: null, confidence: 0, decision: 'FLAGGED',
        reasons: [{ kind: 'detection', message: 'No counter detected -- manual review required' }],
        sensing_score: null, sensing_flag: false, counter_bbox: null, digit_bboxes: null, members: {},
        image_id: check.image_id, image_url: check.image_url, timestamp: check.timestamp,
      };
      state.screen = 'result'; render();
      return;
    }
    placeOverlays(container, img, check, { label: false, crop: false });
    if (hasReason(check.reasons, 'sensing')) {
      state.pendingCheck = check;
      $('#scanStatus').textContent = 'Unusual photo detected…';
      showSensingModal(check.reasons);
      return;
    }
    await continueRecognition(check);
  } catch (e) {
    toast('Could not read the photo: ' + e.message);
    state.screen = 'capture'; render();
  }
}

// Recognition step, after a clean check or "Use Anyway".
async function continueRecognition(check) {
  $('#scanStatus').textContent = 'Reading numbers…';
  try {
    const result = await predictRecognize(check.image_id, check.sensing_flag);
    result.sensing_score = check.sensing_score;
    result.image_id = check.image_id; result.image_url = check.image_url; result.timestamp = check.timestamp;
    state.result = result;
    await new Promise((r) => setTimeout(r, 400));   // let the boxes register before switching
    state.screen = 'result'; render();
  } catch (e) {
    toast('Could not read the photo: ' + e.message);
    state.screen = 'capture'; render();
  }
}

async function onSave() {
  if (state.feedback === 'editing') { toast('Confirm the corrected reading first'); return; }
  if (state.feedback !== 'confirmed' && state.feedback !== 'corrected') { toast('Please say whether the reading is correct before saving'); return; }
  try {
    $('#saveBtn').disabled = true;
    await saveReading();
    await loadHistory();
    state.screen = 'history'; render();
    toast('Saved to history');
  } catch (e) { toast('Save failed: ' + e.message); $('#saveBtn').disabled = false; }
}

async function openDetail(id) {
  await loadHistory();
  state.selected = state.history.find((h) => String(h.id) === String(id)) || null;
  state.screen = 'detail'; render();
}

// sensing modal
// Blocking modal for a sensing-flagged photo; closes only via its two buttons.

function showSensingModal(reasons) {
  const sensing = (reasons || []).find((r) => r.kind === 'sensing');
  $('#sensingModalBody').textContent = sensing ? sensing.message : '';
  $('#sensingModal').hidden = false;
}
function hideSensingModal() { $('#sensingModal').hidden = true; }

// Sample photos (web/samples/).
async function showSamplesModal() {
  const grid = $('#samplesGrid');
  if (!grid.children.length) {
    const list = await (await fetch('/samples/samples.json')).json();
    grid.innerHTML = list.map((s) => `<button data-action="sample" data-file="${s.file}" title="${esc(s.expect)}">
      <img src="/samples/${s.thumb}" alt=""><span class="lbl">${esc(s.label)}</span><span class="badge ${s.kind}">${s.kind === 'good' ? 'good photo' : 'tricky photo'}</span></button>`).join('');
  }
  $('#samplesModal').hidden = false;
}
async function scanSample(file) {
  $('#samplesModal').hidden = true;
  const blob = await (await fetch('/samples/' + file)).blob();
  scan(new File([blob], file, { type: 'image/jpeg' }));
}

// Full-size photo viewer with pinch / double-tap / wheel zoom and pan.
const zoom = { scale: 1, tx: 0, ty: 0, pointers: new Map(), pinchDist: 0, lastTap: 0, tapTimer: null, moved: false };
const ZOOM_MAX = 5, ZOOM_DOUBLE_TAP = 2.5;

function openLightbox(src) { $('#lightboxImg').src = src; $('#lightbox').hidden = false; zoomReset(); }
function closeLightbox() { $('#lightbox').hidden = true; $('#lightboxImg').src = ''; zoomReset(); }

function zoomReset() { zoom.scale = 1; zoom.tx = 0; zoom.ty = 0; zoom.pointers.clear(); zoomApply(); }
function zoomApply() {
  const box = $('#lightbox'), img = $('#lightboxImg');
  // keep the photo covering the viewport
  const maxX = Math.max(0, (img.offsetWidth * zoom.scale - box.clientWidth) / 2);
  const maxY = Math.max(0, (img.offsetHeight * zoom.scale - box.clientHeight) / 2);
  zoom.tx = Math.min(maxX, Math.max(-maxX, zoom.tx));
  zoom.ty = Math.min(maxY, Math.max(-maxY, zoom.ty));
  img.style.transform = `translate(${zoom.tx}px, ${zoom.ty}px) scale(${zoom.scale})`;
  box.classList.toggle('zoomed', zoom.scale > 1.01);
}
// zoom to `next` keeping the screen point (px, py) fixed under the finger / cursor
function zoomTo(next, px, py) {
  const img = $('#lightboxImg'), r = img.getBoundingClientRect();
  const cx = r.left + r.width / 2 - zoom.tx, cy = r.top + r.height / 2 - zoom.ty;   // image centre at scale 1
  next = Math.min(ZOOM_MAX, Math.max(1, next));
  const qx = (px - cx - zoom.tx) / zoom.scale, qy = (py - cy - zoom.ty) / zoom.scale;
  zoom.tx = px - cx - next * qx; zoom.ty = py - cy - next * qy; zoom.scale = next;
  if (next <= 1.001) { zoom.tx = 0; zoom.ty = 0; }
  zoomApply();
}
function pinchDistance() {
  const [a, b] = [...zoom.pointers.values()];
  return Math.hypot(a.x - b.x, a.y - b.y);
}
(function wireLightbox() {
  const box = $('#lightbox');
  box.addEventListener('pointerdown', (ev) => {
    if (ev.target.closest('[data-action]')) return;
    zoom.pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY, x0: ev.clientX, y0: ev.clientY });
    try { box.setPointerCapture(ev.pointerId); } catch (_) { /* pointer already gone */ }
    zoom.moved = false;
    if (zoom.pointers.size === 2) zoom.pinchDist = pinchDistance();
    if (zoom.scale > 1) box.classList.add('dragging');
  });
  box.addEventListener('pointermove', (ev) => {
    const p = zoom.pointers.get(ev.pointerId);
    if (!p) return;
    const dx = ev.clientX - p.x, dy = ev.clientY - p.y;
    p.x = ev.clientX; p.y = ev.clientY;
    if (Math.hypot(ev.clientX - p.x0, ev.clientY - p.y0) > 8) zoom.moved = true;
    if (zoom.pointers.size === 2) {
      const [a, b] = [...zoom.pointers.values()];
      const dist = pinchDistance();
      if (zoom.pinchDist > 0) zoomTo(zoom.scale * dist / zoom.pinchDist, (a.x + b.x) / 2, (a.y + b.y) / 2);
      zoom.pinchDist = dist;
    } else if (zoom.scale > 1) {
      zoom.tx += dx; zoom.ty += dy; zoomApply();
    }
  });
  const release = (ev) => {
    if (!zoom.pointers.has(ev.pointerId)) return;
    zoom.pointers.delete(ev.pointerId);
    box.classList.remove('dragging');
    if (zoom.pointers.size === 1) zoom.pinchDist = 0;   // second finger lifted: back to panning
    if (zoom.pointers.size > 0 || zoom.moved || ev.type !== 'pointerup') return;
    // double-tap toggles zoom; single tap closes when not zoomed
    const now = Date.now(), x = ev.clientX, y = ev.clientY;
    if (now - zoom.lastTap < 300) {
      clearTimeout(zoom.tapTimer); zoom.lastTap = 0;
      zoomTo(zoom.scale > 1.01 ? 1 : ZOOM_DOUBLE_TAP, x, y);
    } else {
      zoom.lastTap = now;
      clearTimeout(zoom.tapTimer);
      zoom.tapTimer = setTimeout(() => { if (zoom.scale <= 1.01) closeLightbox(); }, 300);
    }
  };
  box.addEventListener('pointerup', release);
  box.addEventListener('pointercancel', release);
  box.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    zoomTo(zoom.scale * (ev.deltaY < 0 ? 1.15 : 1 / 1.15), ev.clientX, ev.clientY);
  }, { passive: false });
  // older iOS Safari still runs its page pinch-zoom over touch-action: none
  ['gesturestart', 'gesturechange'].forEach((t) => box.addEventListener(t, (ev) => ev.preventDefault()));
  window.addEventListener('resize', () => { if (!box.hidden) zoomApply(); });
})();

// events (one delegated listener)

document.addEventListener('click', async (ev) => {
  const t = ev.target.closest('[data-action]');
  if (state.screen === 'splash' && !t) { leaveSplash(); return; }
  if (!t) return;
  const action = t.dataset.action;
  switch (action) {
    case 'history': await loadHistory(); state.screen = 'history'; render(); break;
    case 'capture':
    case 'retake': resetCapture(); state.screen = 'capture'; render(); break;
    case 'shutter': $('#cameraInput').click(); break;
    case 'gallery': $('#galleryInput').click(); break;
    case 'samples': await showSamplesModal(); break;
    case 'samples-close': $('#samplesModal').hidden = true; break;
    case 'sample': await scanSample(t.dataset.file); break;
    case 'save': await onSave(); break;
    case 'open': await openDetail(t.dataset.id); break;
    case 'zoom': if (state.selected && state.selected.image_url) openLightbox(state.selected.image_url); break;
    case 'zoom-close': closeLightbox(); break;
    case 'sensing-retake': hideSensingModal(); resetCapture(); state.screen = 'capture'; render(); break;
    case 'sensing-use-anyway': {
      hideSensingModal();
      const check = state.pendingCheck; state.pendingCheck = null;
      if (check) await continueRecognition(check);
      break;
    }
    case 'delete':
      if (state.selected && confirm('Delete this reading?')) {
        await deleteReading(state.selected.id); await loadHistory(); state.screen = 'history'; render();
      }
      break;
    case 'mark-correct': state.feedback = 'confirmed'; render(); break;
    case 'mark-incorrect':
      state.feedback = 'editing'; state.corrected = state.corrected || state.result.reading || '';
      state.invalid = false; render();
      break;
    case 'confirm-correction': {
      const input = t.parentElement.querySelector('[data-role=corrected]');
      const value = (input.value || '').trim();
      if (!VALID_READING.test(value)) { state.invalid = true; toast('Enter exactly 5 digits'); render(); return; }
      state.corrected = value; state.feedback = 'corrected'; state.invalid = false; render();
      break;
    }
  }
});
document.addEventListener('input', (ev) => {
  if (ev.target.dataset.role !== 'corrected') return;
  ev.target.value = ev.target.value.replace(/\D/g, '').slice(0, 5);
  state.corrected = ev.target.value;
});
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && !$('#lightbox').hidden) { closeLightbox(); return; }
  if (ev.key === 'Enter' && ev.target.dataset.role === 'corrected') ev.target.parentElement.querySelector('[data-action=confirm-correction]').click();
});
['#cameraInput', '#galleryInput'].forEach((sel) => $(sel).addEventListener('change', (ev) => {
  const file = ev.target.files && ev.target.files[0];
  ev.target.value = '';
  if (file) scan(file);
}));

// boot
startSplash();
