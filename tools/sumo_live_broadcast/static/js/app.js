/* SUMO 实时转播 —— 应用层。
 * 负责：会话生命周期 UI、SSE 收帧、控制指令、相位面板/决策历史/指标。
 * 地图呈现全部委托给 SLB.createRenderer（契约见 renderer.js），
 * 通过 ?renderer=canvas|deckgl 选择实现，缺依赖自动回退 canvas。
 */
'use strict';

let INIT = null, CATALOG = {}, LANES = [], laneById = {}, BOUNDS = null;
let PANELS = {}, CONTROLLED = [];
let PHASES = [], YELLOW_DICT = {}, PHASE_COUNT = 0, INTER_ID = null;
let lastFrame = null;
let renderer = null;

function post(op, extra) {
  fetch('/api/control', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.assign({op}, extra || {}))}).catch(() => {});
}

// ---------- phase cards ----------
const sigColor = c => c === 'G' || c === 'g' ? '#20a050' : c === 'y' ? '#e0a100' : '#8f2f2f';
// Bearing (deg, world coords y-up) of a lane's junction-adjacent endpoint,
// measured from the junction centre.  `isIn` picks the last shape point for
// approach lanes and the first for exit lanes.
function laneBearing(lane, cx, cy, isIn) {
  const p = isIn ? lane.shape[lane.shape.length - 1] : lane.shape[0];
  return Math.atan2(p[1] - cy, p[0] - cx) * 180 / Math.PI;
}
// Heading of a lane polyline at a given end (deg, direction of travel).
function segHeading(shape, atEnd) {
  const i = atEnd ? shape.length - 1 : 1, j = atEnd ? shape.length - 2 : 0;
  return Math.atan2(shape[i][1] - shape[j][1], shape[i][0] - shape[j][0]) * 180 / Math.PI;
}
// Movement kind from the via-lane's actual geometry: heading change through
// the junction — ~0 = through, ~±180 = u-turn, otherwise a turn.
function viaTurn(via) {
  if (!via || via.shape.length < 2) return 0;
  const d = segHeading(via.shape, true) - segHeading(via.shape, false);
  return ((d + 540) % 360) - 180;
}
const ARM_DEG = {E: 0, N: 90, W: 180, S: 270};
// Snap a world bearing to the nearest cardinal arm.
function snapArm(b) {
  const x = ((b % 360) + 360) % 360;
  let best = 'E', bd = 999;
  for (const [n, a] of Object.entries(ARM_DEG)) {
    const d = Math.min(Math.abs(x - a), 360 - Math.abs(x - a));
    if (d < bd) { bd = d; best = n; }
  }
  return best;
}
// Draw one road arm toward a world bearing (canvas y is flipped).
function drawArm(ctx, cx, cy, deg, len, w, color, alpha) {
  const a = deg * Math.PI / 180;
  ctx.save(); ctx.translate(cx, cy); ctx.rotate(-a);
  ctx.globalAlpha = alpha; ctx.fillStyle = color;
  ctx.fillRect(0, -w / 2, len, w);
  ctx.restore(); ctx.globalAlpha = 1;
}
// Chinese movement summary for a phase, e.g. 东西直行 / 南进 直行·右转.
const DIR_CN = {E: '东', S: '南', W: '西', N: '北'};
const KIND_CN = {thru: '直行', left: '左转', right: '右转', uturn: '掉头'};
const KIND_ORD = {thru: 0, left: 1, right: 2, uturn: 3};
function linkMove(t, jx, jy) {
  const inL = laneById[t.in], outL = laneById[t.out], via = t.via && laneById[t.via];
  if (!inL || !outL) return null;
  const d = via ? viaTurn(via)
    : segHeading(outL.shape, false) - segHeading(inL.shape, true);
  const turn = ((d + 540) % 360) - 180;
  const kind = Math.abs(turn) < 30 ? 'thru' : Math.abs(turn) > 150 ? 'uturn'
    : turn > 0 ? 'left' : 'right';
  return {a: snapArm(laneBearing(inL, jx, jy, true)),
          b: snapArm(laneBearing(outL, jx, jy, false)), kind};
}
function describePhase(state) {
  const item = CATALOG[INTER_ID] || {};
  const jx = item.x || 0, jy = item.y || 0;
  const groups = {};
  (item.links || []).forEach((t, i) => {
    if ((state[i] || 'r').toLowerCase() !== 'g') return;
    const m = linkMove(t, jx, jy); if (!m) return;
    const g = groups[m.a] = groups[m.a] || {outs: new Set(), kinds: new Set()};
    g.outs.add(m.b); g.kinds.add(m.kind);
  });
  const keys = Object.keys(groups);
  if (!keys.length) return '全红';
  if (keys.length === 2) {   // two opposite approaches, same single kind
    // Conventional pairing order: 东西 / 南北.
    const pairOrder = {E: 0, S: 1, W: 2, N: 3};
    const [a, b] = keys.sort((x, y) => pairOrder[x] - pairOrder[y]);
    const opp = Math.abs(ARM_DEG[a] - ARM_DEG[b]) === 180;
    const ka = [...groups[a].kinds], kb = [...groups[b].kinds];
    if (opp && ka.length === 1 && kb.length === 1 && ka[0] === kb[0])
      return DIR_CN[a] + DIR_CN[b] + KIND_CN[ka[0]];
  }
  return keys.sort().map(a => DIR_CN[a] + '进 ' +
    [...groups[a].kinds].sort((x, y) => KIND_ORD[x] - KIND_ORD[y])
      .map(k => KIND_CN[k]).join('·')).join('　');
}
function drawPhaseCard(canvas, state) {
  const ctx = canvas.getContext('2d');
  const item = CATALOG[INTER_ID] || {};
  const links = item.links || [];
  const jx = item.x || 0, jy = item.y || 0;
  const W = canvas.width, H = canvas.height, cx = W / 2, cy = H / 2;
  ctx.clearRect(0, 0, W, H);
  const armLen = Math.min(W, H) * 0.47, armW = Math.min(W, H) * 0.20;
  const box = armW;
  // --- light cross base ----------------------------------------------------
  [0, 90, 180, 270].forEach(dg => drawArm(ctx, cx, cy, dg, armLen, armW, '#e4e9ec', 1));
  ctx.fillStyle = '#dae0e4'; ctx.fillRect(cx - box / 2, cy - box / 2, box, box);
  // --- per-link info -------------------------------------------------------
  const infos = links.map((t, i) => {
    const inL = t.in && laneById[t.in], outL = t.out && laneById[t.out];
    if (!inL || !outL) return null;
    return {inL, outL, via: t.via && laneById[t.via],
            green: (state[i] || 'r').toLowerCase() === 'g'};
  });
  // Arm tints: approach arm stronger green, exit arm lighter.
  const tinted = new Set();
  infos.forEach(inf => {
    if (!inf || !inf.green) return;
    [[laneBearing(inf.inL, jx, jy, true), 0.32],
     [laneBearing(inf.outL, jx, jy, false), 0.14]].forEach(([b, al]) => {
      const arm = snapArm(b), key = arm + ':' + al;
      if (tinted.has(key)) return; tinted.add(key);
      drawArm(ctx, cx, cy, ARM_DEG[arm], armLen, armW, '#239b4a', al);
    });
  });
  // Red stop bars on approaches where nothing is green.
  const approaches = new Set(), served = new Set();
  infos.forEach(inf => {
    if (!inf) return;
    const a = snapArm(laneBearing(inf.inL, jx, jy, true));
    approaches.add(a); if (inf.green) served.add(a);
  });
  approaches.forEach(a => {
    if (served.has(a)) return;
    const b = ARM_DEG[a] * Math.PI / 180, r = box / 2 + 3;
    const px2 = cx + Math.cos(b) * r, py2 = cy - Math.sin(b) * r;
    const tx = Math.cos(b + Math.PI / 2), ty = -Math.sin(b + Math.PI / 2);
    ctx.strokeStyle = '#c0392b'; ctx.lineWidth = 2.4; ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(px2 - tx * armW * 0.42, py2 - ty * armW * 0.42);
    ctx.lineTo(px2 + tx * armW * 0.42, py2 + ty * armW * 0.42);
    ctx.stroke();
  });
  // --- green movement arrows: draw the via lane's real path ----------------
  const s = armLen / 16;                       // ~16 m world -> full arm length
  const mp = p => [cx + (p[0] - jx) * s, cy - (p[1] - jy) * s];
  let drew = false;
  infos.forEach(inf => {
    if (!inf || !inf.green) return;
    const shape = (inf.via && inf.via.shape.length > 1) ? inf.via.shape
      : [inf.inL.shape[inf.inL.shape.length - 1], inf.outL.shape[0]];
    const P = shape.map(mp);
    ctx.strokeStyle = '#1e9e48'; ctx.lineWidth = 3;
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.beginPath();
    P.forEach((q, i) => i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]));
    ctx.stroke();
    const p0 = P[P.length - 2], p1 = P[P.length - 1];
    const an = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);
    ctx.fillStyle = '#1e9e48'; ctx.beginPath();
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p1[0] - 8 * Math.cos(an - 0.5), p1[1] - 8 * Math.sin(an - 0.5));
    ctx.lineTo(p1[0] - 8 * Math.cos(an + 0.5), p1[1] - 8 * Math.sin(an + 0.5));
    ctx.closePath(); ctx.fill();
    drew = true;
  });
  if (!drew) {   // fallback: signal-dot strip
    (state || '').split('').forEach((c, i) => {
      ctx.fillStyle = sigColor(c); ctx.beginPath();
      ctx.arc(10 + (i % 8) * 14, 12 + Math.floor(i / 8) * 16, 5, 0, 7); ctx.fill();
    });
    return;
  }
  // North arrow: world +y maps to canvas -y (up).
  const ax = W - 13, ay = 15;
  ctx.strokeStyle = '#9fb0b9'; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(ax, ay + 8); ctx.lineTo(ax, ay - 6); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ax - 3, ay - 2); ctx.lineTo(ax, ay - 7); ctx.lineTo(ax + 3, ay - 2); ctx.stroke();
}
let pendingPhase = null;   // clicked but not yet confirmed by a frame
function setFocus(id) {
  if (!PANELS[id]) return;
  INTER_ID = id;
  const p = PANELS[id];
  PHASES = p.phases || []; PHASE_COUNT = p.phase_count || 0;
  YELLOW_DICT = p.yellow_dict || {};
  pendingPhase = null;
  buildPhaseCards();
  if (lastFrame) updatePhaseCards(lastFrame);
  const sel = document.getElementById('interSel');
  if (sel.value !== id) sel.value = id;
  if (renderer) renderer.setFocus(id);
}
function buildPhaseCards() {
  const grid = document.getElementById('phaseGrid'); grid.innerHTML = '';
  PHASES.forEach(p => {
    const card = document.createElement('div'); card.className = 'phaseCard'; card.dataset.phase = p.index;
    card.innerHTML = '<span class="badge bEffect">生效</span><span class="badge bTarget">目标</span>' +
                     '<span class="badge bProp">建议</span><span class="badge bChosen">已选</span>';
    const cv = document.createElement('canvas'); cv.width = 150; cv.height = 100;
    const lab = document.createElement('div'); lab.className = 'plabel'; lab.textContent = 'P' + p.index;
    const st = document.createElement('div'); st.className = 'pstate'; st.textContent = describePhase(p.state);
    card.appendChild(cv); card.appendChild(lab); card.appendChild(st); grid.appendChild(card);
    drawPhaseCard(cv, p.state);
    card.onclick = () => {
      pendingPhase = p.index;
      document.querySelectorAll('.phaseCard').forEach(c => c.classList.remove('chosen'));
      card.classList.add('chosen');
      post('phase', {phase: p.index, junction: INTER_ID});
    };
    card.title = `P${p.index}: ${p.state}（点击接管路口 ${INTER_ID}）`;
  });
}
function updatePhaseCards(f) {
  const held = (f.held_actions || {})[INTER_ID];
  const picked = (f.pending_choices || {})[INTER_ID];
  const proposed = f.awaiting ? (f.proposed_actions || {})[INTER_ID] : null;
  const raw = ((f.lights || {})[INTER_ID] || [null])[0];
  if (pendingPhase !== null && held === pendingPhase) pendingPhase = null;
  const inEffect = raw !== null && raw < PHASE_COUNT ? raw : null;
  const via = raw !== null && raw >= PHASE_COUNT
    ? Object.entries(YELLOW_DICT).find(e => e[1] === raw) : null;
  document.getElementById('phaseInEffect').textContent =
    inEffect !== null ? 'P' + inEffect : (via ? '黄灯 ' + via[0].replace('_', '→') : (raw === null ? '-' : 'P' + raw));
  document.getElementById('phaseTarget').textContent = held === undefined ? '-' : 'P' + held;
  document.querySelectorAll('.phaseCard').forEach(c => {
    const i = Number(c.dataset.phase);
    c.classList.toggle('effect', i === inEffect);
    c.classList.toggle('target', i === held);
    c.classList.toggle('proposed', Boolean(f.awaiting) && i === proposed);
    c.classList.toggle('chosen',
      (picked !== undefined && i === picked) ||
      (pendingPhase !== null && i === pendingPhase && i !== held));
    c.classList.toggle('manualOk', Boolean(f.awaiting) || f.controller.kind === 'manual');
  });
  const pinned = (f.manual || {})[INTER_ID];
  document.getElementById('manualTag').style.display = pinned === undefined ? 'none' : 'inline-block';
  document.getElementById('releaseBtn').style.display = pinned === undefined ? 'none' : 'inline-block';
}

// ---------- sidebar ----------
function fmtHistory(f) {
  const ul = document.getElementById('historyList'); ul.innerHTML = '';
  (f.history || []).slice().reverse().forEach(h => {
    const li = document.createElement('li');
    if (h.event) { li.innerHTML = `<em>t=${h.t.toFixed(0)}s</em><span>${h.event}</span>`; }
    else {
      const src = h.source === 'manual' ? '<span class="srcManual">人工</span>' : `<span>${h.source}</span>`;
      const errMark = h.errors ? ' <b class="err">!</b>' : '';
      if (h.actions) {
        // Per-junction vector decision; right column shows the focused one.
        const mine = h.actions[INTER_ID];
        li.innerHTML = `<em>t=${h.t.toFixed(0)}s #${h.decision}</em><span>P${mine ?? '-'} ${src}${errMark}</span>`;
        li.title = Object.entries(h.actions).map(([k, v]) => `${k}: P${v}`).join('  ') +
          (h.note ? `\n${h.note}` : '') +
          (h.errors ? `\n${Object.entries(h.errors).map(([k, v]) => `${k}: ${v}`).join('  ')}` : '');
      } else {
        const j = h.junction ? `${h.junction} ` : '';
        li.innerHTML = `<em>t=${h.t.toFixed(0)}s</em><span>${j}P${h.action} ${src}${errMark}</span>`;
        if (h.note) li.title = h.note;
      }
    }
    ul.appendChild(li);
  });
}
function updateSidebar(f) {
  updatePhaseCards(f);
  document.getElementById('simTime').textContent = f.t.toFixed(0) + 's';
  document.getElementById('nextIn').textContent = f.awaiting ? '暂停等待' : f.next_decision_in.toFixed(0);
  const last = (f.history || []).filter(h => h.action !== undefined || h.actions).slice(-1)[0];
  document.getElementById('lastDecision').textContent = last
    ? (last.actions
      ? `t=${last.t.toFixed(0)}s → ${Object.keys(last.actions).length} 路口（${last.source}）`
      : `t=${last.t.toFixed(0)}s → ${last.junction ? last.junction + ' ' : ''}P${last.action}（${last.source}）`)
    : '-';
  const meta = (window.CTLMETA || {})[f.controller.id] || {};
  const ident = (meta.detail && meta.detail.identity) || {};
  const src = meta.path ? meta.path.split('/').slice(-1)[0] : '';
  document.getElementById('ctlSrc').textContent = f.controller.kind === 'snapshot'
    ? `权重: ${src}${ident.global_episode ? ' · ep' + ident.global_episode : ''}${ident.logical_run_id ? ' · ' + ident.logical_run_id : ''}`
    : '';
  const box = document.getElementById('awaitBar');
  if (f.awaiting === 'confirm') {
    box.classList.add('show');
    const props = f.proposed_actions || {};
    const n = Object.keys(props).length;
    const mine = props[INTER_ID];
    document.getElementById('awaitText').textContent = n > 1
      ? `控制器 ${f.controller.label} 已建议 ${n} 个路口（当前路口 P${mine ?? '-'}）；点击相位卡可单独改选，应用并继续后生效`
      : `控制器 ${f.controller.label} 建议 P${mine ?? '-'}，确认后生效`;
  } else box.classList.remove('show');
  const m = document.getElementById('metrics');
  m.innerHTML = [['排队车辆', f.queue.toFixed(0)], ['累计通过', f.throughput],
                 ['在网车辆', f.vehicles_total], ['停车(<0.1m/s)', f.halting]]
    .map(p => `<div>${p[0]}<strong>${p[1]}</strong></div>`).join('');
  const pill = document.getElementById('statePill');
  if (f.rebuilding) { pill.textContent = '场景重建中'; pill.className = 'pill pause'; }
  else if (f.awaiting) { pill.textContent = '等待决策'; pill.className = 'pill await'; }
  else if (f.paused) { pill.textContent = '已暂停'; pill.className = 'pill pause'; }
  else { pill.textContent = `运行中 ${f.speed}×`; pill.className = 'pill run'; }
  document.getElementById('pauseBtn').textContent = f.paused ? '继续' : '暂停';
  if (document.getElementById('controllerSelect').value !== f.controller.id)
    document.getElementById('controllerSelect').value = f.controller.id;
  if (document.getElementById('waitChk').checked !== f.wait_enabled)
    document.getElementById('waitChk').checked = f.wait_enabled;
  fmtHistory(f);
}

// ---------- session lifecycle UI ----------
let sessionState = 'idle', buildStart = null, liveInited = false;
function fmtElapsed() {
  if (!buildStart) return '';
  return '已用时 ' + ((Date.now() - buildStart) / 1000).toFixed(0) + 's';
}
function renderSession(s) {
  sessionState = s.state;
  const ov = document.getElementById('sessionOverlay');
  const title = document.getElementById('sessTitle');
  const sub = document.getElementById('sessSub');
  const err = document.getElementById('sessErr');
  const list = document.getElementById('progList');
  const startBtn = document.getElementById('startBtn');
  const cancelBtn = document.getElementById('cancelBuildBtn');
  const pill = document.getElementById('statePill');
  if (s.state === 'live') {
    buildStart = null;
    ov.classList.remove('show');
    if (!liveInited || s.scene_key !== currentSceneKey) {
      liveInited = true; reInit();
    }
    return;
  }
  ov.classList.add('show');
  list.innerHTML = '';
  err.style.display = 'none'; err.textContent = '';
  cancelBtn.style.display = 'none';
  if (s.state === 'building') {
    if (!buildStart) buildStart = Date.now();
    title.textContent = `正在开启仿真（${s.scene_key}）`;
    sub.textContent = 'SUMO 世界正在分阶段构建，完成后自动进入转播。';
    const prog = s.progress || [];
    prog.forEach((p, i) => {
      const li = document.createElement('li');
      li.className = i === prog.length - 1 ? 'active' : 'done';
      li.textContent = p.msg;
      list.appendChild(li);
    });
    if (!prog.length) { const li = document.createElement('li'); li.className = 'active'; li.textContent = '排队等待构建…'; list.appendChild(li); }
    startBtn.style.display = 'none';
    cancelBtn.style.display = 'inline-block';
    document.getElementById('startSceneSel').style.display = 'none';
    pill.textContent = '构建中'; pill.className = 'pill pause';
  } else if (s.state === 'stopping') {
    buildStart = null;
    title.textContent = '正在关闭仿真…';
    sub.textContent = '释放 SUMO 世界与推流线程。';
    startBtn.style.display = 'none';
    document.getElementById('startSceneSel').style.display = 'none';
    pill.textContent = '关闭中'; pill.className = 'pill pause';
  } else if (s.state === 'error') {
    buildStart = null;
    title.textContent = '开启仿真失败';
    sub.textContent = '构建过程出错，可修正后重试；错误详情如下。';
    err.style.display = 'block';
    err.textContent = s.error || '未知错误';
    startBtn.style.display = 'inline-block'; startBtn.textContent = '重试开启';
    document.getElementById('startSceneSel').style.display = '';
    pill.textContent = '启动失败'; pill.className = 'pill pause';
  } else {  // idle
    buildStart = null;
    title.textContent = '仿真未运行';
    sub.textContent = '选择仿真包并开启后才会加载 SUMO 世界、占用计算资源；服务本身常驻。';
    startBtn.style.display = 'inline-block'; startBtn.textContent = '开启仿真';
    document.getElementById('startSceneSel').style.display = '';
    pill.textContent = '空闲'; pill.className = 'pill pause';
  }
}
setInterval(() => {
  if (sessionState === 'building')
    document.getElementById('sessionElapsed').textContent = fmtElapsed();
  else document.getElementById('sessionElapsed').textContent = '';
}, 1000);
function onStatus(msg) { renderSession(msg.session || {}); }

// ---------- events ----------
let currentSceneKey = null;
function onFrame(f) {
  // Frames are only published while the session is live: the server switches
  // from 'status' messages to frames on the building -> live transition, so
  // an arriving frame is the only signal that the build finished.  Recover
  // the client state here, otherwise the overlay would stay in 'building'
  // forever and the live-branch reInit() (needed when the page was opened
  // before the world existed) would never run.
  if (sessionState !== 'live')
    renderSession({state: 'live', scene_key: f.scene_key || currentSceneKey});
  if (currentSceneKey !== null && f.scene_key && f.scene_key !== currentSceneKey
      && !f.rebuilding) {
    reInit();  // scene was switched server-side: rebuild geometry/panel
    return;
  }
  lastFrame = f;
  if (renderer) renderer.pushFrame(f);
  updateSidebar(f);
}
document.getElementById('pauseBtn').onclick = () => post(lastFrame && lastFrame.paused ? 'resume' : 'pause');
document.getElementById('speedSel').onchange = e => post('speed', {value: Number(e.target.value)});
document.getElementById('waitChk').onchange = e => post('wait', {value: e.target.checked});
document.getElementById('resetBtn').onclick = () => { if (confirm('重置仿真？')) post('reset'); };
document.getElementById('controllerSelect').onchange = e => post('switch', {controller: e.target.value});
document.getElementById('interSel').onchange = e => setFocus(e.target.value);
document.getElementById('releaseBtn').onclick = () => post('release', {junction: INTER_ID});
document.getElementById('applyBtn').onclick = () => post('resolve', {});
window.addEventListener('resize', () => { if (renderer) renderer.resize(); });

// ---------- boot ----------
let es = null;
function applyInit(init) {
  INIT = init;
  window.CTLMETA = Object.fromEntries((init.controllers || []).map(c => [c.id, c]));
  currentSceneKey = init.scene_key || init.scene;
  document.getElementById('scene').textContent = init.scene || init.scene_key || '';
  // scene pickers (toolbar + overlay) always available
  for (const sid of ['sceneSel', 'startSceneSel']) {
    const sel = document.getElementById(sid); sel.innerHTML = '';
    (init.scenes || []).forEach(s => {
      const o = document.createElement('option'); o.value = s.id;
      o.textContent = s.id + ' · ' + s.network; sel.appendChild(o);
    });
    sel.value = currentSceneKey;
  }
  const sel = document.getElementById('controllerSelect');
  sel.innerHTML = '';
  init.controllers.forEach(c => {
    const o = document.createElement('option'); o.value = c.id;
    o.textContent = c.label + (c.kind === 'snapshot' ? '（快照模型）' : '');
    o.title = c.path || '';
    sel.appendChild(o);
  });
  sel.value = init.default_controller;
  document.getElementById('speedSel').value = String(init.speed);
  // World-dependent payload only exists once a session is built.
  if (!init.network || !init.panels) return;
  CATALOG = init.intersections; LANES = init.network.lanes;
  laneById = Object.fromEntries(LANES.map(l => [l.id, l]));
  BOUNDS = init.network.bounds;
  PANELS = init.panels;
  CONTROLLED = init.controlled || Object.keys(PANELS);
  if (renderer) renderer.setNetwork(init.network);
  const interSel = document.getElementById('interSel');
  interSel.innerHTML = '';
  CONTROLLED.forEach((tl, i) => {
    const o = document.createElement('option'); o.value = tl;
    o.textContent = CONTROLLED.length > 1 ? `${i} · ${tl}` : tl;
    interSel.appendChild(o);
  });
  document.getElementById('interSel').style.display = '';
  setFocus(init.focus || CONTROLLED[0]);
}
function reInit() {
  lastFrame = null;
  fetch('/api/init').then(r => r.json()).then(applyInit);
}
function boot() {
  const mount = document.getElementById('mapHost');
  const params = new URLSearchParams(location.search);
  const kind = window.SLB.pickRendererKind(params.get('renderer'));
  renderer = window.SLB.createRenderer(kind, mount, {
    onJunctionClick: id => setFocus(id),
  });
  const zoom = params.get('zoom');
  if (zoom !== null && renderer.setZoom) renderer.setZoom(zoom);
  const hint = document.getElementById('viewHint');
  const other = kind === 'deckgl' ? 'canvas' : 'deckgl';
  hint.innerHTML = `滚轮缩放 · 拖动平移 · 渲染:${kind}
    <a href="?renderer=${other}" style="color:inherit">切换到 ${other}</a>`;
  fetch('/api/init').then(r => r.json()).then(init => {
    applyInit(init);
    renderSession(init.session || {state: 'idle', scene_key: init.scene_key});
    es = new EventSource('/api/stream');
    es.onmessage = e => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'status') onStatus(msg); else onFrame(msg);
    };
    es.onerror = () => { document.getElementById('conn').textContent = 'SSE 重连中'; };
  });
}
document.getElementById('sceneSel').onchange = e => {
  document.getElementById('statePill').textContent = '场景重建中';
  document.getElementById('statePill').className = 'pill pause';
  post('scene', {scene: e.target.value});
};
document.getElementById('startBtn').onclick = () =>
  post('start', {scene: document.getElementById('startSceneSel').value});
document.getElementById('stopBtn').onclick = () =>
  { if (confirm('关闭仿真并释放资源？')) post('stop'); };
document.getElementById('cancelBuildBtn').onclick = () => post('cancel');

boot();
