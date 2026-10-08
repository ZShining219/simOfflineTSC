"""Self-contained browser page for the live SUMO broadcast.

The page keeps two copies of the last frames pushed over SSE and interpolates
vehicle positions between them, so the one-frame-per-sim-second stream still
renders as continuous real-world motion.  The right-hand panel lists every
selectable green phase as a mini junction diagram, marks the phase currently
in effect and the controller's latest target, and keeps a short decision
history.
"""

from __future__ import annotations

import json


PAGE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>SUMO 实时转播</title>
<style>
  :root { --ink:#1d2a33; --line:#c9d4da; --accent:#1976d2; --warn:#e0a100; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 "Segoe UI", Arial, "PingFang SC", sans-serif; color:var(--ink); background:#eef2f4; height:100vh; display:flex; flex-direction:column; }
  header { display:flex; align-items:center; gap:14px; padding:8px 16px; background:#17202a; color:#fff; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  .pill { padding:2px 10px; border-radius:10px; font-size:12px; background:#37474f; }
  .pill.run { background:#2e7d32; } .pill.pause { background:#e0a100; color:#222; }
  .pill.await { background:#b71c1c; animation:blink 1s infinite alternate; }
  @keyframes blink { from{opacity:.55} to{opacity:1} }
  #toolbar { display:flex; align-items:center; gap:16px; padding:6px 16px; background:#fff; border-bottom:1px solid var(--line); flex-wrap:wrap; }
  #toolbar label { display:flex; align-items:center; gap:4px; }
  button { padding:4px 12px; border:1px solid var(--line); border-radius:4px; background:#fff; cursor:pointer; }
  button:hover { background:#eef4f8; }
  main { flex:1; display:flex; min-height:0; }
  #mapWrap { flex:1; position:relative; min-width:0; }
  #map { width:100%; height:100%; display:block; background:#f4f7f8; }
  #side { width:330px; background:#fff; border-left:1px solid var(--line); display:flex; flex-direction:column; overflow:hidden; }
  #side section { padding:10px 12px; border-bottom:1px solid var(--line); }
  #side h2 { font-size:13px; margin:0 0 8px; color:#45515a; text-transform:uppercase; letter-spacing:.04em; }
  #controllerSelect { min-width:150px; }
  #phaseGrid { display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; }
  .phaseCard { position:relative; border:2px solid var(--line); border-radius:6px; padding:4px; text-align:center; cursor:pointer; background:#fafcfc; transition:border-color .15s, box-shadow .15s, transform .1s; }
  .phaseCard canvas { width:100%; display:block; }
  .phaseCard .plabel { font-size:11px; color:#45515a; }
  .phaseCard .pstate { font-size:10px; color:#5a6a72; word-break:break-all; }
  .phaseCard .badge { position:absolute; top:3px; left:3px; font-size:10px; padding:0 5px; border-radius:8px; display:none; }
  .phaseCard.effect { border-color:#2e7d32; box-shadow:0 0 0 2px #2e7d3244; }
  .phaseCard.effect .badge.bEffect { display:inline-block; background:#2e7d32; color:#fff; }
  .phaseCard.target { border-color:var(--accent); box-shadow:0 0 0 2px #1976d244; }
  .phaseCard.target .badge.bTarget { display:inline-block; background:var(--accent); color:#fff; }
  .phaseCard.proposed { border-color:var(--warn); box-shadow:0 0 0 3px #e0a10055; animation:blink 1s infinite alternate; }
  .phaseCard.proposed .badge.bProp { display:inline-block; background:var(--warn); color:#222; }
  .phaseCard.chosen { border-color:#7b1fa2; box-shadow:0 0 0 3px #7b1fa255; transform:scale(1.04); }
  .phaseCard.chosen .badge.bChosen { display:inline-block; background:#7b1fa2; color:#fff; }
  .phaseCard.manualOk:hover { border-color:#7b1fa2; }
  #decisionBox .prop { font-size:13px; margin:4px 0; }
  #awaitBar { display:none; background:#fff8e1; border:1px solid #e0a100; border-radius:6px; padding:8px; margin-top:6px; }
  #awaitBar.show { display:block; }
  #metrics { display:grid; grid-template-columns:1fr 1fr; gap:4px 10px; font-size:13px; }
  #metrics strong { float:right; }
  #history { flex:1; overflow-y:auto; min-height:80px; }
  #history ul { list-style:none; margin:0; padding:0; font-size:12px; }
  #history li { padding:3px 6px; border-bottom:1px solid #eef2f4; display:flex; justify-content:space-between; }
  #history li em { color:#6a7a84; font-style:normal; }
  .srcManual { color:#7b1fa2; font-weight:600; }
  .err { color:#b71c1c; }
  #conn { margin-left:auto; font-size:12px; opacity:.8; }
  #sessionOverlay { position:fixed; inset:0; background:#f4f7f8f2; display:none; z-index:50; align-items:center; justify-content:center; }
  #sessionOverlay.show { display:flex; }
  #sessionCard { background:#fff; border:1px solid var(--line); border-radius:10px; padding:28px 34px; min-width:380px; max-width:560px; box-shadow:0 6px 30px #00000018; }
  #sessionCard h2 { margin:0 0 6px; font-size:18px; color:#243440; }
  #sessionCard .sub { color:#6a7a84; font-size:13px; margin-bottom:14px; }
  #sessionCard .errBox { background:#fdecea; border:1px solid #e0a1a1; color:#8f1d1d; border-radius:6px; padding:10px 12px; font-size:12px; font-family:monospace; white-space:pre-wrap; word-break:break-all; max-height:180px; overflow:auto; margin-bottom:12px; }
  #progList { list-style:none; margin:0 0 14px; padding:0; font-size:13px; }
  #progList li { padding:4px 0 4px 26px; position:relative; color:#45515a; }
  #progList li.done::before { content:'✓'; position:absolute; left:4px; color:#2e7d32; font-weight:700; }
  #progList li.active::before { content:'●'; position:absolute; left:4px; color:var(--accent); animation:blink 0.9s infinite alternate; }
  #progList li { color:#9aa8b0; }
  #progList li.done, #progList li.active { color:#243440; }
  #sessionCard .row { display:flex; gap:10px; align-items:center; }
  #sessionCard select { flex:1; padding:6px; font-size:14px; }
  #sessionElapsed { font-size:12px; color:#8b98a0; margin-left:auto; }
  .bigBtn { padding:9px 22px; font-size:14px; background:var(--accent); color:#fff; border:none; border-radius:6px; cursor:pointer; }
  .bigBtn:hover { filter:brightness(1.08); }
  .ghostBtn { padding:9px 16px; font-size:13px; background:#fff; color:#45515a; border:1px solid var(--line); border-radius:6px; cursor:pointer; }
</style>
</head>
<body>
<header>
  <h1>SUMO 实时转播 · <span id="scene"></span></h1>
  <span id="statePill" class="pill">连接中</span>
  <span id="conn">SSE</span>
</header>
<div id="sessionOverlay">
  <div id="sessionCard">
    <h2 id="sessTitle">仿真未运行</h2>
    <div class="sub" id="sessSub">开启后才加载 SUMO 世界并占用资源；关闭后回到空闲。</div>
    <ul id="progList"></ul>
    <div class="errBox" id="sessErr" style="display:none"></div>
    <div class="row">
      <select id="startSceneSel"></select>
      <button class="bigBtn" id="startBtn">开启仿真</button>
      <button class="ghostBtn" id="cancelBuildBtn" style="display:none">取消构建</button>
      <span id="sessionElapsed"></span>
    </div>
  </div>
</div>
<div id="toolbar">
  <label>仿真包 <select id="sceneSel"></select></label>
  <button id="pauseBtn">暂停</button>
  <label>速度
    <select id="speedSel">
      <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
      <option value="1" selected>1×(实时)</option>
      <option value="2">2×</option><option value="4">4×</option>
    </select>
  </label>
  <label>控制器 <select id="controllerSelect"></select></label>
  <label><input type="checkbox" id="waitChk"> 决策点暂停等待</label>
  <span>仿真时刻 <b id="simTime">-</b> · 距下次决策 <b id="nextIn">-</b>s</span>
  <button id="resetBtn">重置仿真</button>
  <button id="stopBtn" style="margin-left:auto;border-color:#c0392b;color:#c0392b">关闭仿真</button>
</div>
<main>
  <div id="mapWrap"><canvas id="map"></canvas><div class="viewHint" style="position:absolute;left:10px;bottom:8px;font-size:12px;color:#6a7a84">滚轮缩放 · 拖动平移</div></div>
  <aside id="side">
    <section id="decisionBox">
      <h2>决策状态</h2>
      <div class="prop">当前生效相位 <b id="phaseInEffect">-</b> · 目标相位 <b id="phaseTarget">-</b></div>
      <div class="prop">最近决策 <span id="lastDecision">-</span></div>
      <div class="prop" id="ctlSrc" style="font-size:11px;color:#6a7a84"></div>
      <div id="awaitBar">
        <div id="awaitText">等待决策…</div>
        <button id="applyBtn">应用并继续</button>
      </div>
    </section>
    <section>
      <h2>可选相位（点击 = 人工接管）</h2>
      <div id="phaseGrid"></div>
    </section>
    <section><h2>指标</h2><div id="metrics"></div></section>
    <section id="history"><h2>决策历史</h2><ul id="historyList"></ul></section>
  </aside>
</main>
<script>
let INIT = null, CATALOG = {}, LANES = [], laneById = {}, BOUNDS = null;
let PHASES = [], YELLOW_DICT = {}, PHASE_COUNT = 0, INTER_ID = null;
let lastFrame = null, prevFrame = null, frameStamp = [0, 0];
const map = document.getElementById('map');
const view = {zoom:1, panX:0, panY:0, drag:false, lx:0, ly:0};

function post(op, extra) {
  fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(Object.assign({op}, extra || {}))}).catch(() => {});
}

// ---------- geometry ----------
function mapScale() {
  const m = 20;
  return Math.min((map.width - 2*m) / Math.max(1, BOUNDS.max_x - BOUNDS.min_x),
                  (map.height - 2*m) / Math.max(1, BOUNDS.max_y - BOUNDS.min_y));
}
function mapPt(x, y) {
  const m = 20, s = mapScale();
  return [m + (x - BOUNDS.min_x) * s, map.height - m - (y - BOUNDS.min_y) * s];
}
function trace(ctx, shape) {
  ctx.beginPath();
  shape.forEach((p, i) => { const q = mapPt(p[0], p[1]); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
}
function drawLane(ctx, lane) {
  const s = mapScale(), w = Math.max(2, (lane.internal ? 1.6 : 3.2) * s);
  trace(ctx, lane.shape);
  ctx.strokeStyle = lane.internal ? '#8fa0a8' : '#34424a'; ctx.lineWidth = w + (lane.internal ? 1 : 4); ctx.stroke();
  trace(ctx, lane.shape);
  ctx.strokeStyle = lane.internal ? '#c3ccd1' : '#697780'; ctx.lineWidth = w; ctx.stroke();
}
function drawLights(ctx, lights) {
  for (const j of INIT.network.junctions) {
    const st = lights[j.id]; if (!st) continue;
    const [x, y] = mapPt(j.x, j.y);
    const active = (st[1] || '').split('').find(c => c.toLowerCase() !== 'r') || 'r';
    const col = active.toLowerCase() === 'g' ? '#20a050' : active.toLowerCase() === 'y' ? '#e0a100' : '#d53939';
    ctx.fillStyle = col + '44'; ctx.beginPath(); ctx.arc(x, y, 16, 0, 7); ctx.fill();
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, 10, 0, 7); ctx.fill();
    ctx.strokeStyle = '#17202a'; ctx.lineWidth = 1.6; ctx.stroke();
    ctx.font = 'bold 11px Arial'; ctx.fillStyle = '#17202a';
    const tag = 'P' + (st[0] ?? '-');
    ctx.fillStyle = '#ffffffdd'; const tw = ctx.measureText(tag).width + 8;
    ctx.fillRect(x + 12, y - 8, tw, 15); ctx.strokeStyle = '#536875'; ctx.strokeRect(x + 12, y - 8, tw, 15);
    ctx.fillStyle = '#17202a'; ctx.fillText(tag, x + 16, y + 4);
  }
}
function drawVehicle(ctx, v) {
  const [x, y] = mapPt(v.x, v.y);
  const a = (90 - v.angle) * Math.PI / 180, jam = v.speed < 0.1;
  ctx.save(); ctx.translate(x, y); ctx.rotate(a);
  if (jam) { ctx.fillStyle = '#e5393533'; ctx.beginPath(); ctx.arc(0, 0, 20, 0, 7); ctx.fill(); }
  ctx.fillStyle = jam ? '#e53935' : '#1976d2'; ctx.strokeStyle = jam ? '#8f1d1d' : '#0d3c68';
  ctx.lineWidth = 1.4; ctx.fillRect(-13, -7, 26, 14); ctx.strokeRect(-13, -7, 26, 14);
  ctx.fillStyle = jam ? '#ffd1d1' : '#cce8ff'; ctx.fillRect(-2, -5, 9, 10);
  ctx.restore();
}
function interpVehicles() {
  if (!lastFrame) return [];
  const cur = new Map(lastFrame.vehicles.map(v => [v[0], v]));
  if (!prevFrame) return [...cur.values()].map(v => ({x: v[1], y: v[2], angle: v[3], speed: v[4]}));
  const prev = new Map(prevFrame.vehicles.map(v => [v[0], v]));
  const span = Math.max(1, frameStamp[1] - frameStamp[0]);
  const alpha = Math.min(1, (performance.now() - frameStamp[1]) / span + 0);
  const out = [];
  for (const [id, v] of cur) {
    const p = prev.get(id);
    if (!p || span <= 0) { out.push({x: v[1], y: v[2], angle: v[3], speed: v[4]}); continue; }
    out.push({x: p[1] + (v[1] - p[1]) * alpha, y: p[2] + (v[2] - p[2]) * alpha,
              angle: v[3], speed: v[4]});
  }
  return out;
}
function render() {
  const ctx = map.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, map.width, map.height);
  if (!INIT || !INIT.network || !LANES.length) { requestAnimationFrame(render); return; }
  ctx.fillStyle = '#f4f7f8'; ctx.fillRect(0, 0, map.width, map.height);
  ctx.translate(view.panX, view.panY);
  ctx.translate(map.width/2, map.height/2); ctx.scale(view.zoom, view.zoom);
  ctx.translate(-map.width/2, -map.height/2);
  LANES.forEach(l => drawLane(ctx, l));
  if (lastFrame) {
    drawLights(ctx, lastFrame.lights);
    interpVehicles().forEach(v => drawVehicle(ctx, v));
  }
  requestAnimationFrame(render);
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
      post('phase', {phase: p.index});
    };
    card.title = `P${p.index}: ${p.state}（点击切换，自动转人工接管）`;
  });
}
function updatePhaseCards(f) {
  if (pendingPhase !== null && f.held_action === pendingPhase) pendingPhase = null;
  const inEffect = f.current_phase_raw < PHASE_COUNT ? f.current_phase_raw : null;
  const via = YELLOW_DICT && f.current_phase_raw >= PHASE_COUNT
    ? Object.entries(YELLOW_DICT).find(e => e[1] === f.current_phase_raw) : null;
  document.getElementById('phaseInEffect').textContent =
    inEffect !== null ? 'P' + inEffect : (via ? '黄灯 ' + via[0].replace('_', '→') : 'P' + f.current_phase_raw);
  document.getElementById('phaseTarget').textContent = 'P' + f.held_action;
  document.querySelectorAll('.phaseCard').forEach(c => {
    const i = Number(c.dataset.phase);
    c.classList.toggle('effect', i === inEffect);
    c.classList.toggle('target', i === f.held_action);
    c.classList.toggle('proposed', Boolean(f.awaiting) && i === f.proposed_action);
    c.classList.toggle('chosen', pendingPhase !== null && i === pendingPhase && i !== f.held_action);
    c.classList.toggle('manualOk', f.controller.kind === 'manual' || Boolean(f.awaiting));
  });
}
function yellowPairs() { return YELLOW_DICT; }

// ---------- sidebar ----------
function fmtHistory(f) {
  const ul = document.getElementById('historyList'); ul.innerHTML = '';
  (f.history || []).slice().reverse().forEach(h => {
    const li = document.createElement('li');
    if (h.event) { li.innerHTML = `<em>t=${h.t.toFixed(0)}s</em><span>${h.event}</span>`; }
    else {
      const src = h.source === 'manual' ? '<span class="srcManual">人工</span>' : `<span>${h.source}</span>`;
      li.innerHTML = `<em>t=${h.t.toFixed(0)}s #${h.decision}</em><span>P${h.action} ${src}${h.error ? ' <b class="err">!</b>' : ''}</span>`;
      if (h.note) li.title = h.note;
    }
    ul.appendChild(li);
  });
}
function updateSidebar(f) {
  updatePhaseCards(f);
  document.getElementById('simTime').textContent = f.t.toFixed(0) + 's';
  document.getElementById('nextIn').textContent = f.awaiting ? '暂停等待' : f.next_decision_in.toFixed(0);
  const last = (f.history || []).filter(h => h.action !== undefined).slice(-1)[0];
  document.getElementById('lastDecision').textContent = last
    ? `t=${last.t.toFixed(0)}s → P${last.action}（${last.source}）` : '-';
  const meta = (window.CTLMETA || {})[f.controller.id] || {};
  const ident = (meta.detail && meta.detail.identity) || {};
  const src = meta.path ? meta.path.split('/').slice(-1)[0] : '';
  document.getElementById('ctlSrc').textContent = f.controller.kind === 'snapshot'
    ? `权重: ${src}${ident.global_episode ? ' · ep' + ident.global_episode : ''}${ident.logical_run_id ? ' · ' + ident.logical_run_id : ''}`
    : '';
  const box = document.getElementById('awaitBar');
  if (f.awaiting === 'confirm') {
    box.classList.add('show');
    document.getElementById('awaitText').textContent = `控制器 ${f.controller.label} 建议 P${f.proposed_action}，确认后生效`;
  } else if (f.awaiting === 'manual') {
    box.classList.add('show');
    document.getElementById('awaitText').textContent = '等待人工选择相位…';
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
  prevFrame = lastFrame; lastFrame = f;
  frameStamp = [frameStamp[1], performance.now()];
  updateSidebar(f);
}
function fitCanvas() {
  map.width = map.clientWidth * devicePixelRatio;
  map.height = map.clientHeight * devicePixelRatio;
}
map.addEventListener('wheel', e => {
  e.preventDefault();
  const k = e.deltaY < 0 ? 1.15 : 0.87;
  view.zoom = Math.min(12, Math.max(0.4, view.zoom * k));
});
map.addEventListener('pointerdown', e => { view.drag = true; view.lx = e.clientX; view.ly = e.clientY; map.setPointerCapture(e.pointerId); });
map.addEventListener('pointermove', e => { if (!view.drag) return; view.panX += e.clientX - view.lx; view.panY += e.clientY - view.ly; view.lx = e.clientX; view.ly = e.clientY; });
map.addEventListener('pointerup', () => view.drag = false);
document.getElementById('pauseBtn').onclick = () => post(lastFrame && lastFrame.paused ? 'resume' : 'pause');
document.getElementById('speedSel').onchange = e => post('speed', {value: Number(e.target.value)});
document.getElementById('waitChk').onchange = e => post('wait', {value: e.target.checked});
document.getElementById('resetBtn').onclick = () => { if (confirm('重置仿真？')) post('reset'); };
document.getElementById('controllerSelect').onchange = e => post('switch', {controller: e.target.value});
document.getElementById('applyBtn').onclick = () => post('resolve', {});
window.addEventListener('resize', fitCanvas);

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
  if (!init.network || !init.intersection) return;
  CATALOG = init.intersections; LANES = init.network.lanes;
  laneById = Object.fromEntries(LANES.map(l => [l.id, l]));
  BOUNDS = init.network.bounds;
  INTER_ID = init.intersection.id;
  PHASES = init.intersection.phases; PHASE_COUNT = init.intersection.phase_count;
  YELLOW_DICT = init.intersection.yellow_dict || {};
  buildPhaseCards();
}
function reInit() {
  prevFrame = null; lastFrame = null;
  fetch('/api/init').then(r => r.json()).then(applyInit);
}
fetch('/api/init').then(r => r.json()).then(init => {
  applyInit(init);
  renderSession(init.session || {state: 'idle', scene_key: init.scene_key});
  fitCanvas(); render();
  es = new EventSource('/api/stream');
  es.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'status') onStatus(msg); else onFrame(msg);
  };
  es.onerror = () => { document.getElementById('conn').textContent = 'SSE 重连中'; };
});
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
</script>
</body>
</html>
"""


def build_page():
    return PAGE_HTML
