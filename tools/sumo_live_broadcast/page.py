"""Browser page shell for the live SUMO broadcast.

The HTML/CSS skeleton lives here; behaviour is split into static modules
served from ``static/``:

- ``js/app.js`` — session UI, SSE frames, control ops, phase panel, history;
- ``js/renderer.js`` — renderer contract + factory (``window.SLB``);
- ``js/renderer_canvas.js`` / ``js/renderer_deckgl.js`` — swappable map
  renderers implementing the same contract (``?renderer=`` picks one).

The map renderers own frame interpolation between the one-frame-per-second
SSE pushes, so the stream still renders as continuous real-world motion.
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
  #interSel { max-width:210px; padding:2px 4px; }
  .pinTag { font-size:11px; color:#fff; background:#7b1fa2; border-radius:8px; padding:0 6px; margin-left:6px; }
  #releaseBtn { padding:1px 8px; font-size:12px; margin-left:6px; }
  .jlabel { font-size:10px; }
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
  <div id="mapWrap"><div id="mapHost" style="position:absolute;inset:0"></div><div class="viewHint" id="viewHint" style="position:absolute;left:10px;bottom:8px;font-size:12px;color:#6a7a84;z-index:5">滚轮缩放 · 拖动平移</div></div>
  <aside id="side">
    <section id="decisionBox">
      <h2>决策状态</h2>
      <div class="prop"><label>路口 <select id="interSel"></select></label><span id="manualTag" class="pinTag" style="display:none">人工接管</span><button id="releaseBtn" style="display:none">解除接管</button></div>
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
<script src="/static/vendor/deck.min.js"></script>
<script src="/static/js/renderer.js"></script>
<script src="/static/js/renderer_canvas.js"></script>
<script src="/static/js/renderer_deckgl.js"></script>
<script src="/static/js/app.js" defer></script>
</body>
</html>
"""


def build_page():
    return PAGE_HTML
