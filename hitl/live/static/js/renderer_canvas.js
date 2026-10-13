/* Canvas 2D 渲染器 —— 原内联实现的模块化封装，作为无 deck.gl 依赖时的保底。
 * 接口契约见 renderer.js。坐标系：世界 y 向上（北），canvas y 向下。
 */
window.SLBRenderers = window.SLBRenderers || {};

window.SLBRenderers.canvas = function (mount, opts) {
  const map = document.createElement('canvas');
  map.id = 'map';
  map.style.cssText = 'width:100%;height:100%;display:block;background:#f4f7f8;';
  mount.appendChild(map);

  const view = {zoom: 1, panX: 0, panY: 0, drag: false, lx: 0, ly: 0};
  let NET = null, CATALOG = {}, LANES = [], BOUNDS = null;
  let lastFrame = null, prevFrame = null, frameStamp = [0, 0];
  let focusId = null, raf = null, downPos = null;

  function mapScale() {
    const m = 20;
    return Math.min((map.width - 2 * m) / Math.max(1, BOUNDS.max_x - BOUNDS.min_x),
                    (map.height - 2 * m) / Math.max(1, BOUNDS.max_y - BOUNDS.min_y));
  }
  function mapPt(x, y) {
    const m = 20, s = mapScale();
    return [m + (x - BOUNDS.min_x) * s, map.height - m - (y - BOUNDS.min_y) * s];
  }
  function trace(ctx, shape) {
    ctx.beginPath();
    shape.forEach((p, i) => {
      const q = mapPt(p[0], p[1]);
      i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]);
    });
  }
  function drawLane(ctx, lane) {
    const s = mapScale(), w = Math.max(2, (lane.internal ? 1.6 : 3.2) * s);
    trace(ctx, lane.shape);
    ctx.strokeStyle = lane.internal ? '#8fa0a8' : '#34424a';
    ctx.lineWidth = w + (lane.internal ? 1 : 4);
    ctx.stroke();
    trace(ctx, lane.shape);
    ctx.strokeStyle = lane.internal ? '#c3ccd1' : '#697780';
    ctx.lineWidth = w;
    ctx.stroke();
  }
  function drawLights(ctx, lights) {
    for (const j of NET.junctions) {
      const st = lights[j.id];
      if (!st) continue;
      const [x, y] = mapPt(j.x, j.y);
      const col = window.SLB.lightColor(st[1]);
      ctx.fillStyle = col + '44';
      ctx.beginPath(); ctx.arc(x, y, 16, 0, 7); ctx.fill();
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(x, y, 10, 0, 7); ctx.fill();
      ctx.strokeStyle = '#17202a'; ctx.lineWidth = 1.6; ctx.stroke();
      ctx.font = 'bold 11px Arial';
      const tag = 'P' + (st[0] ?? '-');
      ctx.fillStyle = '#ffffffdd';
      const tw = ctx.measureText(tag).width + 8;
      ctx.fillRect(x + 12, y - 8, tw, 15);
      ctx.strokeStyle = '#536875'; ctx.strokeRect(x + 12, y - 8, tw, 15);
      ctx.fillStyle = '#17202a'; ctx.fillText(tag, x + 16, y + 4);
      const short = String(j.id).length > 16 ? String(j.id).slice(0, 15) + '…' : String(j.id);
      ctx.font = '10px Arial';
      const iw = ctx.measureText(short).width + 6;
      ctx.fillStyle = '#ffffffcc'; ctx.fillRect(x - iw / 2, y + 12, iw, 13);
      ctx.strokeStyle = j.id === focusId ? '#7b1fa2' : '#8fa0a8';
      ctx.lineWidth = j.id === focusId ? 2 : 0.8;
      ctx.strokeRect(x - iw / 2, y + 12, iw, 13);
      ctx.fillStyle = '#45515a'; ctx.fillText(short, x - iw / 2 + 3, y + 22);
    }
  }
  function drawVehicle(ctx, v) {
    const [x, y] = mapPt(v.x, v.y);
    const a = (90 - v.angle) * Math.PI / 180, jam = v.speed < 0.1;
    ctx.save(); ctx.translate(x, y); ctx.rotate(a);
    if (jam) {
      ctx.fillStyle = '#e5393533';
      ctx.beginPath(); ctx.arc(0, 0, 20, 0, 7); ctx.fill();
    }
    ctx.fillStyle = jam ? '#e53935' : '#1976d2';
    ctx.strokeStyle = jam ? '#8f1d1d' : '#0d3c68';
    ctx.lineWidth = 1.4;
    ctx.fillRect(-13, -7, 26, 14); ctx.strokeRect(-13, -7, 26, 14);
    ctx.fillStyle = jam ? '#ffd1d1' : '#cce8ff';
    ctx.fillRect(-2, -5, 9, 10);
    ctx.restore();
  }
  function loop() {
    raf = requestAnimationFrame(loop);
    const ctx = map.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, map.width, map.height);
    if (!NET || !LANES.length) return;
    ctx.fillStyle = '#f4f7f8'; ctx.fillRect(0, 0, map.width, map.height);
    ctx.translate(view.panX, view.panY);
    ctx.translate(map.width / 2, map.height / 2);
    ctx.scale(view.zoom, view.zoom);
    ctx.translate(-map.width / 2, -map.height / 2);
    LANES.forEach(l => drawLane(ctx, l));
    if (lastFrame) {
      drawLights(ctx, lastFrame.lights);
      window.SLB.interpVehicles(prevFrame, lastFrame, frameStamp[0], frameStamp[1])
        .forEach(v => drawVehicle(ctx, v));
    }
  }
  function pickJunction(e) {
    if (!NET || !BOUNDS) return;
    const rect = map.getBoundingClientRect();
    const dpr = devicePixelRatio || 1;
    const px = (e.clientX - rect.left) * dpr, py = (e.clientY - rect.top) * dpr;
    let best = null, bd = Infinity;
    for (const j of NET.junctions) {
      const q = mapPt(j.x, j.y);
      const sx = (q[0] - map.width / 2) * view.zoom + map.width / 2 + view.panX;
      const sy = (q[1] - map.height / 2) * view.zoom + map.height / 2 + view.panY;
      const d = Math.hypot(sx - px, sy - py);
      if (d < bd) { bd = d; best = j.id; }
    }
    if (best && bd < 36 * dpr * Math.max(0.6, view.zoom)
        && opts.onJunctionClick) opts.onJunctionClick(best);
  }

  map.addEventListener('wheel', e => {
    e.preventDefault();
    const k = e.deltaY < 0 ? 1.15 : 0.87;
    view.zoom = Math.min(12, Math.max(0.4, view.zoom * k));
  }, {passive: false});
  map.addEventListener('pointerdown', e => {
    view.drag = true; view.lx = e.clientX; view.ly = e.clientY;
    downPos = [e.clientX, e.clientY]; map.setPointerCapture(e.pointerId);
  });
  map.addEventListener('pointermove', e => {
    if (!view.drag) return;
    view.panX += e.clientX - view.lx; view.panY += e.clientY - view.ly;
    view.lx = e.clientX; view.ly = e.clientY;
  });
  map.addEventListener('pointerup', e => {
    view.drag = false;
    if (downPos && Math.hypot(e.clientX - downPos[0], e.clientY - downPos[1]) < 5)
      pickJunction(e);
    downPos = null;
  });

  raf = requestAnimationFrame(loop);
  return {
    kind: 'canvas',
    setNetwork(network) {
      NET = network; CATALOG = {};
      LANES = (network && network.lanes) || [];
      BOUNDS = network && network.bounds;
      prevFrame = null; lastFrame = null;
      this.resize();
    },
    pushFrame(f) {
      prevFrame = lastFrame; lastFrame = f;
      frameStamp = [frameStamp[1], performance.now()];
    },
    setFocus(id) { focusId = id; },
    fit() { view.zoom = 1; view.panX = 0; view.panY = 0; },
    setZoom(z) { view.zoom = Number(z); },
    resize() {
      map.width = map.clientWidth * devicePixelRatio;
      map.height = map.clientHeight * devicePixelRatio;
    },
    dispose() {
      cancelAnimationFrame(raf); map.remove();
    },
  };
};
