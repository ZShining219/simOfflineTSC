/* deck.gl 渲染器 —— WebGL 图层化地图呈现（默认实现）。
 * 接口契约见 renderer.js；依赖 window.deck（vendored standalone bundle）。
 *
 * 坐标约定：SUMO 路网 y 向北为正，deck.gl 直角坐标系 y 向屏幕下方为正，
 * 因此全部输入点先做 [x - cx, cy - y] 变换（居中 + 翻转为北上）。
 * 车辆朝向角为 SUMO 语义（0=北, 顺时针到 90=东）。
 */
window.SLBRenderers = window.SLBRenderers || {};

window.SLBRenderers.deckgl = function (mount, opts) {
  const deckgl = window.deck;
  if (!deckgl) throw new Error('deck.gl bundle not loaded');

  let deckInst = null, NET = null, LANES = [], JUNCTIONS = [], BOUNDS = null;
  let CX = 0, CY = 0;
  let lanePaths = null;                       // 静态几何，只构建一次
  let lastFrame = null, prevFrame = null, frameStamp = [0, 0];
  let focusId = null, raf = null, pendingZoom = null;

  const W = (x, y) => [x - CX, CY - y];       // world -> deck coords

  function fitZoom() {
    const w = mount.clientWidth || 1, h = mount.clientHeight || 1;
    if (!BOUNDS) return 0;
    const bw = Math.max(1, BOUNDS.max_x - BOUNDS.min_x);
    const bh = Math.max(1, BOUNDS.max_y - BOUNDS.min_y);
    return Math.log2(Math.max(0.05, Math.min((w - 40) / bw, (h - 40) / bh)));
  }
  function ensureDeck() {
    if (deckInst) return;
    deckInst = new deckgl.Deck({
      parent: mount,
      views: new deckgl.OrthographicView({
        id: 'map',
        controller: {inertia: 160, doubleClickZoom: true, touchRotate: false},
      }),
      initialViewState: {target: [0, 0, 0], zoom: fitZoom()},
      layers: [],
      getTooltip: null,
    });
  }
  function vehiclePolygons(vehs) {
    // 朝向向量（deck 坐标：x 东，y 南）。h: 0=北, 90=东。
    return vehs.map(v => {
      const h = v.angle * Math.PI / 180;
      const fx = Math.sin(h), fy = -Math.cos(h);      // 前
      const sx = Math.cos(h), sy = Math.sin(h);       // 右
      const p = W(v.x, v.y), L = 2.45, Hw = 1.05;
      return {
        id: v.id, speed: v.speed, c: p,
        polygon: [
          [p[0] + fx * L + sx * Hw, p[1] + fy * L + sy * Hw],
          [p[0] + fx * L - sx * Hw, p[1] + fy * L - sy * Hw],
          [p[0] - fx * L - sx * Hw, p[1] - fy * L - sy * Hw],
          [p[0] - fx * L + sx * Hw, p[1] - fy * L + sy * Hw],
        ],
      };
    });
  }
  function speedColor(s) {
    return s < 0.1 ? [229, 57, 53] : s < 4 ? [240, 162, 2] : [25, 118, 210];
  }
  function lightRGB(state) {
    const active = (state || '').split('').find(c => c.toLowerCase() !== 'r') || 'r';
    const c = active.toLowerCase();
    return c === 'g' ? [32, 160, 80] : c === 'y' ? [224, 161, 0] : [213, 57, 57];
  }
  function draw() {
    raf = requestAnimationFrame(draw);
    if (!deckInst || !NET) return;
    const lights = (lastFrame && lastFrame.lights) || {};
    const sigJ = [], plainJ = [];
    for (const j of JUNCTIONS)
      (lights[j.id] ? sigJ : plainJ).push(j);
    const vehs = window.SLB.interpVehicles(
      prevFrame, lastFrame, frameStamp[0], frameStamp[1]);
    const polys = vehiclePolygons(vehs);
    const jammed = polys.filter(v => v.speed < 0.1);
    const focusJ = JUNCTIONS.filter(j => j.id === focusId);

    deckInst.setProps({layers: [
      new deckgl.PathLayer({
        id: 'lane-casing', data: lanePaths, pickable: false,
        getPath: d => d.pts,
        getWidth: d => d.internal ? 1.8 : 4.4,
        widthUnits: 'meters',
        getColor: d => d.internal ? [143, 160, 168] : [52, 66, 74],
        parameters: {depthTest: false},
      }),
      new deckgl.PathLayer({
        id: 'lane-surface', data: lanePaths, pickable: false,
        getPath: d => d.pts,
        getWidth: d => d.internal ? 1.2 : 3.2,
        widthUnits: 'meters',
        getColor: d => d.internal ? [199, 208, 213] : [105, 119, 128],
        parameters: {depthTest: false},
      }),
      new deckgl.ScatterplotLayer({   // 非信号路口：小灰点
        id: 'plain-junctions', data: plainJ, pickable: false,
        getPosition: d => W(d.x, d.y), getRadius: 4,
        radiusUnits: 'pixels',
        getFillColor: [143, 160, 168, 170],
        parameters: {depthTest: false},
      }),
      new deckgl.ScatterplotLayer({   // 信号路口光晕
        id: 'junction-halo', data: sigJ, pickable: false,
        getPosition: d => W(d.x, d.y), getRadius: 17,
        radiusUnits: 'pixels',
        getFillColor: d => lightRGB((lights[d.id] || [])[1]).concat(55),
        parameters: {depthTest: false},
      }),
      new deckgl.ScatterplotLayer({   // 信号路口主盘（可点选）
        id: 'junctions', data: sigJ, pickable: true,
        getPosition: d => W(d.x, d.y), getRadius: 9,
        radiusUnits: 'pixels',
        stroked: true, getLineColor: [23, 32, 42], lineWidthMinPixels: 1.5,
        getFillColor: d => lightRGB((lights[d.id] || [])[1]),
        onClick: info => {
          if (info.object && opts.onJunctionClick)
            opts.onJunctionClick(info.object.id);
        },
        parameters: {depthTest: false},
      }),
      new deckgl.ScatterplotLayer({   // 焦点路口紫圈
        id: 'focus-ring', data: focusJ, pickable: false,
        getPosition: d => W(d.x, d.y), getRadius: 20,
        radiusUnits: 'pixels',
        stroked: true, filled: false,
        getLineColor: [123, 31, 162], lineWidthMinPixels: 3,
        parameters: {depthTest: false},
      }),
      new deckgl.ScatterplotLayer({   // 拥堵车辆红色光晕
        id: 'jam-halo', data: jammed, pickable: false,
        getPosition: d => d.c,
        getRadius: 13, radiusUnits: 'pixels',
        getFillColor: [229, 57, 53, 45],
        parameters: {depthTest: false},
      }),
      new deckgl.SolidPolygonLayer({  // 车辆（真实尺寸，放大后呈矩形车体）
        id: 'vehicles', data: polys, pickable: false,
        getPolygon: d => d.polygon,
        getFillColor: d => speedColor(d.speed).concat(235),
        getLineColor: d => speedColor(d.speed).map(c => Math.max(0, c - 80)),
        stroked: true, lineWidthMinPixels: 0.8,
        parameters: {depthTest: false},
      }),
      new deckgl.ScatterplotLayer({   // 车辆像素底点：保证任意缩放可见
        id: 'veh-dots', data: polys, pickable: false,
        getPosition: d => d.c, getRadius: 5,
        radiusUnits: 'pixels',
        getFillColor: d => speedColor(d.speed),
        parameters: {depthTest: false},
      }),
      new deckgl.TextLayer({          // 相位标签 P#
        id: 'phase-tags', data: sigJ, pickable: false,
        getPosition: d => W(d.x, d.y),
        getText: d => 'P' + ((lights[d.id] || [])[0] ?? '-'),
        getPixelOffset: [16, -9], getSize: 12, sizeUnits: 'pixels',
        fontWeight: 'bold', characterSet: 'auto',
        background: true, getBackgroundColor: [255, 255, 255, 215],
        backgroundPadding: [4, 2],
        parameters: {depthTest: false},
      }),
      new deckgl.TextLayer({          // 路口 id
        id: 'junction-ids', data: sigJ, pickable: false,
        getPosition: d => W(d.x, d.y),
        getText: d => String(d.id).length > 16
          ? String(d.id).slice(0, 15) + '…' : String(d.id),
        getPixelOffset: [0, 15], getSize: 10, sizeUnits: 'pixels',
        getColor: [69, 81, 90], characterSet: 'auto',
        background: true, getBackgroundColor: [255, 255, 255, 190],
        backgroundPadding: [3, 1],
        parameters: {depthTest: false},
      }),
    ]});
  }
  raf = requestAnimationFrame(draw);

  return {
    kind: 'deckgl',
    setNetwork(network) {
      NET = network;
      LANES = (network && network.lanes) || [];
      JUNCTIONS = (network && network.junctions) || [];
      BOUNDS = network && network.bounds;
      if (BOUNDS) {
        CX = (BOUNDS.min_x + BOUNDS.max_x) / 2;
        CY = (BOUNDS.min_y + BOUNDS.max_y) / 2;
      }
      lanePaths = LANES.map(l => ({internal: l.internal, pts: l.shape.map(p => W(p[0], p[1]))}));
      prevFrame = null; lastFrame = null;
      ensureDeck();
      if (deckInst) pendingZoom !== null ? this.setZoom(pendingZoom) : this.fit();
    },
    pushFrame(f) {
      prevFrame = lastFrame; lastFrame = f;
      frameStamp = [frameStamp[1], performance.now()];
    },
    setFocus(id) { focusId = id; },
    fit() {
      if (deckInst)
        deckInst.setProps({initialViewState: {target: [0, 0, 0], zoom: fitZoom()}});
    },
    setZoom(z) {   // 调试/深链用：?zoom=
      pendingZoom = z;
      if (deckInst)
        deckInst.setProps({initialViewState: {target: [0, 0, 0], zoom: Number(z)}});
    },
    resize() { /* deck.gl 自适应容器尺寸 */ },
    dispose() {
      cancelAnimationFrame(raf);
      if (deckInst) { deckInst.finalize(); deckInst = null; }
    },
  };
};
