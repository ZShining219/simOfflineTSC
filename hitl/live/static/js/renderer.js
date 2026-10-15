/* 地图渲染器契约 + 工厂。
 *
 * 隔离边界：渲染器对外只见两份输入 —— 静态路网 (setNetwork) 与逐帧动态状态
 * (pushFrame)。插值平滑、缩放/平移、路口点击命中、车辆/灯色的具体画法全部
 * 归渲染器内部实现；上层 (app.js) 只负责 UI/会话/相位面板/控制指令。
 *
 * 用法：
 *   var r = SLB.createRenderer(kind, mountEl, {onJunctionClick: fn});
 *
 * kind: 'deckgl' | 'canvas'；deckgl 依赖 window.deck（standalone bundle）。
 * 实现注册在 window.SLBRenderers 上，新增渲染器 = 写一个同签名工厂即可。
 *
 * 返回对象：
 *   kind                  实际渲染器名
 *   setNetwork(network)   {lanes, junctions, bounds}；场景切换时重发
 *   pushFrame(frame)      新 SSE 帧；内部持有 prev/cur 做插值
 *   setFocus(tlId)        高亮当前相位面板对应的路口；null 清除
 *   fit()                 视野复位到路网范围
 *   resize()              容器尺寸变化
 *   dispose()             销毁（场景切换/停止时调用）
 */
window.SLBRenderers = window.SLBRenderers || {};
window.SLB = window.SLB || {};

window.SLB.createRenderer = function (kind, mount, opts) {
  const factory = window.SLBRenderers[kind];
  if (!factory) throw new Error('unknown renderer: ' + kind);
  return factory(mount, opts || {});
};

// 渲染器可用的选择器：优先 deckgl，缺依赖时退回 canvas。
window.SLB.pickRendererKind = function (requested) {
  if (requested && window.SLBRenderers[requested]
      && (requested !== 'deckgl' || window.deck)) return requested;
  return window.deck ? 'deckgl' : 'canvas';
};

// 两帧间车辆插值（各实现共用）：SUMO 每秒一帧，按到达间隔线性外推。
window.SLB.interpVehicles = function (prevFrame, lastFrame, stamp0, stamp1) {
  if (!lastFrame) return [];
  const cur = new Map(lastFrame.vehicles.map(v => [v[0], v]));
  if (!prevFrame) return [...cur.values()].map(
    v => ({id: v[0], x: v[1], y: v[2], angle: v[3], speed: v[4]}));
  const prev = new Map(prevFrame.vehicles.map(v => [v[0], v]));
  const span = Math.max(1, stamp1 - stamp0);
  const alpha = Math.min(1, (performance.now() - stamp1) / span);
  const out = [];
  for (const [id, v] of cur) {
    const p = prev.get(id);
    if (!p || span <= 0) {
      out.push({id, x: v[1], y: v[2], angle: v[3], speed: v[4]});
      continue;
    }
    out.push({id, x: p[1] + (v[1] - p[1]) * alpha,
              y: p[2] + (v[2] - p[2]) * alpha, angle: v[3], speed: v[4]});
  }
  return out;
};

// 信号灯主色（各实现共用）：取状态串里第一个非红字符的语义。
window.SLB.lightColor = function (state) {
  const active = (state || '').split('').find(c => c.toLowerCase() !== 'r') || 'r';
  const c = active.toLowerCase();
  return c === 'g' ? '#20a050' : c === 'y' ? '#e0a100' : '#d53939';
};
