"""
segmentation_modification: 编辑器 HTML 页面模板
===============================================

独立 HTML 页面，通过占位符替换注入会话数据。
与 visualize_3d.py 的 _HTML_TEMPLATE 模式相同。
"""

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>分割编辑器 — __MASK_DISPLAY_NAME__</title>
<style>
  :root {
    --bg: #f0f0f4;
    --panel: #ffffff;
    --border: #d1d1d6;
    --text: #1c1c1e;
    --text-dim: #8e8e93;
    --accent: #007aff;
    --positive: #34c759;
    --negative: #ff3b30;
    --btn-bg: #e8e8ed;
    --btn-hover: #d1d1d6;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    user-select: none;
    overflow: hidden;
  }

  /* ── 顶部状态栏 ── */
  .top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 16px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    gap: 12px;
    flex-wrap: wrap;
    min-height: 36px;
  }
  .top-bar .title {
    font-weight: 600;
    font-size: 14px;
    white-space: nowrap;
  }
  .top-bar .title span { color: var(--accent); }
  .top-bar .status { font-size: 13px; color: var(--text-dim); }
  .top-bar .status strong { color: var(--text); }

  /* ── 主体布局 ── */
  .main { display: flex; flex: 1; overflow: hidden; }

  /* ── 左侧切片视图（最大化图片区域） ── */
  .viewer-panel {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2px;
    position: relative;
    overflow: hidden;
    background: #e5e5ea;
    min-width: 0;
  }
  .viewer-panel .slice-container {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    overflow: hidden;
  }
  .viewer-panel .slice-container img {
    max-width: 100%;
    max-height: 100%;
    border-radius: 3px;
    border: 1px solid var(--border);
    image-rendering: pixelated;
    display: block;
    object-fit: contain;
    flex-shrink: 0;
    pointer-events: none;
    -webkit-user-drag: none;
    user-select: none;
  }
  .viewer-panel .slice-container canvas {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    cursor: crosshair;
    pointer-events: auto;
    touch-action: none;
    z-index: 10;
  }

  /* ── 右侧控制面板 ── */
  .control-panel {
    width: 240px;
    flex-shrink: 0;
    background: var(--panel);
    border-left: 1px solid var(--border);
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow-y: auto;
    font-size: 13px;
  }
  .control-panel .section-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-bottom: 3px;
  }

  /* 模式按钮 */
  .mode-group { display: flex; gap: 6px; }
  .mode-btn {
    flex: 1;
    padding: 7px 10px;
    border: 2px solid var(--border);
    border-radius: 6px;
    background: transparent;
    color: var(--text);
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.15s;
    text-align: center;
  }
  .mode-btn:hover { background: var(--btn-hover); }
  .mode-btn.active-positive {
    border-color: var(--positive);
    background: rgba(34, 197, 94, 0.15);
    color: var(--positive);
  }
  .mode-btn.active-negative {
    border-color: var(--negative);
    background: rgba(239, 68, 68, 0.15);
    color: var(--negative);
  }

  /* 切片导航 */
  .slice-nav { display: flex; align-items: center; gap: 6px; }
  .slice-nav button {
    width: 32px; height: 32px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--btn-bg);
    color: var(--text);
    cursor: pointer;
    font-size: 14px;
    transition: background 0.15s;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .slice-nav button:hover { background: var(--btn-hover); }
  .slice-nav input[type="range"] {
    flex: 1;
    accent-color: var(--accent);
    cursor: pointer;
    min-width: 0;
  }
  .slice-nav .slice-label {
    min-width: 70px;
    text-align: center;
    font-size: 12px;
    color: var(--text-dim);
    white-space: nowrap;
  }

  /* 操作按钮 */
  .action-btn {
    width: 100%;
    padding: 9px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    transition: all 0.15s;
  }
  .action-btn.refine {
    background: var(--accent);
    color: white;
  }
  .action-btn.refine:hover { background: #1a7de6; }
  .action-btn.refine:disabled {
    background: #555;
    cursor: not-allowed;
    opacity: 0.6;
  }
  .action-btn.save {
    background: var(--positive);
    color: white;
  }
  .action-btn.save:hover { background: #16a34a; }
  .action-btn.undo {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px;
    font-size: 12px;
  }
  .action-btn.undo:hover { background: var(--btn-hover); }
  .propagation-settings {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    color: var(--text-dim);
    font-size: 12px;
  }
  .propagation-settings input {
    width: 64px;
    padding: 5px 7px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--btn-bg);
    color: var(--text);
    text-align: center;
  }

  /* Click 列表 */
  .click-list {
    max-height: 120px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .click-list .click-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(255,255,255,0.03);
    font-size: 11px;
    font-family: monospace;
  }
  .click-list .click-item .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .click-list .click-item .dot.pos { background: var(--positive); }
  .click-list .click-item .dot.neg { background: var(--negative); }

  /* 加载覆盖层 */
  .loading-overlay {
    display: none;
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0,0,0,0.6);
    z-index: 999;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 16px;
  }
  .loading-overlay.active { display: flex; }
  .loading-overlay .spinner {
    width: 40px; height: 40px;
    border: 4px solid rgba(255,255,255,0.1);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-overlay .loading-text { font-size: 14px; color: white; }

  /* Toast 通知 */
  .toast {
    position: fixed;
    bottom: 24px; left: 50%;
    transform: translateX(-50%);
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 13px;
    z-index: 1000;
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
  }
  .toast.show { opacity: 1; }
  .toast.success { background: var(--positive); color: white; }
  .toast.error { background: var(--negative); color: white; }
  .toast.info { background: var(--accent); color: white; }

  /* ── Zoom 控件 ── */
  .zoom-controls {
    display: flex;
    align-items: center;
    gap: 4px;
    justify-content: center;
  }
  .zoom-controls button {
    width: 32px; height: 32px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--btn-bg);
    color: var(--text);
    cursor: pointer;
    font-size: 16px;
    font-weight: 700;
    transition: background 0.15s;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .zoom-controls button:hover { background: var(--btn-hover); }
  .zoom-controls button:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .zoom-controls button:disabled:hover { background: var(--btn-bg); }
  .zoom-controls .zoom-label {
    min-width: 48px;
    text-align: center;
    font-size: 12px;
    color: var(--text-dim);
    font-family: monospace;
  }

  /* ── 纵向布局（竖屏/窄屏模式）── */
  .main.portrait { flex-direction: column; }
  .main.portrait .control-panel {
    width: 100% !important;
    border-left: none !important;
    border-top: 1px solid var(--border);
    max-height: 45vh;
    flex-shrink: 0;
    overflow-y: auto;
  }
  .main.portrait .viewer-panel { flex: 1; min-height: 50vh; }

  /* ── 缩放时图片过渡 ── */
  .viewer-panel .slice-container img.zooming {
    transition: transform 0.12s ease;
  }
</style>
</head>
<body>

<!-- 加载覆盖层 -->
<div class="loading-overlay" id="loading-overlay">
  <div class="spinner"></div>
  <div class="loading-text" id="loading-text">加载中...</div>
</div>

<!-- Toast 通知 -->
<div class="toast" id="toast"></div>

<!-- 顶部状态栏 -->
<div class="top-bar">
  <div class="title">分割编辑器 · <span id="mask-name">__MASK_DISPLAY_NAME__</span></div>
  <div class="status">
    切片 <strong id="slice-display">0/0</strong>
    · 点击 <strong id="click-count">0</strong>
    · 掩码像素 <strong id="mask-pixels">0</strong>
  </div>
</div>

<!-- 主体 -->
<div class="main">
  <!-- 左侧切片视图（最大化区域） -->
  <div class="viewer-panel" id="viewer-panel">
    <div class="slice-container">
      <img id="slice-img" src="" alt="CT Slice" draggable="false">
      <canvas id="click-canvas"></canvas>
    </div>
  </div>

  <!-- 右侧控制面板 -->
  <div class="control-panel">

    <!-- 点击模式 -->
    <div>
      <div class="section-label">点击模式</div>
      <div class="mode-group">
        <button class="mode-btn active-positive" id="mode-positive"
                onclick="setMode('positive')">＋ 正向 (P)</button>
        <button class="mode-btn" id="mode-negative"
                onclick="setMode('negative')">－ 负向 (N)</button>
      </div>
    </div>

    <!-- Mask 切换 -->
    <div>
      <div class="section-label">编辑目标</div>
      <select id="mask-selector" onchange="switchMask(this.value)"
              style="width:100%;padding:6px 8px;border-radius:6px;border:1px solid var(--border);background:var(--btn-bg);color:var(--text);font-size:13px;cursor:pointer;">
        __MASK_NAMES_OPTIONS__
      </select>
    </div>

    <!-- 切片导航 -->
    <div>
      <div class="section-label">切片导航</div>
      <div class="slice-nav">
        <button onclick="prevSlice()">◀</button>
        <input type="range" id="slice-slider" min="0" max="__NUM_SLICES__"
               value="__DEFAULT_SLICE__" oninput="goSlice(parseInt(this.value))">
        <button onclick="nextSlice()">▶</button>
      </div>
      <div style="text-align:center;margin-top:3px;font-size:12px;color:var(--text-dim)">
        <span id="slice-label">__DEFAULT_SLICE__ / __NUM_SLICES__</span>
      </div>
    </div>

    <!-- Zoom 控制 -->
    <div>
      <div class="section-label">缩放</div>
      <div class="zoom-controls">
        <button onclick="zoomOut()" id="btn-zoom-out" title="缩小 (Ctrl+−)">−</button>
        <span class="zoom-label" id="zoom-level">100%</span>
        <button onclick="zoomIn()" id="btn-zoom-in" title="放大 (Ctrl+=)">+</button>
        <button onclick="resetZoom()" id="btn-zoom-reset" title="重置缩放 (Ctrl+0)"
                style="width:auto;padding:0 8px;font-size:11px;font-weight:400;">重置</button>
      </div>
    </div>

    <!-- 旋转 / 翻转 -->
    <div>
      <div class="section-label">旋转 / 翻转</div>
      <div class="zoom-controls">
        <button onclick="rotateCCW()" id="btn-rotate-ccw" title="逆时针旋转 (R)">↺</button>
        <button onclick="rotateCW()" id="btn-rotate-cw" title="顺时针旋转 (Shift+R)">↻</button>
        <button onclick="flipHorizontal()" id="btn-flip-h" title="水平翻转 (H)"
                style="font-size:13px;font-weight:400;">⇔</button>
        <button onclick="flipVertical()" id="btn-flip-v" title="垂直翻转 (V)"
                style="font-size:13px;font-weight:400;">⇕</button>
        <button onclick="resetTransform()" id="btn-reset-transform" title="重置"
                style="width:auto;padding:0 8px;font-size:11px;font-weight:400;">重置</button>
      </div>
    </div>

    <!-- 操作按钮 -->
    <button class="action-btn refine" id="btn-refine" onclick="refineSlice()">
      运行 Refine
    </button>
    <div class="propagation-settings">
      <label for="propagation-radius">3D 传播半径（层）</label>
      <input id="propagation-radius" type="number" min="1" max="50" value="5">
    </div>
    <button class="action-btn save" id="btn-propagate" onclick="propagate3D()"
            style="background:var(--accent);opacity:0.85;"
            disabled title="先运行 Refine 后再传播">
      3D 传播
    </button>
    <button class="action-btn save" onclick="saveMask()">
      保存掩码
    </button>
    <button class="action-btn undo" onclick="undoLastClick()">
      撤销最后点击
    </button>

    <!-- 当前切片的点击列表 -->
    <div>
      <div class="section-label">当前切片的点击</div>
      <div class="click-list" id="click-list">
        <span style="color:var(--text-dim);font-size:12px;">暂无点击</span>
      </div>
    </div>

    <div style="margin-top:auto;text-align:center;font-size:11px;color:var(--text-dim);padding-top:8px;border-top:1px solid var(--border);">
      会话: <span id="session-id-display" style="font-family:monospace">__SESSION_ID__</span>
    </div>
  </div>
</div>

<script>
// ============================================================
// 配置
// ============================================================
const SESSION_ID = '__SESSION_ID__';
const NUM_SLICES = __NUM_SLICES__;
const IMAGE_WIDTH = __IMAGE_WIDTH__;
const IMAGE_HEIGHT = __IMAGE_HEIGHT__;
const MASK_NAMES = __MASK_NAMES_JSON__;
const DISPLAY_NAMES = __DISPLAY_NAMES_JSON__;

// 工具函数：获取显示名称
function displayName(name) { return DISPLAY_NAMES[name] || name; }

// ============================================================
// 状态
// ============================================================
const state = {
  currentSlice: Math.floor(NUM_SLICES / 2),
  mode: 'positive',       // 'positive' | 'negative'
  clicks: [],             // [{sliceIdx, x, y, label}, ...] 本地缓存，即时绘制
  maskPixels: 0,
  isLoading: false,
  zoomLevel: 1.0,         // 缩放倍数 (0.25 ~ 4.0)
  rotation: 0,            // 旋转角度: 0 | 90 | 180 | 270（顺时针）
  flipH: false,           // 水平翻转
  flipV: false,           // 垂直翻转
  isPortrait: false,      // 是否为竖屏布局
};

// ============================================================
// DOM 引用
// ============================================================
const $ = id => document.getElementById(id);
const sliceImg = $('slice-img');
const clickCanvas = $('click-canvas');
const sliceSlider = $('slice-slider');
const sliceLabel = $('slice-label');
const sliceDisplay = $('slice-display');
const clickCount = $('click-count');
const maskPixels = $('mask-pixels');
const clickList = $('click-list');
const loadingOverlay = $('loading-overlay');
const loadingText = $('loading-text');
const btnRefine = $('btn-refine');

// ============================================================
// 切片加载
// ============================================================
function loadSlice(sliceIdx) {
  if (state.isLoading) return;
  if (sliceIdx < 0 || sliceIdx >= NUM_SLICES) return;
  state.currentSlice = sliceIdx;
  sliceSlider.value = sliceIdx;
  sliceLabel.textContent = `${sliceIdx} / ${NUM_SLICES}`;
  sliceDisplay.textContent = `${sliceIdx}/${NUM_SLICES}`;

  // 点击标记统一由透明 Canvas 绘制。若 PNG 也绘制，会出现两个重叠的 P/N。
  const url = `/api/segmentation-editor/slice/${SESSION_ID}/${sliceIdx}?overlay=true&clicks=false&_=${Date.now()}`;

  sliceImg.onload = () => {
    alignCanvas();
    drawClickMarkers();
    state.isLoading = false;
  };
  sliceImg.onerror = () => {
    showToast('加载切片失败', 'error');
    state.isLoading = false;
  };
  state.isLoading = true;
  sliceImg.src = url;
}

function alignCanvas() {
  // 让 Canvas 尺寸和位置匹配图片的实际显示区域
  const rect = sliceImg.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  // 计算 Canvas 相对于容器的偏移（图片居中时与 Canvas 的 top:0;left:0 不对齐）
  const containerRect = clickCanvas.parentElement.getBoundingClientRect();
  clickCanvas.width = rect.width;
  clickCanvas.height = rect.height;
  clickCanvas.style.left = (rect.left - containerRect.left) + 'px';
  clickCanvas.style.top = (rect.top - containerRect.top) + 'px';
  clickCanvas.style.width = rect.width + 'px';
  clickCanvas.style.height = rect.height + 'px';
}

// ============================================================
// Canvas 绘制点击标记（即时反馈，不等后端）
// ============================================================
function drawClickMarkers() {
  const ctx = clickCanvas.getContext('2d');
  ctx.clearRect(0, 0, clickCanvas.width, clickCanvas.height);

  if (clickCanvas.width === 0 || clickCanvas.height === 0) return;

  // 只绘制当前切片的点击
  const sliceClicks = state.clicks.filter(c => c.sliceIdx === state.currentSlice);
  if (sliceClicks.length === 0) return;

  for (const click of sliceClicks) {
    const pos = imageToDisplay(click.x, click.y);
    const cx = pos[0];
    const cy = pos[1];

    // 外圈（白色描边）
    ctx.beginPath();
    ctx.arc(cx, cy, 9, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // 内圈（正/负颜色）
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.fillStyle = click.label === 1 ? '#22c55e' : '#ef4444';
    ctx.fill();

    // 标签
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(click.label === 1 ? 'P' : 'N', cx, cy - 12);
  }
}

// ============================================================
// Mask 切换
// ============================================================
let currentMask = '__MASK_NAME__';  // 会被 init 响应更新

async function switchMask(maskName) {
  if (maskName === currentMask) return;
  try {
    const resp = await fetch('/api/segmentation-editor/switch-mask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ session_id: SESSION_ID, mask_name: maskName })
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    currentMask = data.mask_name;
    document.getElementById('mask-name').textContent = displayName(currentMask);
    document.getElementById('mask-selector').value = currentMask;
    // 恢复目标 mask 尚未 refine 的点击；标记只由 Canvas 绘制。
    state.clicks = (data.clicks || []).map(c => ({
      sliceIdx: c.slice_idx, x: c.x, y: c.y, label: c.label
    }));
    // 重新加载切片显示新 mask 的 overlay
    loadSlice(state.currentSlice);
    updateClickList();
    showToast(`切换到 ${displayName(currentMask)}`, 'info');
  } catch (e) {
    showToast('切换 mask 失败: ' + e.message, 'error');
  }
}

// ============================================================
// 点击模式
// ============================================================
function setMode(mode) {
  state.mode = mode;
  document.getElementById('mode-positive').className =
    'mode-btn' + (mode === 'positive' ? ' active-positive' : '');
  document.getElementById('mode-negative').className =
    'mode-btn' + (mode === 'negative' ? ' active-negative' : '');
}

async function handleClick(event) {
  event.preventDefault();

  // 视觉反馈：点击时边框闪一下
  clickCanvas.style.outline = '2px solid #0f8fff';
  setTimeout(() => clickCanvas.style.outline = '', 200);

  // 使用图片的 getBoundingClientRect（图片加载后才有效，点击时一定已加载）
  if (!sliceImg.complete || sliceImg.naturalWidth === 0) {
    showToast('图片未加载完成，请稍后再点击', 'info');
    return;
  }
  const rect = sliceImg.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  // 获取显示空间坐标
  const displayX = event.clientX - rect.left;
  const displayY = event.clientY - rect.top;
  // 事件绑定在容器上；图片四周的留白不属于可编辑区域。
  if (displayX < 0 || displayY < 0 ||
      displayX > rect.width || displayY > rect.height) return;
  // 逆变换到原始图像像素坐标（处理旋转/翻转/缩放）
  const imgCoord = displayToImage(displayX, displayY, rect.width, rect.height);
  const label = state.mode === 'positive' ? 1 : 0;

  // 边界钳制
  const cx = Math.max(0, Math.min(IMAGE_WIDTH - 1, imgCoord[0]));
  const cy = Math.max(0, Math.min(IMAGE_HEIGHT - 1, imgCoord[1]));

  // 乐观更新：立刻在 Canvas 上绘制标记（不等后端返回）
  const newClick = {sliceIdx: state.currentSlice, x: cx, y: cy, label};
  state.clicks.push(newClick);
  drawClickMarkers();
  updateClickList();

  try {
    const resp = await fetch('/api/segmentation-editor/click', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: SESSION_ID,
        slice_idx: state.currentSlice,
        x: cx,
        y: cy,
        label: label,
      })
    });
    if (!resp.ok) {
      // 后端拒绝：回滚本地状态
      state.clicks = state.clicks.filter(c => c !== newClick);
      drawClickMarkers();
      updateClickList();
      throw new Error(await resp.text());
    }

    // 后端成功：重新加载图片（此时后端也渲染了标记）
    loadSlice(state.currentSlice);
  } catch (e) {
    showToast('点击失败: ' + e.message, 'error');
  }
}

// ============================================================
// 点击列表
// ============================================================
function updateClickList() {
  const sliceClicks = state.clicks.filter(c => c.sliceIdx === state.currentSlice);
  clickCount.textContent = state.clicks.length;

  if (sliceClicks.length === 0) {
    clickList.innerHTML = '<span style="color:var(--text-dim);font-size:12px;">暂无点击</span>';
  } else {
    clickList.innerHTML = sliceClicks.map((c, i) =>
      `<div class="click-item">
        <span class="dot ${c.label === 1 ? 'pos' : 'neg'}"></span>
        #${i + 1} (${c.x}, ${c.y}) ${c.label === 1 ? '正向' : '负向'}
      </div>`
    ).join('');
  }
}

// ============================================================
// Refine
// ============================================================
async function refineSlice() {
  if (btnRefine.disabled) return;

  btnRefine.disabled = true;
  showLoading('MedSAM2 推理中...');

  try {
    const resp = await fetch('/api/segmentation-editor/refine', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: SESSION_ID,
        slice_idx: state.currentSlice,
      })
    });

    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(err);
    }

    const data = await resp.json();

    // Update image with refined overlay
    if (data.overlay_png) {
      sliceImg.src = 'data:image/png;base64,' + data.overlay_png;
    }

    // Update stats
    if (data.mask_pixels !== undefined) {
      state.maskPixels = data.mask_pixels;
      maskPixels.textContent = data.mask_pixels.toLocaleString();
    }

    // Refine 完成 → 清除 P/N 标记
    if (data.clicks_cleared) {
      state.clicks = state.clicks.filter(c => c.sliceIdx !== state.currentSlice);
      drawClickMarkers();
      updateClickList();
    }

    // 启用 3D 传播按钮
    document.getElementById('btn-propagate').disabled = false;

    showToast('Refine 完成 ✓', 'success');
  } catch (e) {
    showToast('Refine 失败: ' + e.message, 'error');
  } finally {
    btnRefine.disabled = false;
    hideLoading();
  }
}

// ============================================================
// 3D Propagation
// ============================================================
async function propagate3D() {
  const btn = document.getElementById('btn-propagate');
  if (btn.disabled) return;
  const radiusInput = document.getElementById('propagation-radius');
  const propagationRadius = Math.max(1, Math.min(50, parseInt(radiusInput.value, 10) || 5));
  radiusInput.value = propagationRadius;

  btn.disabled = true;
  showLoading('3D 传播中（可能需要 30-60 秒）...');

  try {
    const resp = await fetch('/api/segmentation-editor/propagate-3d', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: SESSION_ID,
        slice_idx: state.currentSlice,
        max_propagation_slices: propagationRadius,
      })
    });

    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(err);
    }

    const data = await resp.json();

    // 重新加载当前切片查看传播效果
    loadSlice(state.currentSlice);

    showToast(
      `3D 传播完成 ✓ 在锚点±${data.propagation_radius}层内更新了 ${data.propagated_slices} 层`,
      'success'
    );
  } catch (e) {
    showToast('3D 传播失败: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    hideLoading();
  }
}

// ============================================================
// Save
// ============================================================
async function saveMask() {
  showLoading('保存掩码中...');

  try {
    const resp = await fetch('/api/segmentation-editor/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ session_id: SESSION_ID })
    });

    if (!resp.ok) throw new Error(await resp.text());

    const data = await resp.json();
    showToast('掩码已保存 ✓', 'success');
  } catch (e) {
    showToast('保存失败: ' + e.message, 'error');
  } finally {
    hideLoading();
  }
}

// ============================================================
// Undo（仅撤销当前切片最后一次点击）
// ============================================================
async function undoLastClick() {
  try {
    const resp = await fetch('/api/segmentation-editor/undo-last-click', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: SESSION_ID,
        slice_idx: state.currentSlice,
      })
    });
    if (!resp.ok) throw new Error(await resp.text());

    const data = await resp.json();
    if (data.removed) {
      for (let i = state.clicks.length - 1; i >= 0; i--) {
        if (state.clicks[i].sliceIdx === state.currentSlice) {
          state.clicks.splice(i, 1);
          break;
        }
      }
      drawClickMarkers();
      updateClickList();
      loadSlice(state.currentSlice);
      showToast('已撤销最后一次点击', 'info');
    } else {
      showToast('当前切片没有可撤销的点击', 'info');
    }
  } catch (e) {
    showToast('撤销失败: ' + e.message, 'error');
  }
}

// ============================================================
// 导航
// ============================================================
function prevSlice() {
  if (state.currentSlice > 0) loadSlice(state.currentSlice - 1);
}
function nextSlice() {
  if (state.currentSlice < NUM_SLICES - 1) loadSlice(state.currentSlice + 1);
}
function goSlice(idx) {
  loadSlice(Math.max(0, Math.min(NUM_SLICES - 1, idx)));
}

// ============================================================
// 缩放
// ============================================================
const ZOOM_MIN = 0.25;
const ZOOM_MAX = 4.0;
const ZOOM_STEP = 0.25;

function setZoom(level) {
  state.zoomLevel = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, level));
  document.getElementById('zoom-level').textContent = Math.round(state.zoomLevel * 100) + '%';
  updateImageTransform();
}

function updateImageTransform() {
  // 组合 CSS transform: 缩放 → 旋转 → 翻转
  var t = '';
  var zoom = state.zoomLevel;
  if (zoom !== 1) t += 'scale(' + zoom + ') ';
  if (state.rotation !== 0) t += 'rotate(' + state.rotation + 'deg) ';
  var fx = state.flipH ? -1 : 1;
  var fy = state.flipV ? -1 : 1;
  if (fx !== 1 || fy !== 1) t += 'scale(' + fx + ', ' + fy + ')';
  sliceImg.style.transform = t.trim() || 'none';
  alignCanvas();
  drawClickMarkers();
}

function rotateCW() {
  state.rotation = (state.rotation + 90) % 360;
  updateImageTransform();
}

function rotateCCW() {
  state.rotation = (state.rotation - 90 + 360) % 360;
  updateImageTransform();
}

function flipHorizontal() {
  state.flipH = !state.flipH;
  updateImageTransform();
}

function flipVertical() {
  state.flipV = !state.flipV;
  updateImageTransform();
}

function resetTransform() {
  state.rotation = 0;
  state.flipH = false;
  state.flipV = false;
  state.zoomLevel = 1.0;
  document.getElementById('zoom-level').textContent = '100%';
  updateImageTransform();
}

// ============================================================
// 坐标变换（旋转/翻转后的点击坐标映射）
// ============================================================
// 原始图像像素 → 显示坐标（canvas 像素空间）。
//
// 必须使用 Canvas/图片的实际显示尺寸，不能假设一个屏幕像素等于一个
// 原图像素。竖屏上下布局中图片通常会按可用高度缩小；此前直接使用
// IMAGE_WIDTH/IMAGE_HEIGHT，导致发送给 refine 的点击坐标发生偏移。
function imageToDisplay(ox, oy) {
  const W = IMAGE_WIDTH, H = IMAGE_HEIGHT;
  const displayW = clickCanvas.width;
  const displayH = clickCanvas.height;
  let u = ox / W;
  let v = oy / H;

  // CSS transform 的执行顺序：翻转 → 旋转 → 缩放。
  if (state.flipH) u = 1 - u;
  if (state.flipV) v = 1 - v;

  let displayU = u;
  let displayV = v;
  if (state.rotation === 90) {
    displayU = 1 - v;
    displayV = u;
  } else if (state.rotation === 180) {
    displayU = 1 - u;
    displayV = 1 - v;
  } else if (state.rotation === 270) {
    displayU = v;
    displayV = 1 - u;
  }

  return [displayU * displayW, displayV * displayH];
}

// 显示坐标 → 原始图像像素（用于鼠标点击映射到原始空间）
function displayToImage(dx, dy, displayW, displayH) {
  const W = IMAGE_WIDTH, H = IMAGE_HEIGHT;
  let displayU = dx / displayW;
  let displayV = dy / displayH;

  // 逆旋转
  let u = displayU;
  let v = displayV;
  if (state.rotation === 90) {
    u = displayV;
    v = 1 - displayU;
  } else if (state.rotation === 180) {
    u = 1 - displayU;
    v = 1 - displayV;
  } else if (state.rotation === 270) {
    u = 1 - displayV;
    v = displayU;
  }

  // 逆翻转
  if (state.flipH) u = 1 - u;
  if (state.flipV) v = 1 - v;

  return [Math.round(u * W), Math.round(v * H)];
}

function zoomIn() {
  setZoom(state.zoomLevel + ZOOM_STEP);
}

function zoomOut() {
  setZoom(state.zoomLevel - ZOOM_STEP);
}

function resetZoom() {
  setZoom(1.0);
}

// ============================================================
// 竖屏布局检测
// ============================================================
function checkPortrait() {
  const isPortrait = window.innerWidth / window.innerHeight < 1;
  if (isPortrait !== state.isPortrait) {
    state.isPortrait = isPortrait;
    document.querySelector('.main').classList.toggle('portrait', isPortrait);
  }
}

// ============================================================
// 鼠标滚轮切换切片
// ============================================================
document.getElementById('viewer-panel').addEventListener('wheel', function(e) {
  if (e.target.closest('.zoom-controls')) return;  // 不干扰 zoom 按钮区域
  e.preventDefault();
  if (e.deltaY > 0) {
    nextSlice();
  } else {
    prevSlice();
  }
}, { passive: false });

// ============================================================
// UI 辅助
// ============================================================
function showLoading(msg) {
  loadingText.textContent = msg || '加载中...';
  loadingOverlay.classList.add('active');
}

function hideLoading() {
  loadingOverlay.classList.remove('active');
}

let toastTimer = null;
function showToast(msg, type) {
  const el = $('toast');
  el.textContent = msg;
  el.className = 'toast ' + (type || 'info') + ' show';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2000);
}

// ============================================================
// 键盘快捷键
// ============================================================
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  switch (e.key) {
    case 'ArrowLeft': prevSlice(); e.preventDefault(); break;
    case 'ArrowRight': nextSlice(); e.preventDefault(); break;
    case 'p': case 'P': setMode('positive'); break;
    case 'n': case 'N': setMode('negative'); break;
    case 'Enter': case 'r': case 'R':
      if (e.shiftKey) { rotateCW(); e.preventDefault(); break; }
      refineSlice(); break;
    case 's': case 'S':
      if (e.ctrlKey || e.metaKey) { e.preventDefault(); saveMask(); }
      break;
    case 'h': case 'H': flipHorizontal(); e.preventDefault(); break;
    case 'v': case 'V': flipVertical(); e.preventDefault(); break;
  }

  // Zoom 快捷键：Ctrl +/=/-
  if (e.ctrlKey || e.metaKey) {
    switch (e.key) {
      case '=': case '+': zoomIn(); e.preventDefault(); break;
      case '-': case '_': zoomOut(); e.preventDefault(); break;
      case '0': resetZoom(); e.preventDefault(); break;
    }
  }
});

// ============================================================
// 窗口大小变化时重对齐 Canvas
// ============================================================
window.addEventListener('resize', () => {
  checkPortrait();
  if (sliceImg.complete) {
    alignCanvas();
    drawClickMarkers();
  }
});

// ============================================================
// 点击事件 — 绑定在容器上（比 Canvas 更可靠）
// ============================================================
document.querySelector('.slice-container').addEventListener('click', handleClick);

// ============================================================
// 初始化：从后端恢复未处理点击，随后再加载不含点击标记的底图。
// ============================================================
async function initializeEditor() {
  try {
    const resp = await fetch('/api/segmentation-editor/init', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: SESSION_ID})
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    currentMask = data.mask_name;
    state.clicks = (data.clicks || []).map(c => ({
      sliceIdx: c.slice_idx, x: c.x, y: c.y, label: c.label
    }));
  } catch (e) {
    showToast('编辑器初始化失败: ' + e.message, 'error');
  }
  loadSlice(Math.floor(NUM_SLICES / 2));
  updateClickList();
  checkPortrait();
}

initializeEditor();
</script>

</body>
</html>"""


# 显示名称映射（内部文件名 → 前端展示名，避免歧义）
_DISPLAY_NAMES = {
    "hepatic": "hepatic vessel",
}


def _get_display_name(name: str) -> str:
    """返回前端展示名称，未映射则原样返回。"""
    return _DISPLAY_NAMES.get(name, name)


# 生成最终 HTML（替换占位符）
def make_editor_html(session_id: str, num_slices: int, image_width: int,
                     image_height: int, mask_name: str,
                     mask_names: "list[str]" = None) -> str:
    """返回已替换占位符的完整 HTML 字符串。"""
    import json
    default_slice = num_slices // 2
    mnames = mask_names or [mask_name]

    html = HTML_TEMPLATE
    html = html.replace("__SESSION_ID__", session_id)
    html = html.replace("__NUM_SLICES__", str(num_slices))
    html = html.replace("__DEFAULT_SLICE__", str(default_slice))
    html = html.replace("__IMAGE_WIDTH__", str(image_width))
    html = html.replace("__IMAGE_HEIGHT__", str(image_height))
    # 内部名称（用于 API 调用）
    html = html.replace("__MASK_NAME__", mask_name)
    # 展示名称（用于标题栏等用户可见位置）
    html = html.replace("__MASK_DISPLAY_NAME__", _get_display_name(mask_name))
    # 展示名称映射（前端 JS 使用）
    display_map = {n: _get_display_name(n) for n in mnames}
    html = html.replace("__DISPLAY_NAMES_JSON__", json.dumps(display_map))
    # mask 列表（JSON，用于前端下拉切换）
    html = html.replace("__MASK_NAMES_JSON__", json.dumps(mnames))
    html = html.replace("__MASK_NAMES_OPTIONS__", "".join(
        f'<option value="{n}"{" selected" if n == mask_name else ""}>{_get_display_name(n)}</option>'
        for n in mnames
    ))
    return html
