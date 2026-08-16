import React, { useRef, useState, useEffect } from 'react';
import { Play, Pause, RefreshCw, ZoomIn, ZoomOut, Maximize2, UploadCloud, FileText, AlertCircle, Layers, FlipHorizontal, FlipVertical } from 'lucide-react';
import { drawNiftiSliceOnCanvas, NiftiVolume, WindowPresetKey, WINDOW_PRESETS, computeAutoRange, getPresetWindowRange } from '../utils/niftiLoader';
import { normalizeSlices } from '../utils/portBUrlHelper';

interface SliceViewerProps {
  currentSliceIndex: number;
  setSliceIndex: React.Dispatch<React.SetStateAction<number>>;
  totalSlices: number;
  caseId: string;
  status: string;
  onFileUpload: (file: File, options?: { intensityUnit?: 'HU' | 'unknown' }) => void;
  organ?: string;
  niftiVolume?: NiftiVolume | null;
  onMaximize?: () => void;
  isFullscreen?: boolean;
  lesionImageUrl?: string | null;
  bestSliceIndex?: number | null;
  bestSlices?: any[];
  onUpdateIntensityUnit?: (volumeId: string, unit: 'HU' | 'unknown') => void;
}

export default function SliceViewer({
  currentSliceIndex,
  setSliceIndex,
  totalSlices,
  caseId,
  status,
  onFileUpload,
  organ,
  niftiVolume,
  onMaximize,
  isFullscreen = false,
  lesionImageUrl = null,
  bestSliceIndex = null,
  bestSlices = [],
  onUpdateIntensityUnit,
}: SliceViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Local slider drag value: only commit to parent on mouse up
  const [sliderDragValue, setSliderDragValue] = useState<number | null>(null);

  // Derived display slice index: use drag value while dragging, parent value otherwise
  const displaySliceIndex = sliderDragValue ?? currentSliceIndex;

  // Commit the dragged value to parent on mouse up
  const commitSliderValue = () => {
    if (sliderDragValue !== null) {
      setSliceIndex(sliderDragValue);
      setSliderDragValue(null);
    }
  };

  // View Mode: 'slice' (NIfTI canvas) or 'lesion' (AI overlay gallery)
  // Default to 'slice' so NIfTI canvas stays visible even after AI results return
  const [viewMode, setViewMode] = useState<'slice' | 'lesion'>('slice');

  // Zoom & Pan state
  const [scale, setScale] = useState<number>(1);
  const [offset, setOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Rotation state: 0 | 90 | 180 | 270 (degrees)
  const [rotation, setRotation] = useState<0 | 90 | 180 | 270>(0);

  // Flip state: horizontal/vertical mirror
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);

  // Selected slice index for lesion gallery view
  const [selectedSliceIdx, setSelectedSliceIdx] = useState(0);

  // Auto scanning playback state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const playIntervalRef = useRef<number | null>(null);

  // Drag and Drop State
  const [isDragActive, setIsDragActive] = useState<boolean>(false);
  const [uploadedFile, setUploadedFile] = useState<{ name: string; size: string } | null>(null);

  // 上传时的强度单位选择（默认 CT(HU) —— 项目只用于肝脏 CT，用户主动声明）
  const [uploadIntentUnit, setUploadIntentUnit] = useState<'HU' | 'unknown'>('HU');

  // ── 窗宽窗位状态 ──
  const [windowLevel, setWindowLevel] = useState<number>(90);
  const [windowWidth, setWindowWidth] = useState<number>(150);
  const [activePreset, setActivePreset] = useState<WindowPresetKey>('liver');
  const [currentVolumeId, setCurrentVolumeId] = useState<string | null>(null);
  const [wlInput, setWlInput] = useState<string>('90');
  const [wwInput, setWwInput] = useState<string>('150');
  const prevIntensityUnitRef = useRef<'HU' | 'unknown' | null>(null);

  const LS_WINDOW_PREFIX = 'CLINICAL_WINDOW_SETTINGS_';

  function restoreWindowSettings(volumeId: string): { wl: number; ww: number; preset: string } | null {
    try {
      const raw = localStorage.getItem(LS_WINDOW_PREFIX + volumeId);
      if (raw) return JSON.parse(raw);
    } catch {}
    return null;
  }

  function saveWindowSettings(volumeId: string, wl: number, ww: number, preset: string): void {
    try {
      localStorage.setItem(LS_WINDOW_PREFIX + volumeId, JSON.stringify({ wl, ww, preset }));
    } catch {}
  }

  // 检测新 volume 或 intensityUnit 变化，恢复/设置窗口
  useEffect(() => {
    const v = niftiVolume;
    if (!v) return;

    const newId = v.volumeId;
    const isNewVolume = newId !== currentVolumeId;
    const isUnitChange = !isNewVolume && prevIntensityUnitRef.current != null &&
      prevIntensityUnitRef.current !== v.intensityUnit;
    prevIntensityUnitRef.current = v.intensityUnit;

    if (isNewVolume) {
      setCurrentVolumeId(newId);
      setUploadIntentUnit(v.intensityUnit);

      // 优先从 localStorage 恢复窗口设置
      const saved = restoreWindowSettings(newId);
      if (saved) {
        setWindowLevel(saved.wl);
        setWindowWidth(saved.ww);
        setActivePreset(saved.preset as WindowPresetKey);
        setWlInput(String(saved.wl));
        setWwInput(String(saved.ww));
        return;
      }
    }

    // 新 volume（无缓存）或同 volume intensityUnit 变化 → 默认窗口
    if (isNewVolume || isUnitChange) {
      setUploadIntentUnit(v.intensityUnit);
      if (v.intensityUnit === 'HU') {
        setWindowLevel(90);
        setWindowWidth(150);
        setActivePreset('liver');
        setWlInput('90');
        setWwInput('150');
      } else {
        computeAutoRange(v);
        setActivePreset('auto');
      }
    }
    // 不重置切片位置、缩放、平移、旋转、播放
  }, [niftiVolume?.volumeId, niftiVolume?.intensityUnit]);

  // 窗口设置变化时自动持久化
  useEffect(() => {
    const id = currentVolumeId;
    if (!id) return;
    saveWindowSettings(id, windowLevel, windowWidth, activePreset);
  }, [windowLevel, windowWidth, activePreset, currentVolumeId]);

  // 预设或自定义 → 同步 WL/WW 输入框
  useEffect(() => {
    setWlInput(String(windowLevel));
    setWwInput(String(windowWidth));
  }, [windowLevel, windowWidth]);

  // 当前窗显示范围描述
  const windowRangeLabel = (() => {
    if (activePreset === 'auto' && niftiVolume?.autoRange) {
      const r = niftiVolume.autoRange;
      return `${r.min.toFixed(0)} — ${r.max.toFixed(0)} (自动)`;
    }
    if (windowWidth >= 1) {
      const r = getPresetWindowRange(windowLevel, windowWidth);
      return `${r.low.toFixed(0)} — ${r.high.toFixed(0)}${niftiVolume?.intensityUnit === 'HU' ? ' HU' : ''}`;
    }
    return '—';
  })();

  // ── 真全屏（浏览器 Fullscreen API） ──
  const [isFullscreenActive, setFullscreenActive] = useState(false);
  const fullscreenContainerRef = useRef<HTMLDivElement>(null);

  const requestFullscreen = async () => {
    const el = fullscreenContainerRef.current || document.querySelector('#slice-viewer-workspace');
    if (!el) return;
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await (el as HTMLElement).requestFullscreen();
      }
    } catch (e: any) {
      console.warn('Fullscreen API 不可用:', e?.message);
    }
  };

  useEffect(() => {
    const handler = () => {
      setFullscreenActive(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handler);
    // 全屏失败时 fallback
    document.addEventListener('fullscreenerror', handler);
    return () => {
      document.removeEventListener('fullscreenchange', handler);
      document.removeEventListener('fullscreenerror', handler);
    };
  }, []);

  // Normalized bestSlices (convert _path to _url) + reset index on change
  const normalizedSlices = React.useMemo(() => normalizeSlices(bestSlices), [bestSlices]);
  useEffect(() => {
    setSelectedSliceIdx(0);
  }, [normalizedSlices.length]);

  // Auto-switch to lesion view when slice images arrive from backend
  useEffect(() => {
    if (normalizedSlices.length > 0 && viewMode === 'slice') {
      setViewMode('lesion');
    }
  }, [normalizedSlices.length]);

  // ── Viewport 尺寸状态（ResizeObserver） ──
  const viewportRef = useRef<HTMLDivElement>(null);
  const [viewportSize, setViewportSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const boxSize = entry.borderBoxSize?.[0];
        const inlineSize = boxSize?.inlineSize ?? entry.contentRect.width;
        const blockSize = boxSize?.blockSize ?? entry.contentRect.height;
        setViewportSize((prev) => {
          const w = Math.round(inlineSize);
          const h = Math.round(blockSize);
          if (!prev || prev.w !== w || prev.h !== h) return { w, h };
          return prev;
        });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // 1. Draw NIfTI slice — 使用 ResizeObserver 尺寸适配 Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !niftiVolume || viewMode !== 'slice') return;

    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    if (cw === 0 || ch === 0) return;

    const bw = Math.round(cw * dpr);
    const bh = Math.round(ch * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }

    const ww = activePreset === 'auto'
      ? undefined
      : (windowWidth >= 1 ? windowWidth : undefined);
    const wl = activePreset === 'auto'
      ? undefined
      : (ww !== undefined ? windowLevel : undefined);
    drawNiftiSliceOnCanvas(canvas, niftiVolume, displaySliceIndex - 1, ww, wl);
  }, [displaySliceIndex, niftiVolume, viewMode, activePreset, windowLevel, windowWidth, viewportSize]);

  // 2. Playback Scroll Cycle (Auto-scanning through CT volume)
  useEffect(() => {
    if (isPlaying) {
      playIntervalRef.current = window.setInterval(() => {
        setSliceIndex((prev) => {
          if (prev >= totalSlices) return 1;
          return prev + 1;
        });
      }, 60); // 60ms per frame for a fast volumetric scan feel
    } else {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    }
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [isPlaying, totalSlices, setSliceIndex]);

  // Stop playback if we click or interact with slider manually
  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setIsPlaying(false);
    // Update local slider value WITHOUT committing to parent / re-rendering canvas
    setSliderDragValue(parseInt(e.target.value));
  };

  // Commit slider value to parent (and trigger canvas re-render) on mouse/touch release
  const handleSliderCommit = () => {
    commitSliderValue();
  };

  // 3. Zoom via Mouse Wheel (using native event listener to support preventDefault)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleNativeWheel = (e: WheelEvent) => {
      e.preventDefault();
      const zoomIntensity = 0.1;
      setScale((prevScale) => {
        let newScale = prevScale + (e.deltaY < 0 ? zoomIntensity : -zoomIntensity);
        return Math.max(0.7, Math.min(newScale, 5.0)); // Zoom limit between 0.7x and 5.0x
      });
    };

    container.addEventListener('wheel', handleNativeWheel, { passive: false });
    return () => {
      container.removeEventListener('wheel', handleNativeWheel);
    };
  }, []);

  // 4. Pan via Mouse Dragging
  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging) return;
    setOffset({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // 5. Reset Zoom, Pan, Rotation & Flip
  const handleReset = () => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
    setRotation(0);
    setFlipH(false);
    setFlipV(false);
  };

  // Rotate 90° clockwise
  const handleRotateCW = () => {
    setRotation((r) => (r === 270 ? 0 : (r + 90) as 0 | 90 | 180 | 270));
  };
  // Rotate 90° counter-clockwise
  const handleRotateCCW = () => {
    setRotation((r) => (r === 0 ? 270 : (r - 90) as 0 | 90 | 180 | 270));
  };
  // Toggle horizontal flip (mirror left-right)
  const handleFlipH = () => setFlipH((v) => !v);
  // Toggle vertical flip (mirror top-bottom)
  const handleFlipV = () => setFlipV((v) => !v);

  // 6. Handle File Drops (Registered on the outer container)
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true);
    } else if (e.type === 'dragleave') {
      // Only deactivate if we leave the outer boundaries
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX;
      const y = e.clientY;
      if (x < rect.left || x >= rect.right || y < rect.top || y >= rect.bottom) {
        setIsDragActive(false);
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      processFile(file);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File) => {
    const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
    setUploadedFile({
      name: file.name,
      size: `${sizeInMB} MB`,
    });
    onFileUpload(file, { intensityUnit: uploadIntentUnit });
  };

  // Calculate if ROI box should be visible (only shown near the target slice +/- 12 layers, and when active)
  // NOTE: ROI display removed — roiBox was a heuristic placeholder, not real backend AI lesion data.

  if (isFullscreen) {
    // Fullscreen rendering via props — not used directly
    return (
      <div className="relative w-full h-full bg-black flex items-center justify-center">
        <p className="text-white font-mono">Fullscreen mode</p>
      </div>
    );
  }

  // ── 正常（非全屏）渲染 ──
  return (
    <div
      ref={fullscreenContainerRef}
      className="relative flex flex-col bg-white border border-slate-250/80 h-full overflow-hidden shadow-sm rounded-xl"
    >
      {/* 全屏头部 */}
      {isFullscreenActive && (
        <div className="flex items-center justify-between px-3 py-1.5 bg-black/80 text-white flex-shrink-0">
          <span className="text-[11px] font-mono">2D 全屏</span>
          <button onClick={requestFullscreen}
            className="flex items-center gap-1 px-2 py-1 bg-white/10 hover:bg-white/20 text-white/80 rounded text-[10px] font-mono transition-colors"
            aria-label="退出全屏">
            <Maximize2 size={11} className="rotate-180" /> 退出
          </button>
        </div>
      )}
      {/* 影像视口 */}
      <div className="relative flex-1 flex items-center justify-center bg-black overflow-hidden">
        <div
          ref={containerRef}
          className="relative w-full h-full bg-black cursor-grab select-none overflow-hidden flex items-center justify-center"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <div className="relative flex items-center justify-center w-full h-full" style={{
            transform: `scaleX(${flipH ? -1 : 1}) scaleY(${flipV ? -1 : 1}) scale(${scale}) translate(${offset.x / (scale || 1)}px, ${offset.y / (scale || 1)}px) rotate(${rotation}deg)`,
          }}>
            {niftiVolume ? (
              <div className="flex items-center justify-center w-full h-full">
                <div style={{ aspectRatio: `${niftiVolume.width} / ${niftiVolume.height}`, maxWidth: '100%', maxHeight: '100%' }}>
                  <canvas ref={canvasRef} className="block w-full h-full" />
                </div>
              </div>
            ) : (
              <div className="text-center text-slate-400 p-4"><Layers className="w-10 h-10 mx-auto mb-2 animate-pulse" /><p className="text-xs">等待影像</p></div>
            )}
          </div>
          {/* 四角角标 */}
          {niftiVolume && (<>
          <div className="absolute top-2 left-2 z-10 pointer-events-none select-none">
            <div className="text-sm font-mono text-white/90 drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)] leading-snug">
              <div>{currentSliceIndex} / {totalSlices}</div>
              <div className="text-xs text-white/70">Z {Math.round((currentSliceIndex / totalSlices) * 100)}%</div>
            </div>
          </div>
          <div className="absolute top-2 right-2 z-10 pointer-events-none select-none text-right">
            <div className="text-sm font-mono text-white/90 drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)] leading-snug">
              <div>WL {windowLevel.toFixed(0)} / WW {windowWidth.toFixed(0)}</div>
              <div className="text-xs text-white/70">{WINDOW_PRESETS[activePreset]?.name || activePreset}</div>
            </div>
          </div>
          <div className="absolute bottom-2 left-2 z-10 pointer-events-none select-none">
            <div className="text-sm font-mono text-white/90 drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)] leading-snug">
              <div>{niftiVolume.width}&times;{niftiVolume.height}</div>
              {niftiVolume.pixDims && <div className="text-xs text-white/70">{niftiVolume.pixDims[1]?.toFixed(1)}&times;{niftiVolume.pixDims[2]?.toFixed(1)}&times;{niftiVolume.pixDims[3]?.toFixed(1)}mm</div>}
            </div>
          </div>
          <div className="absolute bottom-2 right-2 z-10 pointer-events-none select-none text-right">
            <div className="text-sm font-mono text-white/90 drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)] leading-snug">
              <div className={niftiVolume.intensityUnit === 'HU' ? 'text-emerald-300' : 'text-amber-300'}>{niftiVolume.intensityUnit === 'HU' ? 'HU' : '强度未知'}</div>
              <div className="text-xs text-white/70">{niftiVolume.datatypeName}</div>
            </div>
          </div>
          </>)}
          {/* 紧凑工具栏 */}
          <div className="absolute bottom-2 left-2 z-20 flex items-center gap-0.5 bg-black/60 backdrop-blur-sm px-1.5 py-1 rounded-md border border-white/10 shadow-lg"
            onMouseDown={(e) => e.stopPropagation()}>
            <button onClick={() => setScale((s) => Math.min(s + 0.25, 4.0))} className="p-1 hover:bg-white/20 text-white/80 rounded" title="放大" aria-label="放大"><ZoomIn size={13} /></button>
            <button onClick={() => setScale((s) => Math.max(s - 0.25, 0.75))} className="p-1 hover:bg-white/20 text-white/80 rounded" title="缩小" aria-label="缩小"><ZoomOut size={13} /></button>
            <span className="w-px h-4 bg-white/20 mx-0.5" />
            <button onClick={handleRotateCCW} className="p-1 hover:bg-white/20 text-white/80 rounded text-[12px]" title="向左旋转" aria-label="向左旋转">&#x21BA;</button>
            <button onClick={handleRotateCW} className="p-1 hover:bg-white/20 text-white/80 rounded text-[12px]" title="向右旋转" aria-label="向右旋转">&#x21BB;</button>
            <button onClick={handleFlipH} className={`p-1 rounded ${flipH ? 'bg-blue-500/40 text-blue-300' : 'hover:bg-white/20 text-white/80'}`} title="水平翻转" aria-label="水平翻转"><FlipHorizontal size={13} /></button>
            <button onClick={handleFlipV} className={`p-1 rounded ${flipV ? 'bg-blue-500/40 text-blue-300' : 'hover:bg-white/20 text-white/80'}`} title="垂直翻转" aria-label="垂直翻转"><FlipVertical size={13} /></button>
            <span className="w-px h-4 bg-white/20 mx-0.5" />
            <button onClick={handleReset} className="px-1.5 py-0.5 hover:bg-white/20 text-white/70 rounded text-[9px] font-mono" title="重置" aria-label="重置">重置</button>
            <button onClick={requestFullscreen} className="p-1 hover:bg-white/20 text-white/80 rounded"
              title={isFullscreenActive ? '退出全屏 (F)' : '全屏浏览 (F)'}
              aria-label={isFullscreenActive ? '退出全屏' : '进入全屏'}>
              {isFullscreenActive ? <Maximize2 size={13} className="rotate-180" /> : <Maximize2 size={13} />}
            </button>
          </div>
        </div>
      </div>

      {/* ── 控制区 ── */}
      {niftiVolume && viewMode !== 'lesion' && (
      <div className="p-4 bg-slate-50 border-t border-slate-200 flex flex-col gap-3.5">
        {/* 切片播放 */}
        <div className="flex items-center gap-3">
          <button onClick={() => setIsPlaying(!isPlaying)}
            className={`p-2 rounded-lg transition-all shadow-sm ${isPlaying ? 'bg-amber-600 hover:bg-amber-700 text-white' : 'bg-blue-600 hover:bg-blue-700 text-white'}`}
            title={isPlaying ? '暂停' : '播放'}>
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <div className="flex-1 relative flex items-center h-5">
            <input type="range" min="1" max={totalSlices} value={displaySliceIndex}
              onChange={handleSliderChange} onMouseUp={handleSliderCommit} onTouchEnd={handleSliderCommit}
              className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600 focus:outline-none" />
          </div>
          <span className="text-xs text-slate-700 font-mono bg-white px-2.5 py-1 rounded border border-slate-250 min-w-[72px] text-center shadow-sm">
            {String(displaySliceIndex).padStart(3, '0')} / {totalSlices}
          </span>
        </div>

        {/* 窗宽窗位 */}
        <div className="border-t border-slate-200 pt-2">
          {/* 强度单位 */}
          {niftiVolume.intensityUnit === 'unknown' ? (
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] font-mono text-amber-600 flex items-center gap-1"><AlertCircle size={11} />无法确认单位为 HU</span>
              <button onClick={() => onUpdateIntensityUnit?.(niftiVolume.volumeId, 'HU')} className="text-[10px] font-semibold text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded hover:bg-amber-100">标记为 CT (HU)</button>
            </div>
          ) : (
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] font-mono text-emerald-600 font-semibold">&#10003; CT HU</span>
              <button onClick={() => onUpdateIntensityUnit?.(niftiVolume.volumeId, 'unknown')} className="text-[10px] text-slate-500 bg-slate-50 border border-slate-200 px-2 py-0.5 rounded hover:bg-slate-100">更改为未知</button>
            </div>
          )}
          {/* 预设 + WL/WW */}
          <div className="flex flex-wrap items-center gap-2">
            <select value={activePreset} onChange={(e) => {
              const key = e.target.value as WindowPresetKey;
              setActivePreset(key);
              if (key === 'auto' && niftiVolume) computeAutoRange(niftiVolume);
              else if (key !== 'custom') { const p = WINDOW_PRESETS[key]; if (p.wl !== null && p.ww !== null) { setWindowLevel(p.wl); setWindowWidth(p.ww); } }
            }} className="text-[10px] font-mono bg-white border border-slate-200 rounded px-1.5 py-1 text-slate-700 focus:outline-none cursor-pointer">
              {(Object.keys(WINDOW_PRESETS) as WindowPresetKey[]).map((k) => <option key={k} value={k}>{WINDOW_PRESETS[k].name}</option>)}
            </select>
            <label className="flex items-center gap-1 text-[10px] font-mono text-slate-500">WL:
              <input type="text" inputMode="numeric" value={wlInput}
                onChange={(e) => setWlInput(e.target.value)}
                onBlur={() => { const v = parseFloat(wlInput); if (Number.isFinite(v) && v >= -32768 && v <= 32767) { setActivePreset('custom'); setWindowLevel(v); } else setWlInput(String(windowLevel)); }}
                onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
                className="w-14 bg-white border border-slate-200 rounded px-1.5 py-1 text-[10px] font-mono focus:outline-none focus:border-blue-400" />
            </label>
            <label className="flex items-center gap-1 text-[10px] font-mono text-slate-500">WW:
              <input type="text" inputMode="numeric" value={wwInput}
                onChange={(e) => setWwInput(e.target.value)}
                onBlur={() => { const v = parseFloat(wwInput); if (Number.isFinite(v) && v >= 1) { setActivePreset('custom'); setWindowWidth(v); } else setWwInput(String(windowWidth)); }}
                onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
                className="w-14 bg-white border border-slate-200 rounded px-1.5 py-1 text-[10px] font-mono focus:outline-none focus:border-blue-400" />
            </label>
            <button onClick={() => {
              if (niftiVolume?.intensityUnit === 'HU') { setWindowLevel(90); setWindowWidth(150); setActivePreset('liver'); setWlInput('90'); setWwInput('150'); }
              else { setActivePreset('auto'); if (niftiVolume) computeAutoRange(niftiVolume); }
            }} className="px-1.5 py-1 text-[10px] font-mono bg-white border border-slate-200 rounded hover:bg-slate-50 text-slate-500" title="恢复默认">默认</button>
          </div>
          {/* 显示范围 */}
          <div className="mt-1"><span className="text-[9px] font-mono text-slate-400">显示范围: {windowRangeLabel}</span></div>
          {/* HCC 提示 */}
          {activePreset === 'hcc_narrow' && (
            <div className="mt-1 flex items-center gap-1 text-[9px] font-mono text-amber-600 bg-amber-50 border border-amber-200 rounded py-1 px-1">
              <AlertCircle size={10} />实验性显示预设，非统一临床标准。不可单独用于诊断。
            </div>
          )}
          {niftiVolume?.intensityUnit === 'unknown' && activePreset === 'liver' && (
            <div className="mt-1 flex items-center gap-1 text-[9px] font-mono text-amber-600 bg-amber-50 border border-amber-200 rounded py-1 px-1">
              <AlertCircle size={10} />研究性肝窗——当前强度单位未知，请确认数据来源。
            </div>
          )}
        </div>

        {/* 上传区域 */}
        <div className="border-t border-slate-200 pt-2">
          <div className="flex items-center gap-3 mb-1">
            <span className="text-[10px] font-mono text-slate-500">新上传的强度类型:</span>
            <label className="flex items-center gap-1 text-[10px] font-mono cursor-pointer"><input type="radio" name="intensityUnit" checked={uploadIntentUnit === 'HU'} onChange={() => setUploadIntentUnit('HU')} className="accent-blue-600" /><span className="text-slate-700">CT（HU）</span></label>
            <label className="flex items-center gap-1 text-[10px] font-mono cursor-pointer"><input type="radio" name="intensityUnit" checked={uploadIntentUnit === 'unknown'} onChange={() => setUploadIntentUnit('unknown')} className="accent-slate-500" /><span className="text-slate-500">未知/其他</span></label>
          </div>
          <label className="flex items-center justify-center gap-2 px-3 py-2 bg-white hover:bg-slate-50 border border-slate-200 hover:border-slate-300 rounded-lg cursor-pointer transition-all shadow-sm text-[10px]">
            <UploadCloud size={14} className="text-blue-600" />浏览并上传 (.nii.gz, .png, .txt, .zip)
            <input type="file" accept=".zip,.nii,.nii.gz,.gz,.dcm,.png,.jpg,.jpeg,.txt" onChange={handleFileChange} className="hidden" />
          </label>
          {uploadedFile && (
            <div className="mt-1 flex items-center justify-between px-3 py-1.5 bg-blue-50/40 rounded-lg border border-blue-200">
              <span className="text-[10px] text-slate-700 truncate font-mono">{uploadedFile.name}</span>
              <span className="text-[9px] text-slate-500 font-mono bg-white px-1.5 py-0.5 rounded border border-slate-150">{uploadedFile.size}</span>
            </div>
          )}
        </div>
      </div>
      )}
    </div>
  );
}
