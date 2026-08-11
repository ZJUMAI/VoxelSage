import React from 'react';
import { GitBranch, MapPin, Eye, Globe, Maximize2, X, ZoomIn, ZoomOut, RotateCcw, RefreshCw, ChevronLeft, ChevronRight, ExternalLink } from 'lucide-react';
import { normalizePortBUrl, normalizeSlices } from '../utils/portBUrlHelper';

// ============================================================
// 类型定义（与 Port B 官方标准输出格式严格对齐）
// ============================================================
interface VesselVolumeItem {
  volume_mm3?: number;
  volume_cm3: number;
  voxel_count: number;
}

/** tumor_vessel_distance 的标准输出结构 */
interface TumorVesselDistanceResult {
  distances: Array<{
    tumor_id: string;
    vessel_name: string;
    min_distance_mm: number;
    interpretation: string;
    tumor_contacts_vessel?: boolean;
    tumor_voxels?: number;
  }>;
}

/** vessel_volume 的标准输出结构 — API 返回 {hepatic: {...}, portal: {...}} 在顶层 */
interface VesselVolumeResult {
  [vesselName: string]: VesselVolumeItem | any;
}

/** slice_selection 的标准输出结构 */
interface SliceSelectionResult {
  slices: Array<{
    index: number;
    score: number;
    png_path?: string;
    overlay_path?: string;
    png_url?: string;
    overlay_url?: string;
  }>;
  top_k?: number;
  scoring_mode?: string;
}

/** three_d_reconstruction 的标准输出结构 */
interface ThreeDReconstructionResult {
  html_path?: string;
  html_url?: string;
  filename?: string;
  organ_count?: number;
  organs?: string[];
}

// ============================================================
// 风险等级颜色
// ============================================================
const riskColor = (interpretation: string) => {
  const i = interpretation.toLowerCase();
  if (i.includes('高度') || i.includes('可疑') || i.includes('<1')) return { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200', badge: 'bg-red-500' };
  if (i.includes('邻近') || i.includes('1-5')) return { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200', badge: 'bg-amber-500' };
  if (i.includes('中等') || i.includes('5-20')) return { bg: 'bg-yellow-50', text: 'text-yellow-700', border: 'border-yellow-200', badge: 'bg-yellow-500' };
  return { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200', badge: 'bg-emerald-500' };
};

// ============================================================
// 中英翻译工具（Port B 返回的血管名/肿瘤名为英文）
// ============================================================
const VESSEL_NAME_MAP: Record<string, string> = {
  'hepatic': '肝静脉',
  'portal': '门静脉',
  'hepatic_vein': '肝静脉',
  'hepatic_veins': '肝静脉',
  'portal_vein': '门静脉',
  'portal_veins': '门静脉',
  'hv': '肝静脉',
  'pv': '门静脉',
  'ivc': '下腔静脉',
};

/** 血管名英译中：hepatic → 肝静脉；未命中时保留原名 */
const translateVesselName = (name: string): string => {
  const key = name.trim().toLowerCase();
  return VESSEL_NAME_MAP[key] ?? name.replace(/_/g, ' ');
};

// ============================================================
// 自动检测 skill 类型
// ============================================================
type SkillType = 'liver_analysis' | 'tumor_diameter' | 'tumor_vessel_distance' | 'vessel_volume' | 'slice_selection' | 'three_d_reconstruction' | 'segmentation_modification' | 'unknown';

/** 检查一个对象是否是 vessel_volume 格式：顶层键为血管名，值为 {volume_cm3, ...} */
function isVesselVolumeData(data: any): boolean {
  if (!data || typeof data !== 'object') return false;
  // 已知非 vessel volume 的顶层 key
  const knownNonVessel = new Set([
    'liver_volume_cm3', 'tumor_results', 'vessel_volumes',
    'tumors', 'distances', 'vessels', 'slices', 'html_path', 'html_url',
    'status', 'message', 'result', 'skill', 'skill_name', 'case_id',
  ]);
  const keys = Object.keys(data);
  if (keys.length === 0) return false;
  // 如果所有 key 都像血管名（不含下划线前缀、不匹配已知字段），则判定为 vessel_volume
  const vesselLikeKeys = keys.filter(k => !knownNonVessel.has(k) && !k.startsWith('_') && k !== 'execution_time_ms');
  if (vesselLikeKeys.length === 0) return false;
  // 检查这些 key 对应的值是否包含 volume_cm3
  return vesselLikeKeys.some(k => data[k] && typeof data[k] === 'object' && 'volume_cm3' in data[k]);
}

function detectSkill(data: any): SkillType {
  if (!data || typeof data !== 'object') return 'unknown';
  if (data.liver_volume_cm3 !== undefined || data.tumor_results) return 'liver_analysis';
  if (data.tumors && Array.isArray(data.tumors)) return 'tumor_diameter';
  if (data.distances && Array.isArray(data.distances)) return 'tumor_vessel_distance';
  // vessel_volume: 兼容两种格式 — {vessels: {...}} 或 顶层直接是血管名
  if (data.vessels && typeof data.vessels === 'object') return 'vessel_volume';
  if (isVesselVolumeData(data)) return 'vessel_volume';
  if (data.slices && Array.isArray(data.slices)) return 'slice_selection';
  if (data.html_path || data.html_url) return 'three_d_reconstruction';
  if (data.editor_url) return 'segmentation_modification';
  return 'unknown';
}

// ============================================================
// 通用小组件
// ============================================================

/** 血管体积列表 */
const VesselVolumeList: React.FC<{ vessels: Record<string, VesselVolumeItem>; liverVolume?: number }> = ({ vessels, liverVolume }) => (
  <div className="bg-white border border-slate-200 rounded-xl p-3.5 shadow-sm">
    <div className="flex items-center gap-1.5 border-b border-slate-100 pb-2 mb-2.5">
      <GitBranch size={14} className="text-purple-600" />
      <h4 className="text-[10px] font-bold text-slate-700 uppercase tracking-wider font-mono">血管体积</h4>
    </div>
    <div className="space-y-2">
      {Object.entries(vessels).map(([name, vv]) => {
        const vol = vv?.volume_cm3;
        if (vol === undefined || vol === null) return null;
        const ratio = liverVolume && liverVolume > 0 ? (vol / liverVolume * 100) : null;
        return (
          <div key={name} className="flex items-center justify-between py-1 border-b border-slate-50 last:border-0">
            <span className="text-[11px] font-medium text-slate-700">{translateVesselName(name)}</span>
            <div className="text-right">
              <span className="text-xs font-bold text-slate-800 font-mono">{vol?.toFixed(2) ?? "0"} cm³</span>
              {ratio !== null && <span className="text-[9px] text-slate-400 ml-1.5">({ratio?.toFixed(1) ?? "0"}%)</span>}
            </div>
          </div>
        );
      })}
    </div>
  </div>
);

// ============================================================
// 各 Skill 主渲染组件
// ============================================================

/** 肿瘤-血管距离 */
const TumorVesselDistanceView: React.FC<{ data: TumorVesselDistanceResult }> = ({ data }) => {
  if (!data.distances || data.distances.length === 0) {
    return <div className="text-[10px] text-slate-500 italic text-center py-4">未返回距离数据</div>;
  }
  return (
    <div className="flex-1 flex flex-col min-h-0 space-y-2">
      <div className="flex items-center gap-1.5 border-b border-slate-100 pb-2">
        <MapPin size={13} className="text-blue-600" />
        <h3 className="text-[10px] font-bold text-slate-700 uppercase tracking-wider font-mono">肿瘤-血管距离</h3>
      </div>
      <div className="space-y-1.5">
        {data.distances.map((d, i) => {
          const rc = riskColor(d.interpretation);
          return (
            <div key={i} className={`${rc.bg} border ${rc.border} rounded-lg p-2.5 flex items-center justify-between`}>
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-[10px] font-bold text-slate-700 font-mono flex-shrink-0">{d.tumor_id}</span>
                <span className="text-[9px] text-slate-400 flex-shrink-0">→</span>
                <span className="text-[10px] text-slate-600 truncate">{d.vessel_name.replace(/_/g, ' ')}</span>
              </div>
              <div className="flex items-center gap-1.5 flex-shrink-0 ml-2">
                {d.tumor_contacts_vessel && (
                  <span className="px-1 py-0.5 rounded text-[8px] font-bold bg-red-100 text-red-600 border border-red-200">接触</span>
                )}
                <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${rc.text} ${rc.bg} border-0`}>{d.min_distance_mm != null ? d.min_distance_mm.toFixed(1) : "-"} mm</span>
                <span className={`px-1.5 py-0.5 rounded text-[8px] font-bold bg-white border ${rc.border} ${rc.text} hidden sm:inline`}>{d.interpretation}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

/** 血管体积 — 适配 API 文档标准格式：顶层键为血管名 */
const VesselVolumeView: React.FC<{ data: VesselVolumeResult }> = ({ data }) => {
  // 兼容两种格式：{vessels: {...}} 或 顶层直接为 {hepatic: {...}, portal: {...}}
  let vessels: Record<string, VesselVolumeItem>;
  if (data.vessels && typeof data.vessels === 'object') {
    vessels = data.vessels as Record<string, VesselVolumeItem>;
  } else {
    // 提取顶层所有类血管的键（有 volume_cm3 属性的）
    vessels = {} as Record<string, VesselVolumeItem>;
    for (const key of Object.keys(data)) {
      if (data[key] && typeof data[key] === 'object' && 'volume_cm3' in data[key]) {
        vessels[key] = data[key] as VesselVolumeItem;
      }
    }
  }
  const entries = Object.entries(vessels);
  if (entries.length === 0) return <div className="text-[10px] text-slate-500 italic text-center py-4">无数据</div>;
  return (
    <div className="flex-1 flex flex-col min-h-0 space-y-2">
      <div className="flex items-center gap-1.5 border-b border-slate-100 pb-2">
        <GitBranch size={13} className="text-purple-600" />
        <h3 className="text-[10px] font-bold text-slate-700 uppercase tracking-wider font-mono">血管体积</h3>
      </div>
      <VesselVolumeList vessels={vessels} />
    </div>
  );
};

/** 切片选取 — 展示实际切片图片，支持 _path/_url 两种格式 */
/** 切片选取 — 展示实际切片图片，支持 _path/_url 两种格式 */
const SliceSelectionView: React.FC<{ data: SliceSelectionResult }> = ({ data }) => {
  const slices = normalizeSlices(data.slices || []);
  if (slices.length === 0) {
    return <div className="text-[10px] text-slate-500 italic text-center py-4">未返回切片数据</div>;
  }
  const [selectedIdx, setSelectedIdx] = React.useState(0);
  const [fullscreenOpen, setFullscreenOpen] = React.useState(false);
  const [fsZoom, setFsZoom] = React.useState(1);
  const [fsRotation, setFsRotation] = React.useState(0);
  const [viewMode, setViewMode] = React.useState<'both' | 'original' | 'overlay'>('both');
  const current = slices[selectedIdx];

  const resetFsState = () => { setFsZoom(1); setFsRotation(0); };
  React.useEffect(() => { if (fullscreenOpen) resetFsState(); }, [fullscreenOpen, selectedIdx]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setFsZoom(z => Math.max(0.25, Math.min(3, z - e.deltaY * 0.005)));
  };

  const renderFsImage = (src: string, label: string) => (
    <div className="flex-1 h-full flex items-center justify-center overflow-hidden relative">
      <img src={src} alt={label}
        style={{ transform: `scale(${fsZoom}) rotate(${fsRotation}deg)`, maxWidth: '100%', maxHeight: '100%' }}
        className="object-contain transition-transform duration-100" referrerPolicy="no-referrer" />
      <span className="absolute bottom-2 left-2 bg-black/70 text-[9px] text-slate-300 font-mono px-2 py-0.5 rounded">{label}</span>
    </div>
  );

  return (
    <div className="flex-1 flex flex-col min-h-0 space-y-2">
      <div className="flex items-center gap-1.5 border-b border-slate-100 pb-2">
        <Eye size={13} className="text-blue-600" />
        <h3 className="text-[10px] font-bold text-slate-700 uppercase tracking-wider font-mono">关键切片选取</h3>
        <span className="text-[8px] text-slate-400 ml-auto">评分模式: {data.scoring_mode || 'crlm'}</span>
      </div>
      <div className="relative rounded-lg overflow-hidden border border-slate-200 bg-black" style={{ minHeight: '70vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div className="grid grid-cols-2 gap-0.5 h-full w-full">
          <div className="relative bg-slate-900 flex items-center justify-center overflow-hidden">
            <img src={current.png_url} alt={`切片 \${current.index}`} className="w-full h-full" style={{ objectFit: "contain" }} referrerPolicy="no-referrer" />
            <div className="absolute bottom-1 left-1 bg-black/70 text-[7px] text-slate-300 font-mono px-1.5 py-0.5 rounded">原CT图像</div>
          </div>
          <div className="relative bg-slate-900 border-l border-slate-700 flex items-center justify-center overflow-hidden">
            <img src={current.overlay_url} alt={`切片 \${current.index} 标注`} className="w-full h-full" style={{ objectFit: "contain" }} referrerPolicy="no-referrer" />
            <div className="absolute bottom-1 right-1 bg-rose-600/70 text-[7px] text-white font-mono px-1.5 py-0.5 rounded">器官标注图像</div>
          </div>
        </div>
        <button onClick={() => setFullscreenOpen(true)} className="absolute top-2 right-2 z-10 flex items-center justify-center p-1.5 bg-black/60 hover:bg-black/80 text-white rounded-md transition-all cursor-pointer" title="全屏查看"><Maximize2 size={14} /></button>
      </div>
      <div className="flex items-center gap-1.5 overflow-x-auto py-1">
        {slices.map((s, i) => (
          <button key={i} onClick={() => setSelectedIdx(i)} className={`flex-shrink-0 flex flex-col items-center gap-0.5 px-2 py-1 rounded-lg transition-all cursor-pointer border \${selectedIdx === i ? 'bg-blue-100 border-blue-300 text-blue-700' : 'bg-white border-slate-200 text-slate-500 hover:bg-slate-50'}`}>
            <span className="text-[9px] font-bold font-mono">#{s.index}</span>
            {s.score != null && <span className="text-[7px] font-mono">{(s.score * 100).toFixed(0)}%</span>}
          </button>
        ))}
      </div>

      {fullscreenOpen && (
        <div className="fixed inset-0 z-50 bg-black/95 flex flex-col" onClick={() => setFullscreenOpen(false)}>
          <div className="flex items-center justify-between px-6 py-3 bg-slate-900/90 border-b border-slate-800 flex-shrink-0 select-none" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-3 text-xs text-slate-300 font-mono">
              <span className="text-white font-bold">切片 #{current.index}</span>
              <span className="text-slate-600">|</span>
              <span>评分: {current.score ? (current.score * 100).toFixed(0) + '%' : 'N/A'}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center bg-slate-800 rounded-lg overflow-hidden border border-slate-700 text-[10px] font-bold">
                <button onClick={() => setViewMode('original')} className={`px-2.5 py-1 transition-colors \${viewMode === 'original' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}>原CT图像</button>
                <button onClick={() => setViewMode('overlay')} className={`px-2.5 py-1 transition-colors border-x border-slate-700 \${viewMode === 'overlay' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}>器官标注图像</button>
                <button onClick={() => setViewMode('both')} className={`px-2.5 py-1 transition-colors \${viewMode === 'both' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}>双图对比</button>
              </div>
              <div className="w-px h-5 bg-slate-700" />
              <button onClick={() => setFsZoom(z => Math.min(z + 0.25, 3))} className="p-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-md transition-all cursor-pointer border border-slate-700" title="放大"><ZoomIn size={14} /></button>
              <button onClick={() => setFsZoom(z => Math.max(z - 0.25, 0.25))} className="p-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-md transition-all cursor-pointer border border-slate-700" title="缩小"><ZoomOut size={14} /></button>
              <span className="text-[10px] text-slate-400 font-mono w-10 text-center">{Math.round(fsZoom * 100)}%</span>
              <div className="w-px h-5 bg-slate-700" />
              <button onClick={() => setFsRotation(r => (r + 90) % 360)} className="p-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-md transition-all cursor-pointer border border-slate-700" title="旋转"><RotateCcw size={14} /></button>
              <button onClick={resetFsState} className="p-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-md transition-all cursor-pointer border border-slate-700" title="复原"><RefreshCw size={14} /></button>
              <div className="w-px h-5 bg-slate-700" />
              <button onClick={() => { setSelectedIdx((selectedIdx - 1 + slices.length) % slices.length); resetFsState(); }} className="p-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-md transition-all cursor-pointer border border-slate-700" title="上一张"><ChevronLeft size={14} /></button>
              <button onClick={() => { setSelectedIdx((selectedIdx + 1) % slices.length); resetFsState(); }} className="p-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-md transition-all cursor-pointer border border-slate-700" title="下一张"><ChevronRight size={14} /></button>
              <div className="w-px h-5 bg-slate-700" />
              <button onClick={() => setFullscreenOpen(false)} className="p-1.5 bg-slate-800 hover:bg-red-700 text-white rounded-md transition-all cursor-pointer border border-slate-700" title="关闭"><X size={14} /></button>
            </div>
          </div>
          <div className="flex-1 flex items-center justify-center min-h-0" onWheel={handleWheel} onClick={e => e.stopPropagation()}>
            {viewMode === 'both' ? (
              <div className="flex items-center justify-center w-full h-full p-4 gap-2">
                {renderFsImage(current.png_url, '原CT图像')}
                <div className="w-px h-3/4 bg-slate-800 flex-shrink-0" />
                {renderFsImage(current.overlay_url, '器官标注图像')}
              </div>
            ) : (
              <div className="flex items-center justify-center w-full h-full p-4">
                {renderFsImage(viewMode === 'original' ? current.png_url : current.overlay_url, viewMode === 'original' ? '原CT图像' : '器官标注图像')}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};const ThreeDReconstructionView: React.FC<{ data: ThreeDReconstructionResult }> = ({ data }) => {
  const htmlUrl = data.html_url || normalizePortBUrl(data.html_path);
  return (
    <div className="flex-1 flex flex-col min-h-0 space-y-2">
      <div className="flex items-center gap-1.5 border-b border-slate-100 pb-2">
        <Globe size={13} className="text-emerald-600" />
        <h3 className="text-[10px] font-bold text-slate-700 uppercase tracking-wider font-mono">3D 模型重建</h3>
        {data.organ_count != null && <span className="text-[8px] text-slate-400 ml-auto">{data.organ_count} 个器官</span>}
      </div>
      {htmlUrl ? (
        <div className="rounded-lg overflow-hidden border border-slate-200 bg-slate-950" style={{ height: '360px' }}>
          <iframe src={htmlUrl} className="w-full h-full" style={{ border: 'none' }} title="3D Reconstructed Model" />
        </div>
      ) : (
        <div className="text-[10px] text-slate-500 italic text-center py-4">未返回 3D 数据</div>
      )}
    </div>
  );
};

/** 未知数据 */

/** 分割编辑 — 返回 MedSAM2 交互式编辑器链接 */
const SegmentationModificationView: React.FC<{ data: any }> = ({ data }) => {
  const editorUrl = data.editor_url || '';
  const [fsOpen, setFsOpen] = React.useState(false);

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {editorUrl ? (
        <div className="flex-1 relative bg-slate-950 overflow-hidden">
          <iframe src={editorUrl} className="w-full h-full" style={{ border: 'none' }} title="分割编辑器" />
          <button onClick={() => setFsOpen(true)}
            className="absolute top-4 right-4 z-10 flex items-center justify-center p-2.5 bg-white/90 hover:bg-white text-slate-700 rounded-lg transition-all cursor-pointer shadow-lg"
            title="全屏查看"><Maximize2 size={22} /></button>
          <a href={editorUrl} target="_blank" rel="noopener noreferrer"
            className="absolute bottom-4 right-4 z-10 inline-flex items-center gap-1.5 px-3.5 py-2 bg-white/90 hover:bg-white text-slate-700 rounded-lg text-xs font-bold transition-all shadow-lg">
            <ExternalLink size={16} />
            <span>新窗口打开</span>
          </a>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-sm text-slate-500 italic">未返回编辑器链接</div>
      )}

      {fsOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950 flex flex-col" onClick={() => setFsOpen(false)}>
          <div className="flex items-center justify-between px-6 py-4 bg-slate-900/90 border-b border-slate-800 flex-shrink-0" onClick={e => e.stopPropagation()}>
            <span className="text-sm text-white font-bold font-mono">分割编辑器</span>
            <div className="flex items-center gap-3">
              <a href={editorUrl} target="_blank" rel="noopener noreferrer"
                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded text-xs font-bold transition-all cursor-pointer">
                <ExternalLink size={14} className="inline mr-1" />新窗口
              </a>
              <button onClick={() => setFsOpen(false)}
                className="px-4 py-1.5 bg-red-800/60 hover:bg-red-700 text-white rounded text-xs font-bold transition-all cursor-pointer">关闭 (ESC)</button>
            </div>
          </div>
          <div className="flex-1 min-h-0" onClick={e => e.stopPropagation()}>
            <iframe src={editorUrl} className="w-full h-full" style={{ border: 'none' }} title="分割编辑器全屏" />
          </div>
        </div>
      )}
    </div>
  );
};

// 确保 Eye 和 ExternalLink 已导入（已有）

const UnknownView: React.FC<{ data: any }> = ({ data }) => (
  <div className="bg-amber-50 border border-amber-200 rounded-xl p-3">
    <div className="text-[10px] font-bold text-amber-700 mb-1">⚠️ 未识别的技能返回数据</div>
    <pre className="text-[8px] text-amber-600 font-mono whitespace-pre-wrap max-h-40 overflow-y-auto">
      {JSON.stringify(data, null, 2).slice(0, 2000)}
    </pre>
  </div>
);

// ============================================================
// 主入口组件
// ============================================================
interface SkillResultRendererProps {
  data: any;
  className?: string;
}

const SkillResultRenderer: React.FC<SkillResultRendererProps> & { detectSkill: typeof detectSkill } = ({ data, className = '' }) => {
  if (!data) return null;
  const skill = detectSkill(data);

  const renderContent = () => {
    switch (skill) {
      case 'tumor_vessel_distance': return <TumorVesselDistanceView data={data as TumorVesselDistanceResult} />;
      case 'vessel_volume': return <VesselVolumeView data={data as VesselVolumeResult} />;
      case 'slice_selection': return <SliceSelectionView data={data as SliceSelectionResult} />;
      case 'three_d_reconstruction': return <ThreeDReconstructionView data={data as ThreeDReconstructionResult} />;
      case 'segmentation_modification': return <SegmentationModificationView data={data} />;
      default: return <UnknownView data={data} />;
    }
  };

  return (
    <div className={`flex-1 flex flex-col min-h-0 ${className}`}>
      {renderContent()}
    </div>
  );
};

SkillResultRenderer.detectSkill = detectSkill;

export default SkillResultRenderer;
