import React, { useState, useEffect, useRef } from 'react';
import { Globe, Sparkles, Layers, Maximize2 } from 'lucide-react';
import SkillResultRenderer from './SkillResultRenderer';
import SliceViewer from './SliceViewer';

function detectSkillName(data: any): string {
  if (!data || typeof data !== 'object') return '技能分析';
  if (data.liver_volume_cm3 !== undefined || data.tumor_results) return '肝脏综合分析';
  if (data.tumors && Array.isArray(data.tumors)) return '肿瘤直径测量';
  if (data.distances && Array.isArray(data.distances)) return '肿瘤-血管距离';
  if ((data.vessels && typeof data.vessels === 'object') || Object.keys(data).some(k => data[k]?.volume_cm3 !== undefined)) return '血管体积';
  if (data.slices && Array.isArray(data.slices)) return '关键切片选取';
  if (data.html_path || data.html_url) return '3D 模型重建';
  if (data.editor_url) return '分割编辑器';
  if (data._skill_name) {
    const nameMap: Record<string, string> = { 'segmentation_modification': '分割编辑器', 'plan_resection': '手术切除规划' };
    return nameMap[data._skill_name] || data._skill_name.replace(/_/g, ' ');
  }
  return '技能分析';
}

interface RendererPanelProps {
  active3dHtmlUrl: string | null;
  currentCase: any;
  workflowState: string;
  onMaximize3d: () => void;
  skillResults: any[];
  navigateTo?: { tab: string; skillIdx?: number; skillName?: string } | null;
  threeDMeta?: any;
  // 2D slice viewer props
  niftiVolume?: any;
  currentSliceIndex?: number;
  setSliceIndex?: (idx: number) => void;
  agentStatus?: string;
  onFileUpload?: (file: File, options?: { intensityUnit?: 'HU' | 'unknown' }) => void;
  lesionImageUrl?: string | null;
  bestSliceIndex?: number | null;
  bestSlices?: any[];
  hasPendingNifti?: boolean;
  onUpdateIntensityUnit?: (volumeId: string, unit: 'HU' | 'unknown') => void;
}

const RendererPanel: React.FC<RendererPanelProps> = ({
  active3dHtmlUrl, currentCase, workflowState, onMaximize3d,
  skillResults, navigateTo,
  niftiVolume, currentSliceIndex, setSliceIndex, agentStatus,
  onFileUpload, lesionImageUrl, bestSliceIndex, bestSlices,
  threeDMeta, hasPendingNifti, onUpdateIntensityUnit,
}) => {
  const show3d = !!active3dHtmlUrl;
  const show2d = !!niftiVolume || !!hasPendingNifti;
  const hasSkills = skillResults.length > 0;

  const LS_RENDERER_TAB = 'CLINICAL_RENDERER_ACTIVE_TAB';

  // 从 localStorage 恢复上次活动的 tab，避免刷新后跳转
  const [activeTab, setActiveTab] = useState<string>(() => {
    try {
      const saved = typeof window !== 'undefined' ? localStorage.getItem(LS_RENDERER_TAB) : null;
      if (saved) {
        const validTabs: string[] = [];
        if (show2d) validTabs.push('2d');
        if (show3d) validTabs.push('3d');
        if (hasSkills) skillResults.forEach((_, i) => validTabs.push(`skills_${i}`));
        if (validTabs.includes(saved)) return saved;
      }
    } catch {}
    return show2d ? '2d' : show3d ? '3d' : hasSkills ? 'skills_0' : '3d';
  });

  // activeTab 变化时自动保存
  useEffect(() => {
    try { localStorage.setItem(LS_RENDERER_TAB, activeTab); } catch {}
  }, [activeTab]);

  useEffect(() => {
    if (!navigateTo) return;
    if (navigateTo.tab === 'skills') {
      let idx = navigateTo.skillIdx;
      if (navigateTo.skillName && skillResults.length > 0) {
        const foundIdx = skillResults.findIndex((r: any) => {
          if (!r || typeof r !== 'object') return false;
          const name = navigateTo.skillName!.toLowerCase();
          return (name === 'slice_selection' && r.slices?.length > 0) ||
                 (name === 'three_d_reconstruction' && !!r.html_url) ||
                 (name === 'liver_analysis' && r.liver_volume_cm3 !== undefined) ||
                 (name === 'tumor_diameter' && r.tumors?.length > 0) ||
                 (name === 'tumor_vessel_distance' && r.distances?.length > 0) ||
                 (name === 'vessel_volume' && (r.vessels || Object.values(r).some((v: any) => v?.volume_cm3 !== undefined)));
        });
        if (foundIdx >= 0) idx = foundIdx;
      }
      setActiveTab(`skills_${idx ?? 0}`);
    } else {
      setActiveTab(navigateTo.tab);
    }
  }, [navigateTo, skillResults]);

  // 标记挂载时 niftiVolume 是否可用（用于判断是否是从 IndexedDB 异步恢复）
  const was2dAvailableOnMountRef = useRef(!!niftiVolume);
  // 记录挂载时 localStorage 中是否有已保存的 tab 偏好（刷新前用户的选择）
  const priorTabRef = useRef<string | null>(null);
  if (priorTabRef.current === null) {
    try { priorTabRef.current = localStorage.getItem(LS_RENDERER_TAB); } catch {}
  }

  // Auto-switch to 2D when niftiVolume becomes available after mount
  //（仅在首次上传场景下自动切到 2D；刷新后如果有保存的 tab 偏好则保留原 tab）
  useEffect(() => {
    if (!was2dAvailableOnMountRef.current && niftiVolume) {
      was2dAvailableOnMountRef.current = true;
      // 没有保存的 tab 偏好时才自动切到 2D（首次上传 NIfTI 的场景）
      // 有保存的偏好时说明是刷新恢复，应保留用户之前的 tab
      if (!priorTabRef.current && activeTab !== '2d') {
        setActiveTab('2d');
      }
    }
  }, [niftiVolume, activeTab]);

  return (
    <div className="flex flex-col h-full min-h-0 bg-white rounded-lg border border-slate-200">
      {/* Tabs Header */}
      <div className="flex items-center px-2 bg-slate-50 border-b border-slate-200 flex-shrink-0 overflow-x-auto">
        {show2d && (
          <button
            onClick={() => setActiveTab('2d')}
            className={`flex items-center gap-1.5 px-3 py-2 text-[11px] font-bold font-mono transition-all cursor-pointer whitespace-nowrap border-b-2 ${
              activeTab === '2d' ? 'text-blue-600 border-blue-600' : 'text-slate-500 border-transparent hover:text-slate-700 hover:border-slate-300'
            }`}
          >
            <Layers size={14} /> 2D 切片
          </button>
        )}
        {show3d && (
          <button
            onClick={() => setActiveTab('3d')}
            className={`flex items-center gap-1.5 px-3 py-2 text-[11px] font-bold font-mono transition-all cursor-pointer whitespace-nowrap border-b-2 ${
              activeTab === '3d' ? 'text-blue-600 border-blue-600' : 'text-slate-500 border-transparent hover:text-slate-700 hover:border-slate-300'
            }`}
          >
            <Globe size={14} /> 3D 重建
          </button>
        )}
        {skillResults.map((result: any, idx: number) => {
          const skillName = detectSkillName(result);
          const tabKey = `skills_${idx}`;
          return (
            <button
              key={tabKey}
              onClick={() => setActiveTab(tabKey)}
              className={`flex items-center gap-1.5 px-3 py-2 text-[11px] font-bold font-mono transition-all cursor-pointer whitespace-nowrap border-b-2 ${
                activeTab === tabKey ? 'text-blue-600 border-blue-600' : 'text-slate-500 border-transparent hover:text-slate-700 hover:border-slate-300'
              }`}
            >
              <Sparkles size={14} /> {skillName}
            </button>
          );
        })}
      </div>

      {/* 2D Slice Tab */}
      {activeTab === '2d' && (
        <div className="flex-1 min-h-0 flex flex-col">
          {hasPendingNifti && !niftiVolume ? (
            <div className="flex-1 flex items-center justify-center bg-slate-50">
              <div className="text-center text-slate-400">
                <Layers className="w-10 h-10 mx-auto mb-3 animate-pulse text-slate-300" />
                <p className="text-xs font-mono text-slate-400">影像数据加载中...</p>
                <p className="text-[10px] text-slate-300 mt-1">正在从本地缓存恢复</p>
              </div>
            </div>
          ) : (
            <SliceViewer
              currentSliceIndex={currentSliceIndex ?? 0}
              setSliceIndex={setSliceIndex ?? (() => {})}
              totalSlices={currentCase?.sliceCount || (niftiVolume?.depth || 120)}
              caseId={currentCase?.id || ''}
              status={agentStatus || 'idle'}
              onFileUpload={onFileUpload || (() => {})}
              organ={currentCase?.organ}
              niftiVolume={niftiVolume || null}
              isFullscreen={false}
              lesionImageUrl={lesionImageUrl || null}
              bestSliceIndex={bestSliceIndex ?? null}
              bestSlices={bestSlices || []}
              onUpdateIntensityUnit={onUpdateIntensityUnit}
            />
          )}
        </div>
      )}

      {/* 3D Tab */}
      {activeTab === '3d' && (
        <div className="flex-1 bg-slate-950 relative overflow-hidden flex flex-col min-h-0">
          {typeof active3dHtmlUrl === 'string' && active3dHtmlUrl ? (
            <>
              <iframe src={active3dHtmlUrl} className="flex-1 bg-slate-950 w-full" style={{ border: 'none' }} title="3D" />
              {onMaximize3d && (
                <button onClick={onMaximize3d}
                  className="absolute bottom-3 right-3 z-20 flex items-center justify-center p-1.5 bg-white/95 hover:bg-white text-slate-700 hover:text-blue-600 rounded-md border border-slate-200 shadow-md hover:shadow-lg transition-all cursor-pointer"
                  title="全屏"><Maximize2 size={13} /></button>
              )}
            </>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-400 font-mono p-6 text-center">
              <Globe className="w-8 h-8 mb-2 text-slate-600 animate-pulse" />
              <p className="text-xs font-semibold text-slate-300">
                {workflowState === 'running' ? '等待后端返回 3D 模型...' : '3D 重建模型'}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Skill Tabs */}
      {skillResults.map((result: any, idx: number) => {
        if (activeTab !== `skills_${idx}`) return null;
        return (
          <div key={idx} className="flex-1 flex flex-col min-h-0 overflow-y-auto scrollbar-thin">
            <SkillResultRenderer data={result} />
          </div>
        );
      })}
    </div>
  );
};

export default RendererPanel;
