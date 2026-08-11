import { useState, useEffect, useReducer, FormEvent, useRef } from 'react';
import { Globe, FileSpreadsheet, Plus } from 'lucide-react';
import { MedicalCase, AgentStatus, ChatMessage, WorkflowState, ExecutionStatus, AgentExecutionState } from './types';
import DeepSeekSidebar from './components/DeepSeekSidebar';
import ChatSection from './components/ChatSection';
import { uploadFileToServer, WS_CHAT_URL, HTTP_BASE_URL } from './api';
import { parseNiftiFile, NiftiVolume } from './utils/niftiLoader';
import { normalizeSkillResult, normalizePortBUrl } from './utils/portBUrlHelper';
import { agentExecutionReducer, executionStatusToAgentStatus, executionStatusToWorkflowState, buildSnapshot } from './utils/agentExecutionReducer';
import { saveNiftiBlobToCache, loadNiftiBlobFromCache, deleteNiftiVolumeFromCache } from './utils/niftiCache';
import SkillManager from './components/SkillManager';
import RendererPanel from './components/RendererPanel';

// ──────────────────────────────────────────────
// 「会话持久化」localStorage 存储键名 & 工具函数
// ──────────────────────────────────────────────
const LS_CASES = 'CLINICAL_PATIENT_CASES';
const LS_CHAT = 'CLINICAL_CHAT_MESSAGES';
const LS_IMAGES = 'CLINICAL_UPLOADED_IMAGES';
const LS_VOLUMES = 'CLINICAL_UPLOADED_VOLUMES';
const LS_TEXTS = 'CLINICAL_UPLOADED_TEXTS';
const LS_SESSION = 'CLINICAL_SESSION_ID';
const LS_SLICE_INDEX = 'CLINICAL_SLICE_INDEX';
const LS_3D_URL = 'CLINICAL_ACTIVE_3D_URL';
const LS_LESION_URL = 'CLINICAL_ACTIVE_LESION_URL';
const LS_SKILL_RESULTS = 'CLINICAL_SKILL_RESULTS';
const LS_BEST_SLICES = 'CLINICAL_BEST_SLICES';
const LS_SELECTED_CASE = 'CLINICAL_SELECTED_CASE_ID';

function safeParse<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try { return JSON.parse(raw); } catch { return fallback; }
}

function safeStringify(val: any): string {
  try { return JSON.stringify(val); } catch { return ''; }
}

function loadFromLS<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') return fallback;
  return safeParse(localStorage.getItem(key), fallback);
}

const filterOut3dSkill = (results: any[]) => {
  if (!results) return results;
  return results.filter((r: any) => {
    if (r.html_url && (r._skill_name === 'three_d_reconstruction' || (!r._skill_name && !r.tumors && !r.slices && !r.distances && !r.liver_volume_cm3))) return false;
    return true;
  });
};

/** 纯文本技能名：肝脏综合分析 / 肿瘤直径测量 — 结果已在对话输出，不进入右侧渲染面板 */
const TEXT_ONLY_SKILL_NAMES = new Set(['liver_analysis', 'tumor_diameter']);

/** 判断一个 skill 结果是否为纯文本技能（无需右侧渲染） */
const isTextOnlySkill = (r: any): boolean => {
  if (!r || typeof r !== 'object') return false;
  const n = r._skill_name;
  if (n && TEXT_ONLY_SKILL_NAMES.has(n)) return true;
  // 兜底：按字段特征识别（与 detectSkillName 一致）
  if (r.liver_volume_cm3 !== undefined || r.tumor_results) return true; // liver_analysis
  if (Array.isArray(r.tumors)) return true;                              // tumor_diameter
  return false;
};

/** 从 skillResults 中剔除纯文本技能，使其不进入右侧渲染面板 */
const filterOutTextOnlySkills = (results: any[]): any[] => {
  if (!results) return results;
  return results.filter((r: any) => !isTextOnlySkill(r));
};

function saveToLS(key: string, val: any): void {
  if (typeof window === 'undefined') return;
  try { localStorage.setItem(key, safeStringify(val)); } catch { /* quota exceeded — silent */ }
}

// 定义空病例列表，不再预置模拟数据
const APP_VERSION = '2026-07-23_v4';
const LS_VERSION_KEY = 'APP_VERSION';
const MOCK_CASES: MedicalCase[] = [];

export default function App() {
  // ── 清理旧版本 localStorage 残留 ──
  if (typeof window !== 'undefined') {
    const OLD_KEYS = [
      'CLINICAL_CASE_RECORD', 'CLINICAL_EVIDENCE_SUMMARY',
      'CLINICAL_ACTIVE_TAB', 'CLINICAL_WORKFLOW_STATE',
    ];
    OLD_KEYS.forEach(key => { try { localStorage.removeItem(key); } catch {} });
    // ── 版本变更时清除所有 localStorage 缓存，避免旧数据冲突 ──
    const prevVersion = localStorage.getItem(LS_VERSION_KEY);
    if (prevVersion !== APP_VERSION) {
      const safeKeys = [
        LS_VERSION_KEY, 'CLINICAL_PATIENT_CASES', '3dmedagent_api_base_url',
        'CLINICAL_CHAT_MESSAGES', 'CLINICAL_UPLOADED_IMAGES',
        'CLINICAL_UPLOADED_VOLUMES', 'CLINICAL_UPLOADED_TEXTS',
        'CLINICAL_SESSION_ID', 'CLINICAL_SLICE_INDEX',
        'CLINICAL_ACTIVE_3D_URL', 'CLINICAL_ACTIVE_LESION_URL',
        'CLINICAL_SKILL_RESULTS', 'CLINICAL_BEST_SLICES',
      ];
      for (let i = localStorage.length - 1; i >= 0; i--) {
        const key = localStorage.key(i);
        if (key && !safeKeys.includes(key)) {
          try { localStorage.removeItem(key); } catch {}
        }
      }
      localStorage.setItem(LS_VERSION_KEY, APP_VERSION);
      console.log(`[版本] ${prevVersion || '首次'} → ${APP_VERSION}，已保留所有会话数据`);
    }
  }

  const filterCases = (parsed: any[]) =>
    parsed.filter((c: any) =>
      c.id &&
      !c.id.includes('case_liver_01') &&
      !c.id.includes('case_pancreas_02') &&
      !c.id.includes('case_lung_03') &&
      !c.id.includes('case_user_01') &&
      !c.id.startsWith('mock_')
    );

  const [cases, setCases] = useState<MedicalCase[]>(() => {
    const saved = localStorage.getItem(LS_CASES);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) return filterCases(parsed);
      } catch (e) {
        console.error('Failed to parse patient cases', e);
      }
    }
    return [];
  });

  const [selectedCaseId, setSelectedCaseId] = useState<string>(() => {
    // 优先使用持久化的选中 case（与模块级预加载一致）
    const savedId = loadFromLS<string | null>(LS_SELECTED_CASE, null);
    if (savedId) {
      const saved = localStorage.getItem(LS_CASES);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          if (Array.isArray(parsed)) {
            const filtered = filterCases(parsed);
            if (filtered.some((c: any) => c.id === savedId)) return savedId;
          }
        } catch {}
      }
    }
    // 回退到首个 case
    const saved = localStorage.getItem(LS_CASES);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) {
          const filtered = filterCases(parsed);
          if (filtered.length > 0) return filtered[0].id;
        }
      } catch (e) {}
    }
    return '';
  });

  // ── 从 localStorage 恢复会话持久状态 ──
  const [currentSliceIndex, setSliceIndex] = useState<number>(
    () => loadFromLS(LS_SLICE_INDEX, 60)
  );
  const [agentStatus, setAgentStatus] = useState<AgentStatus>('idle');

  // Modal toggle state for patient creation
  const [showCreateModal, setShowCreateModal] = useState<boolean>(false);
  const [editingCaseId, setEditingCaseId] = useState<string | null>(null);

  // Form states for creating a new patient record
  const [newPatientName, setNewPatientName] = useState<string>('');
  const [newPatientGender, setNewPatientGender] = useState<string>('');
  const [newPatientAge, setNewPatientAge] = useState<string>('');
  const [newPatientId, setNewPatientId] = useState<string>('');
  const [newPatientOrgan, setNewPatientOrgan] = useState<string>('Liver (肝脏)');
  const [newPatientHistory, setNewPatientHistory] = useState<string>('');
  const [newPatientQuery, setNewPatientQuery] = useState<string>('');
  const [newPatientSlices, setNewPatientSlices] = useState<string>('120');
  const [newPatientTargetSlice, setNewPatientTargetSlice] = useState<string>('60');


  // 硬编码后端地址，见 src/api.ts

  const [sessionId, setSessionId] = useState<string | null>(() => loadFromLS<string | null>(LS_SESSION, null));
  const [uploadedImages, setUploadedImages] = useState<Array<{ file_id: string; path: string; name: string }>>(
    () => loadFromLS(LS_IMAGES, [])
  );
  const [uploadedVolumes, setUploadedVolumes] = useState<Array<{ file_id: string; path: string; name: string; volume_role: string }>>(
    () => loadFromLS(LS_VOLUMES, [])
  );
  const [uploadedTexts, setUploadedTexts] = useState<Array<{ file_id: string; path: string; name: string }>>(
    () => loadFromLS(LS_TEXTS, [])
  );
  const [active3dHtmlUrl, setActive3dHtmlUrl] = useState<string | null>(
    () => loadFromLS<string | null>(LS_3D_URL, null)
  );
  const [activeLesionImageUrl, setActiveLesionImageUrl] = useState<string | null>(
    () => loadFromLS<string | null>(LS_LESION_URL, null)
  );
  const [threeDMeta, setThreeDMeta] = useState<any>(null);

  // Real-time WebSocket Pipeline states
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [workflowState, setWorkflowState] = useState<WorkflowState>('idle');
  const [answerText, setAnswerText] = useState<string>('');
  const [bestSlices, setBestSlices] = useState<any[]>(
    () => loadFromLS(LS_BEST_SLICES, [])
  );
  const [bestSliceIndex, setBestSliceIndex] = useState<number | null>(null);
  const [skillResults, setSkillResults] = useState<any[]>(
    () => loadFromLS(LS_SKILL_RESULTS, [])
  );
  const [showSkillManager, setShowSkillManager] = useState(false);
  const [navigateTarget, setNavigateTarget] = useState<{ tab: string; skillIdx?: number; skillName?: string } | null>(null);
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);
  // ── 追踪 skillResults 变化，自动导航到最新添加的 skill ──
  const prevSkillResultsLenRef = useRef<number>(0);
  useEffect(() => {
    if (skillResults.length > prevSkillResultsLenRef.current) {
      // 新增了 skill → 导航到最后一个（最新添加的）
      setNavigateTarget({ tab: 'skills', skillIdx: skillResults.length - 1 });
    }
    prevSkillResultsLenRef.current = skillResults.length;
  }, [skillResults]);

  // ── 3D 重建完成时自动导航到 3D tab ──
  // three_d_reconstruction 不走 skillResults，单独通过 active3dHtmlUrl 控制，
  // 所以需要独立追踪其变化。初始值（刷新恢复）不触发导航。
  const prev3dUrlRef = useRef<string | null>(active3dHtmlUrl);
  useEffect(() => {
    // active3dHtmlUrl 从 null 变为有值 → 3D 重建刚完成 → 跳转到 3D tab
    if (active3dHtmlUrl && prev3dUrlRef.current === null) {
      setNavigateTarget({ tab: '3d' });
    }
    prev3dUrlRef.current = active3dHtmlUrl;
  }, [active3dHtmlUrl]);

  // Store client-side parsed NIfTI volumes mapped by case ID
  const [niftiVolumes, setNiftiVolumes] = useState<Record<string, NiftiVolume>>({});
  // 用 ref 同步 niftiVolumes，供 case-switch effect 中读取当前值（无需加入依赖数组）
  const niftiVolumesRef = useRef<Record<string, NiftiVolume>>({});
  niftiVolumesRef.current = niftiVolumes;

  // ── Agent 执行追踪状态 ──
  const [agentExecutionStates, dispatchAgentExecution] = useReducer(agentExecutionReducer, {});
  const [currentExecutionId, setCurrentExecutionId] = useState<string | null>(null);
  // ref 追踪最新的 execId — 解决 WebSocket 闭包陈旧问题（multi-round 复用同一个 WS）
  const currentExecIdRef = useRef<string | null>(null);
  // ref 同步 agentExecutionStates — 避免 WS onmessage 中读取到陈旧状态
  const agentExecutionStatesRef = useRef(agentExecutionStates);
  agentExecutionStatesRef.current = agentExecutionStates;



  // ── 分栏拖拽调整 ──
  // 卡顿根源：拖动过程中每次 mousemove 都 setState → 触发整棵 App 树重渲染。
  // 改为：拖动期间用 ref 直接写 DOM 宽度（零 React 重渲染），mouseup 时一次性把最终值提交回 state。
  // 子组件（SliceViewer）自带 ResizeObserver，容器宽度变化会自动适配，无需依赖重渲染。

  // 左侧栏（患者档案）宽度调整
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);
  const [sidebarWidth, setSidebarWidth] = useState<number>(288);
  const [isResizing, setIsResizing] = useState<boolean>(false);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const sidebarInnerRef = useRef<HTMLDivElement>(null);
  const sidebarResizeRef = useRef<{ startX: number; startWidth: number }>({ startX: 0, startWidth: 288 });

  useEffect(() => {
    if (!isResizing) return;
    const finish = () => {
      const el = sidebarRef.current;
      const w = el ? parseFloat(el.style.width) : NaN;
      if (!Number.isNaN(w)) setSidebarWidth(w);
      setIsResizing(false);
    };
    const handleMouseMove = (e: MouseEvent) => {
      if (e.buttons === 0) { finish(); return; } // 在窗口外松开时的兜底
      const { startX, startWidth } = sidebarResizeRef.current;
      const w = Math.max(180, Math.min(500, startWidth + (e.clientX - startX)));
      if (sidebarRef.current) sidebarRef.current.style.width = `${w}px`;
      if (sidebarInnerRef.current) sidebarInnerRef.current.style.width = `${w}px`;
    };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', finish);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', finish);
    };
  }, [isResizing]);

  const handleSidebarDragStart = (e: React.MouseEvent) => {
    e.preventDefault();
    sidebarResizeRef.current = { startX: e.clientX, startWidth: sidebarWidth };
    setIsResizing(true);
  };

  // 中间分界线（对话栏 | 渲染器）调整
  const [middleRatio, setMiddleRatio] = useState(0.5);
  const [isResizingMiddle, setIsResizingMiddle] = useState(false);
  const chatPanelRef = useRef<HTMLDivElement>(null);
  const renderPanelRef = useRef<HTMLDivElement>(null);
  const middleResizeRef = useRef<{ startX: number; startRatio: number; rectLeft: number; rectWidth: number }>({ startX: 0, startRatio: 0.5, rectLeft: 0, rectWidth: 1 });

  useEffect(() => {
    if (!isResizingMiddle) return;
    const finish = () => {
      const el = chatPanelRef.current;
      if (el) {
        const w = parseFloat(el.style.width);
        if (!Number.isNaN(w)) setMiddleRatio(w / 100);
      }
      setIsResizingMiddle(false);
    };
    const handleMouseMove = (e: MouseEvent) => {
      if (e.buttons === 0) { finish(); return; } // 在窗口外松开时的兜底
      const { rectLeft, rectWidth } = middleResizeRef.current;
      if (rectWidth <= 0) return;
      const ratio = Math.max(0.25, Math.min(0.75, (e.clientX - rectLeft) / rectWidth));
      if (chatPanelRef.current) chatPanelRef.current.style.width = `${ratio * 100}%`;
      if (renderPanelRef.current) renderPanelRef.current.style.width = `${(1 - ratio) * 100}%`;
    };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', finish);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', finish);
    };
  }, [isResizingMiddle]);

  // Fullscreen states
  const [isMeshFullscreen, setIsMeshFullscreen] = useState<boolean>(false);

  // ESC key listener to exit full-screen modes
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsMeshFullscreen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  // Loaded case reference (can be null if cases is empty)
  const currentCase = cases.find((c) => c.id === selectedCaseId) || cases[0] || null;

  // Chat messages state — restored from localStorage on mount
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(
    () => loadFromLS<ChatMessage[]>(LS_CHAT, [])
  );

  // ── 离线持久化：每次状态变更自动保存到 localStorage ──
  useEffect(() => { saveToLS(LS_CHAT, chatMessages); }, [chatMessages]);
  useEffect(() => { saveToLS(LS_IMAGES, uploadedImages); }, [uploadedImages]);
  useEffect(() => { saveToLS(LS_VOLUMES, uploadedVolumes); }, [uploadedVolumes]);
  useEffect(() => { saveToLS(LS_TEXTS, uploadedTexts); }, [uploadedTexts]);
  useEffect(() => { saveToLS(LS_SESSION, sessionId); }, [sessionId]);
  useEffect(() => { saveToLS(LS_SLICE_INDEX, currentSliceIndex); }, [currentSliceIndex]);
  useEffect(() => { saveToLS(LS_3D_URL, active3dHtmlUrl); }, [active3dHtmlUrl]);
  useEffect(() => { saveToLS(LS_LESION_URL, activeLesionImageUrl); }, [activeLesionImageUrl]);
  // ── 持久化当前选中的 caseId（供模块级预加载和刷新恢复使用）──
  useEffect(() => { saveToLS(LS_SELECTED_CASE, selectedCaseId); }, [selectedCaseId]);
  // ── 自动保存 cases — 确保刷新后患者列表不丢失 ──
  useEffect(() => { saveToLS(LS_CASES, cases); }, [cases]);
  useEffect(() => { if (skillResults.length > 0) saveToLS(LS_SKILL_RESULTS, skillResults); }, [skillResults]);
  useEffect(() => { if (bestSlices.length > 0) saveToLS(LS_BEST_SLICES, bestSlices); }, [bestSlices]);

  // ── 页面刷新/关闭前保存当前 case 的完整会话到 per-case SESSION_ key ──
  useEffect(() => {
    if (!selectedCaseId) return;
    const handleBeforeUnload = () => {
      if (chatMessages.length === 0) return;
      saveToLS(`SESSION_${selectedCaseId}`, {
        chatMessages: chatMessages.filter(m => m.id !== 'welcome'),
        uploadedImages,
        uploadedVolumes,
        uploadedTexts,
        sessionId,
        currentSliceIndex,
        active3dHtmlUrl,
        activeLesionImageUrl,
        skillResults,
        bestSlices,
        answerText,
        workflowState,
        agentStatus,
      });
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [
    selectedCaseId, chatMessages, uploadedImages, uploadedVolumes,
    uploadedTexts, sessionId, currentSliceIndex,
    active3dHtmlUrl, activeLesionImageUrl,
    skillResults, bestSlices, answerText, workflowState, agentStatus,
  ]);

  // ── 启动时清理旧版 3D skill 结果 + 纯文本技能（保留 3D URL 以支持刷新持久化）──
  useEffect(() => {
    const filtered = filterOutTextOnlySkills(filterOut3dSkill(skillResults));
    if (JSON.stringify(filtered) !== JSON.stringify(skillResults)) { setSkillResults(filtered); }
    // 不再清除 active3dHtmlUrl — 保留以支持刷新后恢复 3D 渲染器
  }, []);

  // ── 刷新页面后，从 IndexedDB 加载已缓存的 NIfTI 原始 Blob 恢复 2D 切片 ──
  useEffect(() => {
    const restoredVolumes = loadFromLS<Array<{ file_id: string; path: string; name: string; volume_role: string }>>(LS_VOLUMES, []);
    const restoredCases = loadFromLS<any[]>(LS_CASES, []);
    if (restoredVolumes.length === 0 || restoredCases.length === 0) return;

    restoredVolumes.forEach(async (vol) => {
      if (!vol.file_id) return;
      const cacheKey = 'nifti:' + vol.file_id;
      try {
        const cached = await loadNiftiBlobFromCache(cacheKey);
        if (!cached) {
          console.log('[niftiCache] READ_MISS', 'key=' + cacheKey, 'fileId=' + vol.file_id, 'fileName=' + vol.name);
          return;
        }
        const arrayBuffer = await cached.blob.arrayBuffer();
        const { parseNiftiFile } = await import('./utils/niftiLoader');
        const volume = await parseNiftiFile(arrayBuffer, cached.fileName);
        const matchCase = restoredCases.find((c: any) => c.id && c.memoryBank?.['影像文件'] === vol.name);
        if (matchCase) {
          setNiftiVolumes((prev) => ({ ...prev, [matchCase.id]: volume }));
          console.log('[niftiCache] RESTORE_OK', 'key=' + cacheKey, 'caseId=' + matchCase.id, 'fileId=' + vol.file_id, 'fileName=' + cached.fileName, 'dims=' + volume.width + 'x' + volume.height + 'x' + volume.depth);
        }
      } catch (e) {
        console.warn('[niftiCache] 恢复失败:', 'key=' + cacheKey, 'err=' + String(e));
      }
    });
  }, []);

  // 追踪当前 case ID，用于会话状态持久化
  const prevCaseIdRef = useRef<string | null>(null);
  const initialLoadDoneRef = useRef(false);

  // Submitting 2D Slice Viewer dedicated query
  const handleSliceQuerySend = () => {
    if (!sliceQuery.trim()) return;
    const textToSend = sliceQuery.trim();
    setSliceQuery('');
    
    // Add user message to chat stream
    const userMsg: ChatMessage = {
      id: `user_slice_${Date.now()}`,
      sender: 'user',
      text: `🔍 **[2D断层切片聚焦提问]** (当前第 **${currentSliceIndex}** 层切片):\n\n${textToSend}`,
      timestamp: new Date().toLocaleTimeString(),
    };
    setChatMessages((prev) => [...prev, userMsg]);
    
    // Trigger diagnostics
    startDiagnostics(textToSend);
  };

  // Submitting 3D Spatial Reconstruction dedicated query
  const handleMeshQuerySend = () => {
    if (!meshQuery.trim()) return;
    const textToSend = meshQuery.trim();
    setMeshQuery('');

    // Add user message to chat stream
    const userMsg: ChatMessage = {
      id: `user_mesh_${Date.now()}`,
      sender: 'user',
      text: `🧊 **[3D空间重建聚焦提问]** (目标脏器: **${currentCase ? currentCase.organ : '未知'}**):\n\n${textToSend}`,
      timestamp: new Date().toLocaleTimeString(),
    };
    setChatMessages((prev) => [...prev, userMsg]);

    // Trigger diagnostics
    startDiagnostics(textToSend);
  };

  // Deleting a custom patient record
  const handleDeleteCase = (idToDelete: string) => {
    const updated = cases.filter((c) => c.id !== idToDelete);
    setCases(updated);
    localStorage.setItem(LS_CASES, JSON.stringify(updated));
    // 同时清理该 case 的持久化会话数据
    try { localStorage.removeItem(`SESSION_${idToDelete}`); } catch { /* ignore */ }
    // 清理 IndexedDB 缓存
    deleteNiftiVolumeFromCache(idToDelete);
    if (selectedCaseId === idToDelete) {
      setSelectedCaseId(updated.length > 0 ? updated[0].id : '');
    }
  };

  // 编辑已有患者档案
  const handleEditCase = (caseId: string) => {
    const target = cases.find(c => c.id === caseId);
    if (!target) return;
    setNewPatientName(target.name);
    setNewPatientGender(target.gender);
    setNewPatientAge(String(target.age));
    setNewPatientId(target.patientId);
    setNewPatientOrgan(target.organ);
    setNewPatientHistory(target.clinicalHistory);
    setNewPatientQuery(target.query);
    setNewPatientSlices(String(target.sliceCount));
    setNewPatientTargetSlice(String(target.targetSliceIndex));
    setEditingCaseId(caseId);
    setShowCreateModal(true);
  };

  // Importing system classical sample cases
  const handleImportSampleCases = () => {
    // Only add samples if they are not already in the list
    const filteredMock = MOCK_CASES.filter(mock => !cases.some(c => c.id === mock.id));
    const updated = [...cases, ...filteredMock];
    setCases(updated);
    localStorage.setItem('CLINICAL_PATIENT_CASES', JSON.stringify(updated));
    if (updated.length > 0) {
      setSelectedCaseId(updated[0].id);
    }
  };

  // Submitting patient creation form
  const handleCreateCaseSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!newPatientName.trim() || !newPatientAge.trim() || isNaN(parseInt(newPatientAge))) {
      return;
    }

    if (editingCaseId) {
      // 更新已有病例
      const updated = cases.map((c) => {
        if (c.id === editingCaseId) {
          return { ...c, name: newPatientName.trim(), patientId: newPatientId.trim() || c.patientId, gender: newPatientGender as 'M' | 'F', age: parseInt(newPatientAge) || c.age, organ: newPatientOrgan, clinicalHistory: newPatientHistory.trim() || c.clinicalHistory, query: newPatientQuery.trim() || c.query };
        }
        return c;
      });
      setCases(updated);
      localStorage.setItem(LS_CASES, JSON.stringify(updated));
      setEditingCaseId(null);
      setShowCreateModal(false);
      return;
    }

    const totalSlices = 120;
    const targetSlice = 60;

    const newCase: MedicalCase = {
      id: 'case_user_' + Date.now(),
      name: newPatientName.trim(),
      patientId: newPatientId.trim() || ('PT-' + Math.floor(1000 + Math.random() * 9000)),
      gender: newPatientGender as 'M' | 'F',
      age: parseInt(newPatientAge) || 45,
      clinicalHistory: newPatientHistory.trim() || '该患者暂无临床主诉背景病史。',
      query: newPatientQuery.trim() || '请评估器官实质形态特征与潜在局部密度异常。',
      organ: newPatientOrgan || 'Liver (肝脏)',
      sliceCount: totalSlices,
      targetSliceIndex: targetSlice,
      roiBox: {
        x: Math.floor(180 + Math.random() * 60),
        y: Math.floor(160 + Math.random() * 60),
        width: 32,
        height: 32,
      },
      slices: [],
      memoryBank: {},
      steps: [],
      finalAnswer: '',
    };

    const updated = [...cases, newCase];
    setCases(updated);
    localStorage.setItem('CLINICAL_PATIENT_CASES', JSON.stringify(updated));
    setSelectedCaseId(newCase.id);
    setShowCreateModal(false);

    // Reset Form fields
    setNewPatientName('');
    setNewPatientId('');
    setNewPatientHistory('');
    setNewPatientQuery('');
    setNewPatientGender('M');
    setNewPatientAge('52');
    setNewPatientOrgan('Liver (肝脏)');
    setNewPatientSlices('120');
    setNewPatientTargetSlice('60');
  };

  // 1. Health check on mount
  useEffect(() => {
    fetch(HTTP_BASE_URL + '/health')
      .then(res => res.json())
      .then(data => {
        console.log("Health response:", data);
        setIsBackendHealthy(true);
      })
      .catch(err => {
        console.warn("Backend health check failed:", err);
        setIsBackendHealthy(false);
      });
  }, []);

  useEffect(() => {
    const newCaseId = currentCase?.id || null;
    const oldCaseId = prevCaseIdRef.current;

    // ── 第1步: Case 切换前 — 保存旧 case 的会话状态（过滤掉欢迎消息，它应动态生成）──
    if (oldCaseId && oldCaseId !== newCaseId && initialLoadDoneRef.current) {
      const filteredChat = chatMessages.filter(m => m.id !== 'welcome');
      saveToLS(`SESSION_${oldCaseId}`, {
        chatMessages: filteredChat,
        uploadedImages,
        uploadedVolumes,
        uploadedTexts,
        sessionId,
        currentSliceIndex,
        active3dHtmlUrl,
        activeLesionImageUrl,
        skillResults,
        bestSlices,
        answerText,
        workflowState,
        agentStatus,
      });
    }

    // ── 第2步: Clean up existing WebSocket if active ──
    if (ws) {
      ws.close();
      setWs(null);
    }

    // ── 第3步: 清空渲染器状态（仅在切换 case 时，不在首次挂载时）──
    // 不清理 niftiVolumes，所有已解析的 volume 保留在内存中，
    // 避免切回已看过的档案时需要从 IndexedDB 异步重读。
    if (initialLoadDoneRef.current) {
      setActive3dHtmlUrl(null);
      setActiveLesionImageUrl(null);
      setSkillResults([]);
      setBestSlices([]);
      setNavigateTarget(null);
    }
    setWorkflowState('idle');
    setAnswerText('');

    if (!newCaseId) {
      setSliceIndex(60);
      setAgentStatus('idle');
      setSessionId(null);
      setUploadedImages([]);
      setUploadedVolumes([]);
      setUploadedTexts([]);
      setChatMessages([
        {
          id: 'welcome',
          sender: 'agent',
          text: `👋 您好！我是 GeoSurge 智能临床辅助诊断系统。\n\n当前系统中没有任何患者病例档案。\n\n- 请点击上方 **「开启新诊断 (录入病例)」** 按钮录入您手头的真实临床病例及问题，并上传 NIfTI / DICOM 医学影像。`,
          timestamp: new Date().toLocaleTimeString(),
        }
      ]);
      prevCaseIdRef.current = null;
      initialLoadDoneRef.current = true;
      return;
    }

    setSliceIndex(currentCase.targetSliceIndex);
    setAgentStatus('idle');
    setSessionId(null);
    setUploadedImages([]);
    setUploadedVolumes([]);
    setUploadedTexts([]);

    // ── 第4步: 从 localStorage 恢复该 case 的会话状态 ──
    const saved = loadFromLS<any>(`SESSION_${newCaseId}`, null);
    if (saved && (saved.chatMessages?.length || saved.sessionId)) {
      // 有历史会话数据 — 恢复 (过滤掉已持久化的 welcome 消息，它应动态生成)
      const restoredChat = saved.chatMessages?.filter((m: any) => m.id !== 'welcome') || [];
      if (restoredChat.length > 0) setChatMessages(restoredChat);
      if (saved.uploadedImages?.length > 0) setUploadedImages(saved.uploadedImages);
      if (saved.uploadedVolumes?.length > 0) setUploadedVolumes(saved.uploadedVolumes);
      if (saved.uploadedTexts?.length > 0) setUploadedTexts(saved.uploadedTexts);
      if (saved.sessionId) setSessionId(saved.sessionId);
      if (saved.currentSliceIndex !== undefined && saved.currentSliceIndex !== null) setSliceIndex(saved.currentSliceIndex);
      if (saved.active3dHtmlUrl) setActive3dHtmlUrl(saved.active3dHtmlUrl);
      if (saved.activeLesionImageUrl) setActiveLesionImageUrl(saved.activeLesionImageUrl);
      if (saved.skillResults?.length > 0) setSkillResults(filterOutTextOnlySkills(filterOut3dSkill(saved.skillResults)));
      if (saved.bestSlices?.length > 0) setBestSlices(saved.bestSlices);
      if (saved.answerText) setAnswerText(saved.answerText);
      if (saved.workflowState) setWorkflowState(saved.workflowState);
      if (saved.agentStatus) setAgentStatus(saved.agentStatus);
    } else if (!initialLoadDoneRef.current) {
      // 首次加载 + 无历史数据 — 尝试从全局键迁移到 per-case session
      const migratedChat = loadFromLS<ChatMessage[]>(LS_CHAT, []).filter(m => m.id !== 'welcome');
      if (migratedChat.length > 0) {
        setChatMessages(migratedChat);
        const migratedSkills = filterOutTextOnlySkills(filterOut3dSkill(loadFromLS(LS_SKILL_RESULTS, [])));
        if (migratedSkills.length > 0) setSkillResults(migratedSkills);
        const migratedSlices = loadFromLS(LS_BEST_SLICES, []);
        if (migratedSlices.length > 0) setBestSlices(migratedSlices);
        // ── 同时恢复上传文件、sessionId、3D/病灶 URL（勿漏）──
        const migImages = loadFromLS(LS_IMAGES, []);
        if (migImages.length > 0) setUploadedImages(migImages);
        const migVolumes = loadFromLS(LS_VOLUMES, []);
        if (migVolumes.length > 0) setUploadedVolumes(migVolumes);
        const migTexts = loadFromLS(LS_TEXTS, []);
        if (migTexts.length > 0) setUploadedTexts(migTexts);
        const migSessionId = loadFromLS<string | null>(LS_SESSION, null);
        if (migSessionId) setSessionId(migSessionId);
        const mig3dUrl = loadFromLS<string | null>(LS_3D_URL, null);
        if (mig3dUrl) setActive3dHtmlUrl(mig3dUrl);
        const migLesionUrl = loadFromLS<string | null>(LS_LESION_URL, null);
        if (migLesionUrl) setActiveLesionImageUrl(migLesionUrl);
        saveToLS(`SESSION_${newCaseId}`, {
          chatMessages: migratedChat,
          uploadedImages: migImages,
          uploadedVolumes: migVolumes,
          uploadedTexts: migTexts,
          sessionId: migSessionId,
          currentSliceIndex: loadFromLS<number>(LS_SLICE_INDEX, 60),
          active3dHtmlUrl: mig3dUrl,
          activeLesionImageUrl: migLesionUrl,
          skillResults: migratedSkills,
          bestSlices: migratedSlices,
        });
      } else {
        setChatMessages([
          {
            id: 'welcome',
            sender: 'agent',
            text: `您好，我是 GeoSurge 智能临床辅助诊断系统。我已成功载入您录入的真实患者 **${currentCase.name}** (${currentCase.gender === 'M' ? '男' : '女'}, ${currentCase.age}岁) 的临床档案。\n\n**临床背景描述**：${currentCase.clinicalHistory}\n\n*请在下方会诊终端中输入您的提问，或通过切片区域或终端上传/拖入该患者的真实多维影像文件。*`,
            timestamp: new Date().toLocaleTimeString(),
          }
        ]);
      }
    } else {
      // Case 切换 + 无历史数据 → 显示欢迎消息
      setChatMessages([
        {
          id: 'welcome',
          sender: 'agent',
          text: `您好，我是 GeoSurge 智能临床辅助诊断系统。我已成功载入您录入的真实患者 **${currentCase.name}** (${currentCase.gender === 'M' ? '男' : '女'}, ${currentCase.age}岁) 的临床档案。\n\n**临床背景描述**：${currentCase.clinicalHistory}\n\n*请在下方会诊终端中输入您的提问，或通过切片区域或终端上传/拖入该患者的真实多维影像文件。*`,
          timestamp: new Date().toLocaleTimeString(),
        }
      ]);
    }

    // ── 第5步: 从 IndexedDB 加载该 case 的 NIfTI 原始 Blob ──
    // 通过 uploadedVolumes 中的 file_id 构造 cacheKey，读取原始 Blob 后重新解析。
    if (newCaseId) {
      // 查找该 case 关联的 volume 条目（通过 fileName 匹配）
      const volEntry = uploadedVolumes.find(v =>
        v.volume_role === 'raw_volume' &&
        currentCase?.memoryBank?.['影像文件'] === v.name
      );
      if (volEntry?.file_id) {
        const cacheKey = 'nifti:' + volEntry.file_id;
        loadNiftiBlobFromCache(cacheKey).then(async (cached) => {
          if (!cached) {
            console.log('[niftiCache] READ_MISS', 'key=' + cacheKey, 'fileId=' + volEntry.file_id, 'caseId=' + newCaseId);
            return;
          }
          try {
            const arrayBuffer = await cached.blob.arrayBuffer();
            const { parseNiftiFile } = await import('./utils/niftiLoader');
            const volume = await parseNiftiFile(arrayBuffer, cached.fileName);
            setNiftiVolumes(prev => {
              if (prev[newCaseId]) return prev;
              return { ...prev, [newCaseId]: volume };
            });
            console.log('[niftiCache] RESTORE_OK', 'key=' + cacheKey, 'caseId=' + newCaseId, 'fileId=' + volEntry.file_id, 'fileName=' + cached.fileName, 'dims=' + volume.width + 'x' + volume.height + 'x' + volume.depth);
          } catch (e) {
            console.warn('[niftiCache] case switch 解析失败:', 'key=' + cacheKey, 'err=' + String(e));
          }
        });
      }
    }

    prevCaseIdRef.current = newCaseId;
    initialLoadDoneRef.current = true;
  }, [selectedCaseId, currentCase?.id]);

  // 2. Real Port A WebSocket Diagnosing — supports multi-round conversation
  const startDiagnostics = async (question: string) => {
    setAnswerText('');

    const customUrl = WS_CHAT_URL;

    // Check if we already have an active WebSocket in 'done' state → multi-round followup
    const isMultiRound = ws && ws.readyState === WebSocket.OPEN && workflowState === 'done' && sessionId;

    // ── Agent 执行追踪：生成 executionId ──
    const execId = `exec_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    currentExecIdRef.current = execId;
    setCurrentExecutionId(execId);
    dispatchAgentExecution({ type: 'EXECUTION_CREATED', executionId: execId, userMessageId: `user_${Date.now()}` });

    // Always add user message to chat (no stale-closure de-dup)
    const userMsg: ChatMessage = {
      id: `user_${Date.now()}`,
      sender: 'user',
      text: question,
      timestamp: new Date().toLocaleTimeString(),
      executionId: execId,
      files: [
        ...uploadedImages.map(img => ({ name: img.name, type: 'image' })),
        ...uploadedVolumes.map(vol => ({ name: vol.name, type: 'volume' })),
        ...uploadedTexts.map(txt => ({ name: txt.name, type: 'text' }))
      ]
    };
    setChatMessages((prev) => [...prev, userMsg]);

    if (isMultiRound) {
      setAgentStatus('processing');
      setWorkflowState('connected');
      // 为 multi-round 创建新的执行追踪状态
      currentExecIdRef.current = execId;
      setCurrentExecutionId(execId);
      dispatchAgentExecution({ type: 'EXECUTION_CREATED', executionId: execId, userMessageId: userMsg.id });
      // 本轮关注的文件的路径全部传给后端，由后端判重和决定处理目标
      const chatPayload: Record<string, any> = {
        event: 'chat_message',
        session_id: sessionId,
        user_message: question,
        images: uploadedImages.map(img => ({ file_id: img.file_id, path: img.path, name: img.name })),
        medical_volumes: uploadedVolumes.map(vol => ({
          file_id: vol.file_id, path: vol.path, name: vol.name, volume_role: vol.volume_role || 'raw_volume'
        })),
        options: { max_new_tokens: 4096, max_agent_rounds: 12 }
      };
      ws.send(JSON.stringify(chatPayload));
      return;
    }

    // ── 首次对话：建立新 WS 连接 ──
    setAgentStatus('processing');
    setWorkflowState('connecting');

    // Agent v2.0: no step tracking — simplified pipeline

    try {
      if (ws) {
        ws.close();
      }

      const socket = new WebSocket(customUrl);
      setWs(socket);

      socket.onopen = () => {
        console.log("WebSocket connected.");
        setWorkflowState('connected');

        // Send initial payload — new backend agent v2.0 format
        const initPayload = {
          event: 'initial',
          session_id: sessionId || null,
          user_message: question,
          inputs: {
            images: uploadedImages.map(img => ({
              file_id: img.file_id,
              path: img.path,
              name: img.name
            })),
            medical_volumes: uploadedVolumes.map(vol => ({
              file_id: vol.file_id,
              path: vol.path,
              name: vol.name,
              volume_role: vol.volume_role || 'raw_volume'
            }))
          },
          options: {
            max_new_tokens: 4096,
            max_agent_rounds: 12
          }
        };

        socket.send(JSON.stringify(initPayload));
        setWorkflowState('submitted');
      };

      let streamingMsgId: string | null = null;
      let currentStreamingText = '';

      socket.onmessage = (event) => {
        try {
          const rawMsg = JSON.parse(event.data);
          console.log("WS Received Event:", rawMsg);
          const ev = rawMsg.event;
          const data = rawMsg.data || rawMsg;
          // 从 ref 读取最新 execId，解决 multi-round 闭包陈旧问题
          const currentId = currentExecIdRef.current;

          switch (ev) {
            case 'session_created':
              const sessId = data.session_id || rawMsg.session_id;
              if (sessId) {
                setSessionId(sessId);
              }
              if (currentId) dispatchAgentExecution({ type: 'WS_CONNECTED', executionId: currentId, sessionId: sessId || '' });
              break;

            case 'workflow_started':
              setWorkflowState('running');
              if (currentId) dispatchAgentExecution({ type: 'WORKFLOW_STARTED', executionId: currentId });
              break;

            case 'progress':
              if (data.message) {
                // Show progress info as system message in chat
                setChatMessages((prev) => {
                  const last = prev[prev.length - 1];
                  if (last && last.sender === 'system' && last.id.startsWith('progress_')) {
                    return prev.map(m => m.id === last.id ? { ...m, text: data.message } : m);
                  }
                  return [...prev, {
                    id: `progress_${Date.now()}`,
                    sender: 'system' as const,
                    text: data.message,
                    timestamp: new Date().toLocaleTimeString(),
                  }];
                });
              }
              break;

            case 'medical_input_detected':
              setChatMessages((prev) => [...prev, {
                id: `medical_input_${Date.now()}`,
                sender: 'system' as const,
                text: data.message || `检测到医学影像: ${data.input_type === 'dicom_folder' ? 'DICOM 文件夹' : 'NIfTI 文件'}`,
                timestamp: new Date().toLocaleTimeString(),
              }]);
              break;

            case 'segmentation_start':
              setAgentStatus('running');
              break;

            case 'segmentation_done':
              {
                const maskCount = data.mask_files ? Object.keys(data.mask_files).length : 0;
                setChatMessages((prev) => [...prev, {
                  id: `seg_done_${Date.now()}`,
                  sender: 'system' as const,
                  text: `影像分割完成 (耗时 ${data.elapsed_seconds || '?'} 秒, ${maskCount} 个器官)`,
                  timestamp: new Date().toLocaleTimeString(),
                }]);
              }
              break;

            case 'skills_loaded':
              if (data.count) {
                setChatMessages((prev) => [...prev, {
                  id: `skills_${Date.now()}`,
                  sender: 'system' as const,
                  text: `已加载 ${data.count} 个诊断技能`,
                  timestamp: new Date().toLocaleTimeString(),
                }]);
              }
              break;

            case 'agent_started':
              setWorkflowState('running');
              setAgentStatus('running');
              if (currentId) dispatchAgentExecution({ type: 'AGENT_STARTED', executionId: currentId });
              break;

            case 'agent_round_start':
              if (currentId) dispatchAgentExecution({ type: 'ROUND_START', executionId: currentId });
              break;

            case 'execution_plan':
              if (currentId) {
                dispatchAgentExecution({ type: 'EXECUTION_PLAN', executionId: currentId, plan: data.skill_calls || [] });
              }
              break;

            case 'skill_call_start':
              if (currentId) {
                dispatchAgentExecution({
                  type: 'CALL_STARTED',
                  executionId: currentId,
                  callId: data.call_id || `call_${Date.now()}`,
                  skillName: data.skill_name || 'unknown',
                });
              }
              break;

            case 'skill_call_result':
              if (currentId) {
                dispatchAgentExecution({
                  type: 'CALL_COMPLETED',
                  executionId: currentId,
                  callId: data.call_id || `call_${Date.now()}`,
                });
              }
              break;

            case 'skill_call_error':
              console.warn('Skill call failed:', data.skill_name, data.message);
              if (currentId) {
                dispatchAgentExecution({
                  type: 'CALL_ERROR',
                  executionId: currentId,
                  callId: data.call_id || `call_${Date.now()}`,
                  errorMessage: data.message || 'Skill 调用失败',
                });
              }
              break;

            case 'agent_round_end':
              if (currentId) dispatchAgentExecution({ type: 'ROUND_END', executionId: currentId });
              break;

            case 'answer_start':
              setWorkflowState('streaming_answer');
              if (currentId) dispatchAgentExecution({ type: 'STREAM_CHUNK', executionId: currentId, chunk: '' });
              streamingMsgId = `streaming_ans_${Date.now()}`;
              currentStreamingText = '';
              setAnswerText('');
              setChatMessages((prev) => [
                ...prev,
                {
                  id: streamingMsgId!,
                  sender: 'agent',
                  text: '',
                  timestamp: new Date().toLocaleTimeString(),
                  status: 'typing' as any
                }
              ]);
              break;

            case 'answer_delta':
              if (data.delta || data.text) {
                const delta = data.delta || data.text;
                currentStreamingText += delta;
                setAnswerText(currentStreamingText);
                if (streamingMsgId) {
                  setChatMessages((prev) => prev.map(m => m.id === streamingMsgId ? {
                    ...m,
                    text: currentStreamingText
                  } : m));
                }
              }
              break;

            case 'answer_end':
              if (streamingMsgId) {
                setChatMessages((prev) => prev.map(m => m.id === streamingMsgId ? {
                  ...m,
                  status: 'done' as any
                } : m));
              }
              break;

            case 'final':
              setWorkflowState('done');
              setAgentStatus('completed');
              if (currentId) {
                dispatchAgentExecution({ type: 'EXECUTION_COMPLETED', executionId: currentId });
                // 同时保存执行快照到聊天消息（从 ref 读取最新状态，避免闭包陈旧）
                const snapExec = agentExecutionStatesRef.current[currentId];
                if (snapExec) {
                  setChatMessages((prev) => prev.map(m =>
                    m.executionId === currentId && m.sender === 'user'
                      ? { ...m, executionSnapshot: buildSnapshot(snapExec) }
                      : m
                  ));
                }
              }

              // 1) 从 outputs 提取 best_slices
              const outputs = data.outputs || {};
              const hasBestSlices = outputs.best_slices?.length > 0;
              if (hasBestSlices) setBestSlices(outputs.best_slices);

              // 2) 从 skill_calls 提取技能结果数据用于 SkillResultRenderer
              const skillCalls = data.skill_calls || [];
              let extractedResults: any[] = [];
              const navTags: string[] = [];
              const newSkillNames: string[] = [];

              skillCalls.forEach((sc: any) => {
                if (sc.status !== 'ok') return;
                const result = normalizeSkillResult(sc.result || {});
                const hasData = result && typeof result === 'object' && Object.keys(result).length > 0;

                // three_d_reconstruction 被实际调用时才设置 3D tab
                if (sc.skill_name === 'three_d_reconstruction' && result.html_url) {
                  setActive3dHtmlUrl(normalizePortBUrl(result.html_url));
                  setThreeDMeta({ organ_count: result.organ_count, organs: result.organs });
                } else if (hasData) {
                  extractedResults.push({ ...result, _skill_name: sc.skill_name });
                  newSkillNames.push(sc.skill_name);
                }
                const tag = `[渲染:${sc.skill_name}]`;
                if (!navTags.includes(tag)) navTags.push(tag);
              });

              // 纯文本技能（肝脏综合分析/肿瘤直径测量）结果已在对话输出，不进入右侧渲染面板
              // 同时移除对它们的"右侧渲染器…查看"提示
              extractedResults = filterOutTextOnlySkills(extractedResults);

              if (extractedResults.length > 0) {
                setSkillResults((prev: any[]) => {
                  // 去重策略：用 skill 类型特征键而非 JSON.stringify
                  // 相同类型的 skill 只保留最新一条
                  const keyed = new Map<string, any>();
                  // 先保留所有旧结果
                  const extractSkillKey = (r: any): string => {
                    // 优先使用记录的 skill_name（通配所有 skill）
                    if (r._skill_name) return r._skill_name;
                    if (r.slices?.length > 0) return 'slice_selection';
                    if (r.html_url) return 'three_d_reconstruction';
                    if (r.liver_volume_cm3 !== undefined) return 'liver_analysis';
                    if (r.tumors?.length > 0) return 'tumor_diameter';
                    if (r.distances?.length > 0) return 'tumor_vessel_distance';
                    if (r.vessels || typeof r === 'object' && Object.values(r).some((v: any) => v?.volume_cm3 !== undefined)) return 'vessel_volume';
                    return 'unknown_skill';
                  };
                  // 保留旧结果（如果新结果中没有同类型覆盖的话）
                  prev.forEach(r => {
                    const k = extractSkillKey(r);
                    if (!keyed.has(k)) keyed.set(k, r);
                  });
                  // 新结果覆盖旧结果（或新增）
                  extractedResults.forEach(r => {
                    const k = extractSkillKey(r);
                    keyed.set(k, r);
                  });
                  return Array.from(keyed.values());
                });
                // 导航由 useEffect(skillResults.length) 自动处理
              }

              // 3) Handle full final answer + 追加导航标签
              {
                let finalAnsText = '';
                if (data.answer && data.answer.text) {
                  finalAnsText = data.answer.text;
                } else if (data.answer_text) {
                  finalAnsText = data.answer_text;
                }

                // 追加 [渲染:xxx] 导航标签（Qwen 可能已包含部分，额外补充 skill 级别的）
                if (navTags.length > 0) {
                  const missingTags = navTags.filter(t => !finalAnsText.includes(t));
                  if (missingTags.length > 0) {
                    finalAnsText += '\n\n' + missingTags.join(' ');
                  }
                }

                // 当右侧渲染器已展示切片图片时，从对话中移除 markdown 图片，改为简洁提示
                if (extractedResults.length > 0 || hasBestSlices) {
                  const hadImages = /!\[.*?\]\((https?:\/\/[^\)]+\.(png|jpg|jpeg|webp)[^\)]*)\)/gi.test(finalAnsText);
                  finalAnsText = finalAnsText.replace(/!\[.*?\]\((https?:\/\/[^\)]+\.(png|jpg|jpeg|webp)[^\)]*)\)/gi, '');
                  if (hadImages) {
                    finalAnsText = finalAnsText.trim();
                    // 根据已有的 skill 类型追加简洁提示
                    const hasSliceSkill = extractedResults.some((r: any) => r.slices?.length > 0);
                    const hasLiver = extractedResults.some((r: any) => r.liver_volume_cm3 !== undefined);
                    const hasTumor = extractedResults.some((r: any) => r.tumors?.length > 0);
                    if (hasSliceSkill && !finalAnsText.includes('右侧')) {
                      finalAnsText = '已为您生成最有信息含量切片，您可以在右侧渲染器「关键切片选取」中查看。\n\n' +
                        (finalAnsText ? finalAnsText.replace(/^(已为您生成)?.*?切片[，。]?\s*/i, '') : '');
                    }
                    if (hasLiver && !finalAnsText.includes('右侧')) {
                      finalAnsText = finalAnsText || '已为您生成肝脏综合分析结果，您可以在右侧渲染器「肝脏综合分析」中查看。';
                    }
                    if (hasTumor && !finalAnsText.includes('右侧')) {
                      finalAnsText = finalAnsText || '已为您生成肿瘤测量结果，您可以在右侧渲染器「肿瘤直径测量」中查看。';
                    }
                  }
                }

                if (finalAnsText) {
                  setAnswerText(finalAnsText);
                  setChatMessages((prev) => {
                    const exists = streamingMsgId ? prev.some(m => m.id === streamingMsgId) : false;
                    if (exists && streamingMsgId) {
                      return prev.map(m => m.id === streamingMsgId ? { ...m, text: finalAnsText, status: 'done' as any } : m);
                    } else {
                      const lastMsg = prev[prev.length - 1];
                      if (lastMsg && lastMsg.sender === 'agent' && lastMsg.status === 'typing') {
                        return prev.map((m, idx) => idx === prev.length - 1 ? { ...m, text: finalAnsText, status: 'done' as any } : m);
                      } else {
                        return [
                          ...prev,
                          {
                            id: `final_ans_msg_${Date.now()}`,
                            sender: 'agent',
                            text: finalAnsText,
                            timestamp: new Date().toLocaleTimeString(),
                            status: 'done' as any
                          }
                        ];
                      }
                    }
                  });
                }
              }
              break;

            case 'cancelled':
              setWorkflowState('cancelled');
              setAgentStatus('idle');
              if (currentId) dispatchAgentExecution({ type: 'EXECUTION_CANCELLED', executionId: currentId });
              setChatMessages((prev) => [
                ...prev,
                {
                  id: `agent_cancel_${Date.now()}`,
                  sender: 'agent' as const,
                  text: `🛑 **任务已取消**`,
                  timestamp: new Date().toLocaleTimeString(),
                }
              ]);
              break;

           case 'error':
              setWorkflowState('error');
              setAgentStatus('error');
              if (currentId) dispatchAgentExecution({ type: 'EXECUTION_ERROR', executionId: currentId, errorMessage: data.message || '未知错误' });
              {
                const stage = data.stage || rawMsg.stage || '';
                const errMsg = data.message || rawMsg.message || '发生未知错误';
                const errDetail = data.detail || rawMsg.detail || null;
                let detailStr = '';
                if (errDetail) {
                  if (typeof errDetail === 'string') {
                    detailStr = `\n\n\`\`\`\n${errDetail.slice(0, 2000)}\n\`\`\``;
                  } else {
                    detailStr = `\n\n\`\`\`json\n${JSON.stringify(errDetail, null, 2).slice(0, 2000)}\n\`\`\``;
                  }
                }
                const header = stage ? `**阶段: \`${stage}\`**` : '**错误**';
                setChatMessages((prev) => [
                  ...prev,
                  {
                    id: `agent_err_${Date.now()}`,
                    sender: 'agent' as const,
                    text: `❌ ${header}\n\n${errMsg}${detailStr}`,
                    timestamp: new Date().toLocaleTimeString(),
                  }
                ]);
              }
              break;

            default:
              console.log("Unhandled WS Event type:", ev);
          }
        } catch (err) {
          console.error("Error parsing WS message:", err);
        }
      };

      socket.onerror = (err) => {
        console.error("WebSocket Error:", err);
        setWorkflowState('error');
        setAgentStatus('error');
      };

      socket.onclose = () => {
        console.log("WebSocket Connection Closed.");
        setWs(null);
        const closeId = currentExecIdRef.current;
        if (closeId) dispatchAgentExecution({ type: 'EXECUTION_DISCONNECTED', executionId: closeId });
      };

    } catch (err: any) {
      console.error("WS Connect error:", err);
      setWorkflowState('error');
      setAgentStatus('error');
    }
  };

  // WS action: cancel the current session
  const handleCancelAnalysis = () => {
    if (currentExecutionId) dispatchAgentExecution({ type: 'EXECUTION_CANCELLING', executionId: currentExecutionId });
    if (ws && ws.readyState === WebSocket.OPEN && sessionId) {
      ws.send(JSON.stringify({ event: 'cancel', session_id: sessionId }));
    } else {
      setWorkflowState('cancelled');
      setAgentStatus('idle');
      if (currentExecutionId) dispatchAgentExecution({ type: 'EXECUTION_CANCELLED', executionId: currentExecutionId });
    }
  };

  /** 拦截聊天中的超链接点击 — 路由到右侧面板 */
  const handleLinkClick = (url: string) => {
    const normalizedUrl = normalizePortBUrl(url);
    // 如果是 3D HTML 链接 → 设置 3D URL 并导航到 3D tab
    if (url.includes('_3d.html') || url.includes('3d.html') || url.includes('/3d_') || url.includes('three_d')) {
      setActive3dHtmlUrl(normalizedUrl);
      setNavigateTarget({ tab: '3d' });
    }
    // 如果是图片链接 → 导航到 skills tab
    else if (url.match(/\.(png|jpe?g|webp)(\?|$)/i)) {
      setActiveLesionImageUrl(normalizedUrl);
      setNavigateTarget({ tab: 'skills' });
    }
    // 其他链接：视为 3D 或页面链接，路由到右侧面板
    else {
      setActive3dHtmlUrl(normalizedUrl);
      setNavigateTarget({ tab: '3d' });
    }
  };

  // Callback when user uploads custom files (supports dragging or browsing)
  const handleCustomFileUpload = async (file: File) => {
    const isImage = /\.(png|jpe?g|webp)$/i.test(file.name);
    const isText = /\.txt$/i.test(file.name);
    const isNifti = /\.(nii|nii\.gz)$/i.test(file.name);
    const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);

    // ── NIfTI 特殊流程：先本地解析（2D 立即渲染），再上传到服务器 ──
    if (isNifti) {
      try {
        // Step 1a: 本地解析 NIfTI，立即可用于 2D 切片渲染（不等待服务器上传）
        const arrayBuffer = await file.arrayBuffer();
        const volume = await parseNiftiFile(arrayBuffer, file.name);

        let activeId = '';
        if (currentCase) {
          activeId = currentCase.id;
          setNiftiVolumes((prev) => ({ ...prev, [activeId]: volume }));
          // 不再缓存 parsed volume — 改为上传后缓存原始 blob（见 Step 1c）

          const updated = cases.map((c) => {
            if (c.id === activeId) {
              return {
                ...c,
                sliceCount: volume.depth,
                targetSliceIndex: Math.floor(volume.depth / 2),
                roiBox: {
                  x: Math.floor(volume.width / 2),
                  y: Math.floor(volume.height / 2),
                  width: Math.floor(volume.width * 0.15) || 40,
                  height: Math.floor(volume.height * 0.15) || 40,
                },
                memoryBank: {
                  ...c.memoryBank,
                  '影像文件': volume.fileName,
                  '图像尺寸': volume.width + ' x ' + volume.height + ' x ' + volume.depth,
                  '数值精度': volume.datatypeName,
                  '体素大小': volume.pixDims[1].toFixed(2) + 'x' + volume.pixDims[2].toFixed(2) + 'x' + volume.pixDims[3].toFixed(2) + ' mm',
                },
                finalAnswer: '成功解析本地 NIfTI 医疗影像并绑定至当前患者！\n- **文件名**: \`' + volume.fileName + '\`\n- **分辨率 (宽 x 高)**: \`' + volume.width + ' x ' + volume.height + '\`\n- **切片总数**: \`' + volume.depth + '\`\n- **数值类型**: \`' + volume.datatypeName + '\`\n- **体素大小 (dx, dy, dz)**: \`' + volume.pixDims[1].toFixed(2) + ' x ' + volume.pixDims[2].toFixed(2) + ' x ' + volume.pixDims[3].toFixed(2) + ' mm\`',
              };
            }
            return c;
          });
          setCases(updated);
          localStorage.setItem('CLINICAL_PATIENT_CASES', JSON.stringify(updated));
          setSliceIndex(Math.floor(volume.depth / 2));
        } else {
          // 没有当前病例 → 不再自动创建病例，提示用户先通过「开启新诊断 (录入病例)」建档
          setChatMessages((prev) => [...prev, {
            id: 'file_need_case_' + Date.now(),
            sender: 'system',
            text: '⚠️ **请先创建患者档案再上传影像**：当前没有选中的患者病例，无法绑定影像。\n请点击左侧上方 **「开启新诊断 (录入病例)」** 按钮，录入患者档案后再上传 \`' + file.name + '\`。',
            timestamp: new Date().toLocaleTimeString(),
          }]);
          return;
        }

        // Step 1b: 现在上传到服务器（不阻塞 2D 渲染）
        let fileId = '', serverPath = '', uploadSuccess = false;
        setChatMessages((prev) => [...prev, {
          id: 'file_uploading_' + Date.now(),
          sender: 'agent',
          text: '📎 **正在上传文件到服务器**: \`' + file.name + '\` (' + sizeInMB + ' MB)，上传完成后可启用后端 AI 分析…',
          timestamp: new Date().toLocaleTimeString(),
        }]);
        try {
          const res = await uploadFileToServer(file);
          fileId = res.file_id; serverPath = res.path; uploadSuccess = true;
        } catch (err) {
          console.warn('文件上传到服务器失败:', err);
        }

        // Step 1c: 上传成功后注册到 uploadedVolumes 并缓存原始 Blob
        if (uploadSuccess && serverPath) {
          setUploadedVolumes((prev) => [...prev, {
            file_id: fileId, path: serverPath, name: file.name, volume_role: 'raw_volume', caseId: activeId
          }]);
          // 缓存原始文件 Blob 到 IndexedDB（只缓存原始二进制，不缓存 parsed volume）
          try {
            const cacheKey = 'nifti:' + fileId;
            const cacheBlob = file.slice(0, file.size, file.type);
            await saveNiftiBlobToCache(cacheKey, {
              fileId,
              fileName: file.name,
              blob: cacheBlob,
              size: file.size,
              mimeType: file.type,
            });
            console.log('[niftiCache] WRITE_OK', 'key=' + cacheKey, 'caseId=' + activeId, 'fileId=' + fileId, 'fileName=' + file.name, 'sizeMB=' + (file.size / 1024 / 1024).toFixed(1));
          } catch (e) {
            console.warn('[niftiCache] WRITE_ERROR', 'fileId=' + fileId, 'err=' + String(e));
          }
          setChatMessages((prev) => [...prev, {
            id: 'file_ok_' + Date.now(),
            sender: 'agent',
            text: '🎉 **NIfTI 3D 序列影像加载成功**！\n\n已成功为当前患者 **' + (currentCase ? currentCase.name : '未命名') + '** 载入 3D 影像文件：\n- **文件名**: \`' + file.name + '\`\n- **分辨率 (宽 x 高)**: ' + volume.width + ' x ' + volume.height + '\n- **切片总数**: ' + volume.depth + ' 层\n- **体素大小**: ' + volume.pixDims[1].toFixed(2) + ' x ' + volume.pixDims[2].toFixed(2) + ' x ' + volume.pixDims[3].toFixed(2) + ' mm\n\n您可以立即在 **2D 切片查看器** 中逐层阅片。',
            timestamp: new Date().toLocaleTimeString(),
          }]);
        } else {
          setChatMessages((prev) => [...prev, {
            id: 'file_note_' + Date.now(),
            sender: 'system',
            text: 'ℹ️ 文件未上传到服务器，仅前端 2D 切片查看器可用。后端 AI 分析和 3D 重建需要先成功上传文件。',
            timestamp: new Date().toLocaleTimeString(),
          }]);
        }
      } catch (e) {
        console.warn('NIfTI 文件处理异常:', e);
        setChatMessages((prev) => [...prev, {
          id: 'file_err_' + Date.now(),
          sender: 'system',
          text: '❌ **文件处理失败**: \`' + file.name + '\` — ' + (e.message || String(e)),
          timestamp: new Date().toLocaleTimeString(),
        }]);
      }
      return;
    }

    // ── 非 NIfTI 文件：先上传到服务器 ──
    let fileId = '';
    let serverPath = '';
    let uploadSuccess = false;
    try {
      const res = await uploadFileToServer(file);
      fileId = res.file_id;
      serverPath = res.path;
      uploadSuccess = true;
    } catch (err) {
      console.warn('文件上传到服务器失败:', err);
    }

    setChatMessages((prev) => [
      ...prev,
      {
        id: 'file_start_' + Date.now(),
        sender: 'agent',
        text: uploadSuccess
          ? '📎 **文件已上传**: \`' + file.name + '\` (' + sizeInMB + ' MB) → ' + serverPath
          : '📎 **文件已加载到前端**: \`' + file.name + '\` (' + sizeInMB + ' MB) (未上传到服务器，后端分析不可用)',
        timestamp: new Date().toLocaleTimeString(),
      }
    ]);

    if (isImage) {
        if (uploadSuccess && serverPath) {
          setUploadedImages((prev) => [...prev, {
            file_id: fileId,
            path: serverPath,
            name: file.name
          }]);
        }
        setChatMessages((prev) => [
          ...prev,
          {
            id: `file_ok_${Date.now()}`,
            sender: 'agent',
            text: uploadSuccess
              ? `✓ **截图上传完成**！\n- 路径: \`${serverPath}\``
              : `✓ **截图已加载到前端**（未上传到服务器）`,
            timestamp: new Date().toLocaleTimeString(),
          }
        ]);
      } else if (isText) {
        let textPreview = "";
        try {
          const fileContent = await new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target?.result as string || "");
            reader.onerror = (e) => reject(e);
            reader.readAsText(file);
          });
          textPreview = fileContent.substring(0, 200);
          if (fileContent.length > 200) textPreview += "...";
        } catch (e) {
          console.warn("Error reading text file:", e);
        }

        if (uploadSuccess && serverPath) { setUploadedTexts((prev) => [...prev, {
          file_id: fileId,
          path: serverPath,
          name: file.name
        }]); }
        setChatMessages((prev) => [
          ...prev,
          {
            id: `file_ok_${Date.now()}`,
            sender: 'agent',
            text: `✓ **临床文本病历加载成功**！\n- 文件ID: \`${fileId}\`\n- 文件名称: \`${file.name}\`\n- 服务器路径: \`${serverPath}\`\n${textPreview ? `\n**病历内容预览**:\n> ${textPreview}\n` : ''}\n已成功作为文本参考数据集暂存于多模态待发输入区。`,
            timestamp: new Date().toLocaleTimeString(),
          }
        ]);
      } else {
        setUploadedVolumes((prev) => [...prev, {
          file_id: fileId,
          path: serverPath,
          name: file.name,
          volume_role: 'raw_volume'
        }]);
        setChatMessages((prev) => [
          ...prev,
          {
            id: `file_ok_${Date.now()}`,
            sender: 'agent',
            text: `✓ **医学 3D 序列容积数据解析成功**！\n- 文件ID: \`${fileId}\`\n- 序列层级: \`120 层切片对齐\`\n- 服务器路径: \`${serverPath}\`\n\n已成功作为原始诊断数据集暂存于终端输入区。请在右下方交互终端中提出具体的临床提问以开启多模态全链路诊断。`,
            timestamp: new Date().toLocaleTimeString(),
          }
        ]);
      }
  };

  const handleRemoveImage = (index: number) => {
    setUploadedImages((prev) => prev.filter((_, i) => i !== index));
  };

  const handleRemoveVolume = (index: number) => {
    setUploadedVolumes((prev) => prev.filter((_, i) => i !== index));
  };

  const handleRemoveText = (index: number) => {
    setUploadedTexts((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpdateVolumeRole = (index: number, role: string) => {
    setUploadedVolumes((prev) => prev.map((vol, i) => i === index ? { ...vol, volume_role: role } : vol));
  };

  /** 更新 NIfTI volume 的强度单位（HU / unknown） */
  const handleUpdateIntensityUnit = (volumeId: string, unit: 'HU' | 'unknown') => {
    setNiftiVolumes((prev) => {
      const updated = { ...prev };
      for (const key of Object.keys(updated)) {
        if (updated[key].volumeId === volumeId) {
          updated[key] = { ...updated[key], intensityUnit: unit };
        }
      }
      return updated;
    });
  };

  const uploadFileDirectly = async (file: File) => {
    return await uploadFileToServer(file);
  };

  return (
    <div className="h-screen overflow-hidden text-slate-850 flex flex-col font-sans selection:bg-blue-100 selection:text-blue-900 bg-slate-50">
      
      {/* 录入并更新真实患者档案 Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 max-w-lg w-full shadow-2xl space-y-4 animate-scaleUp max-h-[90vh] overflow-y-auto scrollbar-thin">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2 text-blue-400">
                <FileSpreadsheet size={16} />
                <h3 className="text-xs font-bold uppercase tracking-wider font-mono">
                  {editingCaseId ? "编辑患者档案" : "录入 / 更新真实临床患者病例档案"}
                </h3>
              </div>
              <button
                onClick={() => { setShowCreateModal(false); setEditingCaseId(null); }}
                className="text-slate-500 hover:text-slate-300 text-sm font-bold"
              >
                ×
              </button>
            </div>

            <form onSubmit={handleCreateCaseSubmit} className="space-y-4 text-xs">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-[10px] font-mono text-slate-500 uppercase">患者姓名 *</label>
                  <input
                    type="text"
                    required
                    value={newPatientName}
                    onChange={(e) => setNewPatientName(e.target.value)}
                    placeholder="例如：张伟、李华"
                    className="w-full bg-slate-950 border border-slate-800 focus:border-blue-500 rounded px-3 py-2 text-slate-200 focus:outline-none"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] font-mono text-slate-500 uppercase">病历卡号 (患者ID)</label>
                  <input
                    type="text"
                    value={newPatientId}
                    onChange={(e) => setNewPatientId(e.target.value)}
                    placeholder="例如：PT-4931 (不填则自动生成)"
                    className="w-full bg-slate-950 border border-slate-800 focus:border-blue-500 rounded px-3 py-2 text-slate-200 focus:outline-none font-mono"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-[10px] font-mono text-slate-500 uppercase">性别 *</label>
                  <select
                    required
                    value={newPatientGender}
                    onChange={(e) => setNewPatientGender(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 focus:border-blue-500 rounded px-3 py-2 text-slate-200 focus:outline-none"
                  >
                    <option value="">-- 请选择 --</option>
                    <option value="M">男 (Male)</option>
                    <option value="F">女 (Female)</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] font-mono text-slate-500 uppercase">年龄 (岁) *</label>
                  <input
                    type="number"
                    required
                    min="0"
                    max="150"
                    value={newPatientAge}
                    onChange={(e) => setNewPatientAge(e.target.value)}
                    placeholder="例如：52"
                    className="w-full bg-slate-950 border border-slate-800 focus:border-blue-500 rounded px-3 py-2 text-slate-200 focus:outline-none font-mono"
                  />
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-[10px] font-mono text-slate-500 uppercase">检查部位与目标脏器 *</label>
                <select
                  value={newPatientOrgan}
                  onChange={(e) => setNewPatientOrgan(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 focus:border-blue-500 rounded px-3 py-2 text-slate-200 focus:outline-none"
                >
                  <option value="Liver (肝脏)">Liver (肝脏)</option>
                  <option value="Pancreas (胰腺)">Pancreas (胰腺)</option>
                  <option value="Lungs (肺部)">Lungs (肺部)</option>
                  <option value="Kidneys (肾脏)">Kidneys (肾脏 - 使用动态通用病灶绘图)</option>
                  <option value="Brain (脑部)">Brain (脑部 - 使用动态通用病灶绘图)</option>
                  <option value="Heart (心脏)">Heart (心脏 - 使用动态通用病灶绘图)</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-[10px] font-mono text-slate-500 uppercase">临床历史背景 / 医生主诉</label>
                <textarea
                  value={newPatientHistory}
                  onChange={(e) => setNewPatientHistory(e.target.value)}
                  rows={2}
                  placeholder="请输入该患者临床背景（例如：常规检查发现胰腺饱满、既往高血压病史等）..."
                  className="w-full bg-slate-950 border border-slate-800 focus:border-blue-500 rounded px-3 py-2 text-slate-200 focus:outline-none resize-none"
                />
              </div>

              <div className="space-y-1">
                <label className="text-[10px] font-mono text-slate-500 uppercase">会诊诊断聚焦诉求 (Query)</label>
                <textarea
                  value={newPatientQuery}
                  onChange={(e) => setNewPatientQuery(e.target.value)}
                  rows={2}
                  placeholder="例如：对该局灶密度降低区域进行精细提取并推荐良恶性分级？"
                  className="w-full bg-slate-950 border border-slate-800 focus:border-blue-500 rounded px-3 py-2 text-slate-200 focus:outline-none resize-none"
                />
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-800/60 font-semibold">
                <button
                  type="button"
                  onClick={() => { setShowCreateModal(false); setEditingCaseId(null); }}
                  className="px-4 py-2 border border-slate-800 hover:bg-slate-850 rounded text-slate-400 hover:text-slate-200 transition-all"
                >
                  取消
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded font-bold transition-all shadow-md shadow-blue-600/15"
                >
                  {editingCaseId ? "保存修改" : "保存并载入系统"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* 2. 主工作台分栏布局 - 统一高保真响应式布局 */}
      <main className="flex-1 w-full flex overflow-hidden min-h-0 bg-slate-50 relative">
        
        {/* ==================== 极简左侧栏: 患者历史列表 (可折叠, 可拖拽调宽) ==================== */}
        <div ref={sidebarRef} style={{ width: sidebarCollapsed ? 0 : sidebarWidth, minWidth: 0, overflow: 'hidden', flexShrink: 0, transition: isResizing ? 'none' : 'width 0.3s ease-in-out' }}>
          <div ref={sidebarInnerRef} className="h-full" style={{ width: sidebarWidth }}>
            <DeepSeekSidebar
              cases={cases}
              selectedCaseId={selectedCaseId}
              onSelectCase={(id) => setSelectedCaseId(id)}
              onDeleteCase={handleDeleteCase}
              onOpenCreateModal={() => setShowCreateModal(true)}
              onEditCase={handleEditCase}
              isCollapsed={sidebarCollapsed}
              onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
              onOpenSkillManager={() => setShowSkillManager(true)}
            />
          </div>
        </div>
        {/* 拖拽调整宽度的手柄 */}
        {!sidebarCollapsed && (
          <div
            onMouseDown={handleSidebarDragStart}
            className={`flex-shrink-0 w-1.5 cursor-col-resize hover:bg-blue-400/40 active:bg-blue-500/60 transition-colors duration-150 ${isResizing ? 'bg-blue-500/60' : 'bg-transparent'}`}
            style={{ touchAction: 'none' }}
          />
        )}

        {/* ==================== 双栏布局: 对话 | 渲染器 (可拖拽调宽) ==================== */}
        <div className="flex-1 flex min-h-0 overflow-hidden main-content-area">

          {/* 左栏: 智能对话会诊终端 */}
          <div ref={chatPanelRef} className="flex flex-col min-h-0" style={{ width: (active3dHtmlUrl || skillResults.length > 0 || niftiVolumes[currentCase?.id || ''] || uploadedVolumes.length > 0) ? `${middleRatio * 100}%` : '100%', minWidth: 0, flexShrink: 0 }}>
            <div className="flex-1 min-h-0 flex flex-col bg-white">
              <ChatSection
                messages={chatMessages}
                onSendMessage={startDiagnostics}
                status={agentStatus}
                querySuggestion={currentCase ? currentCase.query : ""}
                uploadedImages={uploadedImages}
                uploadedVolumes={uploadedVolumes}
                uploadedTexts={uploadedTexts}
                onRemoveImage={handleRemoveImage}
                onRemoveVolume={handleRemoveVolume}
                onRemoveText={handleRemoveText}
                onFileUpload={handleCustomFileUpload}
                isSidebarCollapsed={sidebarCollapsed}
                onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
                onUpdateVolumeRole={handleUpdateVolumeRole}
                onUploadFileDirectly={uploadFileDirectly}
                workflowState={workflowState}
                onCancel={handleCancelAnalysis}
                onOpenSkillManager={() => setShowSkillManager(true)}
                onNavigate={(tab, skillIdx, skillName) => setNavigateTarget({ tab, skillIdx, skillName })}
                onLinkClick={handleLinkClick}
                agentExecutionStates={agentExecutionStates}
                currentExecutionId={currentExecutionId}
              />
            </div>
          </div>

          {/* 中间拖拽分界线 + 右栏: 渲染器 (有内容才显示) */}
          {(active3dHtmlUrl || skillResults.length > 0 || niftiVolumes[currentCase?.id || ''] || uploadedVolumes.length > 0) && (
            <>
          <div
            onMouseDown={(e) => {
              e.preventDefault();
              const container = e.currentTarget.parentElement;
              const rect = container ? container.getBoundingClientRect() : null;
              middleResizeRef.current = {
                startX: e.clientX,
                startRatio: middleRatio,
                rectLeft: rect ? rect.left : 0,
                rectWidth: rect ? rect.width : 1,
              };
              setIsResizingMiddle(true);
            }}
            className={`flex-shrink-0 w-1.5 cursor-col-resize transition-colors duration-150 ${
              isResizingMiddle ? 'bg-blue-500/60' : 'bg-slate-200 hover:bg-blue-400/40'
            }`}
            style={{ touchAction: 'none' }}
          />

          <div ref={renderPanelRef} className="flex flex-col min-h-0" style={{ width: `${(1 - middleRatio) * 100}%`, minWidth: 0, flexShrink: 0 }}>
            <div className="flex-1 min-h-0 flex flex-col rounded-xl overflow-hidden">
              <RendererPanel
                active3dHtmlUrl={active3dHtmlUrl}
                currentCase={currentCase}
                workflowState={workflowState}
                onMaximize3d={() => setIsMeshFullscreen(true)}
                skillResults={skillResults}
                navigateTo={navigateTarget}
                threeDMeta={threeDMeta}
                niftiVolume={niftiVolumes[currentCase?.id || ''] || null}
                currentSliceIndex={currentSliceIndex}
                setSliceIndex={setSliceIndex}
                agentStatus={agentStatus}
                onFileUpload={handleCustomFileUpload}
                lesionImageUrl={activeLesionImageUrl}
                bestSliceIndex={bestSliceIndex}
                bestSlices={bestSlices}
                onUpdateIntensityUnit={handleUpdateIntensityUnit}
                hasPendingNifti={uploadedVolumes.length > 0 && !niftiVolumes[currentCase?.id || '']}
              />
            </div>
          </div>
            </>
          )}

        </div>
      </main>

      {/* 3. 页脚信息 */}
      <footer className="py-3 border-t border-slate-200 bg-white text-center text-[10px] text-slate-550 font-mono flex flex-col sm:flex-row justify-between px-6 gap-2 flex-shrink-0">
        <div>GeoSurge AI 临床辅助诊断协同终端 | 仅用于医学科研与决策辅助，不可直接作为临床治疗决策依据</div>
        <div className="flex justify-center items-center gap-4 text-slate-500">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          <span>后端连接: 已配置</span>
        </div>
      </footer>

      {/* 3D 空间重建全屏浏览模式 */}
      {isMeshFullscreen && currentCase && (
        <div className="fixed inset-0 bg-slate-950/95 z-50 flex flex-col p-4 md:p-6">
          {/* Header Panel */}
          <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4 flex-shrink-0">
            <div className="flex items-center gap-2">
              <Globe className="text-emerald-500 w-5 h-5 animate-spin-slow" />
              <div>
                <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider font-mono">
                  3D 解剖空间重建全屏会诊模式
                </h3>
                <p className="text-[10px] text-slate-400 font-mono mt-0.5">
                  患者: {currentCase.name} | 病案号: {currentCase.patientId} | 器官: {currentCase.organ}
                </p>
              </div>
            </div>
            <button
              onClick={() => setIsMeshFullscreen(false)}
              className="px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white rounded-lg text-xs font-semibold transition-all border border-slate-700/60"
            >
              退出全屏 (ESC)
            </button>
          </div>

          {/* Large Viewport Workspace - fully stretched to the edge */}
          <div className="flex-1 min-h-0 bg-slate-950 overflow-hidden flex items-center justify-center relative font-sans">
            {typeof active3dHtmlUrl === 'string' && active3dHtmlUrl ? (
              <div className="w-full h-full flex flex-col relative">
                <iframe
                  src={active3dHtmlUrl}
                  className="w-full flex-1 bg-slate-950"
                  style={{ width: '100%', height: '100%', border: 'none' }}
                  title="3D Dynamic Mesh Render Fullscreen"
                />
              </div>
            ) : workflowState === 'done' ? (
              <div className="w-full h-full flex flex-col items-center justify-center bg-slate-950 text-center p-6 text-slate-400 font-mono">
                <Globe className="w-12 h-12 mx-auto mb-3 text-slate-600 animate-bounce" />
                <p className="text-sm font-semibold text-slate-300">本次未返回 3D 可视化页面。</p>
              </div>
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center bg-slate-950 text-center p-6 text-slate-400 font-mono">
                <Globe className="w-12 h-12 mx-auto mb-3 text-slate-600 animate-pulse" />
                <p className="text-sm font-semibold text-slate-300">等待后端返回 3D 空间解剖重建模型...</p>
              </div>
            )}
          </div>
        </div>
      )}
    <SkillManager isOpen={showSkillManager} onClose={() => setShowSkillManager(false)} />
    </div>
  );
}
