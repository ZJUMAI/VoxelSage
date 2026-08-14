import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Sparkles, User, Terminal, Paperclip, Image as ImageIcon, FileText, PanelLeftOpen, XCircle, Zap, Slash, Brain, Eye, Globe, GitBranch, Crosshair, Ruler, MapPin, Activity, Scissors, Database } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage, AgentExecutionState } from '../types';
import AgentExecutionCard from './AgentExecutionCard';
import { formatExecutionStatus } from '../utils/agentExecutionReducer';
import { SKILLS_LIST_URL } from '../api';

interface ChatSectionProps {
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  status: string;
  querySuggestion: string;
  uploadedImages: any[];
  uploadedVolumes: any[];
  uploadedTexts: any[];
  onRemoveImage: (index: number) => void;
  onRemoveVolume: (index: number) => void;
  onRemoveText: (index: number) => void;
  onFileUpload: (file: File) => void;
  isSidebarCollapsed?: boolean;
  onToggleSidebar?: () => void;
  onUpdateVolumeRole?: (index: number, role: string) => void;
  onUploadFileDirectly?: (file: File) => Promise<{ file_id: string; path: string; name: string }>;
  workflowState?: string;
  onCancel?: () => void;
  onOpenSkillManager?: () => void;
  onNavigate?: (tab: '3d' | '2d' | 'skills', skillIdx?: number, skillName?: string) => void;
  /** 拦截聊天中的超链接点击，路由到右侧面板 */
  onLinkClick?: (url: string) => void;
  /** Agent 执行状态（按 executionId 索引） */
  agentExecutionStates?: Record<string, AgentExecutionState>;
  /** 当前活跃的 executionId */
  currentExecutionId?: string | null;
}

/** 提取文本中的 [渲染:xxx] 标签，在文本下方生成快速导航按钮 */
function extractRenderTags(text: string): string[] {
  const tags: string[] = [];
  const re = /\[渲染:([^\]]+)\]/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (!tags.includes(m[1])) tags.push(m[1]);
  }
  return tags;
}

const SKILL_LABEL_MAP: Record<string, string> = {
  'liver_analysis': '肝脏综合分析',
  'slice_selection': '关键切片选取',
  'three_d_reconstruction': '3D 模型重建',
  'tumor_diameter': '肿瘤直径测量',
  'tumor_vessel_distance': '肿瘤-血管距离',
  'vessel_volume': '血管体积',
  'plan_resection': '手术切除规划',
  'plan_resection_sequence': '手术切除规划',
  'segmentation_modification': '分割编辑器',
};

const SKILL_ICON_MAP: Record<string, React.ReactNode> = {
  'liver_analysis': <Activity size={10} />,
  'slice_selection': <Eye size={10} />,
  'three_d_reconstruction': <Globe size={10} />,
  'tumor_diameter': <Ruler size={10} />,
  'tumor_vessel_distance': <MapPin size={10} />,
  'vessel_volume': <GitBranch size={10} />,
  'plan_resection': <Crosshair size={10} />,
  'segmentation_modification': <Scissors size={10} />,
};

export default function ChatSection({
  messages,
  onSendMessage,
  status,
  querySuggestion,
  uploadedImages,
  uploadedVolumes,
  uploadedTexts,
  onRemoveImage,
  onRemoveVolume,
  onRemoveText,
  onFileUpload,
  isSidebarCollapsed = false,
  onToggleSidebar,
  onUpdateVolumeRole,
  onUploadFileDirectly,
  workflowState,
  onCancel,
  onOpenSkillManager,
  onNavigate,
  onLinkClick,
  agentExecutionStates = {},
  currentExecutionId,
}: ChatSectionProps) {
  const [inputText, setInputText] = useState('');
  const [availableSkills, setAvailableSkills] = useState<any[]>([]);

  // ── Slash 命令自动补全状态 ──
  const [showSlashDropdown, setShowSlashDropdown] = useState(false);
  const [slashFilter, setSlashFilter] = useState('');
  const [activeSlashIndex, setActiveSlashIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const slashDropdownRef = useRef<HTMLDivElement>(null);

  // Filter skills based on slash input (the text after "/")
  const filteredSkills = availableSkills.filter(s => {
    const label = SKILL_LABEL_MAP[s.name] || s.name.replace(/_/g, ' ');
    return label.includes(slashFilter) || s.name.includes(slashFilter);
  });

  // Fetch available skills on mount — 从后端获取
  useEffect(() => {
    fetch(`${SKILLS_LIST_URL}`)
      .then(r => {
        if (!r.ok) {
          console.warn(`[Skills] 获取列表失败: HTTP ${r.status} (${SKILLS_LIST_URL})`);
          return { skills: [] };
        }
        return r.json();
      })
      .then(d => {
        const skills = d.skills || [];
        console.log(`[Skills] 已加载 ${skills.length} 个技能`, skills.map((s: any) => s.name));
        setAvailableSkills(skills);
      })
      .catch((err) => {
        console.warn(`[Skills] 请求异常 (${SKILLS_LIST_URL}):`, err);
      });
  }, []);

  const chatBottomRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // Auto scroll to bottom
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [messages]);

  // Close slash dropdown on outside click
  useEffect(() => {
    if (!showSlashDropdown) return;
    const handleClick = (e: MouseEvent) => {
      if (slashDropdownRef.current && !slashDropdownRef.current.contains(e.target as Node) &&
          inputRef.current && !inputRef.current.contains(e.target as Node)) {
        setShowSlashDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [showSlashDropdown]);

  // Reset active index when filtered list changes
  useEffect(() => {
    setActiveSlashIndex(0);
  }, [filteredSkills.length]);

  /** 填充 slash 命令到输入框 */
  const applySlashCommand = useCallback((skillName: string) => {
    setInputText(` /${skillName} `);
    setShowSlashDropdown(false);
    inputRef.current?.focus();
  }, []);

  /** 处理输入变化：检测 "/" 触发下拉菜单 */
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setInputText(val);

    // 检测是否正在输入 slash 命令（兼容 " / " 和 "/" 两种格式）
    const trimmed = val.trimStart();
    if (trimmed.startsWith('/')) {
      const afterSlash = trimmed.slice(1).split(' ')[0];
      setSlashFilter(afterSlash);
      setShowSlashDropdown(true);
    } else {
      setShowSlashDropdown(false);
    }
  };

  /** 处理按键：Tab 补全、Enter 提交、上下键导航 */
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // 跳过 IME 输入法组合中的 Enter（例如中文拼音选字）
    if ((e.nativeEvent as any).isComposing || e.keyCode === 229) {
      return;
    }

    if (!showSlashDropdown || filteredSkills.length === 0) {
      // 正常 Enter 提交
      if (e.key === 'Enter' && !e.shiftKey) {
        handleSubmitDirect();
      }
      return;
    }

    switch (e.key) {
      case 'Tab':
        e.preventDefault();
        if (filteredSkills[activeSlashIndex]) {
          applySlashCommand(filteredSkills[activeSlashIndex].name);
        }
        break;
      case 'Enter':
        // Enter 不自动补全，直接提交当前输入内容
        e.preventDefault();
        setShowSlashDropdown(false);
        handleSubmitDirect();
        break;
      case 'ArrowDown':
        e.preventDefault();
        setActiveSlashIndex(prev => Math.min(prev + 1, filteredSkills.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setActiveSlashIndex(prev => Math.max(prev - 1, 0));
        break;
      case 'Escape':
        e.preventDefault();
        setShowSlashDropdown(false);
        break;
    }
  };

  /** 直接提交流程（由 handleSubmitDirect / handleSubmit 调用） */
  const handleSubmitDirect = () => {
    if (!inputText.trim()) return;
    onSendMessage(inputText.trim());
    setInputText('');
    setShowSlashDropdown(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSubmitDirect();
  };

  const handleSuggestionClick = () => {
    onSendMessage(querySuggestion);
  };

  const handleChatFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  /** 点击技能按钮时，填充 " /skill_name " 到输入框，用户可继续输入 */
  const handleSkillButtonClick = (skillName: string) => {
    setInputText(` /${skillName} `);
    setShowSlashDropdown(false);
    inputRef.current?.focus();
  };

  return (
    <div id="chat-section-container" className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-50 border-b border-slate-200/85 flex-shrink-0">
        <div className="flex items-center gap-2">
          {isSidebarCollapsed && onToggleSidebar && (
            <button
              type="button"
              onClick={onToggleSidebar}
              className="p-1 text-slate-500 hover:text-slate-800 hover:bg-slate-200/80 rounded transition-all mr-1 flex items-center justify-center cursor-pointer"
              title="展开患者病例档案"
            >
              <PanelLeftOpen size={14} className="text-slate-600" />
            </button>
          )}
          <Terminal size={14} className="text-blue-600" />
          <h3 className="text-xs font-semibold text-slate-800 uppercase tracking-wider font-mono">
            智能交互诊断终端
          </h3>
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_6px_rgba(16,185,129,0.8)]"></span>
          <span className="text-[9px] text-slate-500 font-mono font-semibold">智能体已激活</span>
        </div>
      </div>

      {/* Message Area */}
      <div ref={messagesContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-200">
        {messages.map((msg) => {
          const isSystem = msg.sender === 'system';
          const isAgent = msg.sender === 'agent';

          if (isSystem) {
            // 系统状态消息 — 紧凑单行小字（类似 deepseek 思考模式）
            if (msg.id.startsWith('progress_') || msg.id.startsWith('seg_') || msg.id.startsWith('skills_') || msg.id.startsWith('medical_input_')) {
              return (
                <div key={msg.id} className="flex justify-center my-0.5">
                  <span className="text-[9.5px] text-slate-400 font-mono bg-slate-50/80 px-2.5 py-0.5 rounded-full border border-slate-200/50 leading-tight">
                    {msg.text}
                  </span>
                </div>
              );
            }
            return (
              <div
                id={`chat-msg-${msg.id}`}
                key={msg.id}
                className="flex items-start gap-2.5 max-w-[95%] mr-auto p-3 bg-slate-50 border border-slate-150 rounded-xl shadow-sm border-l-3 border-l-blue-500/80 animate-fadeIn"
              >
                <div className="w-5 h-5 rounded bg-blue-50 border border-blue-150 text-blue-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Terminal size={10} />
                </div>
                <div className="flex-1 space-y-1">
                  <div className="text-[11px] font-medium text-slate-700 leading-relaxed">
                    {msg.text}
                  </div>
                  <div className="text-[8px] text-slate-400 font-mono">
                    {msg.timestamp}
                  </div>
                </div>
              </div>
            );
          }

          return (
            <div
              id={`chat-msg-${msg.id}`}
              key={msg.id}
              className={`flex gap-3 max-w-[85%] ${
                isAgent ? 'mr-auto' : 'ml-auto flex-row-reverse'
              }`}
            >
              {/* Avatar Icon */}
              <div
                className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 border ${
                  isAgent
                    ? 'bg-blue-50 border-blue-200 text-blue-600'
                    : 'bg-slate-100 border-slate-200 text-slate-600'
                }`}
              >
                {isAgent ? <Sparkles size={13} /> : <User size={13} />}
              </div>

              {/* Message Bubble */}
              <div className="space-y-1">
                <div
                  className={`px-3.5 py-2.5 rounded-xl text-xs leading-relaxed ${
                    isAgent
                      ? 'bg-slate-50 border border-slate-200/85 text-slate-800 rounded-tl-none shadow-sm'
                      : 'bg-blue-600 text-white rounded-tr-none shadow-sm whitespace-pre-wrap'
                  }`}
                >
                  {isAgent ? (
                    <div className="markdown-body">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          a: ({ href, children }) => (
                            <a
                              href={href}
                              onClick={(e) => {
                                if (href && (href.startsWith('http://') || href.startsWith('https://'))) {
                                  e.preventDefault();
                                  onLinkClick?.(href);
                                }
                              }}
                              className="text-blue-600 hover:text-blue-800 underline cursor-pointer"
                            >
                              {children}
                            </a>
                          ),
                          img: ({ src, alt }) => (
                            <img
                              src={src}
                              alt={alt || ''}
                              onClick={() => src && onLinkClick?.(src)}
                              className="max-w-full rounded-lg border border-slate-200 my-2 cursor-pointer hover:opacity-90 transition-opacity"
                              referrerPolicy="no-referrer"
                            />
                          ),
                        }}
                      >{msg.text.replace(/\[渲染:[^\]]+\]/g, '')}</ReactMarkdown>
                      {/* 提取 [渲染:xxx] 标签生成快速导航按钮 */}
                      {(() => {
                        const tags = extractRenderTags(msg.text);
                        if (tags.length === 0) return null;
                        return (
                          <div className="flex flex-wrap gap-1 mt-2 pt-2 border-t border-slate-200/60">
                            <span className="text-[8px] text-slate-400 font-mono mr-0.5 self-center">查看:</span>
                            {tags.map((tag, i) => {
                              const t = tag.toLowerCase();
                              let tab: '3d' | '2d' | 'skills' = 'skills';
                              let skillIdx: number | undefined;

                              // ── 直接匹配 skill 名称 ──
                              let skillName: string | undefined;
                              if (t === 'three_d_reconstruction' || t === '3d' || t === '3d模型' || t === '三维') {
                                tab = '3d';
                              } else if (t === 'slice_selection' || t === '2d' || t === '切片' || t === '关键') {
                                tab = 'skills'; skillName = 'slice_selection';
                              } else if (t === 'liver_analysis' || t.includes('肝脏') || t === '综合分析') {
                                tab = 'skills'; skillName = 'liver_analysis';
                              } else if (t === 'tumor_diameter' || t.includes('肿瘤') || t.includes('直径')) {
                                tab = 'skills'; skillName = 'tumor_diameter';
                              } else if (t === 'tumor_vessel_distance' || t.includes('血管距离') || t === 'vessel') {
                                tab = 'skills'; skillName = 'tumor_vessel_distance';
                              } else if (t === 'vessel_volume' || t.includes('血管体积')) {
                                tab = 'skills'; skillName = 'vessel_volume';
                              } else if (t === 'plan_resection' || t.includes('手术切除')) {
                                tab = 'skills'; skillName = 'plan_resection';
                              } else if (t === 'segmentation_modification' || t === '分割编辑器' || t.includes('分割')) {
                                tab = 'skills'; skillName = 'segmentation_modification';
                              } else {
                                tab = 'skills'; skillName = tag;
                              }

                              // 纯文本技能（肝脏综合分析/肿瘤直径测量）：结果已在对话输出，
                              // 无右侧面板可跳转，不生成导航按钮
                              if (skillName === 'liver_analysis' || skillName === 'tumor_diameter') return null;
                              return (
                                <button key={i} onClick={() => onNavigate?.(tab, skillIdx, skillName)}
                                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[9px] font-bold font-mono bg-blue-50 text-blue-700 border border-blue-200 hover:bg-blue-100 hover:border-blue-400 cursor-pointer transition-all">
                                  📊 {SKILL_LABEL_MAP[tag.toLowerCase()] || tag}
                                </button>
                              );
                            })}
                          </div>
                        );
                      })()}
                    </div>
                  ) : (
                    <div className="space-y-1.5">
                      <div>{msg.text}</div>
                      {msg.files && msg.files.length > 0 && (
                        <div className="pt-1.5 border-t border-blue-500/40 flex flex-col gap-1">
                          <span className="text-[9px] text-blue-200 font-bold">📎 上传文件附件：</span>
                          <div className="flex flex-wrap gap-1">
                            {msg.files.map((file, fIdx) => (
                              <div
                                key={fIdx}
                                className="flex items-center gap-1 bg-blue-750/90 text-[10px] text-white px-2 py-0.5 rounded border border-blue-500/30 font-mono"
                              >
                                {file.type.includes('image') || file.name.endsWith('.png') || file.name.endsWith('.jpg') || file.name.endsWith('.jpeg') ? (
                                  <ImageIcon size={9} />
                                ) : (
                                  <Database size={9} />
                                )}
                                <span>{file.name}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
                {/* Agent 执行追踪卡：在用户消息下方显示 */}
                {!isAgent && msg.executionId && (
                  <AgentExecutionCard
                    state={agentExecutionStates[msg.executionId]}
                    snapshot={msg.executionSnapshot}
                    isActive={msg.executionId === currentExecutionId}
                    roundLabel={
                      msg.executionId === currentExecutionId && agentExecutionStates[msg.executionId]
                        ? `第 ${agentExecutionStates[msg.executionId].roundNumber} 轮`
                        : undefined
                    }
                  />
                )}
                <div
                  className={`text-[9px] text-slate-400 font-mono ${
                    isAgent ? 'text-left pl-1' : 'text-right pr-1'
                  }`}
                >
                  {msg.timestamp}
                </div>
              </div>
            </div>
          );
        })}

        {/* Agent processing indicator */}
        {status !== 'idle' && status !== 'completed' && status !== 'error' && (
          <div className="flex gap-3 max-w-[85%] mr-auto">
            <div className="w-7 h-7 rounded-lg bg-blue-50 border border-blue-200 text-blue-600 flex items-center justify-center flex-shrink-0 animate-pulse">
              <Sparkles size={13} />
            </div>
            <div className="space-y-1.5">
              <div className="bg-slate-50 border border-slate-200 text-slate-700 px-3.5 py-3 rounded-xl rounded-tl-none text-xs flex flex-col gap-1.5 min-w-[200px] shadow-sm">
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-600 animate-ping"></span>
                  <span className="font-mono text-[9px] text-blue-600 font-bold uppercase tracking-wider">
                    VoxelSage 智能体分析中...
                  </span>
                </div>
                <p className="text-[10px] text-slate-500 font-sans italic leading-none">
                  {status === 'uploading' && '正在上传文件...'}
                  {status === 'processing' && '正在调用 Qwen3-VL 多模态诊断引擎...'}
                  {status === 'running' && 'AI 智能体正在执行诊断分析...'}
                  {(status !== 'uploading' && status !== 'processing' && status !== 'running') && '正在分析...'}
                </p>
                <div className="flex gap-1 items-center mt-1">
                  <span className="w-1 h-1 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="w-1 h-1 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-1 h-1 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={chatBottomRef} />
      </div>

      {/* Input bar, Attachments & Recommendations */}
      <div className="p-3 bg-slate-50 border-t border-slate-200 space-y-2 flex-shrink-0">

        {/* Render Attachments Queue */}
        {(uploadedImages.length > 0 || uploadedVolumes.length > 0 || uploadedTexts.length > 0) && (
          <div className="flex flex-wrap gap-1.5 px-2 py-1.5 bg-white border border-slate-200 rounded-lg shadow-sm">
            {uploadedImages.map((img, idx) => (
              <div key={`img-${idx}`} className="flex items-center gap-1 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded text-[10px] text-blue-700 font-mono">
                <ImageIcon size={9} className="text-blue-600" />
                <span>{img.name}</span>
                <button type="button" onClick={() => onRemoveImage(idx)} className="hover:text-red-500 font-bold ml-1 transition-colors">×</button>
              </div>
            ))}
            {uploadedVolumes.map((vol, idx) => (
              <div key={`vol-${idx}`} className="flex items-center gap-1.5 bg-purple-50 border border-purple-200 px-2 py-0.5 rounded text-[10px] text-purple-700 font-mono">
                <Database size={9} className="text-purple-600" />
                <span className="max-w-[120px] truncate" title={vol.name}>{vol.name}</span>
                <span className="text-purple-300">|</span>
                <span className="text-[9px] text-purple-600 font-sans">角色:</span>
                <select
                  value={vol.volume_role || 'raw_volume'}
                  onChange={(e) => onUpdateVolumeRole && onUpdateVolumeRole(idx, e.target.value)}
                  className="bg-white border border-purple-200 text-purple-700 rounded px-1.5 py-0.2 text-[9px] font-sans font-semibold focus:outline-none cursor-pointer"
                >
                  <option value="raw_volume">raw_volume (原始)</option>
                  <option value="mask_volume">mask_volume (分割)</option>
                  <option value="enhanced_volume">enhanced_volume (增强)</option>
                  <option value="unknown">unknown (未知)</option>
                </select>
                <button type="button" onClick={() => onRemoveVolume(idx)} className="hover:text-red-500 font-bold ml-0.5 transition-colors cursor-pointer text-xs">×</button>
              </div>
            ))}
            {uploadedTexts.map((txt, idx) => (
              <div key={`txt-${idx}`} className="flex items-center gap-1 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded text-[10px] text-amber-700 font-mono">
                <FileText size={9} className="text-amber-600" />
                <span>{txt.name}</span>
                <button type="button" onClick={() => onRemoveText(idx)} className="hover:text-red-500 font-bold ml-1 transition-colors">×</button>
              </div>
            ))}
          </div>
        )}

        {/* Rec Suggestion Bubble */}
        {status === 'idle' && (
          <div className="flex items-center gap-1.5">
            <span className="text-[8px] font-mono bg-slate-100 border border-slate-200 text-slate-500 px-1.5 py-0.5 rounded flex items-center gap-1 font-semibold select-none">
              <Zap size={10} /> 推荐提问
            </span>
            <button
              onClick={handleSuggestionClick}
              className="text-[10px] font-sans font-semibold text-blue-600 hover:text-blue-700 bg-blue-50 hover:bg-blue-100 border border-blue-200 hover:border-blue-300 rounded-full px-2.5 py-0.5 text-left truncate transition-all max-w-[90%] shadow-sm"
            >
              {querySuggestion}
            </button>
          </div>
        )}

        {/* ── 输入框 + Slash 下拉菜单 ── */}
        <div className="relative">
          <form onSubmit={handleSubmit} className="flex gap-2 items-center">
            {/* Chat Attachment button */}
            <label className="p-2 bg-white hover:bg-slate-50 border border-slate-200 hover:border-slate-300 text-slate-600 hover:text-slate-800 rounded-lg cursor-pointer transition-all flex items-center justify-center flex-shrink-0 shadow-sm" title="添加文件附件 (支持 .nii.gz, .png, .txt, .zip, .dcm)">
              <Paperclip size={13} />
              <input
                type="file"
                accept=".png,.jpg,.jpeg,.webp,.nii,.nii.gz,.gz,.txt,.zip,.dcm"
                onChange={handleChatFileChange}
                className="hidden"
              />
            </label>

            <input
              ref={inputRef}
              id="chat-input-field"
              type="text"
              value={inputText}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              disabled={status !== 'idle' && status !== 'completed' && status !== 'error'}
              placeholder={
                status !== 'idle' && status !== 'completed' && status !== 'error'
                  ? '智能体诊断分析中...'
                  : status === 'error'
                  ? '诊断异常，可修改问题后重试...'
                  : '输入 "/" 选择技能，或直接输入临床提问...'
              }
              className="flex-1 bg-white border border-slate-200 focus:border-blue-500 hover:border-slate-300 rounded-lg px-3 py-2 text-xs text-slate-850 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-blue-500/30 transition-all disabled:opacity-50 shadow-sm"
            />

            <button
              id="chat-submit-btn"
              type="submit"
              disabled={(status !== 'idle' && status !== 'completed' && status !== 'error') || (!inputText.trim() && uploadedImages.length === 0 && uploadedVolumes.length === 0 && uploadedTexts.length === 0)}
              className={`px-3 py-2 rounded-lg transition-all flex items-center justify-center gap-1.5 self-stretch flex-shrink-0 shadow-sm font-semibold cursor-pointer text-xs ${
                (!workflowState || workflowState === 'idle' || workflowState === 'connected' || workflowState === 'error' || workflowState === 'done') && (inputText.trim() || uploadedImages.length > 0 || uploadedVolumes.length > 0 || uploadedTexts.length > 0)
                  ? 'bg-blue-600 hover:bg-blue-700 text-white px-4'
                  : 'bg-blue-600 hover:bg-blue-500 text-white disabled:bg-slate-100 disabled:text-slate-300'
              }`}
            >
              {(!workflowState || workflowState === 'idle' || workflowState === 'connected' || workflowState === 'error' || workflowState === 'done') && (inputText.trim() || uploadedImages.length > 0 || uploadedVolumes.length > 0 || uploadedTexts.length > 0) ? (
                <>
                  <Sparkles size={13} className="animate-pulse" />
                  <span>开始分析</span>
                </>
              ) : (
                <Send size={13} />
              )}
            </button>

            {onCancel && (workflowState === 'submitted' || workflowState === 'running' || workflowState === 'streaming_answer') && (
              <button
                type="button"
                onClick={onCancel}
                className="px-3 py-2 bg-red-50 hover:bg-red-100 border border-red-200 text-red-650 rounded-lg text-xs font-bold flex items-center gap-1 cursor-pointer transition-all shadow-sm"
                title="取消当前诊断工作流"
              >
                <XCircle size={13} />
                <span>取消</span>
              </button>
            )}
          </form>

          {/* ── Slash 命令下拉菜单 ── */}
          {showSlashDropdown && filteredSkills.length > 0 && (
            <div
              ref={slashDropdownRef}
              className="absolute bottom-full left-0 right-0 mb-1 z-50 bg-white border border-slate-200 rounded-lg shadow-xl overflow-hidden max-h-64 overflow-y-auto"
            >
              <div className="px-3 py-1.5 bg-slate-50 border-b border-slate-100 text-[9px] text-slate-400 font-mono flex items-center gap-2">
                <Slash size={10} />
                <span>Skills — 选择后按 <kbd className="px-1 py-0.5 bg-white border border-slate-200 rounded text-[8px] font-bold">Tab</kbd> 或 <kbd className="px-1 py-0.5 bg-white border border-slate-200 rounded text-[8px] font-bold">↵</kbd> 确认</span>
              </div>
              {filteredSkills.map((skill, idx) => {
                const label = SKILL_LABEL_MAP[skill.name] || skill.name.replace(/_/g, ' ');
                const isActive = idx === activeSlashIndex;
                return (
                  <div
                    key={skill.name}
                    onClick={() => applySlashCommand(skill.name)}
                    onMouseEnter={() => setActiveSlashIndex(idx)}
                    className={`flex items-center justify-between px-3 py-2 cursor-pointer text-xs border-b border-slate-50 last:border-0 transition-colors ${
                      isActive ? 'bg-blue-50 text-blue-700' : 'text-slate-700 hover:bg-slate-50'
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-[9px] font-mono font-bold text-blue-500 flex-shrink-0">/{skill.name}</span>
                      <span className="text-[10px] text-slate-500 truncate">{label}</span>
                    </div>
                    <span className={`text-[8px] font-mono px-1.5 py-0.5 rounded flex-shrink-0 ${
                      skill.type === 'user' ? 'bg-purple-50 text-purple-500' : 'bg-slate-100 text-slate-400'
                    }`}>
                      {skill.type === 'user' ? '自定义' : '内置'}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* ── 模式/技能快捷按钮（点击填充 /skill_name） ── */}
        {status === 'idle' || status === 'completed' ? (
          <div className="flex items-center gap-1.5 overflow-x-auto py-1 scrollbar-thin">
            <button
              onClick={() => {
                setInputText('');
                inputRef.current?.focus();
              }}
              className={`flex items-center gap-1 px-2 py-1 rounded-full text-[9px] font-bold font-mono transition-all cursor-pointer whitespace-nowrap border ${
                !inputText.trimStart().startsWith('/') || inputText.trim() === '/'
                  ? 'bg-blue-600 text-white border-blue-600 shadow-sm'
                  : 'bg-white text-slate-500 border-slate-200 hover:border-blue-300 hover:text-blue-600'
              }`}
            >
              <Zap size={10} />
              <span>智能自动</span>
            </button>
            <button
              onClick={() => onOpenSkillManager?.()}
              className="flex items-center gap-1 px-2.5 py-1 rounded-full text-[9px] font-bold font-mono transition-all cursor-pointer whitespace-nowrap border bg-blue-50 text-blue-600 border-blue-200 hover:bg-blue-100 hover:border-blue-400 shadow-sm"
              title="上传或管理自定义 Skills"
            >
              <span className="text-xs font-bold leading-none">+</span>
              <span>管理</span>
            </button>
            <div className="w-px h-4 bg-slate-200 mx-0.5 flex-shrink-0" />
            {availableSkills.slice(0, 10).map((skill: any) => {
              const label = SKILL_LABEL_MAP[skill.name] || skill.name.replace(/_/g, ' ');
              const isUserSkill = skill.type === 'user';
              return (
                <button
                  key={skill.name}
                  onClick={() => handleSkillButtonClick(skill.name)}
                  className={`flex items-center gap-1 px-2 py-1 rounded-full text-[9px] font-bold font-mono transition-all cursor-pointer whitespace-nowrap border ${
                    inputText === ` /${skill.name} ` || inputText === ` /${skill.name}` || inputText === `/${skill.name} ` || inputText === `/${skill.name}`
                      ? 'bg-emerald-600 text-white border-emerald-600 shadow-sm'
                      : isUserSkill
                        ? 'bg-white text-purple-600 border-purple-200 hover:border-purple-400 hover:text-purple-700'
                        : 'bg-white text-slate-500 border-slate-200 hover:border-emerald-300 hover:text-emerald-600'
                  }`}
                >
                  {SKILL_ICON_MAP[skill.name] || <Zap size={10} />}
                  <span>{label}</span>
                  {isUserSkill && <span className="text-[7px] opacity-60 ml-0.5">·自定义</span>}
                </button>
              );
            })}
          </div>
        ) : null}
      </div>
    </div>
  );
}
