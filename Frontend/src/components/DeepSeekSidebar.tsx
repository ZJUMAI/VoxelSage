import { useState } from 'react';
import { Plus, Search, Trash2, MessageSquare, Activity, PanelLeftClose, Pencil } from 'lucide-react';
import { MedicalCase } from '../types';

interface DeepSeekSidebarProps {
	cases: MedicalCase[];
	selectedCaseId: string;
	onSelectCase: (caseId: string) => void;
	onDeleteCase: (caseId: string) => void;
	onOpenCreateModal: () => void;
	isCollapsed?: boolean;
	onToggleCollapse?: () => void;
	onOpenSkillManager?: () => void;
	onEditCase?: (caseId: string) => void;
}

export default function DeepSeekSidebar({
  cases,
  selectedCaseId,
  onSelectCase,
  onDeleteCase,
  onOpenCreateModal,
  isCollapsed = false,
  onToggleCollapse,
  onOpenSkillManager,
  onEditCase,
}: DeepSeekSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);

  const handleToggleSearch = () => {
    if (showSearch) setSearchQuery('');
    setShowSearch(!showSearch);
  };

  const filteredCases = cases.filter(c =>
    c.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.patientId.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.organ.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div
      id="deepseek-cases-sidebar"
      className={`bg-white text-slate-800 flex flex-col h-full border-r border-slate-200/80 flex-shrink-0 font-sans select-none relative transition-all duration-300 ease-in-out ${
        isCollapsed ? 'w-0 overflow-hidden opacity-0 border-r-0 p-0' : 'w-full opacity-100'
      }`}
    >
      {/* Header */}
      <div className="px-4 py-4 flex items-center justify-between border-b border-slate-100 flex-shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <div className="flex items-center justify-center w-7 h-7 bg-blue-600 rounded-lg flex-shrink-0">
            <Activity size={15} className="text-white animate-pulse" />
          </div>
          <div className="truncate">
            <h1 className="text-sm font-bold tracking-tight text-slate-900 uppercase font-mono leading-none truncate">VoxelSage</h1>
            <span className="text-[8px] text-slate-400 font-mono block mt-0.5 truncate">多模态会诊系统 v1.2</span>
          </div>
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          <button onClick={handleToggleSearch}
            className={`p-1 rounded transition-all ${showSearch ? 'text-blue-600 bg-blue-50' : 'text-slate-400 hover:text-slate-700 hover:bg-slate-100'}`}
            title="检索患者病例"><Search size={13} /></button>
          {onToggleCollapse && (
            <button onClick={onToggleCollapse} className="p-1 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded transition-all" title="收起侧边栏">
              <PanelLeftClose size={13} />
            </button>
          )}
        </div>
      </div>

      {/* New Dialog Button */}
      <div className="px-4 py-3.5 flex-shrink-0">
        <button onClick={onOpenCreateModal}
          className="w-full flex items-center justify-center gap-2 py-2 px-4 bg-blue-50 hover:bg-blue-100/90 border border-blue-200/60 text-blue-700 hover:text-blue-800 rounded-full text-xs font-semibold transition-all shadow-sm whitespace-nowrap">
          <Plus size={14} className="text-blue-600" />
          <span>开启新诊断 (录入病例)</span>
        </button>
      </div>

      {/* Search */}
      <div className={`px-4 pb-2 flex-shrink-0 transition-all duration-300 ease-in-out ${showSearch ? 'opacity-100 max-h-12 mt-1' : 'opacity-0 max-h-0 overflow-hidden'}`}>
        <div className="relative">
          <Search size={12} className="absolute left-3 top-2.5 text-slate-400" />
          <input type="text" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="搜索患者姓名、ID或器官..."
            className="w-full bg-slate-50 border border-slate-200 focus:border-blue-400 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-800 focus:outline-none placeholder-slate-400 font-mono" />
        </div>
      </div>

      {/* Cases List */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-4 scrollbar-thin scrollbar-thumb-slate-200 scrollbar-track-transparent">
        {filteredCases.length === 0 ? (
          <div className="text-center py-8 px-4 text-slate-400 space-y-2">
            <p className="text-xs">无活跃会诊病例记录</p>
            <p className="text-[10px] text-slate-400 leading-normal max-w-[180px] mx-auto">
              请点击上方录入真实病例，或直接在会诊终端上传影像，系统将自动帮您建立会诊记录。
            </p>
          </div>
        ) : (
          <div className="space-y-1.5">
            <h3 className="px-3 text-[10px] font-bold text-slate-400 font-mono tracking-wider uppercase">活跃会诊记录 ({filteredCases.length})</h3>
            <div className="space-y-1.5">
              {filteredCases.map((patientCase) => {
                const isSelected = patientCase.id === selectedCaseId;
                return (
                  <div key={patientCase.id} onClick={() => onSelectCase(patientCase.id)}
                    className={`group relative flex items-center justify-between px-3 py-2.5 rounded-xl cursor-pointer border transition-all ${
                      isSelected ? 'bg-blue-600 border-blue-600 text-white font-medium shadow-sm' : 'bg-blue-50/75 hover:bg-blue-100/90 border-blue-100/60 text-slate-800'
                    }`}>
                    <div className="flex items-center gap-2.5 min-w-0 flex-1">
                      <MessageSquare size={14} className={isSelected ? 'text-white' : 'text-blue-500 group-hover:text-blue-600'} />
                      <div className="truncate text-left leading-tight">
                        <span className={`text-xs font-semibold block truncate ${isSelected ? 'text-white' : 'text-slate-800'}`}>{patientCase.name}</span>
                        <span className={`text-[10px] font-mono block mt-0.5 truncate ${isSelected ? 'text-blue-100/90' : 'text-slate-500'}`}>{patientCase.patientId} | {patientCase.organ}</span>
                      </div>
                    </div>
                    <button type="button" onClick={(e) => { e.stopPropagation(); onDeleteCase(patientCase.id); }}
                      className={`p-1.5 rounded-lg transition-all border ${
                        isSelected ? 'text-white/95 hover:text-white bg-blue-700/60 hover:bg-blue-700 border-white/20' : 'text-slate-400 hover:text-red-600 bg-white hover:bg-red-50 border-slate-200 hover:border-red-200 shadow-sm'
                      }`} title="删除病例"><Trash2 size={11} /></button>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
