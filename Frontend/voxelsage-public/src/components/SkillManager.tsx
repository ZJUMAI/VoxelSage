import React, { useState, useEffect } from 'react';
import { Activity, Trash2, Upload, X } from 'lucide-react';
import { SKILLS_BASE_URL } from '../api';

interface SkillItem {
  name: string;
  version?: string;
  description?: string;
  type?: 'builtin' | 'user';
}

interface SkillManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

const SkillManager: React.FC<SkillManagerProps> = ({ isOpen, onClose }) => {
  const [skills, setSkills] = useState<SkillItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  // Fetch skills when opened
  useEffect(() => {
    if (!isOpen) return;
    setMessage('');
    setLoading(true);
    fetch(`${SKILLS_BASE_URL}/api/skills/list`)
      .then(r => r.ok ? r.json() : { skills: [] })
      .then(d => setSkills(d.skills || []))
      .catch(() => setSkills([]))
      .finally(() => setLoading(false));
  }, [isOpen]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setMessage('');
    try {
      const fd = new FormData();
      fd.append('file', file);
      const resp = await fetch(`${SKILLS_BASE_URL}/api/skills/upload`, { method: 'POST', body: fd });
      const data = await resp.json();
      setMessage(resp.ok ? `✅ 注册成功：${data.name} v${data.version}` : `❌ ${data.detail || '注册失败'}`);
      if (resp.ok) {
        const listResp = await fetch(`${SKILLS_BASE_URL}/api/skills/list`);
        if (listResp.ok) setSkills((await listResp.json()).skills || []);
      }
    } catch (err: any) {
      setMessage(`❌ 上传失败: ${err.message}`);
    }
    setUploading(false);
    e.target.value = '';
  };

  const handleDelete = async (name: string) => {
    if (!window.confirm(`确定要注销 Skill「${name}」吗？`)) return;
    try {
      const resp = await fetch(`${SKILLS_BASE_URL}/api/skills/${name}`, { method: 'DELETE' });
      const data = await resp.json();
      setMessage(resp.ok ? `✅ 已注销 ${name}` : `❌ ${data.detail || '注销失败'}`);
      if (resp.ok) {
        const listResp = await fetch(`${SKILLS_BASE_URL}/api/skills/list`);
        if (listResp.ok) setSkills((await listResp.json()).skills || []);
      }
    } catch (err: any) {
      setMessage(`❌ 注销失败: ${err.message}`);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      style={{ backgroundColor: 'rgba(15, 23, 42, 0.6)', backdropFilter: 'blur(4px)' }}
      onClick={onClose}
    >
      <div
        className="bg-white rounded-xl shadow-2xl w-full max-w-lg max-h-[85vh] flex flex-col border border-slate-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200 flex-shrink-0">
          <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2">
            <Activity size={16} className="text-blue-600" />
            <span className="font-mono">Skills 管理</span>
          </h3>
          <button onClick={onClose} className="p-1 text-slate-400 hover:text-slate-700 rounded hover:bg-slate-100 transition-all cursor-pointer">
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* Message */}
          {message && (
            <div className="px-3 py-2 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700">
              {message}
            </div>
          )}

          {/* Upload Area */}
          <label className="block border-2 border-dashed border-slate-300 hover:border-blue-400 rounded-lg p-5 text-center cursor-pointer transition-colors">
            <input type="file" accept=".zip" onChange={handleUpload} className="hidden" disabled={uploading} />
            <Upload size={20} className="mx-auto mb-1.5 text-slate-400" />
            <p className="text-xs text-slate-500 font-medium">
              {uploading ? '⏳ 上传注册中...' : '点击上传自定义 Skill（.zip，需含 skill.yaml + main.py）'}
            </p>
          </label>

          {/* Skills List */}
          <div>
            <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider font-mono mb-2">
              已注册 Skills {loading ? '(加载中...)' : `(${skills.length})`}
            </h4>
            {skills.length === 0 && !loading && (
              <p className="text-xs text-slate-400 italic text-center py-6">暂无已注册 Skills</p>
            )}
            {skills.length > 0 && (
              <div className="space-y-1.5">
                {skills.map((s, i) => (
                  <div key={i} className="flex items-center justify-between px-3 py-2.5 bg-slate-50 rounded-lg border border-slate-100 hover:border-slate-200 transition-colors">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-slate-800 font-mono">{s.name}</span>
                        {s.version && <span className="text-[9px] text-slate-400 bg-white border border-slate-200 px-1 rounded">v{s.version}</span>}
                        <span className={`text-[8px] px-1.5 py-0.5 rounded-full border ${
                          s.type === 'user'
                            ? 'bg-emerald-50 text-emerald-600 border-emerald-200'
                            : 'bg-blue-50 text-blue-600 border-blue-200'
                        }`}>
                          {s.type === 'user' ? '自定义' : '内置'}
                        </span>
                      </div>
                      {s.description && (
                        <p className="text-[10px] text-slate-500 mt-0.5 truncate">{s.description}</p>
                      )}
                    </div>
                    {s.type === 'user' && (
                      <button onClick={() => handleDelete(s.name)} className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded transition-all flex-shrink-0 cursor-pointer" title="注销">
                        <Trash2 size={12} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SkillManager;
