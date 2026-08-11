import React, { useState } from 'react';
import { AgentExecutionState, AgentExecutionSnapshot, ExecutionStatus } from '../types';
import { formatExecutionStatus } from '../utils/agentExecutionReducer';
import { CheckCircle, XCircle, Loader2, Clock, AlertTriangle, ChevronDown, ChevronRight, Activity } from 'lucide-react';

interface AgentExecutionCardProps {
  /** 当前活跃的执行状态（内存中），或 undefined 表示该执行已结束且无内存状态 */
  state?: AgentExecutionState;
  /** 持久化的快照（从聊天消息恢复时使用） */
  snapshot?: AgentExecutionSnapshot;
  /** 是否是最新的活跃执行 */
  isActive: boolean;
  /** 当前轮次号文本（仅活跃执行时显示） */
  roundLabel?: string;
}

const STATUS_ICONS: Record<ExecutionStatus, React.ReactNode> = {
  idle: <Clock size={12} className="text-slate-400" />,
  connecting: <Loader2 size={12} className="text-blue-500 animate-spin" />,
  connected: <Activity size={12} className="text-emerald-500" />,
  processing: <Loader2 size={12} className="text-blue-500 animate-spin" />,
  agent_round: <Loader2 size={12} className="text-blue-500 animate-spin" />,
  streaming: <Loader2 size={12} className="text-blue-500 animate-spin" />,
  completed: <CheckCircle size={12} className="text-emerald-500" />,
  cancelling: <Loader2 size={12} className="text-amber-500 animate-spin" />,
  cancelled: <XCircle size={12} className="text-slate-400" />,
  error: <AlertTriangle size={12} className="text-red-500" />,
  disconnected: <XCircle size={12} className="text-red-400" />,
};

const STATUS_COLORS: Record<ExecutionStatus, string> = {
  idle: 'border-slate-200',
  connecting: 'border-blue-300',
  connected: 'border-emerald-300',
  processing: 'border-blue-300',
  agent_round: 'border-blue-300',
  streaming: 'border-blue-300',
  completed: 'border-emerald-300',
  cancelling: 'border-amber-300',
  cancelled: 'border-slate-300',
  error: 'border-red-300',
  disconnected: 'border-red-300',
};

/** 从 state 或 snapshot 获取状态值 */
function getStatus(
  state?: AgentExecutionState,
  snapshot?: AgentExecutionSnapshot
): ExecutionStatus {
  if (state) return state.status;
  if (snapshot) return snapshot.status;
  return 'completed';
}

function getRoundCount(
  state?: AgentExecutionState,
  snapshot?: AgentExecutionSnapshot
): number {
  if (state) return state.roundNumber;
  if (snapshot) return snapshot.roundCount;
  return 0;
}

function getCallSummary(
  state?: AgentExecutionState,
  snapshot?: AgentExecutionSnapshot
): { total: number; ok: number; error: number; unresolved: number } {
  if (state) {
    const calls = Object.values(state.calls);
    return {
      total: calls.length,
      ok: calls.filter((c) => c.status === 'ok').length,
      error: calls.filter((c) => c.status === 'error').length,
      unresolved: calls.filter((c) => c.status === 'unresolved').length,
    };
  }
  if (snapshot) {
    return {
      total: snapshot.totalCalls,
      ok: snapshot.successCalls,
      error: snapshot.errorCalls,
      unresolved: snapshot.unresolvedCalls,
    };
  }
  return { total: 0, ok: 0, error: 0, unresolved: 0 };
}

const AgentExecutionCard: React.FC<AgentExecutionCardProps> = ({
  state,
  snapshot,
  isActive,
  roundLabel,
}) => {
  const [expanded, setExpanded] = useState(false);
  const status = getStatus(state, snapshot);
  const roundCount = getRoundCount(state, snapshot);
  const calls = getCallSummary(state, snapshot);
  const isTerminal = ['completed', 'cancelled', 'error', 'disconnected'].includes(status);

  // Compact summary for completed cards
  if (!isActive && isTerminal && !expanded) {
    return (
      <div
        className={`mt-2 mb-2 border rounded-lg overflow-hidden cursor-pointer transition-all hover:shadow-sm ${STATUS_COLORS[status]} bg-white`}
        onClick={() => setExpanded(true)}
      >
        <div className="flex items-center gap-2 px-3 py-2 text-[10px]">
          <ChevronRight size={10} className="text-slate-400 flex-shrink-0" />
          {STATUS_ICONS[status]}
          <span className="font-semibold text-slate-700">{formatExecutionStatus(status)}</span>
          <span className="text-slate-300">·</span>
          <span className="text-slate-500">{roundCount} 轮</span>
          {calls.total > 0 && (
            <>
              <span className="text-slate-300">·</span>
              <span className="text-slate-500">
                {calls.ok}✓ {calls.error > 0 && `${calls.error}✗ `}
                {calls.unresolved > 0 && `${calls.unresolved}?`}
              </span>
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div
      className={`mt-2 mb-2 border rounded-lg overflow-hidden transition-all ${
        isActive ? 'border-blue-300 shadow-sm bg-blue-50/30' : STATUS_COLORS[status] + ' bg-white'
      }`}
    >
      {/* Header */}
      <div
        className={`flex items-center justify-between px-3 py-2 ${
          isActive ? 'bg-blue-50/50' : 'bg-slate-50/50'
        }`}
      >
        <div className="flex items-center gap-2 text-[10px]">
          {STATUS_ICONS[status]}
          <span
            className={`font-semibold ${
              status === 'error' || status === 'disconnected'
                ? 'text-red-700'
                : status === 'completed'
                ? 'text-emerald-700'
                : 'text-blue-700'
            }`}
          >
            {formatExecutionStatus(status)}
          </span>
          {roundCount > 0 && (
            <span className="text-slate-400 font-mono">
              · {roundCount} 轮
            </span>
          )}
          {roundLabel && isActive && (
            <span className="text-blue-500 font-mono">{roundLabel}</span>
          )}
        </div>
        {isTerminal && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-slate-400 hover:text-slate-600 transition-colors cursor-pointer p-0.5"
          >
            {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          </button>
        )}
      </div>

      {/* Body (expanded or active) */}
      {(expanded || !isTerminal || isActive) && (
        <div className="px-3 py-2 space-y-1.5 text-[10px]">
          {/* Skill calls summary */}
          {calls.total > 0 && (
            <div className="flex flex-wrap gap-1">
              {state &&
                Object.values(state.calls).map((call) => (
                  <span
                    key={call.callId}
                    className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-mono font-semibold ${
                      call.status === 'ok'
                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                        : call.status === 'error'
                        ? 'bg-red-50 text-red-700 border border-red-200'
                        : call.status === 'unresolved'
                        ? 'bg-slate-50 text-slate-500 border border-slate-200'
                        : call.status === 'running'
                        ? 'bg-blue-50 text-blue-700 border border-blue-200'
                        : 'bg-slate-50 text-slate-400 border border-slate-200'
                    }`}
                  >
                    {call.status === 'ok' && <CheckCircle size={8} />}
                    {call.status === 'error' && <XCircle size={8} />}
                    {call.status === 'running' && <Loader2 size={8} className="animate-spin" />}
                    {call.status === 'unresolved' && <AlertTriangle size={8} />}
                    {call.skillName.replace(/_/g, ' ')}
                  </span>
                ))}
              {snapshot && (
                <span className="text-slate-500">
                  {calls.ok} 成功{calls.error > 0 && `, ${calls.error} 失败`}{' '}
                  {calls.unresolved > 0 && `, ${calls.unresolved} 未完成`}
                </span>
              )}
            </div>
          )}

          {/* Progress / error message */}
          {isActive && !isTerminal && (
            <div className="flex items-center gap-1.5 text-slate-500">
              <span className="w-1 h-1 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-1 h-1 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-1 h-1 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          )}

          {/* Error info */}
          {status === 'error' && (
            <div className="text-red-600 font-medium truncate max-w-full">
              {state?.errorMessage || snapshot?.errorMessage || '执行异常'}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AgentExecutionCard;
