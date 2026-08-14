import { AgentExecutionState, AgentExecutionSnapshot, ExecutionStatus, CallState } from '../types';

// ── Action Types ──

type ExecutionAction =
  | { type: 'EXECUTION_CREATED'; executionId: string; userMessageId: string }
  | { type: 'WS_CONNECTING'; executionId: string }
  | { type: 'WS_CONNECTED'; executionId: string; sessionId: string }
  | { type: 'WORKFLOW_STARTED'; executionId: string }
  | { type: 'AGENT_STARTED'; executionId: string }
  | { type: 'ROUND_START'; executionId: string }
  | { type: 'EXECUTION_PLAN'; executionId: string; plan: any[] }
  | { type: 'CALL_STARTED'; executionId: string; callId: string; skillName: string }
  | { type: 'CALL_COMPLETED'; executionId: string; callId: string }
  | { type: 'CALL_ERROR'; executionId: string; callId: string; errorMessage: string }
  | { type: 'STREAM_CHUNK'; executionId: string; chunk: string }
  | { type: 'ROUND_END'; executionId: string }
  | { type: 'EXECUTION_COMPLETED'; executionId: string }
  | { type: 'EXECUTION_ERROR'; executionId: string; errorMessage: string }
  | { type: 'EXECUTION_CANCELLING'; executionId: string }
  | { type: 'EXECUTION_CANCELLED'; executionId: string }
  | { type: 'EXECUTION_DISCONNECTED'; executionId: string };

// ── Reducer ──

export function agentExecutionReducer(
  state: Record<string, AgentExecutionState>,
  action: ExecutionAction
): Record<string, AgentExecutionState> {
  const exec = state[action.executionId];

  switch (action.type) {
    case 'EXECUTION_CREATED':
      return {
        ...state,
        [action.executionId]: {
          executionId: action.executionId,
          userMessageId: action.userMessageId,
          status: 'idle',
          roundNumber: 0,
          calls: {},
          streamedText: '',
          streamingMsgId: null,
          createdAt: new Date().toISOString(),
        },
      };

    case 'WS_CONNECTING':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec, status: 'connecting' as ExecutionStatus } };

    case 'WS_CONNECTED':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec, status: 'connected' as ExecutionStatus, sessionId: action.sessionId } };

    case 'WORKFLOW_STARTED':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec, status: 'processing' as ExecutionStatus, roundNumber: 1 } };

    case 'AGENT_STARTED':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec, status: 'agent_round' as ExecutionStatus } };

    case 'ROUND_START':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec, roundNumber: exec.roundNumber + 1 } };

    case 'EXECUTION_PLAN':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec, status: 'agent_round' as ExecutionStatus } };

    case 'CALL_STARTED': {
      if (!exec) return state;
      const newCall: CallState = {
        callId: action.callId,
        skillName: action.skillName,
        status: 'running',
        startedAt: new Date().toISOString(),
      };
      return {
        ...state,
        [action.executionId]: {
          ...exec,
          calls: { ...exec.calls, [action.callId]: newCall },
        },
      };
    }

    case 'CALL_COMPLETED': {
      if (!exec || !exec.calls[action.callId]) return state;
      return {
        ...state,
        [action.executionId]: {
          ...exec,
          calls: {
            ...exec.calls,
            [action.callId]: { ...exec.calls[action.callId], status: 'ok', completedAt: new Date().toISOString() },
          },
        },
      };
    }

    case 'CALL_ERROR': {
      if (!exec || !exec.calls[action.callId]) return state;
      return {
        ...state,
        [action.executionId]: {
          ...exec,
          calls: {
            ...exec.calls,
            [action.callId]: { ...exec.calls[action.callId], status: 'error', errorMessage: action.errorMessage, completedAt: new Date().toISOString() },
          },
        },
      };
    }

    case 'STREAM_CHUNK':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec, status: 'streaming' as ExecutionStatus, streamedText: exec.streamedText + action.chunk } };

    case 'ROUND_END':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec } };

    case 'EXECUTION_COMPLETED':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec, status: 'completed' as ExecutionStatus, completedAt: new Date().toISOString() } };

    case 'EXECUTION_ERROR':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec, status: 'error' as ExecutionStatus, errorMessage: action.errorMessage, completedAt: new Date().toISOString() } };

    case 'EXECUTION_CANCELLING':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec, status: 'cancelling' as ExecutionStatus } };

    case 'EXECUTION_CANCELLED':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec, status: 'cancelled' as ExecutionStatus, completedAt: new Date().toISOString() } };

    case 'EXECUTION_DISCONNECTED':
      if (!exec) return state;
      return { ...state, [action.executionId]: { ...exec, status: 'disconnected' as ExecutionStatus } };

    default:
      return state;
  }
}

// ── Helper: AgentStatus 映射 ──

export function executionStatusToAgentStatus(status: ExecutionStatus): string {
  switch (status) {
    case 'idle': return 'idle';
    case 'connecting':
    case 'connected': return 'uploading';
    case 'processing':
    case 'agent_round':
    case 'streaming': return 'processing';
    case 'completed': return 'completed';
    case 'cancelling':
    case 'cancelled': return 'completed';
    case 'error':
    case 'disconnected': return 'error';
    default: return 'idle';
  }
}

// ── Helper: WorkflowState 映射 ──

export function executionStatusToWorkflowState(status: ExecutionStatus): string {
  switch (status) {
    case 'idle': return 'idle';
    case 'connecting': return 'connecting';
    case 'connected': return 'connected';
    case 'processing':
    case 'agent_round': return 'running';
    case 'streaming': return 'streaming_answer';
    case 'completed': return 'done';
    case 'cancelling': return 'cancelled';
    case 'cancelled': return 'cancelled';
    case 'error':
    case 'disconnected': return 'error';
    default: return 'idle';
  }
}

// ── Helper: 构建轻量快照 ──

export function buildSnapshot(exec: AgentExecutionState): AgentExecutionSnapshot {
  const callList = exec.calls ? Object.values(exec.calls) : [];
  return {
    status: exec.status || 'completed',
    roundCount: exec.roundNumber || 0,
    totalCalls: callList.length,
    successCalls: callList.filter(c => c.status === 'ok').length,
    errorCalls: callList.filter(c => c.status === 'error').length,
    unresolvedCalls: callList.filter(c => c.status === 'unresolved').length,
    completedAt: exec.completedAt,
    errorMessage: exec.errorMessage,
  };
}

// ── Helper: 格式化执行状态为中文标签 ──

export function formatExecutionStatus(status: ExecutionStatus): string {
  const labels: Record<ExecutionStatus, string> = {
    idle: '等待中',
    connecting: '连接中',
    connected: '已连接',
    processing: '数据准备中',
    agent_round: '智能体分析中',
    streaming: '生成回答中',
    completed: '分析完成',
    cancelling: '正在取消',
    cancelled: '已取消',
    error: '执行异常',
    disconnected: '连接断开',
  };
  return labels[status] || status;
}
