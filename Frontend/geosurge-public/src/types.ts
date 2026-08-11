export interface RoiBox {
  x: number;      // percentage or pixel coordinate
  y: number;
  width: number;
  height: number;
}

export interface MemoryBank {
  [key: string]: string;
}

export type AgentStatus = 'idle' | 'uploading' | 'processing' | 'running' | 'completed' | 'error';

export type WorkflowState =
  | "idle"
  | "connecting"
  | "connected"
  | "submitted"
  | "running"
  | "streaming_answer"
  | "done"
  | "error"
  | "cancelled";

export interface CaseRecordField {
  label: string;
  value: string;
  editable: boolean;
  type?: 'text' | 'textarea' | 'select';
  options?: string[];
}

export interface CaseRecord {
  patient_info?: {
    name?: string;
    age?: number;
    gender?: string;
    [key: string]: any;
  };
  clinical_findings?: string;
  imaging_findings?: string;
  diagnosis_recommendations?: string;
  [key: string]: any;
}


export interface ReasoningStep {
  id: string;
  phase: string;
  title: string;
  status: 'pending' | 'processing' | 'success' | 'error';
  timestamp: string;
  message: string;
  data?: {
    [key: string]: any;
  };
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'agent' | 'system';
  text: string;
  timestamp: string;
  status?: 'typing' | 'done';
  files?: Array<{ name: string; type: string }>;
  /** 关联的 Agent 执行 ID — 用于在用户消息下方渲染执行卡 */
  executionId?: string;
  /** 执行完成后的轻量快照（随聊天消息持久化，不含 params/result） */
  executionSnapshot?: AgentExecutionSnapshot;
}

/** 执行状态 — Agent 任务的单一事实来源 */
export type ExecutionStatus =
  | 'idle'
  | 'connecting'       // WebSocket 连接中
  | 'connected'        // WebSocket 已连接
  | 'processing'       // 正在准备医学数据/分割中
  | 'agent_round'      // 第 N 轮分析中
  | 'streaming'        // 正在生成回答
  | 'completed'        // 分析完成
  | 'cancelling'       // 正在取消
  | 'cancelled'        // 已取消
  | 'error'            // 执行异常
  | 'disconnected';    // 实时连接已中断

/** 单个 skill 调用的状态 */
export interface CallState {
  callId: string;
  skillName: string;
  status: 'pending' | 'running' | 'ok' | 'error' | 'unresolved';
  startedAt?: string;
  completedAt?: string;
  /** 仅保留轻量错误消息，不含 stack trace / detail */
  errorMessage?: string;
}

/** 轻量执行快照 — 用于持久化到聊天消息，不含 params/result/path */
export interface AgentExecutionSnapshot {
  status: ExecutionStatus;
  roundCount: number;
  totalCalls: number;
  successCalls: number;
  errorCalls: number;
  unresolvedCalls: number;
  completedAt?: string;
  errorMessage?: string;
}

/** Agent 执行完整状态 — 保存在内存中，不直接持久化 */
export interface AgentExecutionState {
  executionId: string;
  userMessageId: string;
  status: ExecutionStatus;
  roundNumber: number;
  calls: Record<string, CallState>;
  streamedText: string;
  streamingMsgId: string | null;
  errorMessage?: string;
  sessionId?: string;
  createdAt: string;
  completedAt?: string;
}

export interface MedicalCase {
  id: string;
  name: string;
  patientId: string;
  gender: 'M' | 'F';
  age: number;
  clinicalHistory: string;
  query: string;
  organ: string;
  sliceCount: number;
  targetSliceIndex: number;
  slices: string[]; // placeholder or mock CT slice images/canvas patterns
  roiBox?: RoiBox;
  memoryBank: MemoryBank;
  steps: ReasoningStep[];
  finalAnswer: string;
}
