import { AgentStatus, ReasoningStep } from './types';

const runtimeHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
const isSecure = typeof window !== 'undefined' && window.location.protocol === 'https:';
const defaultHttpProtocol = isSecure ? 'https:' : 'http:';
const defaultWsProtocol = isSecure ? 'wss:' : 'ws:';
const stripTrailingSlash = (value: string) => value.replace(/\/$/, '');

const portAHttpRoot = stripTrailingSlash(
  import.meta.env.VITE_PORT_A_HTTP_URL || `${defaultHttpProtocol}//${runtimeHost}:8900`,
);
const portAWsRoot = stripTrailingSlash(
  import.meta.env.VITE_PORT_A_WS_URL || `${defaultWsProtocol}//${runtimeHost}:8900`,
);
const portBHttpRoot = stripTrailingSlash(
  import.meta.env.VITE_PORT_B_HTTP_URL || `${defaultHttpProtocol}//${runtimeHost}:8765`,
);

// Backend URLs can be injected at build time; defaults follow the page hostname.
export const WS_CHAT_URL = `${portAWsRoot}/ws/frontend/chat`;
export const HTTP_BASE_URL = portAHttpRoot;
export const HEALTH_URL = `${portAHttpRoot}/health`;
export const UPLOAD_URL = `${portAHttpRoot}/api/upload`;

// Port B — Skill 列表 & 渲染工具
export const SKILLS_BASE_URL = portBHttpRoot;
export const SKILLS_LIST_URL = `${SKILLS_BASE_URL}/api/skills/list`;

/** @deprecated 直接使用配置后的 WS_CHAT_URL */
export function getApiBaseUrl(): string {
  return WS_CHAT_URL;
}

/** @deprecated 直接使用配置后的 HTTP_BASE_URL */
export function getHttpRootUrl(): string {
  return HTTP_BASE_URL;
}

/**
 * 上传文件到后端服务器 (支持 .nii.gz, .png 等)
 * 新后端 agent v2.0 返回 FileUploadItem[]
 */
export async function uploadFileToServer(
  file: File,
): Promise<{ file_id: string; path: string; name: string }> {
  const formData = new FormData();
  formData.append('files', file);

  const response = await fetch(`${UPLOAD_URL}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`文件上传失败: ${response.statusText}`);
  }

  // 后端返回 files 数组
  const result: any = await response.json();
  if (result.files && result.files.length > 0) {
    return {
      file_id: result.files[0].file_id,
      path: result.files[0].path,
      name: result.files[0].name,
    };
  }
  throw new Error('后端返回的 files 数组为空');
}

/**
 * 检查后端健康状态
 */
export async function checkBackendHealth(): Promise<any> {
  const response = await fetch(`${HEALTH_URL}`);
  if (!response.ok) {
    throw new Error(`健康检查失败: ${response.statusText}`);
  }
  return await response.json();
}
