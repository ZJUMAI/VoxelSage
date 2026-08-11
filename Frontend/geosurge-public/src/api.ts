import { AgentStatus, ReasoningStep } from './types';

// 后端地址 — 请根据实际部署修改（公开版默认使用本机 localhost）
export const WS_CHAT_URL = 'ws://localhost:8900/ws/frontend/chat';
export const HTTP_BASE_URL = 'http://localhost:8900';
export const HEALTH_URL = 'http://localhost:8900/health';
export const UPLOAD_URL = 'http://localhost:8900/api/upload';

// Port B — Skill 列表 & 渲染工具
export const SKILLS_BASE_URL = 'http://localhost:8765';
export const SKILLS_LIST_URL = `${SKILLS_BASE_URL}/api/skills/list`;

/** @deprecated 改用硬编码 WS_CHAT_URL */
export function getApiBaseUrl(): string {
  return WS_CHAT_URL;
}

/** @deprecated 改用硬编码 HTTP_BASE_URL */
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
