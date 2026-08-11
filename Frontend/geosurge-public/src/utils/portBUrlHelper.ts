/**
 * Port B 路径 → 完整 URL 转换工具
 *
 * Port B 返回的结果中可能包含相对路径（如 /process-output/xxx/xxx.png）
 * 而非完整 URL。这个工具统一将路径转换为前端可访问的完整 URL。
 */

// Port B 基地址 — 用于将完整 URL 转为相对路径，通过 Port A 代理
const PORT_B_PUBLIC = 'http://localhost:8765';

/**
 * 将 Port B 路径转为同源 URL（通过 Port A 代理，避免跨域）
 * - Port B 完整 URL → 转为相对路径（/process-output/...）
 * - 已是相对路径 → 直接返回
 */
export function normalizePortBUrl(pathOrUrl: string | undefined | null): string {
  if (!pathOrUrl) return '';
  if (pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')) {
    if (pathOrUrl.startsWith(PORT_B_PUBLIC)) {
      return pathOrUrl.replace(PORT_B_PUBLIC, '');
    }
    return pathOrUrl;
  }
  const clean = pathOrUrl.startsWith('/') ? pathOrUrl : `/${pathOrUrl}`;
  return clean;
}

/**
 * 对 slice_selection 的 slices 数组做 URL 补全
 * 优先: png_url → normalizePortBUrl(png_path) → url(后端outputs格式)
 * 同样处理 overlay_url → overlay_path
 */
export function normalizeSlices(slices: any[]): any[] {
  if (!slices || !Array.isArray(slices)) return [];
  return slices.map((s) => ({
    ...s,
    png_url: s.png_url || normalizePortBUrl(s.png_path) || s.url || '',
    overlay_url: s.overlay_url || normalizePortBUrl(s.overlay_path) || s.url || '',
  }));
}

/**
 * 通用：将 result 对象中的 _path 字段补全为 _url
 */
export function normalizeSkillResult(result: any): any {
  if (!result || typeof result !== 'object') return result;

  const out = { ...result };

  // 3D HTML
  if (out.html_path && !out.html_url) {
    out.html_url = normalizePortBUrl(out.html_path);
  }

  // editor_url (分割编辑器 — 保留原始 URL，iframe 直接加载)
  // iframe 加载跨域页面不受 CORS 限制
  // 保持原样传递给 SegmentationModificationView

  // slices
  if (out.slices && Array.isArray(out.slices)) {
    out.slices = normalizeSlices(out.slices);
  }

  return out;
}
