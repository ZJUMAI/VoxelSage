// @ts-ignore
import * as nifti from 'nifti-reader-js';

export interface NiftiVolume {
  fileName: string;
  width: number;
  height: number;
  depth: number;
  datatypeCode: number;
  datatypeName: string;
  pixDims: number[]; // [dx, dy, dz, ...]
  header: any;
  typedArray: any;
  /** 验证有效的 scl_slope，为 0/NaN/Infinity 时设为 1 */
  sclSlope: number;
  /** 验证有效的 scl_inter，为 NaN/Infinity 时设为 0 */
  sclInter: number;
  /** 用户确认或来源可推导的强度单位 */
  intensityUnit: 'HU' | 'unknown';
  /** 稳定标识（优先 file_id），用于判断是否为新影像 */
  volumeId: string;
  /** Auto 范围缓存（延迟计算），null=未计算 */
  autoRange: { min: number; max: number } | null;
}

/** 窗宽窗位预设键 */
export type WindowPresetKey = 'liver' | 'hcc_narrow' | 'soft_tissue' | 'auto' | 'custom';

/** 窗宽窗位预设定义 */
export interface WindowPreset {
  name: string;
  wl: number | null;
  ww: number | null;
  experimental: boolean;
}

export const WINDOW_PRESETS: Record<WindowPresetKey, WindowPreset> = {
  liver:      { name: '传统肝窗 (研究显示预设)',           wl: 90,  ww: 150, experimental: false },
  hcc_narrow: { name: 'HCC 窄窗 (实验性预设，非统一临床标准)', wl: 90,  ww: 120, experimental: true  },
  soft_tissue:{ name: '软组织窗',                          wl: 50,  ww: 350, experimental: false },
  auto:       { name: '自动范围',                          wl: null, ww: null, experimental: false },
  custom:     { name: '自定义',                            wl: null, ww: null, experimental: false },
};

/**
 * 将原始体素值转换为物理强度值。
 * 纯函数——不修改任何传入参数。
 * physicalValue = rawValue * sclSlope + sclInter
 */
export function toPhysicalValue(rawValue: number, volume: NiftiVolume): number {
  return rawValue * volume.sclSlope + volume.sclInter;
}

/**
 * 验证 scl_slope：无效/0/NaN/Infinity → 1
 */
export function validateSclSlope(value: number): number {
  if (value === 0 || !Number.isFinite(value) || Number.isNaN(value)) return 1;
  return value;
}

/**
 * 验证 scl_inter：无效/NaN/Infinity → 0
 */
export function validateSclInter(value: number): number {
  if (!Number.isFinite(value) || Number.isNaN(value)) return 0;
  return value;
}

/**
 * 窗宽窗位预设的显示范围
 */
export function getPresetWindowRange(wl: number, ww: number): { low: number; high: number } {
  return { low: wl - ww / 2, high: wl + ww / 2 };
}

/**
 * 计算 volume 的稳健物理强度范围（延迟计算，结果缓存到 volume.autoRange）。
 *
 * 算法：按大步长对整个 volume 分层采样 → 2%–98% 百分位。
 * 采样步长 = max(1, floor(totalVoxels / 10000))，约 10 000 个样本点。
 * 对 512³ Int16 约 256 M 体素，采样约 10 000 点，计算耗时 < 5 ms。
 *
 * 边界处理：
 * - 所有体素相同时返回 [sample, sample+1]
 * - 非有限值时返回 [0, 1]
 */
export function computeAutoRange(volume: NiftiVolume): { min: number; max: number } {
  if (volume.autoRange) return volume.autoRange;

  const data = volume.typedArray;
  const total = data.length;
  const sampleCount = 10000;
  const step = Math.max(1, Math.floor(total / sampleCount));

  const samples: number[] = [];
  for (let i = 0; i < total; i += step) {
    const raw = data[i];
    if (raw === undefined || raw === null) continue;
    samples.push(raw * volume.sclSlope + volume.sclInter);
  }

  if (samples.length === 0) {
    volume.autoRange = { min: 0, max: 1 };
    return volume.autoRange;
  }

  samples.sort((a, b) => a - b);
  const loIdx = Math.floor(samples.length * 0.02);
  const hiIdx = Math.floor(samples.length * 0.98);
  let min = samples[Math.max(0, loIdx)];
  let max = samples[Math.min(samples.length - 1, hiIdx)];

  // 边界处理
  if (!Number.isFinite(min)) min = 0;
  if (!Number.isFinite(max)) max = 1;
  if (max - min < 1) max = min + 1;

  volume.autoRange = { min, max };
  return volume.autoRange;
}

/**
 * Maps NIfTI datatype codes to their human-readable type names and TypedArray constructor fallbacks
 */
export function getDatatypeInfo(code: number) {
  switch (code) {
    case 2:
      return { name: 'Uint8 (8-bit unsigned)', TypedArray: Uint8Array };
    case 4:
      return { name: 'Int16 (16-bit signed)', TypedArray: Int16Array };
    case 8:
      return { name: 'Int32 (32-bit signed)', TypedArray: Int32Array };
    case 16:
      return { name: 'Float32 (32-bit float)', TypedArray: Float32Array };
    case 32:
      return { name: 'Float64 (64-bit float)', TypedArray: Float64Array };
    case 256:
      return { name: 'Int8 (8-bit signed)', TypedArray: Int8Array };
    case 512:
      return { name: 'Uint16 (16-bit unsigned)', TypedArray: Uint16Array };
    case 768:
      return { name: 'Uint32 (32-bit unsigned)', TypedArray: Uint32Array };
    default:
      return { name: `Unknown (${code})`, TypedArray: Int16Array };
  }
}

export interface ParseNiftiOptions {
  intensityUnit?: 'HU' | 'unknown';
  volumeId?: string;
}

/**
 * Parses a NIfTI file (either .nii or .nii.gz) from an ArrayBuffer
 */
export async function parseNiftiFile(
  arrayBuffer: ArrayBuffer,
  fileName: string,
  options?: ParseNiftiOptions,
): Promise<NiftiVolume> {
  let data = arrayBuffer;

  // Decompress if compressed (.nii.gz)
  if (nifti.isCompressed(data)) {
    try {
      data = nifti.decompress(data);
    } catch (decompressErr: any) {
      // fflate throws "unexpected EOF" for truncated gzip files
      const msg = decompressErr?.message || String(decompressErr);
      if (msg.includes('unexpected EOF') || msg.includes('unexpected end') || msg.includes('invalid')) {
        throw new Error(
          `文件 ${fileName} 的 gzip 压缩数据不完整（已被截断或损坏），` +
          `无法解压。请检查源文件完整性或重新下载。`
        );
      }
      throw new Error(`gzip 解压失败: ${msg}`);
    }
  }

  if (!nifti.isNIFTI(data)) {
    throw new Error('该文件不符合 NIfTI 标准格式！请上传有效的 .nii 或 .nii.gz 文件。');
  }

  const header = nifti.readHeader(data);
  const imageBuffer = nifti.readImage(header, data);

  const dims = header.dims; // [ndim, Nx, Ny, Nz, Nt, ...]
  const width = dims[1] || 1;
  const height = dims[2] || 1;
  const depth = dims[3] || 1;

  const datatypeCode = header.datatypeCode;
  const { name: datatypeName, TypedArray } = getDatatypeInfo(datatypeCode);

  // Parse voxel spacings (pixDims)
  const pixDims = header.pixDims || [1, 1, 1, 1];

  // Wrap raw image ArrayBuffer in the correct TypedArray — 原始值，永不写回物理值
  const typedArray = new TypedArray(imageBuffer);

  // 验证 scl_slope / scl_inter
  const sclSlope = validateSclSlope(header.scl_slope);
  const sclInter = validateSclInter(header.scl_inter);

  return {
    fileName,
    width,
    height,
    depth,
    datatypeCode,
    datatypeName,
    pixDims,
    header,
    typedArray,
    sclSlope,
    sclInter,
    intensityUnit: options?.intensityUnit || 'unknown',
    volumeId: options?.volumeId || fileName,
    autoRange: null,
  };
}

/**
 * Renders a single 2D slice from a NIfTI volume onto an HTML Canvas.
 *
 * 窗宽窗位映射（DICOM 线性窗）：
 *   1. 原始体素值 → 物理值（raw * sclSlope + sclInter）
 *   2. 物理值按窗口 [level - width/2, level + width/2] 归一化到 0～1
 *   3. clamp 到 [0, 1] → 映射到 0～255
 *
 * 当 windowWidth / windowLevel 都提供且 ww >= 1 时使用指定窗口；
 * 否则回退到当前切片的物理值 min/max。
 *
 * 不修改 volume.typedArray。
 */
export function drawNiftiSliceOnCanvas(
  canvas: HTMLCanvasElement,
  volume: NiftiVolume,
  sliceIndex: number, // 0-based index
  windowWidth?: number,
  windowLevel?: number
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const Nx = volume.width;
  const Ny = volume.height;
  const Nz = volume.depth;

  const z = Math.max(0, Math.min(sliceIndex, Nz - 1));
  const sliceSize = Nx * Ny;
  const sliceOffset = z * sliceSize;
  const dataArray = volume.typedArray;
  const slope = volume.sclSlope;
  const inter = volume.sclInter;

  if (!dataArray || sliceOffset + sliceSize > dataArray.length) {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#666666';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('切片索引超出数据范围', canvas.width / 2, canvas.height / 2);
    return;
  }

  const imgData = ctx.createImageData(Nx, Ny);

  // 确定窗口范围（物理值空间）
  let currentMin: number;
  let currentMax: number;
  let useSliceFallback = false;

  if (windowWidth !== undefined && windowLevel !== undefined && windowWidth >= 1) {
    currentMin = windowLevel - windowWidth / 2;
    currentMax = windowLevel + windowWidth / 2;
  } else {
    // 回退：计算当前切片的物理值 min/max
    useSliceFallback = true;
    let sliceMin = Infinity;
    let sliceMax = -Infinity;
    for (let i = 0; i < sliceSize; i++) {
      const raw = dataArray[sliceOffset + i];
      if (raw === undefined || raw === null) continue;
      const phys = raw * slope + inter;
      if (phys < sliceMin) sliceMin = phys;
      if (phys > sliceMax) sliceMax = phys;
    }
    currentMin = sliceMin === Infinity ? 0 : sliceMin;
    currentMax = sliceMax === -Infinity ? 1 : sliceMax;
  }

  const range = currentMax - currentMin || 1;

  for (let y = 0; y < Ny; y++) {
    for (let x = 0; x < Nx; x++) {
      const niftiIndex = sliceOffset + (Ny - 1 - y) * Nx + x;
      const raw = dataArray[niftiIndex];
      if (raw === undefined || raw === null) {
        // 缺省像素写黑
        const canvasIndex = (y * Nx + x) * 4;
        imgData.data[canvasIndex] = 0;
        imgData.data[canvasIndex + 1] = 0;
        imgData.data[canvasIndex + 2] = 0;
        imgData.data[canvasIndex + 3] = 255;
        continue;
      }
      const phys = raw * slope + inter;

      // DICOM 线性窗 + clamp
      let normVal = (phys - currentMin) / range;
      normVal = Math.max(0, Math.min(1, normVal));

      const gray = Math.floor(normVal * 255);

      const canvasIndex = (y * Nx + x) * 4;
      imgData.data[canvasIndex] = gray;
      imgData.data[canvasIndex + 1] = gray;
      imgData.data[canvasIndex + 2] = gray;
      imgData.data[canvasIndex + 3] = 255;
    }
  }

  const offscreen = document.createElement('canvas');
  offscreen.width = Nx;
  offscreen.height = Ny;
  const offscreenCtx = offscreen.getContext('2d');
  if (offscreenCtx) {
    offscreenCtx.putImageData(imgData, 0, 0);
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
  }
}
