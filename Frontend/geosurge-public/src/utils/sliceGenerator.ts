import { RoiBox } from '../types';

/**
 * 医疗 CT 切片动态 Canvas 绘制工具
 * 用于模拟 3D 卷扫切片效果，随着滑动条变化，器官轮廓、位置和病灶大小会发生真实形态变化。
 */

interface DrawSliceParams {
  canvas: HTMLCanvasElement;
  sliceIndex: number;
  totalSlices: number;
  caseId: string;
  roiBox: RoiBox;
  targetSliceIndex: number;
  highlightRoi: boolean;
  organ?: string;
}

export function drawMedicalSlice({
  canvas,
  sliceIndex,
  totalSlices,
  caseId,
  roiBox,
  targetSliceIndex,
  highlightRoi,
  organ,
}: DrawSliceParams) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;

  // 1. 清空画布 (医疗 CT 背景通常是纯黑)
  ctx.fillStyle = '#060a12';
  ctx.fillRect(0, 0, w, h);

  // 绘制一个带有网格线的医学级背景，突出科技感
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
  ctx.lineWidth = 1;
  const gridSize = 40;
  for (let x = 0; x < w; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // 比例因子，用于匹配 512x512 基准
  const scaleX = w / 512;
  const scaleY = h / 512;

  // 2. 绘制人体躯干外轮廓 (椭圆形)
  const centerX = w / 2;
  const centerY = h / 2 + 10 * scaleY;
  const bodyRadiusX = 180 * scaleX;
  const bodyRadiusY = 130 * scaleY;

  ctx.beginPath();
  ctx.ellipse(centerX, centerY, bodyRadiusX, bodyRadiusY, 0, 0, 2 * Math.PI);
  ctx.fillStyle = '#111723'; // 软肌肉组织灰度
  ctx.fill();
  ctx.strokeStyle = '#2b354a';
  ctx.lineWidth = 3;
  ctx.stroke();

  // 3. 绘制脊椎骨 (Spine) - 位于后背正中偏下
  const spineX = centerX;
  const spineY = centerY + bodyRadiusY - 40 * scaleY;
  const spineR = 25 * scaleX;

  // 脊椎主体 (白色骨质)
  ctx.beginPath();
  ctx.arc(spineX, spineY, spineR, 0, 2 * Math.PI);
  ctx.fillStyle = '#e2e8f0';
  ctx.fill();
  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // 椎管内部 (黑色或软组织)
  ctx.beginPath();
  ctx.arc(spineX, spineY, spineR * 0.4, 0, 2 * Math.PI);
  ctx.fillStyle = '#060a12';
  ctx.fill();

  // 脊椎横突/棘突骨刺突起
  ctx.fillStyle = '#e2e8f0';
  ctx.beginPath();
  ctx.moveTo(spineX - spineR, spineY);
  ctx.quadraticCurveTo(spineX - spineR - 15 * scaleX, spineY + 5 * scaleY, spineX - spineR - 5 * scaleX, spineY + 15 * scaleY);
  ctx.lineTo(spineX - spineR + 5 * scaleX, spineY + 5 * scaleY);
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(spineX + spineR, spineY);
  ctx.quadraticCurveTo(spineX + spineR + 15 * scaleX, spineY + 5 * scaleY, spineX + spineR + 5 * scaleX, spineY + 15 * scaleY);
  ctx.lineTo(spineX + spineR - 5 * scaleX, spineY + 5 * scaleY);
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(spineX, spineY + spineR);
  ctx.lineTo(spineX - 5 * scaleX, spineY + spineR + 20 * scaleY);
  ctx.lineTo(spineX + 5 * scaleX, spineY + spineR + 20 * scaleY);
  ctx.fill();

  // 4. 绘制肋骨断影 (Ribs) - 胸壁侧骨骼结节
  ctx.fillStyle = '#e2e8f0';
  const ribCount = 8;
  for (let i = 0; i < ribCount; i++) {
    const angleLeft = Math.PI / 2 + 0.3 + (i * (Math.PI - 0.6)) / (ribCount - 1);
    const angleRight = Math.PI / 2 - 0.3 - (i * (Math.PI - 0.6)) / (ribCount - 1);

    const rlX = centerX + Math.cos(angleLeft) * (bodyRadiusX - 10 * scaleX);
    const rlY = centerY + Math.sin(angleLeft) * (bodyRadiusY - 10 * scaleY);
    ctx.beginPath();
    ctx.ellipse(rlX, rlY, 12 * scaleX, 5 * scaleY, angleLeft + Math.PI/2, 0, 2*Math.PI);
    ctx.fill();

    const rrX = centerX + Math.cos(angleRight) * (bodyRadiusX - 10 * scaleX);
    const rrY = centerY + Math.sin(angleRight) * (bodyRadiusY - 10 * scaleY);
    ctx.beginPath();
    ctx.ellipse(rrX, rrY, 12 * scaleX, 5 * scaleY, angleRight + Math.PI/2, 0, 2*Math.PI);
    ctx.fill();
  }

  // 5. 根据不同的病例/器官类型绘制特异的实质器官
  const organLower = (organ || '').toLowerCase();
  const isLiver = caseId === 'case_liver_01' || organLower.includes('liver') || organLower.includes('肝');
  const isPancreas = caseId === 'case_pancreas_02' || organLower.includes('pancreas') || organLower.includes('胰');
  const isLung = caseId === 'case_lung_03' || organLower.includes('lung') || organLower.includes('肺');

  if (isLiver) {
    // ---- 腹部 CT：肝脏 + 脾脏 + 胃 + 胆囊 ----
    // 肝脏在右侧 (CT标准图像的左侧)，占比非常大
    // 肝脏随着层数动态膨胀和收缩：40 ~ 95
    if (sliceIndex >= 30 && sliceIndex <= 105) {
      const liverProgress = 1 - Math.abs(sliceIndex - 72) / 40; // 0 到 1 之间的饱满度
      if (liverProgress > 0) {
        const liverX = centerX - 60 * scaleX;
        const liverY = centerY - 10 * scaleY;
        const rx = 100 * scaleX * Math.max(0.4, liverProgress);
        const ry = 75 * scaleY * Math.max(0.4, liverProgress);

        ctx.save();
        // 绘制不规则肝脏形状
        ctx.beginPath();
        ctx.ellipse(liverX, liverY, rx, ry, -0.2, 0, 2 * Math.PI);
        // 稍作不规则裁剪变形，模拟真实肝脏弯曲
        ctx.fillStyle = '#2d1e1a'; // 肝脏暗红色/深灰褐色
        ctx.fillStyle = 'rgba(56, 44, 44, 0.95)'; // 肝密度稍微低一些 (脂肪肝 HU=38)
        ctx.fill();
        ctx.strokeStyle = '#4a3737';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();

        // 绘制胆囊 (Gallbladder) - 肝脏下方的梨形小空腔
        if (sliceIndex >= 65 && sliceIndex <= 85) {
          ctx.beginPath();
          ctx.ellipse(liverX + 40 * scaleX, liverY + 30 * scaleY, 15 * scaleX, 10 * scaleY, 0.5, 0, 2 * Math.PI);
          ctx.fillStyle = '#1c281a'; // 液性暗区
          ctx.fill();
          ctx.strokeStyle = '#2d402b';
          ctx.stroke();
        }
      }
    }

    // 脾脏在左侧 (CT标准图像的右侧)
    if (sliceIndex >= 45 && sliceIndex <= 90) {
      const spleenProgress = 1 - Math.abs(sliceIndex - 68) / 25;
      if (spleenProgress > 0) {
        const spleenX = centerX + 95 * scaleX;
        const spleenY = centerY - 5 * scaleY;
        ctx.beginPath();
        ctx.ellipse(spleenX, spleenY, 45 * scaleX * spleenProgress, 30 * scaleY * spleenProgress, 0.4, 0, 2 * Math.PI);
        ctx.fillStyle = '#2d2836'; // 脾脏颜色
        ctx.fill();
        ctx.strokeStyle = '#3e374b';
        ctx.stroke();
      }
    }

    // 胃部 (Stomach) - 中间偏左上方的巨大空腔，通常含有气体或食物
    if (sliceIndex >= 40 && sliceIndex <= 80) {
      const stomachProgress = 1 - Math.abs(sliceIndex - 55) / 25;
      if (stomachProgress > 0) {
        const stomachX = centerX + 20 * scaleX;
        const stomachY = centerY - 45 * scaleY;
        ctx.beginPath();
        ctx.ellipse(stomachX, stomachY, 45 * scaleX * stomachProgress, 35 * scaleY * stomachProgress, -0.3, 0, 2 * Math.PI);
        ctx.fillStyle = '#1c212d'; // 混杂内容物
        ctx.fill();
        ctx.strokeStyle = '#2d354a';
        ctx.stroke();

        // 胃内气液平面 (胃泡)
        ctx.beginPath();
        ctx.ellipse(stomachX - 10 * scaleX, stomachY - 10 * scaleY, 20 * scaleX, 12 * scaleY, -0.3, 0, 2 * Math.PI);
        ctx.fillStyle = '#060a12'; // 极黑的气体
        ctx.fill();
      }
    }

    // 6. 绘制肝脏内部的病灶 (ROI Lesion)
    // 只有在接近 targetSliceIndex 左右时病灶才可见，尺寸呈现高斯分布
    const distToTarget = Math.abs(sliceIndex - targetSliceIndex);
    if (distToTarget <= 15) {
      const factor = Math.exp(-Math.pow(distToTarget / 6, 2)); // 高斯衰减系数
      const lSize = 18 * scaleX * factor;

      // 映射 ROI 物理坐标
      const rx = roiBox.x * scaleX;
      const ry = roiBox.y * scaleY;

      ctx.beginPath();
      ctx.arc(rx, ry, lSize, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(25, 25, 25, 0.85)'; // 肝脏低密度囊肿 (Hypodense)
      ctx.fill();
      ctx.strokeStyle = '#524343';
      ctx.lineWidth = 1;
      ctx.stroke();

      // 如果需要高亮病灶
      if (highlightRoi) {
        ctx.beginPath();
        ctx.arc(rx, ry, lSize + 4 * scaleX, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)'; // 红色呼吸闪烁高亮
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]); // 还原
      }
    }

  } else if (isPancreas) {
    // ---- 腹部下段 CT：胰腺 + 肾脏 + 腹主动脉 ----
    // 肾脏 (Kidneys) - 左右各一，位于腹壁脊柱两侧
    if (sliceIndex >= 50 && sliceIndex <= 100) {
      const kidneyProgress = 1 - Math.abs(sliceIndex - 75) / 30;
      if (kidneyProgress > 0) {
        // 右肾
        ctx.beginPath();
        ctx.ellipse(centerX - 70 * scaleX, centerY + 30 * scaleY, 28 * scaleX * kidneyProgress, 38 * scaleY * kidneyProgress, -0.1, 0, 2 * Math.PI);
        ctx.fillStyle = '#222d3d';
        ctx.fill();
        ctx.strokeStyle = '#324259';
        ctx.stroke();

        // 左肾
        ctx.beginPath();
        ctx.ellipse(centerX + 70 * scaleX, centerY + 30 * scaleY, 28 * scaleX * kidneyProgress, 38 * scaleY * kidneyProgress, 0.1, 0, 2 * Math.PI);
        ctx.fillStyle = '#222d3d';
        ctx.fill();
        ctx.strokeStyle = '#324259';
        ctx.stroke();
      }
    }

    // 腹主动脉 (Aorta) - 脊椎前方的明亮白色圆形管腔
    ctx.beginPath();
    ctx.arc(centerX - 15 * scaleX, centerY + bodyRadiusY - 70 * scaleY, 12 * scaleX, 0, 2 * Math.PI);
    ctx.fillStyle = '#475569'; // 强化后的血管
    ctx.fill();
    ctx.strokeStyle = '#64748b';
    ctx.stroke();

    // 胰腺 (Pancreas) - 横跨中上腹呈条带状
    if (sliceIndex >= 50 && sliceIndex <= 90) {
      const pancProgress = 1 - Math.abs(sliceIndex - 68) / 22;
      if (pancProgress > 0) {
        // 绘制条带状的胰腺
        ctx.beginPath();
        ctx.moveTo(centerX - 50 * scaleX, centerY - 25 * scaleY);
        ctx.quadraticCurveTo(
          centerX, centerY - 45 * scaleY,
          centerX + 70 * scaleX, centerY - 15 * scaleY
        );
        ctx.lineTo(centerX + 65 * scaleX, centerY - 5 * scaleY);
        ctx.quadraticCurveTo(
          centerX, centerY - 32 * scaleY,
          centerX - 55 * scaleX, centerY - 15 * scaleY
        );
        ctx.closePath();
        ctx.fillStyle = '#2a3429'; // 胰腺微绿/灰组织
        ctx.fill();
        ctx.strokeStyle = '#3c4a3b';
        ctx.stroke();
      }
    }

    // 胰头病灶 (Pancreatic Head Lesion)
    const distToTarget = Math.abs(sliceIndex - targetSliceIndex);
    if (distToTarget <= 12) {
      const factor = Math.exp(-Math.pow(distToTarget / 5, 2));
      const lSize = 12 * scaleX * factor;

      const rx = roiBox.x * scaleX;
      const ry = roiBox.y * scaleY;

      ctx.beginPath();
      ctx.arc(rx, ry, lSize, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(239, 68, 68, 0.2)'; // 占位病灶
      ctx.fill();
      ctx.beginPath();
      ctx.arc(rx, ry, lSize * 0.7, 0, 2 * Math.PI);
      ctx.fillStyle = '#3a2d24'; // 实质低强化结节
      ctx.fill();
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1.2;
      ctx.stroke();

      if (highlightRoi) {
        ctx.beginPath();
        ctx.arc(rx, ry, lSize + 5 * scaleX, 0, 2 * Math.PI);
        ctx.strokeStyle = '#f59e0b'; // 黄色警告框
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

  } else if (isLung) {
    // ---- 胸部肺窗 CT：左右大肺野 + 纵隔心脏 ----
    // 肺腔充满空气，在 CT 肺窗上呈极深的纯黑色大斑块
    if (sliceIndex >= 10 && sliceIndex <= 115) {
      const lungProgress = 1 - Math.abs(sliceIndex - 55) / 50;
      if (lungProgress > 0) {
        const shrinkFactor = Math.max(0.4, lungProgress);

        // 左肺腔 (对应屏幕右侧)
        ctx.beginPath();
        ctx.ellipse(centerX + 65 * scaleX, centerY - 5 * scaleY, 65 * scaleX * shrinkFactor, 105 * scaleY * shrinkFactor, -0.15, 0, 2 * Math.PI);
        ctx.fillStyle = '#05070a'; // 空气深度黑
        ctx.fill();
        ctx.strokeStyle = '#2b354a';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // 绘制左肺内部的树枝状肺纹理 (气管/肺血管分支)
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(centerX + 35 * scaleX, centerY - 10 * scaleY);
        ctx.lineTo(centerX + 75 * scaleX, centerY - 25 * scaleY);
        ctx.moveTo(centerX + 35 * scaleX, centerY - 10 * scaleY);
        ctx.lineTo(centerX + 85 * scaleX, centerY + 15 * scaleY);
        ctx.stroke();

        // 右肺腔 (对应屏幕左侧)
        ctx.beginPath();
        ctx.ellipse(centerX - 65 * scaleX, centerY - 5 * scaleY, 65 * scaleX * shrinkFactor, 105 * scaleY * shrinkFactor, 0.15, 0, 2 * Math.PI);
        ctx.fillStyle = '#05070a';
        ctx.fill();
        ctx.strokeStyle = '#2b354a';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // 右肺内部纹理
        ctx.beginPath();
        ctx.moveTo(centerX - 35 * scaleX, centerY - 10 * scaleY);
        ctx.lineTo(centerX - 75 * scaleX, centerY - 25 * scaleY);
        ctx.moveTo(centerX - 35 * scaleX, centerY - 10 * scaleY);
        ctx.lineTo(centerX - 85 * scaleX, centerY + 15 * scaleY);
        ctx.stroke();
      }
    }

    // 纵隔与心脏 (Mediastinum & Heart) - 中间偏左的实体灰区
    if (sliceIndex >= 25 && sliceIndex <= 95) {
      const heartProgress = 1 - Math.abs(sliceIndex - 60) / 40;
      if (heartProgress > 0) {
        ctx.beginPath();
        ctx.ellipse(centerX + 10 * scaleX, centerY + 10 * scaleY, 42 * scaleX * heartProgress, 55 * scaleY * heartProgress, 0.1, 0, 2 * Math.PI);
        ctx.fillStyle = '#1e293b'; // 心肌密度
        ctx.fill();
        ctx.strokeStyle = '#334155';
        ctx.stroke();
      }
    }

    // 肺部结节 (Lung Nodule) - 位于右肺 (屏幕左侧) 的一粒小白点
    const distToTarget = Math.abs(sliceIndex - targetSliceIndex);
    if (distToTarget <= 15) {
      const factor = Math.exp(-Math.pow(distToTarget / 5, 2));
      const noduleSize = 6 * scaleX * factor;

      const rx = roiBox.x * scaleX;
      const ry = roiBox.y * scaleY;

      ctx.beginPath();
      ctx.arc(rx, ry, noduleSize, 0, 2 * Math.PI);
      ctx.fillStyle = '#f1f5f9'; // 结节属于高密度软组织，表现为亮白色点
      ctx.fill();
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1;
      ctx.stroke();

      // 磨玻璃晕轮 (Ground Glass Opacity halo around it)
      ctx.beginPath();
      ctx.arc(rx, ry, noduleSize * 2.5, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(241, 245, 249, 0.18)';
      ctx.fill();

      if (highlightRoi) {
        ctx.beginPath();
        ctx.arc(rx, ry, noduleSize * 3 + 4 * scaleX, 0, 2 * Math.PI);
        ctx.strokeStyle = '#10b981'; // 绿色呼吸闪烁高亮
        ctx.lineWidth = 1.5;
        ctx.setLineDash([2, 2]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  } else {
    // ---- 其他自定义器官（例如：脑部、肾脏等）的通用占位及病灶渲染 ----
    const distToTarget = Math.abs(sliceIndex - targetSliceIndex);
    if (distToTarget <= 15) {
      const factor = Math.exp(-Math.pow(distToTarget / 5, 2));
      const noduleSize = 10 * scaleX * factor;
      const rx = roiBox.x * scaleX;
      const ry = roiBox.y * scaleY;

      // 绘制一个抽象的中等密度腺瘤样结节
      ctx.beginPath();
      ctx.arc(rx, ry, noduleSize, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(234, 179, 8, 0.25)'; // 浅黄色实质病变
      ctx.fill();
      ctx.beginPath();
      ctx.arc(rx, ry, noduleSize * 0.6, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(239, 68, 68, 0.4)'; // 核心病变
      ctx.fill();
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.2;
      ctx.stroke();

      if (highlightRoi) {
        ctx.beginPath();
        ctx.arc(rx, ry, noduleSize + 5 * scaleX, 0, 2 * Math.PI);
        ctx.strokeStyle = '#ef4444'; // 红色呼吸闪烁
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  // 7. 在左上角绘制方向切片标识
  ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
  ctx.font = `bold ${Math.round(11 * scaleX)}px monospace`;
  ctx.fillText('R', 15 * scaleX, centerY); // 医疗影像右侧对应屏幕左边
  ctx.fillText('L', w - 25 * scaleX, centerY);
  ctx.fillText('A', centerX, 25 * scaleY); // Anterior 前侧
  ctx.fillText('P', centerX, h - 15 * scaleY); // Posterior 后侧

  // 右下角标尺 (Scale Indicator)
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
  ctx.lineWidth = 2;
  const barWidth = 50 * scaleX;
  ctx.beginPath();
  ctx.moveTo(w - 70 * scaleX, h - 25 * scaleY);
  ctx.lineTo(w - 20 * scaleX, h - 25 * scaleY);
  ctx.moveTo(w - 70 * scaleX, h - 28 * scaleY);
  ctx.lineTo(w - 70 * scaleX, h - 22 * scaleY);
  ctx.moveTo(w - 20 * scaleX, h - 28 * scaleY);
  ctx.lineTo(w - 20 * scaleX, h - 22 * scaleY);
  ctx.stroke();

  ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
  ctx.font = `${Math.round(9 * scaleX)}px monospace`;
  ctx.fillText('5 cm', w - 55 * scaleX, h - 10 * scaleY);
}
