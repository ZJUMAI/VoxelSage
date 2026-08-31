/**
 * BezierSurface.js — 浏览器端 Bézier 曲面渲染模块
 *
 * 根据 GeoSurge 的 bicubic_bezier 切面定义（4×4 控制点 + 参考平面），
 * 在 Three.js 中生成带有距离颜色映射的半透明曲面。
 *
 * 使用方式:
 *   import { buildBezierSurface } from './BezierSurface.js';
 *   const { mesh, line, label } = buildBezierSurface(planeData);
 *   scene.add(mesh);
 *   scene.add(line);
 *   scene.add(label);
 */

import * as THREE from 'three';

// ----------------------------------------------------------------
//  Bernstein 基函数 (degree 3)
// ----------------------------------------------------------------
function bernstein3(t) {
  const u = 1.0 - t;
  return [
    u * u * u,
    3.0 * u * u * t,
    3.0 * u * t * t,
    t * t * t,
  ];
}

// ----------------------------------------------------------------
//  距离 → 颜色映射
//    ≥ 5.0 mm  → 蓝色 (安全, 醒目不刺眼)
//    2.0~5.0   → 黄到橙色渐变 (警戒)
//    < 2.0 mm  → 红色 (违规)
// ----------------------------------------------------------------
// ----------------------------------------------------------------
//  距离 → 颜色映射 (临床切缘着色)
//    < MARGIN (5mm)   → 红→橙 (切缘不足)
//    MARGIN ~ 2×MARGIN → 橙→黄 (切缘临界)
//    2×MARGIN ~ 4×MARGIN → 黄绿→蓝绿 (切缘充足)
//    ≥ 4×MARGIN        → 深蓝 (安全)
// ----------------------------------------------------------------
const MARGIN_MM = 5.0;
function distanceToColor(d) {
  if (d < MARGIN_MM) {
    // 0 → MARGIN: 深红 → 橙色
    const t = d / MARGIN_MM;
    return [1.0, t * 0.65, t * 0.15];
  }
  if (d < MARGIN_MM * 2) {
    // MARGIN → 2×MARGIN: 橙色 → 黄色
    const t = (d - MARGIN_MM) / MARGIN_MM;
    return [1.0 - t * 0.2, 0.65 + t * 0.35, 0.15 + t * 0.55];
  }
  if (d < MARGIN_MM * 4) {
    // 2×MARGIN → 4×MARGIN: 黄绿 → 蓝绿
    const t = (d - MARGIN_MM * 2) / (MARGIN_MM * 2);
    return [0.8 - t * 0.7, 1.0 - t * 0.3, t * 0.9];
  }
  // ≥ 4×MARGIN: 深蓝（远处）
  const t = Math.min((d - MARGIN_MM * 4) / (MARGIN_MM * 4), 1.0);
  return [0.1 + t * 0.05, 0.3 + t * 0.25, 0.9 + t * 0.05];
}

// ----------------------------------------------------------------
//  构建 Bézier 曲面网格
// ----------------------------------------------------------------
function buildBezierGeometry(planeData) {
  const ref = planeData.reference_plane;
  const origin = new THREE.Vector3().fromArray(ref.origin_mm);
  const normal = new THREE.Vector3().fromArray(ref.normal_world);
  const uAxis  = new THREE.Vector3().fromArray(ref.u_axis_world);
  const vAxis  = new THREE.Vector3().fromArray(ref.v_axis_world);
  const uRange = ref.u_range_mm;  // [u_min, u_max]
  const vRange = ref.v_range_mm;  // [v_min, v_max]
  const grid = planeData.height_control_4x4_mm;  // 4x4
  const distGrid = planeData.vertex_distances_mm; // 可选
  const centerOffset = planeData.center_offset;
  if (centerOffset) {
    origin.sub(new THREE.Vector3().fromArray(centerOffset));
  }
  const res = planeData.surface_resolution || [20, 20];
  const nU = res[0];
  const nV = res[1];

  const positions = [];
  const colors = [];
  const indices = [];
  const hasDist = distGrid && distGrid.length === nU && distGrid[0] && distGrid[0].length === nV;
  if (hasDist) {
    let dMin = Infinity, dMax = -Infinity;
    for (let i = 0; i < nU; i++) {
      for (let j = 0; j < nV; j++) {
        const d = distGrid[i][j];
        if (d < dMin) dMin = d;
        if (d > dMax) dMax = d;
      }
    }
    console.log('[buildBezierGeometry] hasDist=true, nU=' + nU + ' nV=' + nV,
      'distMin=' + dMin.toFixed(2), 'distMax=' + dMax.toFixed(2),
      'planeData.candidate_name=' + (planeData.candidate_name || 'N/A').substring(0, 50));
  } else {
    console.warn('[buildBezierGeometry] hasDist=false, distGrid type=' + typeof distGrid,
      'nU=' + nU + ' nV=' + nV,
      distGrid ? ('len=' + distGrid.length + ' row0=' + (distGrid[0] ? distGrid[0].length : 'null')) : 'null');
  }

  for (let i = 0; i < nU; i++) {
    const uNorm = i / (nU - 1);  // [0, 1]
    const u = uRange[0] + uNorm * (uRange[1] - uRange[0]);
    const bu = bernstein3(uNorm);

    for (let j = 0; j < nV; j++) {
      const vNorm = j / (nV - 1);  // [0, 1]
      const v = vRange[0] + vNorm * (vRange[1] - vRange[0]);
      const bv = bernstein3(vNorm);

      // 计算 Bézier 高度
      let height = 0.0;
      for (let ci = 0; ci < 4; ci++) {
        for (let cj = 0; cj < 4; cj++) {
          height += bu[ci] * bv[cj] * grid[ci][cj];
        }
      }

      // 参考平面上的点 + 沿法线偏移
      const point = new THREE.Vector3()
        .copy(origin)
        .addScaledVector(uAxis, u)
        .addScaledVector(vAxis, v)
        .addScaledVector(normal, height);

      positions.push(point.x, point.y, point.z);

      // 颜色
      if (hasDist) {
        const d = distGrid[i][j];
        const c = distanceToColor(d);
        colors.push(c[0], c[1], c[2]);
      } else {
        // 无距离数据 → 默认蓝色
        colors.push(0.3, 0.5, 1.0);
      }
    }
  }

  // 索引 (四边形拆分为两个三角形)
  for (let i = 0; i < nU - 1; i++) {
    for (let j = 0; j < nV - 1; j++) {
      const a = i * nV + j;
      const b = i * nV + (j + 1);
      const c = (i + 1) * nV + j;
      const d = (i + 1) * nV + (j + 1);
      indices.push(a, b, c);
      indices.push(b, d, c);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  return geometry;
}

// ----------------------------------------------------------------
//  构建曲面边界线
// ----------------------------------------------------------------
function buildBoundaryLine(planeData) {
  const ref = planeData.reference_plane;
  const origin = new THREE.Vector3().fromArray(ref.origin_mm);
  const normal = new THREE.Vector3().fromArray(ref.normal_world);
  const uAxis  = new THREE.Vector3().fromArray(ref.u_axis_world);
  const vAxis  = new THREE.Vector3().fromArray(ref.v_axis_world);
  const uRange = ref.u_range_mm;
  const vRange = ref.v_range_mm;
  const grid = planeData.height_control_4x4_mm;
  const res = planeData.surface_resolution || [20, 20];

  // 沿四个边采样 (共 4 × (res-1) 段)
  const nU = res[0];
  const nV = res[1];
  const points = [];

  const centerOffset = planeData.center_offset;
  function sample(uNorm, vNorm) {
    const u = uRange[0] + uNorm * (uRange[1] - uRange[0]);
    const v = vRange[0] + vNorm * (vRange[1] - vRange[0]);
    const bu = bernstein3(uNorm);
    const bv = bernstein3(vNorm);
    let height = 0.0;
    for (let ci = 0; ci < 4; ci++) {
      for (let cj = 0; cj < 4; cj++) {
        height += bu[ci] * bv[cj] * grid[ci][cj];
      }
    }
    const pt = new THREE.Vector3()
      .copy(origin)
      .addScaledVector(uAxis, u)
      .addScaledVector(vAxis, v)
      .addScaledVector(normal, height);
    if (centerOffset) {
      pt.sub(new THREE.Vector3().fromArray(centerOffset));
    }
    return pt;
  }

  // 下边 (u=0..1, v=0)
  for (let i = 0; i < nU; i++) points.push(sample(i / (nU - 1), 0.0));
  // 右边 (u=1, v=0..1)
  for (let j = 1; j < nV; j++) points.push(sample(1.0, j / (nV - 1)));
  // 上边 (u=1..0, v=1)
  for (let i = nU - 2; i >= 0; i--) points.push(sample(i / (nU - 1), 1.0));
  // 左边 (u=0, v=1..0)
  for (let j = nV - 2; j >= 1; j--) points.push(sample(0.0, j / (nV - 1)));

  const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(
    lineGeo,
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 })
  );
}

// ----------------------------------------------------------------
//  构建距离标注精灵 (Canvas Sprite)
// ----------------------------------------------------------------
function buildSurfaceGrid(planeData) {
  const ref = planeData.reference_plane;
  const origin = new THREE.Vector3().fromArray(ref.origin_mm);
  const normal = new THREE.Vector3().fromArray(ref.normal_world);
  const uAxis  = new THREE.Vector3().fromArray(ref.u_axis_world);
  const vAxis  = new THREE.Vector3().fromArray(ref.v_axis_world);
  const uRange = ref.u_range_mm;
  const vRange = ref.v_range_mm;
  const grid = planeData.height_control_4x4_mm;
  const centerOffset = planeData.center_offset;
  function sample(uNorm, vNorm) {
    const u = uRange[0] + uNorm * (uRange[1] - uRange[0]);
    const v = vRange[0] + vNorm * (vRange[1] - vRange[0]);
    const bu = bernstein3(uNorm);
    const bv = bernstein3(vNorm);
    let height = 0.0;
    for (let ci = 0; ci < 4; ci++) {
      for (let cj = 0; cj < 4; cj++) {
        height += bu[ci] * bv[cj] * grid[ci][cj];
      }
    }
    const pt = new THREE.Vector3()
      .copy(origin)
      .addScaledVector(uAxis, u)
      .addScaledVector(vAxis, v)
      .addScaledVector(normal, height);
    if (centerOffset) {
      pt.sub(new THREE.Vector3().fromArray(centerOffset));
    }
    return pt;
  }
  const div = 10;
  const positions = [];
  const stride = div + 1;
  for (let i = 0; i <= div; i++) {
    for (let j = 0; j < div; j++) {
      const a = sample(j / div, i / div);
      const b = sample((j + 1) / div, i / div);
      positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
  }
  for (let j = 0; j <= div; j++) {
    for (let i = 0; i < div; i++) {
      const a = sample(j / div, i / div);
      const b = sample(j / div, (i + 1) / div);
      positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  return new THREE.LineSegments(
    geo,
    new THREE.LineBasicMaterial({ color: 0x222222, transparent: true, opacity: 0.7 })
  );
}

export function buildMarginLabel(planeData, maxDim) {
  const margin = planeData.margin_min_mm;
  const p05 = planeData.margin_p05_mm;
  if (margin == null) return null;

  const labelText = `${isFinite(margin) ? margin.toFixed(1) : '—'} mm` +
    (p05 != null ? `  ·  P5 ${p05.toFixed(1)}` : '');

  // 选择曲面上一个合适位置放标签: 取 u=0.5, v=0.5 处
  const ref = planeData.reference_plane;
  const origin = new THREE.Vector3().fromArray(ref.origin_mm);
  const normal = new THREE.Vector3().fromArray(ref.normal_world);
  const uAxis  = new THREE.Vector3().fromArray(ref.u_axis_world);
  const vAxis  = new THREE.Vector3().fromArray(ref.v_axis_world);
  const uRange = ref.u_range_mm;
  const vRange = ref.v_range_mm;
  const grid = planeData.height_control_4x4_mm;
  const centerOffset = planeData.center_offset;
  const bu = bernstein3(0.5);
  const bv = bernstein3(0.5);
  let height = 0.0;
  for (let ci = 0; ci < 4; ci++) {
    for (let cj = 0; cj < 4; cj++) {
      height += bu[ci] * bv[cj] * grid[ci][cj];
    }
  }
  const centerPos = new THREE.Vector3()
    .copy(origin)
    .addScaledVector(uAxis, (uRange[0] + uRange[1]) / 2)
    .addScaledVector(vAxis, (vRange[0] + vRange[1]) / 2)
    .addScaledVector(normal, height)
    .addScaledVector(normal, maxDim * 0.03);  // 沿法线方向偏移一点，避免遮挡
  if (centerOffset) {
    centerPos.sub(new THREE.Vector3().fromArray(centerOffset));
  }

  // Canvas 绘制文本
  const canvas = document.createElement('canvas');
  canvas.width = 600;
  canvas.height = 100;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 600, 100);

  // 背景 (more opaque).  Threshold reads the same configurable target the
  // server uses (margin_target_mm), so the label colour stays consistent
  // with the rest of the UI when tumor_margin_mm is overridden.
  const marginTarget = (planeData.margin_target_mm != null)
    ? planeData.margin_target_mm
    : 5.0;
  const isSafe = margin >= marginTarget;
  ctx.fillStyle = isSafe ? 'rgba(0,130,50,0.95)' : 'rgba(200,50,50,0.95)';
  ctx.beginPath();
  ctx.roundRect(10, 6, 580, 62, 12);
  ctx.fill();

  // Plain clinical status text keeps the label consistent across platforms.
  ctx.font = 'Bold 18px Arial';
  ctx.fillStyle = 'rgba(255,255,255,0.86)';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(isSafe ? 'SAFE' : 'RISK', 24, 27);

  ctx.font = 'Bold 22px Arial';
  ctx.textAlign = 'left';
  ctx.fillStyle = '#fff';
  ctx.fillText(labelText, 24, 51);

  ctx.font = '13px Arial';
  ctx.fillStyle = 'rgba(255,255,255,0.78)';
  ctx.fillText(isSafe ? 'Margin target met' : 'Below margin target', 220, 27);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const spriteMat = new THREE.SpriteMaterial({
    map: texture,
    depthTest: false,
    sizeAttenuation: true,
  });
  const sprite = new THREE.Sprite(spriteMat);
  sprite.scale.set(maxDim * 0.22, maxDim * 0.038, 1);
  sprite.position.copy(centerPos);

  return sprite;
}

// ----------------------------------------------------------------
//  buildBezierSurface — 主入口
// ----------------------------------------------------------------
export function buildBezierSurface(planeData, maxDim) {
  const geometry = buildBezierGeometry(planeData);
  const res = planeData.surface_resolution || [20, 20];
  const nU = res[0];
  const nV = res[1];
  const cps3d = planeData.control_points_3d;
  const hasControlPoints = Array.isArray(cps3d) && cps3d.length === 4
    && cps3d.every(row => Array.isArray(row) && row.length === 4
      && row.every(point => Array.isArray(point) && point.length === 3));

  // control_points_3d is the authoritative geometry after browser editing.
  // Keep the existing color attribute, but replace all rendered positions so
  // the surface, planning grid and sequence path share exactly one geometry.
  if (hasControlPoints) {
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(
      computeSurfacePositions(cps3d, nU, nV), 3));
    geometry.computeVertexNormals();
  }
  const material = new THREE.MeshPhysicalMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.55,
    side: THREE.DoubleSide,
    depthWrite: false,
    metalness: 0.02,
    roughness: 0.4,
    envMapIntensity: 0.2,
  });

  const mesh = new THREE.Mesh(geometry, material);
  // Increase render order to ensure plane renders on top of organs
  mesh.renderOrder = 5;
  mesh.userData = {
    name: 'plan_resection_plane',
    type: 'resection_plane',
  };

  // Draw visible grid lines on the surface (paper-style visualization)
  const gridHelper = hasControlPoints
    ? buildSurfaceGridFromCP(cps3d, nU, nV)
    : buildSurfaceGrid(planeData);
  gridHelper.renderOrder = 6;

  const line = hasControlPoints
    ? buildBoundaryLineFromCP(cps3d, nU, nV)
    : buildBoundaryLine(planeData);
  line.renderOrder = 6;

  const label = buildMarginLabel(planeData, maxDim || 300);

  return { mesh, line, gridHelper, label };
}

// ================================================================
//  Phase 2: Interactive Editing Functions
// ================================================================

// ----------------------------------------------------------------
//  computeSurfacePositions — 从 4x4 3D 控制点计算曲面顶点位置
//  使用通用 Bézier 公式: S(u,v) = ΣΣ C[i][j] * B_i(u) * B_j(v)
// ----------------------------------------------------------------
export function computeSurfacePositions(cps3d, nU, nV) {
  const positions = new Float32Array(nU * nV * 3);
  let idx = 0;
  for (let i = 0; i < nU; i++) {
    const uNorm = i / (nU - 1);
    const bu = bernstein3(uNorm);
    for (let j = 0; j < nV; j++) {
      const vNorm = j / (nV - 1);
      const bv = bernstein3(vNorm);
      let x = 0, y = 0, z = 0;
      for (let ci = 0; ci < 4; ci++) {
        for (let cj = 0; cj < 4; cj++) {
          const w = bu[ci] * bv[cj];
          const cp = cps3d[ci][cj];
          x += w * cp[0];
          y += w * cp[1];
          z += w * cp[2];
        }
      }
      positions[idx++] = x;
      positions[idx++] = y;
      positions[idx++] = z;
    }
  }
  return positions;
}

// ----------------------------------------------------------------
//  computeVertexDistances — 计算每个曲面顶点到最近肿瘤点的距离
// ----------------------------------------------------------------
export function computeVertexDistances(surfacePositions, tumorCloud) {
  const nVerts = surfacePositions.length / 3;
  const nTumor = tumorCloud.length / 3;
  const distances = new Float32Array(nVerts);

  if (nTumor === 0) {
    distances.fill(99.9);
    return distances;
  }

  for (let vi = 0; vi < nVerts; vi++) {
    const i3 = vi * 3;
    const vx = surfacePositions[i3];
    const vy = surfacePositions[i3 + 1];
    const vz = surfacePositions[i3 + 2];
    let minDistSq = Infinity;
    for (let ti = 0; ti < nTumor; ti++) {
      const t3 = ti * 3;
      const dx = vx - tumorCloud[t3];
      const dy = vy - tumorCloud[t3 + 1];
      const dz = vz - tumorCloud[t3 + 2];
      const dSq = dx * dx + dy * dy + dz * dz;
      if (dSq < minDistSq) minDistSq = dSq;
    }
    distances[vi] = Math.sqrt(minDistSq);
  }
  return distances;
}

// ----------------------------------------------------------------
//  distancesToColors — 将距离数组转换为顶点颜色数组
// ----------------------------------------------------------------
export function distancesToColors(distances) {
  const n = distances.length;
  const colors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const c = distanceToColor(distances[i]);
    colors[i * 3] = c[0];
    colors[i * 3 + 1] = c[1];
    colors[i * 3 + 2] = c[2];
  }
  return colors;
}

// ----------------------------------------------------------------
//  buildControlPoints — 构建 16 个控制点球体
//  内 4 个 (i=1,2 × j=1,2) 青色，外 12 个白色
// ----------------------------------------------------------------
export function buildControlPoints(planeData, maxDim) {
  const cps3d = planeData.control_points_3d;
  if (!cps3d) return new THREE.Group();

  const group = new THREE.Group();
  group.userData.name = 'control_points';

  const radius = maxDim * 0.008;
  const sphereGeo = new THREE.SphereGeometry(radius, 12, 12);

  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      const isInner = (i >= 1 && i <= 2 && j >= 1 && j <= 2);
      const mat = new THREE.MeshPhysicalMaterial({
        color: isInner ? 0x00ffff : 0xffffff,
        emissive: isInner ? 0x00ffff : 0x222222,
        emissiveIntensity: isInner ? 0.4 : 0.0,
        transparent: true,
        opacity: 0.9,
        depthWrite: false,
      });
      const sphere = new THREE.Mesh(sphereGeo.clone(), mat);
      const pos = cps3d[i][j];
      sphere.position.set(pos[0], pos[1], pos[2]);
      sphere.userData = { type: 'control_point', gridI: i, gridJ: j };
      group.add(sphere);
    }
  }
  return group;
}

// ----------------------------------------------------------------
//  updateSurfaceGeometry — 原地更新曲面几何体的顶点和颜色
// ----------------------------------------------------------------
export function updateSurfaceGeometry(geometry, positions, colors) {
  const posAttr = geometry.attributes.position;
  const colAttr = geometry.attributes.color;

  posAttr.array.set(positions);
  posAttr.needsUpdate = true;

  if (colors) {
    colAttr.array.set(colors);
    colAttr.needsUpdate = true;
  }

  geometry.computeVertexNormals();
}

// ----------------------------------------------------------------
//  buildBoundaryLineFromCP — 从 4x4 3D 控制点构建曲面边界线
// ----------------------------------------------------------------
export function buildBoundaryLineFromCP(cps3d, nU, nV) {
  function sample(uNorm, vNorm) {
    const bu = bernstein3(uNorm);
    const bv = bernstein3(vNorm);
    let x = 0, y = 0, z = 0;
    for (let ci = 0; ci < 4; ci++) {
      for (let cj = 0; cj < 4; cj++) {
        const w = bu[ci] * bv[cj];
        const cp = cps3d[ci][cj];
        x += w * cp[0];
        y += w * cp[1];
        z += w * cp[2];
      }
    }
    return new THREE.Vector3(x, y, z);
  }

  const points = [];
  // Bottom edge (vNorm=0)
  for (let i = 0; i < nU; i++) points.push(sample(i / (nU - 1), 0));
  // Right edge (uNorm=1)
  for (let j = 1; j < nV; j++) points.push(sample(1, j / (nV - 1)));
  // Top edge (vNorm=1, reversed)
  for (let i = nU - 2; i >= 0; i--) points.push(sample(i / (nU - 1), 1));
  // Left edge (uNorm=0, reversed)
  for (let j = nV - 2; j >= 1; j--) points.push(sample(0, j / (nV - 1)));

  const geo = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(
    geo,
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 })
  );
}

// Build a reference grid from the exact control points and resolution used by
// a sequence planner. This keeps the black reference grid aligned with the
// planner's cell partition, including user-edited surfaces.
export function buildSurfaceGridFromCP(cps3d, nU, nV) {
  const positions = computeSurfacePositions(cps3d, nU, nV);
  const linePositions = [];
  const index = (u, v) => (u * nV + v) * 3;
  const pushSegment = (a, b) => {
    linePositions.push(positions[a], positions[a + 1], positions[a + 2]);
    linePositions.push(positions[b], positions[b + 1], positions[b + 2]);
  };
  for (let i = 0; i < nU; i++) {
    for (let j = 0; j < nV - 1; j++) pushSegment(index(i, j), index(i, j + 1));
  }
  for (let j = 0; j < nV; j++) {
    for (let i = 0; i < nU - 1; i++) pushSegment(index(i, j), index(i + 1, j));
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
  return new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color: 0x222222, transparent: true, opacity: 0.7 })
  );
}
