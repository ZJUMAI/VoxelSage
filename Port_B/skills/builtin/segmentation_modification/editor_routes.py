"""
segmentation_modification: FastAPI 路由
========================================

注册分割编辑器的所有 HTTP 端点。
导出 register_routes(app) 供 API.py 调用。
"""

import asyncio
import base64
import io
import os
import cv2
import numpy as np
from fastapi import HTTPException
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
from typing import Optional

from skills.builtin.segmentation_modification.session_manager import editor_sessions
from skills.builtin.segmentation_modification.html_template import make_editor_html

# ============================================================
# Pydantic 请求模型
# ============================================================

class InitRequest(BaseModel):
    session_id: str

class ClickRequest(BaseModel):
    session_id: str
    slice_idx: int
    x: int
    y: int
    label: int  # 1=foreground(positive), 0=background(negative)

class RefineRequest(BaseModel):
    session_id: str
    slice_idx: int

class SaveRequest(BaseModel):
    session_id: str

class ClearClicksRequest(BaseModel):
    session_id: str
    slice_idx: int

class UndoLastClickRequest(BaseModel):
    session_id: str
    slice_idx: int

class SwitchMaskRequest(BaseModel):
    session_id: str
    mask_name: str

class Propagate3DRequest(BaseModel):
    session_id: str
    slice_idx: int  # 用户当前正在 refine 的切片，作为传播锚点

# ============================================================
# 切片渲染（CT + mask overlay + click 标记）
# ============================================================

# 器官调色板（用于不同 mask 的 overlay 颜色）
_OVERLAY_COLORS = {
    "liver": (0, 200, 0),
    "liver_remnant": (0, 200, 0),
    "hepatic_tumor": (255, 0, 0),
    "tumor": (255, 0, 0),
    "hepatic": (0, 150, 255),
    "hepatic_vessel": (0, 150, 255),
    "portal": (255, 150, 0),
    "portal_vein": (255, 150, 0),
    "spleen": (200, 0, 200),
    "kidney": (200, 200, 0),
    "pancreas": (0, 200, 200),
}
_DEFAULT_OVERLAY_COLOR = (0, 255, 0)  # 默认绿色

# VISTA3D label → 器官名称映射（用于 _split_multilabel_masks 拆分后的 label_N 着色）
_VISTA3D_LABEL_MAP = {
    1: "liver",
    3: "spleen",
    4: "pancreas",
    17: "portal_vein",
    25: "hepatic_vessel",
    26: "hepatic_tumor",
}
# 通用调色板（用于未知 label 值）
_LABEL_PALETTE = [
    (0, 200, 0),     # 0: green
    (255, 0, 0),     # 1: red
    (0, 150, 255),   # 2: blue
    (255, 150, 0),   # 3: orange
    (200, 0, 200),   # 4: purple
    (200, 200, 0),   # 5: yellow
    (0, 200, 200),   # 6: cyan
    (255, 100, 100), # 7: pink
    (100, 255, 100), # 8: lime
    (100, 100, 255), # 9: cornflower
]


def _get_color_for_mask(mask_name: str):
    """根据 mask 名称获取 overlay 颜色。"""
    # 精确匹配
    if mask_name in _OVERLAY_COLORS:
        return _OVERLAY_COLORS[mask_name]
    # 前缀匹配（如 tumor_1 → tumor）
    for key, color in _OVERLAY_COLORS.items():
        if mask_name.startswith(key):
            return color
    # label_N 模式 → 查 VISTA3D label map
    if mask_name.startswith("label_"):
        try:
            label_val = int(mask_name[len("label_"):])
            organ_name = _VISTA3D_LABEL_MAP.get(label_val)
            if organ_name and organ_name in _OVERLAY_COLORS:
                return _OVERLAY_COLORS[organ_name]
            # 未映射的 label 用调色板
            return _LABEL_PALETTE[label_val % len(_LABEL_PALETTE)]
        except ValueError:
            pass
    return _DEFAULT_OVERLAY_COLOR


def _get_color_for_label(label_value: int, default_color: tuple = None) -> tuple:
    """根据 label 数值获取对应渲染颜色（多标签 mask 用）。"""
    organ_name = _VISTA3D_LABEL_MAP.get(label_value)
    if organ_name and organ_name in _OVERLAY_COLORS:
        return _OVERLAY_COLORS[organ_name]
    # 用调色板
    palette = _LABEL_PALETTE + [_DEFAULT_OVERLAY_COLOR]
    return palette[label_value % len(palette)]


def _render_slice_png(
    ct_uint8_slice: np.ndarray,
    mask_2d: Optional[np.ndarray],
    click_points: list,
    overlay_color: tuple,
) -> bytes:
    """将 CT 切片 + 可选 mask overlay + click 标记渲染为 PNG bytes。

    支持多标签 mask：当 mask 中有多个不同的 label 值时，
    每个 label 自动使用独立的颜色渲染。

    Args:
        ct_uint8_slice: (H, W) uint8 CT 切片
        mask_2d: (H, W) int/uint8 mask，或 None
        click_points: [(x, y, label), ...]
        overlay_color: mask overlay 的 BGR 颜色（单标签时用）

    Returns:
        PNG bytes
    """
    # 灰度 → BGR
    img = cv2.cvtColor(ct_uint8_slice, cv2.COLOR_GRAY2BGR)

    # Mask overlay（半透明）
    if mask_2d is not None and mask_2d.any():
        overlay = img.copy()

        # 检查是否为多标签 mask（>2 个唯一值，含 0）
        unique_vals = np.unique(mask_2d)
        non_zero_vals = unique_vals[unique_vals > 0]

        # 限制最大 label 数，防止 CT 体数据被误当 mask（含 970+ 个"label"）
        MAX_LABELS = 50
        if len(non_zero_vals) > MAX_LABELS:
            # 极多 unique 值 → 不是 label mask（可能是 CT 误存），回退到二值化单色显示
            mask_bool = mask_2d > 0
            color = np.array(overlay_color, dtype=np.uint8)
            overlay[mask_bool] = (overlay[mask_bool] * 0.5 + color * 0.5).astype(np.uint8)
        elif len(non_zero_vals) > 1:
            # —— 多标签渲染：每个 label 独立颜色 ——
            for label_val in non_zero_vals:
                label_mask = mask_2d == label_val
                color = _get_color_for_label(int(label_val), overlay_color)
                color_arr = np.array(color, dtype=np.uint8)
                overlay[label_mask] = (overlay[label_mask] * 0.4 + color_arr * 0.6).astype(np.uint8)
        else:
            # —— 单标签渲染（原有逻辑）——
            mask_bool = mask_2d > 0
            color = np.array(overlay_color, dtype=np.uint8)
            overlay[mask_bool] = (overlay[mask_bool] * 0.5 + color * 0.5).astype(np.uint8)

        img = overlay

    # Click 标记
    for (x, y, label) in click_points:
        if label == 1:  # 前景 — 绿色圆点
            color_click = (0, 255, 0)
            label_text = "P"
        else:  # 背景 — 红色圆点
            color_click = (255, 0, 0)
            label_text = "N"
        # 外圈（白色描边）
        cv2.circle(img, (int(x), int(y)), 8, (255, 255, 255), 2)
        # 内圈（正/负颜色）
        cv2.circle(img, (int(x), int(y)), 6, color_click, -1)
        # 小标签
        cv2.putText(img, label_text, (int(x) - 4, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 编码为 PNG
    success, png_bytes = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise RuntimeError("Failed to encode slice PNG")
    return png_bytes.tobytes()


# ============================================================
# Mock Refine（安装 MedSAM2 前使用的占位实现）
# ============================================================

def _mock_refine(session, slice_idx: int, points: np.ndarray,
                 labels: np.ndarray, overlay_color: tuple) -> np.ndarray:
    """Mock refine：在已有 mask 上叠加/减去点击区域的膨胀。

    安装 MedSAM2 后替换为真实 inference。
    此函数仅用于前端开发和测试。
    """
    # 从已有 mask 开始（而不是空白），仅修改点击区域
    existing_mask = session.get_current_mask()[:, :, slice_idx]
    result = existing_mask.copy().astype(np.uint8)

    if len(points) == 0:
        return result

    # 正点击（label=1）: 在已有 mask 上增加区域
    has_positive = False
    for pt, lb in zip(points, labels):
        if lb == 1:
            cv2.circle(result, (int(pt[0]), int(pt[1])), 30, 1, -1)
            has_positive = True

    # 负点击（label=0）: 在已有 mask 上减去区域
    for pt, lb in zip(points, labels):
        if lb == 0:
            cv2.circle(result, (int(pt[0]), int(pt[1])), 20, 0, -1)

    # 形态学闭合填补空洞 + 开运算去除孤立点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return result


# ============================================================
# 路由注册
# ============================================================

def register_routes(app):
    """注册所有分割编辑器路由到 FastAPI app。"""

    # ── 编辑器页面 ──

    @app.get("/segmentation-editor/{session_id}", response_class=HTMLResponse)
    def get_editor_page(session_id: str):
        """返回分割编辑器 HTML 页面。"""
        session = editor_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        h, w = session._ct_shape[0], session._ct_shape[1]
        mask_names = session.get_mask_names()
        html = make_editor_html(
            session_id=session_id,
            num_slices=session._ct_shape[2],
            image_width=w,
            image_height=h,
            mask_name=session.current_mask_name,
            mask_names=mask_names,
        )
        return HTMLResponse(content=html)

    # ── 初始化 ──

    @app.post("/api/segmentation-editor/init")
    def init_editor(req: InitRequest):
        """初始化会话：预加载 CT 和 mask 数据。"""
        session = editor_sessions.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        # 触发惰性加载
        session.get_ct_array()
        session.get_current_mask()

        return {
            "status": "ok",
            "num_slices": int(session._ct_shape[2]),
            "image_size": [int(session._ct_shape[0]), int(session._ct_shape[1])],
            "mask_name": session.current_mask_name,
            "mask_names": session.get_mask_names(),
            "click_count": len(session.click_history),
            "clicks": [
                {"slice_idx": s, "x": x, "y": y, "label": label}
                for (s, x, y, label) in session.get_clicks_for_current_mask()
            ],
        }

    # ── 获取切片 PNG ──

    @app.get("/api/segmentation-editor/slice/{session_id}/{slice_idx}")
    def get_slice(session_id: str, slice_idx: int,
                  overlay: bool = True, clicks: bool = True):
        """返回单张 CT 切片的 PNG。

        overlay=true   — 叠加 mask 半透明层（支持多标签 masks 不同颜色）
        clicks=true    — 显示 click 标记
        """
        session = editor_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        slice_idx = int(slice_idx)
        if slice_idx < 0 or slice_idx >= session._ct_shape[2]:
            raise HTTPException(
                status_code=400,
                detail=f"Slice index out of range (0-{session._ct_shape[2] - 1})"
            )

        ct_uint8 = session.get_ct_uint8()
        ct_slice = ct_uint8[:, :, slice_idx].copy()

        # Mask overlay — 使用 get_raw_mask 保留多标签值
        mask_2d = None
        if overlay:
            try:
                raw_mask = session.get_raw_mask(session.current_mask_name)
                mask_2d = raw_mask[:, :, slice_idx]
            except Exception:
                # fallback 到二值化 mask
                fallback = session.get_current_mask()
                mask_2d = fallback[:, :, slice_idx]

        # Click markers
        click_points = []
        if clicks:
            click_points = session.get_clicks_for_slice(slice_idx)

        color = _get_color_for_mask(session.current_mask_name)

        png_bytes = _render_slice_png(ct_slice, mask_2d, click_points, color)

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    # ── 添加点击 ──

    @app.post("/api/segmentation-editor/click")
    def add_click(req: ClickRequest):
        """记录一个点击点。"""
        session = editor_sessions.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        slice_idx = int(req.slice_idx)
        if slice_idx < 0 or slice_idx >= session._ct_shape[2]:
            raise HTTPException(status_code=400, detail="Slice index out of range")

        label = int(req.label)
        if label not in (0, 1):
            raise HTTPException(status_code=400, detail="Label must be 0 (bg) or 1 (fg)")

        session.add_click(slice_idx, int(req.x), int(req.y), label)

        return {
            "status": "ok",
            "click_count": len(session.click_history),
            "slice_clicks": len(session.get_clicks_for_slice(slice_idx)),
        }

    # ── Refine ──

    _REFINE_TIMEOUT = 120  # 秒 — 超过此时间自动返回超时错误

    @app.post("/api/segmentation-editor/refine")
    async def refine_mask(req: RefineRequest):
        """使用 MedSAM2 优化当前切片的掩码。

        MVP 阶段使用 mock refine（形态学膨胀），
        安装 MedSAM2 后自动切换到真实推理。
        """
        session = editor_sessions.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        slice_idx = int(req.slice_idx)
        if slice_idx < 0 or slice_idx >= session._ct_shape[2]:
            raise HTTPException(status_code=400, detail="Slice index out of range")

        # 获取该切片的 click
        clicks = session.get_clicks_for_slice(slice_idx)
        if not clicks:
            raise HTTPException(
                status_code=400,
                detail="No clicks on this slice. Add positive/negative clicks first."
            )

        points = np.array([[x, y] for (x, y, _) in clicks], dtype=np.float32)
        labels = np.array([label for (_, _, label) in clicks], dtype=np.int32)

        color = _get_color_for_mask(session.current_mask_name)

        # 将耗时的 refine 逻辑放到线程池执行，避免阻塞事件循环
        def _do_refine():
            refined_mask_2d = None
            medsam2_available = False

            try:
                from skills.builtin.segmentation_modification.medsam2_wrapper import (
                    MedSAM2Model, init_session_inference, refine_with_clicks
                )

                MedSAM2Model.get_instance()

                if session.medsam2_inference_state is None:
                    ct_volume = session.get_ct_array()
                    mask_volume = session.get_current_mask()
                    h, w, d = ct_volume.shape
                    inference_state, obj_id, orig_shape = init_session_inference(
                        ct_volume, mask_volume, video_height=h, video_width=w
                    )
                    session.medsam2_inference_state = inference_state
                    session.medsam2_obj_id = obj_id
                    session._medsam_orig_shape = orig_shape

                orig_shape = getattr(session, '_medsam_orig_shape', None)
                refined_mask_2d = refine_with_clicks(
                    inference_state=session.medsam2_inference_state,
                    obj_id=session.medsam2_obj_id,
                    frame_idx=slice_idx,
                    points=points,
                    labels=labels,
                    orig_shape=orig_shape,
                )
                medsam2_available = True

            except FileNotFoundError:
                pass
            except ImportError:
                pass
            except Exception as e:
                session.log(f"MedSAM2 refine failed, using mock: {e}")

            if not medsam2_available:
                refined_mask_2d = _mock_refine(session, slice_idx, points, labels, color)
                final_mask = refined_mask_2d
            else:
                existing_slice = session.get_current_mask()[:, :, slice_idx]
                merged = existing_slice.copy().astype(np.uint8)
                merged[refined_mask_2d > 0] = 1
                for pt, lb in zip(points, labels):
                    if lb == 0:
                        cv2.circle(merged, (int(pt[0]), int(pt[1])), 25, 0, -1)
                final_mask = merged

            session.update_mask_slice(slice_idx, final_mask)
            session.clear_clicks_for_slice(slice_idx)

            ct_slice = session.get_ct_uint8()[:, :, slice_idx]
            updated_png = _render_slice_png(ct_slice, final_mask, [], color)
            png_b64 = base64.b64encode(updated_png).decode('utf-8')

            return {
                "status": "ok",
                "medsam2": medsam2_available,
                "slice_idx": slice_idx,
                "overlay_png": png_b64,
                "mask_pixels": int(final_mask.sum()),
                "clicks_cleared": True,
            }

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_do_refine),
                timeout=_REFINE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Refine timed out after {_REFINE_TIMEOUT}s. Try a simpler region.",
            )

    # ── 3D Propagation（MedSAM2 视频传播）──

    @app.post("/api/segmentation-editor/propagate-3d")
    def propagate_3d(req: Propagate3DRequest):
        """将当前切片的 refine 变化传播到相邻切片，实现 3D 平滑。

        使用 MedSAM2 的 propagate_in_video，以 refine 过的切片为锚点，
        让模型填充其他切片，使相邻层之间过渡平滑。
        """
        session = editor_sessions.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        slice_idx = int(req.slice_idx)
        if slice_idx < 0 or slice_idx >= session._ct_shape[2]:
            raise HTTPException(status_code=400, detail="Slice index out of range")

        # 确保当前 mask 已加载（含 refine 修改）
        session.get_current_mask()
        session.log("Starting 3D propagation...")

        try:
            from skills.builtin.segmentation_modification.medsam2_wrapper import (
                MedSAM2Model, propagate_3d as run_propagate
            )

            # 确保 MedSAM2 已加载
            MedSAM2Model.get_instance()

            ct_volume = session.get_ct_array()
            mask_volume = session.get_current_mask()
            h, w, d = ct_volume.shape

            # 判断哪些切片被 refine 过（与原始 mask 不同）
            # 用 refined_indices 记录所有有变化的切片
            orig_raw = getattr(session, '_original_masks', {}).get(session.current_mask_name)
            if orig_raw is not None:
                refined_indices = [
                    z for z in range(d)
                    if not np.array_equal(mask_volume[:, :, z], orig_raw[:, :, z])
                ]
            else:
                refined_indices = [slice_idx]

            if not refined_indices:
                refined_indices = [slice_idx]

            session.log(f"3D propagation from {len(refined_indices)} refined anchor(s): {refined_indices[:5]}...")

            # 运行 MedSAM2 3D propagation
            propagated = run_propagate(
                ct_volume=ct_volume,
                mask_volume=mask_volume,
                refined_indices=refined_indices,
                video_height=h,
                video_width=w,
                num_anchor_slices=5,
            )

            # 更新 session 中的 mask
            changed_count = 0
            for frame_idx, new_mask_2d in propagated.items():
                session.update_mask_slice(frame_idx, new_mask_2d)
                changed_count += 1

            session.log(f"3D propagation updated {changed_count} slices")

            return {
                "status": "ok",
                "propagated_slices": changed_count,
                "total_slices": d,
                "anchors": refined_indices[:10],
            }

        except (FileNotFoundError, ImportError) as e:
            raise HTTPException(
                status_code=503,
                detail=f"MedSAM2 not available: {e}. "
                       f"Please download the checkpoint first."
            )
        except Exception as e:
            session.log(f"3D propagation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"3D propagation failed: {e}",
            )

    # ── 保存 ──

    @app.post("/api/segmentation-editor/save")
    def save_mask_endpoint(req: SaveRequest):
        """将当前 mask 写回 NIfTI。"""
        session = editor_sessions.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        mask = session.get_current_mask()
        affine = session._affine
        current_name = session.current_mask_name
        mask_path = os.path.join(session.mask_dir, f"{current_name}.nii.gz")

        # 创建 .bak 备份（使用当前 mask 名称查找原始文件）
        pattern = os.path.join(session.mask_dir, f"{current_name}*")
        orig_files = sorted(__import__('glob').glob(pattern))
        if orig_files and os.path.exists(orig_files[0]):
            import shutil
            backup_path = orig_files[0] + ".bak"
            shutil.copy2(orig_files[0], backup_path)

        # 保存
        import nibabel as nib
        nii = nib.Nifti1Image(mask.astype(np.float32), affine)
        nib.save(nii, mask_path)

        return {
            "status": "ok",
            "mask_path": mask_path,
            "backup_path": backup_path if orig_files and os.path.exists(orig_files[0]) else None,
            "dtype": "float32",
        }

    # ── 清除点击 ──

    @app.post("/api/segmentation-editor/undo-last-click")
    def undo_last_click(req: UndoLastClickRequest):
        """撤销当前 mask、当前切片最后添加的一个点击。"""
        session = editor_sessions.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        slice_idx = int(req.slice_idx)
        if slice_idx < 0 or slice_idx >= session._ct_shape[2]:
            raise HTTPException(status_code=400, detail="Slice index out of range")

        removed = session.pop_last_click(slice_idx=slice_idx)
        return {
            "status": "ok",
            "removed": None if removed is None else {
                "slice_idx": removed[0],
                "x": removed[1],
                "y": removed[2],
                "label": removed[3],
                "mask_name": removed[4],
            },
            "click_count": len(session.click_history),
            "slice_clicks": len(session.get_clicks_for_slice(slice_idx)),
        }

    @app.post("/api/segmentation-editor/clear-clicks")
    def clear_clicks(req: ClearClicksRequest):
        """清除指定切片上的所有点击。"""
        session = editor_sessions.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        slice_idx = int(req.slice_idx)
        if slice_idx < 0 or slice_idx >= session._ct_shape[2]:
            raise HTTPException(status_code=400, detail="Slice index out of range")

        session.clear_clicks_for_slice(slice_idx)

        return {
            "status": "ok",
            "click_count": len(session.click_history),
        }

    # ── 切换 Mask ──

    @app.post("/api/segmentation-editor/switch-mask")
    def switch_mask(req: SwitchMaskRequest):
        """切换到指定 mask，不会丢失当前 mask 的编辑。"""
        session = editor_sessions.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        if req.mask_name not in session.get_mask_names():
            raise HTTPException(
                status_code=400,
                detail=f"Mask '{req.mask_name}' not available. Available: {session.get_mask_names()}"
            )

        session.switch_mask(req.mask_name)

        # 各 mask 的点击在后端分别保留，并把目标 mask 的点击返回给前端。

        return {
            "status": "ok",
            "mask_name": session.current_mask_name,
            "mask_names": session.get_mask_names(),
            "click_count": len(session.click_history),
            "clicks": [
                {"slice_idx": s, "x": x, "y": y, "label": label}
                for (s, x, y, label) in session.get_clicks_for_current_mask()
            ],
        }

    # ── 获取当前 mask 列表（无需切换时查询）──

    @app.get("/api/segmentation-editor/masks/{session_id}")
    def get_masks(session_id: str):
        """返回会话中所有可用 mask 及其当前状态。"""
        session = editor_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        return {
            "status": "ok",
            "current_mask": session.current_mask_name,
            "mask_names": session.get_mask_names(),
        }
