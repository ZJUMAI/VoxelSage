"""
segmentation_modification: Skill 入口
=======================================

run(ctx) 被 SkillEngine 调用，返回编辑器 URL。
"""

import os
import uuid
from skills.builtin.segmentation_modification.session_manager import (
    editor_sessions,
    _SKIP_MASK_NAMES,
)


def _get_base_url() -> str:
    """获取服务基础 URL，优先用 PUBLIC_BASE_URL 环境变量。"""
    env_url = os.environ.get("PUBLIC_BASE_URL")
    if env_url:
        return env_url.rstrip("/")
    return "http://127.0.0.1:8898"


def run(ctx):
    """创建编辑器会话并返回 Web UI URL。

    参数（通过 ctx.params 传入）:
        mask_name: 要编辑的掩码名称（可选，默认使用第一个掩码）

    返回:
        dict: {editor_url, session_id, mask_name, num_slices, image_size}
    """
    masks = ctx.list_masks()
    if not masks:
        # 如果 list_masks() 为空，可能是因为只有多标签汇总文件
        #（如 all.nii.gz，已被 models.py _scan_mask_dir 排除）。
        # 检查是否有可拆分的大文件。
        import glob as _glob
        _multi_label_candidates = [_f for _f in sorted(
            _glob.glob(os.path.join(ctx.mask_dir, "*.nii.gz"))
        )]
        if _multi_label_candidates:
            # 传入候选文件名给 session，由 _split_multilabel_masks 拆分
            _raw_stems = []
            for _f in _multi_label_candidates:
                _stem = os.path.basename(_f)
                if _stem.endswith(".nii.gz"):
                    _stem = _stem[:-7]
                elif _stem.endswith(".nii"):
                    _stem = _stem[:-4]
                _raw_stems.append(_stem)
            masks = _raw_stems
        else:
            return {"status": "error", "message": "未找到可编辑的掩码文件"}

    # 排除 VISTA3D 的多标签汇总文件（如 all.nii.gz），它不是独立可编辑掩码
    editable_masks = [m for m in masks if m not in _SKIP_MASK_NAMES]

    # 如果过滤后没有掩码，则保留原列表（可能只有 all.nii.gz，
    # 由 session 内部的 _split_multilabel_masks 拆分）
    if editable_masks:
        masks = editable_masks

    # 默认选中 "liver"（如果存在），否则选第一个
    default_mask = "liver" if "liver" in masks else masks[0]
    mask_name = ctx.params.get("mask_name", default_mask)
    if mask_name not in masks:
        return {
            "status": "error",
            "message": f"掩码 '{mask_name}' 不存在。可用: {', '.join(masks)}",
        }

    # 生成唯一会话 ID
    session_id = f"{ctx.case_id}_{mask_name}_{uuid.uuid4().hex[:8]}"

    # 获取 CT 信息
    ct_array = ctx.get_ct_array()  # (H, W, D)
    affine = ctx.get_affine()

    # 创建会话（传入所有可用 mask，支持前端切换）
    editor_sessions.create(
        session_id=session_id,
        case_id=ctx.case_id,
        ct_nifti_path=ctx.ct_nifti_path,
        mask_dir=ctx.mask_dir,
        output_dir=ctx.get_output_dir(),
        mask_name=mask_name,
        mask_names=masks,
        ct_shape=ct_array.shape,
        affine=affine,
    )

    # 预热缓存：提前加载 CT 窗宽窗位数据和 mask
    # 这样用户首次打开编辑器时无需等待 ~10s 的 NIfTI 加载
    session = editor_sessions.get(session_id)
    if session is not None:
        session.get_ct_uint8()
        session.get_current_mask()
        # 回读 session 中实际生效的 mask 名称（可能已被拆分 label_N）
        actual_mask_name = session.current_mask_name
        actual_mask_names = session.get_mask_names()
        ctx.log(f"Preloaded CT (windowed) + mask '{actual_mask_name}' for editor session")
    else:
        actual_mask_name = mask_name
        actual_mask_names = masks

    ctx.log(f"Created editor session {session_id} for mask '{mask_name}'")

    base = _get_base_url()
    return {
        "editor_url": f"{base}/segmentation-editor/{session_id}",
        "session_id": session_id,
        "mask_name": actual_mask_name,
        "mask_names": actual_mask_names,
        "num_slices": int(ct_array.shape[2]),
        "image_size": f"{ct_array.shape[0]}x{ct_array.shape[1]}",
    }
