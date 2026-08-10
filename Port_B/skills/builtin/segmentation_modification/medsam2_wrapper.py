"""
segmentation_modification: MedSAM2 封装层
==========================================

提供 MedSAM2 模型的单例加载、推理状态初始化和单切片 click refinement。

安装要求（用户手动操作）:
    cd MedSAM2-main && pip install -e ".[dev]" && bash download.sh

缺少 checkpoint 或包未安装时，所有函数抛出 FileNotFoundError，
由调用方（editor_routes.py）捕获并回退到 mock refine。
"""

import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# MedSAM2 路径配置
_MEDSAM2_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "MedSAM2-main"
)
_CHECKPOINT_DIR = os.path.join(_MEDSAM2_ROOT, "checkpoints")
_CFG_DIR = os.path.join(_MEDSAM2_ROOT, "sam2", "configs")

# 默认检查点名称（CT 病灶专用）
_DEFAULT_CHECKPOINT = "MedSAM2_CTLesion.pt"
_DEFAULT_CFG = "sam2.1_hiera_t512.yaml"


class MedSAM2Model:
    """MedSAM2 模型单例。

    首次调用 get_instance() 时加载模型，
    之后复用同一个 predictor 实例。
    """

    _instance = None
    _predictor = None
    _device = None
    _checkpoint_path = None
    _cfg_path = None

    def __init__(self):
        if MedSAM2Model._predictor is not None:
            return
        self._load_model()

    @classmethod
    def get_instance(cls):
        """获取 MedSAM2 单例。缺失 checkpoint 时抛出 FileNotFoundError。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_checkpoint(cls, checkpoint_name: str, cfg_name: str = _DEFAULT_CFG):
        """设置检查点文件名（在首次 get_instance() 前调用）。"""
        cls._checkpoint_path = os.path.join(_CHECKPOINT_DIR, checkpoint_name)
        cls._cfg_path = os.path.join(_CFG_DIR, cfg_name)
        if not os.path.exists(cls._checkpoint_path):
            raise FileNotFoundError(
                f"MedSAM2 checkpoint not found: {cls._checkpoint_path}\n"
                f"Download with: cd {_MEDSAM2_ROOT} && bash download.sh"
            )

    def _load_model(self):
        """加载 MedSAM2 模型（仅首次调用执行）。"""
        # 确定 checkpoint 路径（支持用户通过 set_checkpoint 自定义）
        ckpt = (
            MedSAM2Model._checkpoint_path
            or os.path.join(_CHECKPOINT_DIR, _DEFAULT_CHECKPOINT)
        )
        cfg_name = _DEFAULT_CFG  # 配置文件名（Hydra 在 config_dir 中查找）

        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"MedSAM2 checkpoint not found at: {ckpt}\n"
                f"Please download first:\n"
                f"  cd {_MEDSAM2_ROOT} && bash download.sh\n"
                f"Or create the file manually."
            )

        # 添加 MedSAM2 到 sys.path
        import sys
        if _MEDSAM2_ROOT not in sys.path:
            sys.path.insert(0, _MEDSAM2_ROOT)

        import torch
        from sam2.build_sam import build_sam2_video_predictor_npz

        # 使用 GPUManager 选剩余显存最大的 GPU，避免与 VISTA3D 冲突
        try:
            from Tool_Box.gpu_manager import query_gpu_info as _query_gpu
            _gpus = _query_gpu()
            if _gpus:
                _best = max(_gpus, key=lambda g: g["memory_free_mb"])
                device = f"cuda:{_best['index']}"
            elif torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        except Exception:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"[MedSAM2] Loading checkpoint {ckpt} on {device}..."
        )

        # Hydra compose() 必须在 initialize_config_dir() 上下文内调用
        # 注意：from sam2.build_sam 这个 import 会初始化 Hydra，
        # 所以 clear() 必须放在 import 之后、initialize_config_dir 之前
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        try:
            GlobalHydra.instance().clear()
        except ValueError:
            pass  # 首次调用时还没有 Hydra 实例

        cfg_dir = str(_CFG_DIR)
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
            predictor = build_sam2_video_predictor_npz(
                config_file=cfg_name,
                ckpt_path=ckpt,
                device=device,
            )
        predictor.eval()

        MedSAM2Model._predictor = predictor
        MedSAM2Model._device = device
        MedSAM2Model._checkpoint_path = ckpt
        logger.info(
            f"[MedSAM2] Model loaded successfully on {device} "
            f"({os.path.getsize(ckpt) // 1024 // 1024} MB)"
        )

    @property
    def predictor(self):
        return MedSAM2Model._predictor

    @property
    def device(self):
        return MedSAM2Model._device


# ============================================================
# Inference state 初始化
# ============================================================

def init_session_inference(
    ct_volume: "np.ndarray",
    mask_volume: "np.ndarray",
    video_height: int,
    video_width: int,
):
    """创建 MedSAM2 video predictor inference state，注入已有 mask。

    MedSAM2 内部使用 512x512 分辨率，输入图像会自动 resize。
    输出的 mask 也是 512x512，调用方需要 resize 回原始尺寸。

    Args:
        ct_volume: (H, W, D) float32 CT 体素值（原始 HU）
        mask_volume: (H, W, D) uint8 二值 mask（0/1）
        video_height: H（原始尺寸）
        video_width: W（原始尺寸）

    Returns:
        (inference_state, obj_id, orig_shape):
        推理状态、对象 ID、原始 (H, W) — 用于后续 output resize
    """
    import numpy as np
    import torch
    from Tool_Box.io import apply_ct_window

    model = MedSAM2Model.get_instance()
    predictor = model.predictor
    device = model.device

    # MedSAM2 模型内部使用 512x512
    model_size = predictor.image_size  # 通常是 512

    # CT 窗宽窗位 → uint8 (H, W, D)
    windowed = apply_ct_window(ct_volume, wl=40.0, ww=400.0)
    ct_uint8 = (windowed * 255.0).astype(np.uint8)
    num_frames = ct_volume.shape[2]

    # Resize 到模型输入尺寸 (512x512)
    # 注意：resize 同时做了 (H, W, D) → (D, H, W) 的转置
    import cv2
    if video_height == model_size and video_width == model_size:
        # 尺寸匹配，只需转置 (H, W, D) → (D, H, W)
        ct_uint8 = np.transpose(ct_uint8, (2, 0, 1))  # (D, H, W)
    else:
        # resize 并转置
        ct_resized = np.zeros(
            (num_frames, model_size, model_size), dtype=np.uint8
        )
        for z in range(num_frames):
            ct_resized[z] = cv2.resize(
                ct_uint8[:, :, z], (model_size, model_size),
                interpolation=cv2.INTER_LINEAR
            )
        ct_uint8 = ct_resized  # (D, H, W)

    # MedSAM2 输入格式: (D, 3, H, W) torch tensor, 归一化到 [0,1]
    # (D, H, W) → (D, 1, H, W) → (D, 3, H, W)
    ct_uint8 = ct_uint8[:, None, :, :]              # (D, 1, H, W)
    ct_uint8 = np.repeat(ct_uint8, 3, axis=1)       # (D, 3, H, W)
    ct_uint8 = ct_uint8.astype(np.float32) / 255.0   # [0, 1]

    # 转为 torch tensor 并送到 GPU
    images_tensor = torch.from_numpy(ct_uint8).to(device)

    # ImageNet 归一化（MedSAM2 训练使用的预处理）
    img_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images_tensor = (images_tensor - img_mean) / img_std

    logger.info(
        f"[MedSAM2] Initializing inference state for {num_frames} frames "
        f"({images_tensor.shape}) on {device}..."
    )

    # 初始化 state（使用模型内部尺寸 512x512）
    inference_state = predictor.init_state(
        images=images_tensor,
        video_height=model_size,
        video_width=model_size,
    )

    # 注入已有 mask（add_new_mask 内部会 resize 到模型尺寸）
    obj_id = 1
    injected = 0
    for z in range(num_frames):
        mask_2d = mask_volume[:, :, z]
        if mask_2d.sum() > 0:
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=z,
                obj_id=obj_id,
                mask=mask_2d,
            )
            injected += 1

    logger.info(f"[MedSAM2] Injected mask for {injected}/{num_frames} frames")
    # 返回 orig_shape 供调用方 resize output mask
    return inference_state, obj_id, (video_height, video_width)


# ============================================================
# Refinement
# ============================================================

def refine_with_clicks(
    inference_state,
    obj_id: int,
    frame_idx: int,
    points: "np.ndarray",
    labels: "np.ndarray",
    orig_shape: "Optional[Tuple[int, int]]" = None,
) -> "np.ndarray":
    """在指定帧上运行点击 refinement。

    Args:
        inference_state: MedSAM2 inference state
        obj_id: 对象 ID（通常为 1）
        frame_idx: 切片索引
        points: (N, 2) float32 数组，(x, y) 像素坐标
        labels: (N,) int32 数组，1=前景, 0=背景
        orig_shape: 原始 (H, W)，若提供则 output resize 回此尺寸

    Returns:
        refined_mask: (H, W) uint8 二值 mask
    """
    import numpy as np

    model = MedSAM2Model.get_instance()
    predictor = model.predictor

    # 添加点击并获取结果
    out_frame_idx, out_obj_ids, out_masks = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
        clear_old_points=True,
    )

    # 提取目标对象的 mask
    obj_index = out_obj_ids.index(obj_id)
    refined_mask = out_masks[obj_index, 0].cpu().numpy()

    # logits → 二值（threshold = 0）
    refined_binary = (refined_mask > 0).astype(np.uint8)

    # 如果提供了原始尺寸且与当前尺寸不同，resize 回去
    if orig_shape is not None:
        orig_h, orig_w = orig_shape
        if refined_binary.shape[0] != orig_h or refined_binary.shape[1] != orig_w:
            import cv2
            refined_binary = cv2.resize(
                refined_binary, (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

    return refined_binary


# ============================================================
# 3D Propagation
# ============================================================

def propagate_3d(
    ct_volume: "np.ndarray",
    mask_volume: "np.ndarray",
    refined_indices: "list[int]",
    video_height: int,
    video_width: int,
    num_anchor_slices: int = 5,
) -> dict:
    """在 3D 体积上运行 MedSAM2 propagation（双向传播）。

    对 refine 后的切片运行 propagation，使相邻切片的 mask 平滑连续。
    策略：以 refine 过的切片为锚点（anchor），让模型传播到其他切片。

    改进说明：
    - 只注入 refine 过的切片作为锚点，不再添加密集邻域切片
      （旧 bug：邻域作为 anchor 被 add_new_mask 注册为 ground truth，
       propagate_in_video 不会修改它们，导致相邻切片 mask 不变）
    - 双向传播：forward（从第一个 refine 切片→末尾）+ backward（从最后一个→开头）
    - 合并两方向结果：frames 优先使用离它最近的传播方向的结果
    - 对于传播未覆盖的切片（如 obj_id 未被检测到），复制最近 refine 切片的 mask

    Args:
        ct_volume: (H, W, D) float32 CT
        mask_volume: (H, W, D) uint8 当前 mask（含 refine 修改）
        refined_indices: 被用户 refine 过的切片索引列表
        video_height: H
        video_width: W
        num_anchor_slices: 保留参数，不再使用（仅向后兼容）

    Returns:
        Dict[int, np.ndarray]: {frame_idx: (H, W) uint8 二值 mask, ...}
    """
    import numpy as np
    import torch
    import cv2
    from Tool_Box.io import apply_ct_window

    model = MedSAM2Model.get_instance()
    predictor = model.predictor
    device = model.device

    model_size = predictor.image_size  # 通常是 512
    num_frames = ct_volume.shape[2]

    # ── 1. CT 预处理 ──
    windowed = apply_ct_window(ct_volume, wl=40.0, ww=400.0)
    ct_uint8 = (windowed * 255.0).astype(np.uint8)

    if video_height == model_size and video_width == model_size:
        ct_uint8 = np.transpose(ct_uint8, (2, 0, 1))  # (D, H, W)
    else:
        ct_resized = np.zeros((num_frames, model_size, model_size), dtype=np.uint8)
        for z in range(num_frames):
            ct_resized[z] = cv2.resize(
                ct_uint8[:, :, z], (model_size, model_size),
                interpolation=cv2.INTER_LINEAR
            )
        ct_uint8 = ct_resized

    # (D, 1, H, W) → (D, 3, H, W) → [0,1] → ImageNet norm
    ct_uint8 = ct_uint8[:, None, :, :]
    ct_uint8 = np.repeat(ct_uint8, 3, axis=1)
    ct_uint8 = ct_uint8.astype(np.float32) / 255.0
    images_tensor = torch.from_numpy(ct_uint8).to(device)
    img_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images_tensor = (images_tensor - img_mean) / img_std

    # ── 2. 确定锚点切片范围 ──
    # BUG FIX: 只使用 refine 过的切片作为锚点，不再添加密集邻域切片。
    # 旧逻辑会添加 refine 切片前后各 num_anchor_slices 层（如 refine 42 → 锚点 37-47），
    # 这些邻域切片通过 add_new_mask 被注册为 ground truth。
    # propagate_in_video 遇到已有 mask 的帧时直接返回原始 mask，不会做 propagation，
    # 导致相邻切片的 mask 没有任何变化。
    #
    # 修复后只注入用户真正修改过的切片，让 MedSAM2 将编辑效果传播到所有其他帧。
    anchor_slices = sorted(set(refined_indices))
    logger.info(
        f"[MedSAM2] Propagation anchors: {len(anchor_slices)} refined slice(s): "
        f"{anchor_slices[:10]}{'...' if len(anchor_slices) > 10 else ''}"
    )

    first_anchor = min(anchor_slices)
    last_anchor = max(anchor_slices)

    obj_id = 1

    def _resize_if_needed(mask_2d: np.ndarray) -> np.ndarray:
        """将模型输出 resize 回原始尺寸。"""
        if mask_2d.shape[0] != video_height or mask_2d.shape[1] != video_width:
            return cv2.resize(
                mask_2d, (video_width, video_height),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask_2d

    def _run_forward(state) -> dict:
        """单向 forward propagation，返回 {frame_idx: (H,W) uint8}。"""
        result = {}
        for frame_idx, fwd_obj_ids, fwd_masks in predictor.propagate_in_video(state):
            if obj_id in fwd_obj_ids:
                obj_idx = fwd_obj_ids.index(obj_id)
                mask_tensor = fwd_masks[obj_idx, 0]
                binary = (mask_tensor.cpu().numpy() > 0).astype(np.uint8)
                result[frame_idx] = _resize_if_needed(binary)
        return result

    def _init_state(images):
        """创建 inference_state。"""
        inf_state = predictor.init_state(
            images=images,
            video_height=model_size,
            video_width=model_size,
        )
        return inf_state

    def _inject_anchors(state, anchor_list):
        """向 state 中注入锚点 mask。"""
        for z in anchor_list:
            mask_2d = mask_volume[:, :, z]
            if mask_2d.sum() > 0:
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=z,
                    obj_id=obj_id,
                    mask=mask_2d,
                )

    # ================================================================
    # 方向 A: Forward — 从第一个锚点出发，向 volume 末尾传播
    # ================================================================
    state_fwd = _init_state(images_tensor)
    _inject_anchors(state_fwd, anchor_slices)
    logger.info("[MedSAM2] Running forward propagation...")
    fwd_result = _run_forward(state_fwd)
    logger.info(f"[MedSAM2] Forward done: {len(fwd_result)} frames with obj detected")

    # ================================================================
    # 方向 B: Backward — 翻转帧序列，从最后一个锚点出发，向 volume 开头传播
    # ================================================================
    # 翻转帧顺序: images_rev[i] = images_tensor[num_frames-1-i]
    images_rev = images_tensor.flip(0)
    # 锚点在反转后的索引
    rev_anchors = [num_frames - 1 - z for z in anchor_slices]

    state_bwd = _init_state(images_rev)
    _inject_anchors(state_bwd, rev_anchors)
    logger.info("[MedSAM2] Running backward propagation (reversed volume)...")
    rev_result = _run_forward(state_bwd)
    # 将反向传播结果映射回原始帧索引
    bwd_result = {}
    for rev_idx, mask in rev_result.items():
        orig_idx = num_frames - 1 - rev_idx
        bwd_result[orig_idx] = mask
    logger.info(f"[MedSAM2] Backward done: {len(bwd_result)} frames with obj detected")

    # ================================================================
    # 合并两个方向的结果
    # ================================================================
    logger.info("[MedSAM2] Merging forward + backward results...")

    def _distance_to_anchors(z: int) -> int:
        """切片到最近锚点的距离。"""
        return min(abs(z - a) for a in anchor_slices)

    merged = {}
    for z in range(num_frames):
        in_fwd = z in fwd_result
        in_bwd = z in bwd_result

        if in_fwd and in_bwd:
            # 两个方向都有 → 选择离锚点更近的方向的结果
            dist_fwd = _distance_to_anchors(z) if z >= first_anchor else abs(z - first_anchor)
            dist_bwd = _distance_to_anchors(z) if z <= last_anchor else abs(z - last_anchor)
            if dist_fwd <= dist_bwd:
                merged[z] = fwd_result[z]
            else:
                merged[z] = bwd_result[z]
        elif in_fwd:
            merged[z] = fwd_result[z]
        elif in_bwd:
            merged[z] = bwd_result[z]
        else:
            # 两个方向都没检测到 → 复制最近锚点的 mask
            nearest = min(anchor_slices, key=lambda a: abs(a - z))
            merged[z] = mask_volume[:, :, nearest].copy()

    # 统计实际变化的切片数
    changed = sum(
        1 for z in merged
        if not np.array_equal(merged[z], mask_volume[:, :, z])
    )
    logger.info(
        f"[MedSAM2] Merged: {len(merged)}/{num_frames} slices, "
        f"{changed} changed from original"
    )
    return merged
