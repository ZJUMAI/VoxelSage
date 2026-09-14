"""
segmentation_modification: 编辑器会话状态管理
==============================================

每个 EditorSession 对应一个浏览器标签页，包含：
- CT volume、多个 mask、原始 mask（用于参考）
- Click 历史 (slice, x, y, label)
- MedSAM2 inference state
- TTL 自动过期

支持同一切片切换编辑不同 mask（多元观测/器官）。
"""

import os
import glob
import time
import threading
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 1800     # 30 分钟
_MAX_SESSIONS = 5        # 最大并发 session 数（防内存溢出）
_MAX_MASKS_LOADED = 4    # 每个 session 最多同时加载的 mask 数

# ---- 多标签掩码关键词（需被排除或拆分） ----
# VISTA3D 会生成一个包含所有标签的全量 all.nii.gz 文件，
# 它不是可编辑的单个掩码，应排除或拆分为独立二值掩码。
_SKIP_MASK_NAMES = {"all", "mask"}

# VISTA3D label → 器官名称映射（用于拆分为可读的 mask 名称）
_VISTA3D_LABEL_MAP = {
    1: "liver",
    3: "spleen",
    4: "pancreas",
    17: "portal_vein",
    25: "hepatic_vessel",
    26: "hepatic_tumor",
}


class EditorSession:
    """单个编辑器会话，支持多个 mask 切换编辑。"""

    def __init__(
        self,
        session_id: str,
        case_id: str,
        ct_nifti_path: str,
        mask_dir: str,
        output_dir: str,
        mask_name: str,            # 初始/默认 mask
        mask_names: List[str],     # 所有可用 mask 名称
        ct_shape: Tuple[int, int, int],
        affine: np.ndarray,
    ):
        self.session_id = session_id
        self.case_id = case_id
        self.ct_nifti_path = ct_nifti_path
        self.mask_dir = mask_dir
        self.output_dir = output_dir

        # 多 mask 支持：过滤掉 "all" 等中间多标签文件
        # scene_manager 的上游（ctx.list_masks()）会扫描 *.nii.gz，
        # 其中 all.nii.gz 是 VISTA3D 多标签汇总，非独立可编辑掩码。
        _raw_names = mask_names[:]
        self.mask_names = [n for n in mask_names if n not in _SKIP_MASK_NAMES]

        # 如果过滤后没有任何掩码，说明只有多标签文件 → 需要拆分为独立标签
        if not self.mask_names:
            self.mask_names = self._split_multilabel_masks(mask_dir, _raw_names, ct_shape)

        # 如果过滤/拆分后初始掩码名不可用，仍优先选 liver，再退到第一个。
        # 这覆盖了会话收到 all、随后才拆分多标签掩码的路径。
        if mask_name not in self.mask_names:
            mask_name = (
                "liver"
                if "liver" in self.mask_names
                else (self.mask_names[0] if self.mask_names else "")
            )

        self.current_mask_name = mask_name
        self._masks: Dict[str, Optional[np.ndarray]] = {n: None for n in self.mask_names}
        self._original_masks: Dict[str, Optional[np.ndarray]] = {n: None for n in self.mask_names}
        # 仅记录用户实际运行过 Refine 的切片。传播写回的切片不能成为下一次
        # 传播的新锚点，否则多次传播会让影响范围逐步向远端扩张。
        self._refined_slices: Dict[str, set] = {n: set() for n in self.mask_names}

        # 惰性加载数据
        self._ct_array: Optional[np.ndarray] = None
        self._ct_uint8: Optional[np.ndarray] = None
        self._affine = affine
        self._ct_shape = ct_shape

        # Click 历史：(slice_idx, x, y, label, mask_name)
        self.click_history: List[Tuple[int, int, int, int, str]] = []

        # MedSAM2 inference state（首次 refine 时初始化）
        self.medsam2_inference_state = None
        self.medsam2_obj_id = 1

        # LRU 追踪（实例变量，必须 __init__ 中初始化，非类变量）
        self._mask_access_order: List[str] = []

        # TTL
        self.last_access: float = time.time()
        self.created_at: float = time.time()

    @staticmethod
    def _split_multilabel_masks(
        mask_dir: str,
        raw_names: List[str],
        ct_shape: Tuple[int, int, int],
    ) -> List[str]:
        """拆分多标签掩码（如 all.nii.gz）为独立二值标签掩码。

        当 mask_dir 中只有多标签文件（如 all.nii.gz）而没有独立二值掩码时，
        读取该多标签文件，按唯一标签值拆分为多个独立二值掩码，
        并保存到 mask_dir 中供编辑器使用。

        Args:
            mask_dir: 掩码目录
            raw_names: 从 ctx.list_masks() 获取的原始名称列表
            ct_shape: CT 体素形状 (H, W, D)

        Returns:
            List[str]: 拆分后的独立掩码名称列表
        """
        import nibabel as nib

        # 找到多标签文件
        multi_label_path = None
        for name in raw_names:
            if name in _SKIP_MASK_NAMES:
                pattern = os.path.join(mask_dir, f"{name}.nii.gz")
                if os.path.exists(pattern):
                    multi_label_path = pattern
                    break
                # 也尝试带通配符匹配
                files = sorted(glob.glob(os.path.join(mask_dir, f"{name}*")))
                if files:
                    multi_label_path = files[0]
                    break

        if multi_label_path is None:
            # 尝试直接找 *.nii.gz 文件
            all_files = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
            if all_files:
                multi_label_path = all_files[0]

        if multi_label_path is None:
            return []

        try:
            nii = nib.load(multi_label_path)
            data = np.array(nii.get_fdata(), dtype=np.int32)
            unique_vals = np.unique(data)
            # 过滤背景 0 和负值（CT HU 值 → 不是标签掩码）
            label_vals = sorted(v for v in unique_vals if v > 0)

            # 如果非零值太多，说明这不是标签掩码（可能是 CT 误存）
            if len(label_vals) > 50:
                logger.warning(
                    f"[EditorSession] '{os.path.basename(multi_label_path)}' has "
                    f"{len(label_vals)} unique non-zero values — not a label mask, skipping split"
                )
                return []

            new_names = []
            for label_val in label_vals:
                # 优先使用器官名称，fallback 到 label_N
                organ_name = _VISTA3D_LABEL_MAP.get(int(label_val))
                if organ_name:
                    label_mask_name = organ_name
                else:
                    label_mask_name = f"label_{int(label_val)}"

                label_mask_path = os.path.join(mask_dir, f"{label_mask_name}.nii.gz")
                # 如果同名文件已存在（独立 mask），跳过不覆盖
                if os.path.exists(label_mask_path):
                    new_names.append(label_mask_name)
                    logger.info(
                        f"[EditorSession] Label {int(label_val)} → {label_mask_name}.nii.gz (already exists, skipped)"
                    )
                    continue

                # 创建二值掩码并保存
                binary_mask = (data == label_val).astype(np.uint8)
                nii_out = nib.Nifti1Image(binary_mask, nii.affine)
                nib.save(nii_out, label_mask_path)
                new_names.append(label_mask_name)
                logger.info(
                    f"[EditorSession] Split label {int(label_val)} → {label_mask_name}.nii.gz"
                )

            if new_names:
                logger.info(
                    f"[EditorSession] Split {len(new_names)} labels from "
                    f"'{os.path.basename(multi_label_path)}'"
                )
            return new_names

        except Exception as e:
            logger.error(f"[EditorSession] Failed to split multi-label mask: {e}")
            return []

    def touch(self):
        """更新最后访问时间。"""
        self.last_access = time.time()

    def is_expired(self, ttl: float = _DEFAULT_TTL) -> bool:
        return (time.time() - self.last_access) > ttl

    # ── 数据加载 ──

    def get_ct_array(self) -> np.ndarray:
        """返回 CT 体素数组 (H, W, D) float32。"""
        if self._ct_array is None:
            import nibabel as nib
            nii = nib.load(self.ct_nifti_path)
            self._ct_array = np.array(nii.get_fdata(), dtype=np.float32)
        return self._ct_array

    def get_ct_uint8(self) -> np.ndarray:
        """返回窗宽窗位后的 uint8 CT (H, W, D)，范围 [0, 255]。"""
        if self._ct_uint8 is None:
            from Tool_Box.io import apply_ct_window
            ct = self.get_ct_array()
            windowed = apply_ct_window(ct, wl=40.0, ww=400.0)
            self._ct_uint8 = (windowed * 255.0).astype(np.uint8)
        return self._ct_uint8

    def _load_aligned_mask_data(self, path: str) -> np.ndarray:
        """Load a mask in the CT voxel grid, respecting both NIfTI affines.

        Segmentation backends may write an anatomically aligned mask with a
        different voxel-axis orientation (for example LPS-like versus RAS).
        Indexing that array directly makes the overlay appear flipped or
        shifted.  Nearest-neighbour resampling preserves discrete labels.
        """
        import nibabel as nib

        mask_nii = nib.load(path)
        target_shape = tuple(int(v) for v in self._ct_shape)
        target_affine = np.asarray(self._affine)
        same_grid = (
            tuple(mask_nii.shape[:3]) == target_shape
            and np.allclose(mask_nii.affine, target_affine, rtol=1e-5, atol=1e-4)
        )
        if not same_grid:
            from nibabel.processing import resample_from_to

            logger.info(
                f"[EditorSession:{self.session_id}] Aligning mask "
                f"'{os.path.basename(path)}' from shape {mask_nii.shape[:3]} "
                f"to CT shape {target_shape}"
            )
            mask_nii = resample_from_to(
                mask_nii,
                (target_shape, target_affine),
                order=0,
                mode="constant",
                cval=0,
            )
        return np.asarray(mask_nii.get_fdata())

    def _load_mask_from_disk(self, name: str) -> np.ndarray:
        """从磁盘加载指定 mask，返回 (H, W, D) uint8。

        处理两种情形:
        1) 文件是标准二值掩码 → 直接返回 (>0)
        2) 文件是多标签掩码（如 all.nii.gz）且 name 为 label_N → 只提取标签 N

        注意: 如果 label_N.nii.gz 已存在（可能是用户编辑保存的），
        优先使用该独立文件，而非从多标签源文件提取。
        """
        import nibabel as nib

        # 如果 name 是 label_N 形式
        _label_prefix = "label_"
        if name.startswith(_label_prefix):
            # 优先检查是否有已保存的独立 label_N.nii.gz 文件
            label_file = os.path.join(self.mask_dir, f"{name}.nii.gz")
            if os.path.exists(label_file):
                return (self._load_aligned_mask_data(label_file) > 0).astype(np.uint8)

            # 没有独立文件 → 从多标签源文件中提取对应标签
            try:
                label_val = int(name[len(_label_prefix):])
            except ValueError:
                label_val = None
            if label_val is not None:
                for src_name in _SKIP_MASK_NAMES:
                    src_path = os.path.join(self.mask_dir, f"{src_name}.nii.gz")
                    if os.path.exists(src_path):
                        data = self._load_aligned_mask_data(src_path)
                        return (np.array(data == label_val, dtype=np.uint8) > 0).astype(np.uint8)
                for src_name in _SKIP_MASK_NAMES:
                    files = sorted(glob.glob(os.path.join(self.mask_dir, f"{src_name}*")))
                    if files:
                        data = self._load_aligned_mask_data(files[0])
                        return (np.array(data == label_val, dtype=np.uint8) > 0).astype(np.uint8)

        # 标准模式：直接通过名称查找文件
        pattern = os.path.join(self.mask_dir, f"{name}*")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"Mask '{name}' not found in {self.mask_dir}")
        path = files[0]
        return (self._load_aligned_mask_data(path) > 0).astype(np.uint8)

    def get_mask(self, name: str) -> np.ndarray:
        """获取指定 mask（惰性加载 + LRU 淘汰）。"""
        if name not in self._masks:
            raise KeyError(f"Mask '{name}' not available in this session")
        if self._masks[name] is None:
            self._masks[name] = self._load_mask_from_disk(name)
            self._original_masks[name] = self._masks[name].copy()
            logger.info(
                f"[EditorSession:{self.session_id}] Loaded mask '{name}' from disk"
            )

        # 更新 LRU：从列表移除再追加到末尾（最近访问）
        if name in self._mask_access_order:
            self._mask_access_order.remove(name)
        self._mask_access_order.append(name)

        # 超限时淘汰最久未访问的 mask（保留当前正在用的）
        while len(self._mask_access_order) > _MAX_MASKS_LOADED:
            evict_candidate = self._mask_access_order.pop(0)
            if evict_candidate != self.current_mask_name:
                self._masks[evict_candidate] = None
                self._original_masks[evict_candidate] = None
                logger.info(
                    f"[EditorSession:{self.session_id}] "
                    f"Evicted mask '{evict_candidate}' from memory (LRU)"
                )
            else:
                # 当前 mask 不能淘汰，放回去
                self._mask_access_order.append(evict_candidate)
                break

        return self._masks[name]

    def get_current_mask(self) -> np.ndarray:
        """返回当前 mask (H, W, D) uint8。"""
        return self.get_mask(self.current_mask_name)

    # ── 原始（非二值化）Mask 加载 — 用于多标签渲染 ──

    def _load_raw_mask_from_disk(self, name: str) -> np.ndarray:
        """从磁盘加载指定 mask，保留原始 label 值（不做 >0 二值化）。

        用于前端渲染，让不同 label 值显示不同颜色。
        注意：返回的数据可能包含多个标签值（如 1, 17, 25, 26），
        而非标准二值 mask 的 0/1。
        """
        import nibabel as nib

        _label_prefix = "label_"
        if name.startswith(_label_prefix):
            # 优先检查独立文件
            label_file = os.path.join(self.mask_dir, f"{name}.nii.gz")
            if os.path.exists(label_file):
                return np.asarray(
                    np.rint(self._load_aligned_mask_data(label_file)),
                    dtype=np.int32,
                )
            # 从多标签源文件提取
            try:
                label_val = int(name[len(_label_prefix):])
            except ValueError:
                label_val = None
            if label_val is not None:
                for src_name in _SKIP_MASK_NAMES:
                    src_path = os.path.join(self.mask_dir, f"{src_name}.nii.gz")
                    if os.path.exists(src_path):
                        data = self._load_aligned_mask_data(src_path)
                        return (np.array(data == label_val, dtype=np.int32) * label_val)

        # 标准模式
        pattern = os.path.join(self.mask_dir, f"{name}*")
        files = sorted(__import__('glob').glob(pattern))
        if not files:
            raise FileNotFoundError(f"Mask '{name}' not found in {self.mask_dir}")
        path = files[0]
        return np.asarray(
            np.rint(self._load_aligned_mask_data(path)),
            dtype=np.int32,
        )

    def get_raw_mask(self, name: str) -> np.ndarray:
        """获取指定 mask 的原始 label 值（非二值化），用于前端多标签着色。"""
        # 对于已知的二值 mask，直接从缓存返回（已缓存的是二值化的）
        if name in self._masks and self._masks[name] is not None:
            return self._masks[name]
        # 否则从磁盘加载原始值
        raw = self._load_raw_mask_from_disk(name)
        # 判断是否多标签（>2 个唯一非零值）
        unique_vals = set(np.unique(raw)) - {0}
        if len(unique_vals) <= 1:
            # 单标签 mask → 回退到缓存的二值化版本
            self.get_mask(name)  # 触发缓存
            return self._masks[name]
        # 多标签 mask → 返回原始值（不缓存，避免污染二值逻辑）
        return raw

    def get_mask_names(self) -> List[str]:
        """返回所有可用 mask 名称。"""
        return self.mask_names

    def switch_mask(self, name: str) -> None:
        """切换到指定 mask（会延迟加载）。"""
        if name not in self._masks:
            raise KeyError(f"Mask '{name}' not available in this session")
        self.current_mask_name = name
        # 确保已加载（MedSAM2 推理状态不变，但 refine 会用当前 mask）
        self.get_current_mask()

    # ── Mask 更新 ──

    def update_mask_slice(self, slice_idx: int, new_mask_2d: np.ndarray):
        """更新当前 mask 指定切片的 2D mask（原地修改）。"""
        mask = self.get_current_mask()
        h, w = mask.shape[0], mask.shape[1]
        if new_mask_2d.shape != (h, w):
            from scipy.ndimage import zoom
            new_mask_2d = zoom(
                new_mask_2d,
                (h / new_mask_2d.shape[0], w / new_mask_2d.shape[1]),
                order=0,
            )
        mask[:, :, slice_idx] = (new_mask_2d > 0).astype(np.uint8)

    def mark_refined_slice(self, slice_idx: int) -> None:
        """记录当前 mask 上由用户 Refine 产生的真实传播锚点。"""
        self._refined_slices.setdefault(self.current_mask_name, set()).add(int(slice_idx))

    def get_refined_slices(self) -> List[int]:
        """返回当前 mask 的用户 Refine 切片，不包含传播生成的切片。"""
        return sorted(self._refined_slices.get(self.current_mask_name, set()))

    # ── Click 管理 ──

    def add_click(self, slice_idx: int, x: int, y: int, label: int):
        """添加一个点击点。label=1 前景, label=0 背景。"""
        self.click_history.append((slice_idx, x, y, label, self.current_mask_name))

    def get_clicks_for_slice(self, slice_idx: int) -> List[Tuple[int, int, int]]:
        """获取当前 mask 在指定切片上的所有点击，返回 [(x, y, label), ...]。"""
        return [
            (x, y, label)
            for (s, x, y, label, mn) in self.click_history
            if s == slice_idx and mn == self.current_mask_name
        ]

    def get_clicks_for_current_mask(self) -> List[Tuple[int, int, int, int]]:
        """获取当前 mask 的点击，返回 [(slice_idx, x, y, label), ...]。"""
        return [
            (s, x, y, label)
            for (s, x, y, label, mask_name) in self.click_history
            if mask_name == self.current_mask_name
        ]

    def clear_clicks_for_slice(self, slice_idx: int):
        """清除当前 mask 在指定切片上的所有点击。"""
        self.click_history = [
            (s, x, y, l, mn)
            for (s, x, y, l, mn) in self.click_history
            if not (s == slice_idx and mn == self.current_mask_name)
        ]

    def pop_last_click(self, slice_idx: Optional[int] = None) -> Optional[Tuple[int, int, int, int, str]]:
        """移除并返回当前 mask 的最后一个点击点，可限定切片。"""
        for index in range(len(self.click_history) - 1, -1, -1):
            click = self.click_history[index]
            click_slice, _, _, _, mask_name = click
            if mask_name == self.current_mask_name and (
                slice_idx is None or click_slice == slice_idx
            ):
                return self.click_history.pop(index)
        return None

    def log(self, msg: str):
        """记录日志。"""
        logger.info(f"[EditorSession:{self.session_id}] {msg}")


class EditorSessionManager:
    """线程安全的会话管理器，支持 TTL 自动过期。"""

    def __init__(self, ttl: float = _DEFAULT_TTL):
        self._sessions: Dict[str, EditorSession] = {}
        self._lock = threading.Lock()
        self._ttl = ttl

    def create(self, session_id: str, **kwargs) -> EditorSession:
        """创建新 session。可能触发 GPU 清理的淘汰操作在锁外执行。"""
        evicted = None
        with self._lock:
            # 防内存溢出：超限时淘汰最旧的 session
            if len(self._sessions) >= _MAX_SESSIONS:
                oldest_id = min(
                    self._sessions,
                    key=lambda sid: self._sessions[sid].last_access,
                )
                evicted = self._sessions.pop(oldest_id, None)
                if evicted:
                    logger.warning(
                        f"[SessionManager] Max sessions ({_MAX_SESSIONS}) reached, "
                        f"evicting oldest session '{oldest_id}' "
                        f"(idle for {(time.time() - evicted.last_access) / 60:.1f} min)"
                    )

            session = EditorSession(session_id=session_id, **kwargs)
            self._sessions[session_id] = session
            logger.info(
                f"[SessionManager] Created session '{session_id}'. "
                f"Active: {len(self._sessions)}/{_MAX_SESSIONS}"
            )

        # 锁已释放 — 淘汰 session 的 GPU 清理在锁外执行
        if evicted is not None:
            try:
                evicted.medsam2_inference_state = None
            except Exception:
                pass

        return session

    def get(self, session_id: str) -> Optional[EditorSession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired(self._ttl):
                self._sessions.pop(session_id, None)
                return None
            session.touch()
            return session

    def remove(self, session_id: str):
        """移除 session。GPU 清理在锁外执行，防止 cudaFree 阻塞其他请求。"""
        session = None
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            try:
                session.medsam2_inference_state = None
            except Exception:
                pass

    def cleanup_expired(self):
        """清理所有过期会话。

        设计要点：
        1. 过期 session 的摘除在锁内完成（快速 dict 操作）
        2. GPU 推理状态释放（可能触发 cudaFree 同步阻塞）在锁外执行
        3. 防止持锁线程卡在 CUDA 调用 → 所有其他请求阻塞
        """
        expired_sessions = []
        with self._lock:
            now = time.time()
            for sid, s in list(self._sessions.items()):
                if (now - s.last_access) > self._ttl:
                    expired_sessions.append(s)
                    del self._sessions[sid]

            active = len(self._sessions)

        # 锁已释放 — GPU 清理在锁外执行
        for s in expired_sessions:
            try:
                s.medsam2_inference_state = None
                logger.info(
                    f"[SessionManager] Expired session '{s.session_id}' cleaned up "
                    f"(idle for {(time.time() - s.last_access) / 60:.1f} min)"
                )
            except Exception:
                pass

        if len(self._sessions) > _MAX_SESSIONS * 0.7:
            logger.warning(
                f"[SessionManager] High session usage: "
                f"{len(self._sessions)}/{_MAX_SESSIONS}"
            )

    def count_active(self) -> int:
        """返回当前活跃 session 数（会触发过期清理）。"""
        # cleanup_expired 已改为锁内只做 dict 操作，安全
        self.cleanup_expired()
        with self._lock:
            return len(self._sessions)

    def get_active_count(self) -> int:
        """返回当前活跃 session 数（不触发清理）。"""
        with self._lock:
            return len(self._sessions)

    def get_session_ids(self) -> List[str]:
        """返回所有 session ID 列表（用于监控）。"""
        with self._lock:
            return list(self._sessions.keys())


# 模块级单例
editor_sessions = EditorSessionManager()
