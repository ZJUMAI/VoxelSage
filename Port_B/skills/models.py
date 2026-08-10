"""
Skill 数据模型
===============
定义 SkillManifest（元数据）和 SkillContext（运行时上下文）。
内置 Skills 和用户上传 Skills 共用同一套接口。
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from Tool_Box.mask_resolution import scan_logical_masks

logger = logging.getLogger(__name__)


# ============================================================
# SkillManifest — 从 skill.yaml 解析的元数据
# ============================================================

@dataclass
class SkillManifest:
    """单个 Skill 的元数据描述。

    与 skill.yaml 文件内容一一对应，供 LLM 构建 function calling schema 使用。
    """
    name: str
    version: str
    description: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    type: str = "builtin"  # "builtin" | "user"

    def _outputs_summary(self) -> str:
        """从 outputs schema 生成一段描述返回值主要结构的文本。

        追加到 tool description 末尾，让 LLM 提前知道调用结果的结构。
        """
        if not self.outputs:
            return ""
        props = self.outputs.get("properties", {})
        if not props:
            return ""
        lines = []
        for name, schema in props.items():
            ptype = schema.get("type", "any")
            desc = schema.get("description", "")
            if desc:
                lines.append(f"  {name} ({ptype}): {desc}")
            else:
                lines.append(f"  {name} ({ptype})")
            # array items 的子结构简要提示
            if ptype == "array" and "items" in schema:
                items = schema["items"]
                items_desc = items.get("description", "")
                if items_desc:
                    lines[-1] += f" — {items_desc}"
                items_props = items.get("properties", {})
                if items_props:
                    keys = sorted(items_props.keys())
                    lines[-1] += f" 每项包含: {', '.join(keys)}"
        return "返回:\n" + "\n".join(lines)

    def to_function_calling_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI-compatible function calling tool schema。

        供 Port A 的 LLM 识别和调用此 Skill。
        description 末尾自动追加返回值结构概要，让 LLM 提前知道返回内容。
        """
        parameters = {"type": "object", "properties": {}, "required": []}
        for param_name, param_schema in self.inputs.items():
            source = param_schema.get("source", "")
            # source=context 的参数由框架自动注入，不需要 LLM 传
            if source == "context":
                continue
            parameters["properties"][param_name] = {
                "type": param_schema.get("type", "string"),
                "description": param_schema.get("description", ""),
            }
            if param_schema.get("required", False):
                parameters["required"].append(param_name)

        # 在 description 末尾追加返回值结构（LLM 能直接看到）
        suffix = self._outputs_summary()
        enriched_description = self.description
        if suffix:
            enriched_description = enriched_description.rstrip("。\n") + "。\n" + suffix

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": enriched_description,
                "parameters": parameters,
            },
        }

    def to_list_entry(self) -> Dict[str, Any]:
        """GET /api/skills/list 返回的单条记录。

        除了元数据和输入参数 schema 外，额外返回 returns 字段，
        包含完整的输出 JSON Schema，供 API 消费者（Port A / 前端）查阅。
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "type": self.type,
            "triggers": self.triggers,
            "parameters": self.to_function_calling_schema(),
            "returns": self.outputs if self.outputs else None,
        }


# ============================================================
# SkillContext — 运行时数据访问接口
# ============================================================

class SkillContext:
    """Skill 执行时的数据上下文。

    提供统一的 API 让 Skill 访问当前病例的 CT 数据、掩码文件和分析结果。
    内置 Skills 和用户上传的 Skills 使用同一套接口。
    """

    def __init__(
        self,
        case_id: str,
        ct_nifti_path: str,
        mask_dir: str,
        output_dir: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.case_id = case_id
        self.ct_nifti_path = ct_nifti_path
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.params = params or {}

        # 缓存（惰性加载，多次调用只加载一次）
        self._ct_nii: Optional[nib.Nifti1Image] = None
        self._ct_array: Optional[np.ndarray] = None
        self._affine: Optional[np.ndarray] = None
        self._spacing: Optional[Tuple[float, float, float]] = None
        self._mask_paths: Optional[Dict[str, str]] = None
        self._mask_cache: Dict[str, np.ndarray] = {}

    # ── 影像数据 ──

    def get_ct_array(self) -> np.ndarray:
        """返回 CT 体素数组 (H, W, D)。"""
        if self._ct_array is None:
            self._load_ct()
        return self._ct_array

    def get_affine(self) -> np.ndarray:
        """返回 NIfTI affine 矩阵 (4, 4)。"""
        if self._affine is None:
            self._load_ct()
        return self._affine

    def get_voxel_spacing(self) -> Tuple[float, float, float]:
        """返回体素间距 (sx, sy, sz) mm。"""
        if self._spacing is None:
            if self._affine is None:
                self._load_ct()
            spacing = nib.affines.voxel_sizes(self._affine)
            self._spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        return self._spacing

    def _load_ct(self):
        """惰性加载 CT NIfTI。"""
        self._ct_nii = nib.load(self.ct_nifti_path)
        self._ct_array = self._ct_nii.get_fdata()
        self._affine = self._ct_nii.affine

    # ── 掩码数据 ──

    def list_masks(self) -> List[str]:
        """返回掩码名称列表，如 ['liver', 'hepatic', 'tumor_1', ...]。"""
        self._scan_mask_dir()
        return list(self._mask_paths.keys())

    def get_mask_path(self, name: str) -> str:
        """返回指定掩码的文件路径。"""
        self._scan_mask_dir()
        path = self._mask_paths.get(name)
        if path is None:
            raise FileNotFoundError(
                f"Mask '{name}' not found in {self.mask_dir}. "
                f"Available: {list(self._mask_paths.keys())}"
            )
        return path

    def get_mask(self, name: str) -> np.ndarray:
        """返回指定掩码的 NumPy 二值矩阵。"""
        if name in self._mask_cache:
            return self._mask_cache[name]
        path = self.get_mask_path(name)
        nii = nib.load(path)
        mask = nii.get_fdata() > 0
        self._mask_cache[name] = mask
        return mask

    def _scan_mask_dir(self):
        """扫描 mask_dir 建立名称→路径映射。

        排除 VISTA3D 生成的多标签汇总文件（all.nii.gz / mask.nii.gz），
        它们不是独立可编辑/可分析的二值掩码。
        """
        if self._mask_paths is not None:
            return
        self._mask_paths = {
            name: resolved.path
            for name, resolved in scan_logical_masks(self.mask_dir).items()
        }

    # ── 输出 ──

    def get_output_dir(self) -> str:
        """返回输出目录路径。"""
        return self.output_dir

    # ── 写入（特批 — segmentation_modification skill 专用） ──

    def save_mask(self, name: str, mask_array: np.ndarray,
                  dtype: type = np.float32) -> str:
        """将掩码数组保存为 NIfTI 到 mask_dir。

        **注意**：此方法会写文件。目前仅 segmentation_modification skill 使用，
        其他 skill 应保持 read-only。

        Args:
            name: 掩码名称（如 'liver', 'tumor_1'）
            mask_array: 二值掩码数组 (H, W, D) 或 (H, W)
            dtype: NIfTI 数据类型（默认 np.float32）

        Returns:
            str: 保存的 .nii.gz 文件路径
        """
        import os
        import shutil
        import nibabel as nib

        mask_path = os.path.join(self.mask_dir, f"{name}.nii.gz")

        # 备份已有文件
        if os.path.exists(mask_path):
            backup_path = mask_path + ".bak"
            shutil.copy2(mask_path, backup_path)
            self.log(f"Backed up existing mask to {backup_path}")

        # 确保二进制
        mask_binary = (mask_array > 0).astype(dtype)

        # 使用 CT 的 affine
        affine = self.get_affine()

        nii = nib.Nifti1Image(mask_binary, affine)
        nib.save(nii, mask_path)

        # 清空缓存，下次 get_mask 重新加载
        if name in self._mask_cache:
            del self._mask_cache[name]
        if self._mask_paths is not None:
            self._mask_paths[name] = mask_path

        self.log(f"Mask '{name}' saved to {mask_path}")
        return mask_path

    # ── 工具 ──

    def log(self, msg: str):
        """记录日志（集成到 API.py 的日志系统）。"""
        logger.info(f"[Skill:{self.case_id}] {msg}")
