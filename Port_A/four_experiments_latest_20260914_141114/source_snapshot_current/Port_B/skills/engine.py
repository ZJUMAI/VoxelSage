"""
Skill 执行引擎
===============
负责 Skills 的注册、查找和执行。

- 内置 Skills（type=builtin）：直接 import 执行，不沙箱
- 用户 Skills（type=user）：subprocess 沙箱隔离 + 超时限制
"""

from __future__ import annotations

import os
import re
import sys
import json
import shutil
import subprocess
import tempfile
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import SkillContext, SkillManifest

logger = logging.getLogger(__name__)


# ============================================================
# Skill 定义
# ============================================================

@dataclass
class SkillDef:
    """内存中的 Skill 定义。"""
    meta: SkillManifest
    module: Optional[Any] = None   # builtin 时：imported module
    skill_dir: Optional[str] = None  # user 时：skill 目录路径


# ============================================================
# 自定义异常
# ============================================================

class SkillError(Exception):
    pass

class SkillNotFoundError(SkillError):
    def __init__(self, name: str):
        super().__init__(f"Skill '{name}' not found")
        self.skill_name = name

class SkillExecutionError(SkillError):
    def __init__(self, name: str, cause: Exception):
        super().__init__(f"Skill '{name}' execution failed: {cause}")
        self.skill_name = name
        self.cause = cause


# ============================================================
# 沙箱执行常量
# ============================================================

_USER_SKILL_TIMEOUT = 120          # 用户 Skill 最大执行时间（秒）
_USER_SKILL_MAX_OUTPUT = 10 * 1024 * 1024  # 最大输出 10MB
_MAX_USER_SKILL_TOTAL_BYTES = 5 * 1024 * 1024    # 用户 Skill 目录总大小上限 5MB
_MAX_USER_SINGLE_FILE_BYTES = 2 * 1024 * 1024    # 单文件大小上限 2MB


# ============================================================
# SkillEngine
# ============================================================

class SkillEngine:
    """全局 Skills 注册表和执行器。"""

    def __init__(self):
        self._skills: Dict[str, SkillDef] = {}
        self._initialized = False

    # ── 注册 ──

    def register_builtin(self, skill_dir: str):
        """扫描 skill_dir 下所有子目录，注册内置 Skills（直接 import 执行）。"""
        if not os.path.isdir(skill_dir):
            logger.warning(f"Built-in skills directory not found: {skill_dir}")
            return

        count = 0
        for entry in sorted(os.scandir(skill_dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            yaml_path = os.path.join(entry.path, "skill.yaml")
            main_path = os.path.join(entry.path, "main.py")
            if not (os.path.isfile(yaml_path) and os.path.isfile(main_path)):
                continue

            try:
                meta = self._load_manifest(yaml_path)
                meta.type = "builtin"
                mod = self._load_module(meta.name, main_path)
                self._skills[meta.name] = SkillDef(meta=meta, module=mod)
                count += 1
                logger.info(f"Registered built-in skill: {meta.name} v{meta.version}")
            except Exception as e:
                logger.error(f"Failed to load skill '{entry.name}': {e}")

        self._initialized = True
        logger.info(f"Registered {count} built-in skill(s)")

    def register_user_skill(self, skill_dir: str) -> SkillManifest:
        """注册一个用户上传的 Skill（subprocess 沙箱执行）。

        校验 skill_dir 中必须包含 skill.yaml + main.py。
        将技能文件复制到 skills/user/<name>/ 目录下持久化管理。
        返回解析后的 SkillManifest。
        """
        yaml_path = os.path.join(skill_dir, "skill.yaml")
        main_path = os.path.join(skill_dir, "main.py")

        if not os.path.isfile(yaml_path):
            raise ValueError(f"Missing skill.yaml in {skill_dir}")
        if not os.path.isfile(main_path):
            raise ValueError(f"Missing main.py in {skill_dir}")

        meta = self._load_manifest(yaml_path)
        meta.type = "user"

        # 校验 main.py 中有 run 函数（简单文本扫描）
        with open(main_path, "r") as f:
            source = f.read()
        if "def run(" not in source:
            raise ValueError("main.py must define a run(ctx) function")

        # ── 安全校验 ──
        # 1. 名称合法性 — 防止路径穿越
        if not re.match(r'^[a-zA-Z0-9_-]+$', meta.name):
            raise ValueError(
                f"Invalid skill name '{meta.name}': only letters, digits, hyphens, "
                f"and underscores allowed"
            )
        # 2. 文件大小校验
        total_size = 0
        for root, dirs, files in os.walk(skill_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    fsize = os.path.getsize(fpath)
                except OSError:
                    continue
                if fsize > _MAX_USER_SINGLE_FILE_BYTES:
                    raise ValueError(
                        f"File '{fname}' is {fsize} bytes, exceeds "
                        f"single-file limit of {_MAX_USER_SINGLE_FILE_BYTES} bytes"
                    )
                total_size += fsize
        if total_size > _MAX_USER_SKILL_TOTAL_BYTES:
            raise ValueError(
                f"Skill directory total size {total_size} bytes exceeds "
                f"limit of {_MAX_USER_SKILL_TOTAL_BYTES} bytes"
            )

        # ── 复制到 skills/user/<name>/ ──
        # engine.py 位于 Port_B/skills/engine.py。
        # 两级 dirname → Port_B/（与 API.py 的 _PROJECT_ROOT 一致）。
        project_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))
        user_skills_dir = os.path.join(project_root, "skills", "user")
        target_dir = os.path.join(user_skills_dir, meta.name)

        os.makedirs(user_skills_dir, exist_ok=True)

        # 目标已存在则覆盖（更新技能）
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)

        shutil.copytree(skill_dir, target_dir, dirs_exist_ok=True)

        # 注册（指向复制后的目录，确保持久化和统一管理）
        self._skills[meta.name] = SkillDef(meta=meta, skill_dir=target_dir)
        logger.info(
            f"Registered user skill: {meta.name} v{meta.version} "
            f"(copied to {target_dir})"
        )
        return meta

    def unregister_skill(self, name: str):
        """注销 Skill，同时删除 skills/user/<name>/ 下的文件。"""
        if name not in self._skills:
            raise SkillNotFoundError(name)

        skill = self._skills[name]

        # 如果是用户技能（存在 skills/user/ 下的对应目录），删除文件
        if skill.skill_dir and os.path.isdir(skill.skill_dir):
            shutil.rmtree(skill.skill_dir)
            logger.info(f"Deleted skill directory: {skill.skill_dir}")

        del self._skills[name]
        logger.info(f"Unregistered skill: {name}")

    def _load_manifest(self, yaml_path: str) -> SkillManifest:
        """从 skill.yaml 加载元数据。"""
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for key in ("name", "version", "description"):
            if key not in data:
                raise ValueError(f"skill.yaml missing required field: {key}")

        return SkillManifest(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            triggers=data.get("triggers", []),
            type="builtin",
        )

    def _load_module(self, skill_name: str, main_path: str):
        """动态 import 一个 Skill 的 main.py。"""
        module_name = f"_skill_builtin_{skill_name}"
        spec = importlib.util.spec_from_file_location(module_name, main_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {main_path}")

        mod = importlib.util.module_from_spec(spec)
        # Add paths so skill modules can resolve intra-package imports
        # (e.g. from skills.models import ..., from Tool_Box import ...)
        skills_root = os.path.dirname(os.path.dirname(main_path))  # skills/builtin/
        project_root = os.path.dirname(skills_root)                 # skills/
        for _p in (skills_root, project_root):
            if _p not in sys.path:
                sys.path.insert(0, _p)

        spec.loader.exec_module(mod)
        if not hasattr(mod, "run"):
            raise AttributeError(f"Skill '{skill_name}' main.py must define run(ctx) function")
        return mod

    # ── 查询 ──

    def get_skill(self, name: str) -> SkillDef:
        skill = self._skills.get(name)
        if skill is None:
            raise SkillNotFoundError(name)
        return skill

    def list_skills(self) -> List[Dict[str, Any]]:
        return [
            skill.meta.to_list_entry()
            for skill in sorted(self._skills.values(), key=lambda s: s.meta.name)
        ]

    def list_skills_openai_tools(self) -> List[Dict[str, Any]]:
        return [
            skill.meta.to_function_calling_schema()
            for skill in sorted(self._skills.values(), key=lambda s: s.meta.name)
        ]

    # ── 执行 ──

    def run(
        self,
        skill_name: str,
        ctx: SkillContext,
    ) -> Dict[str, Any]:
        """执行指定 Skill。

        - 内置 Skills：直接 import 执行
        - 用户 Skills：subprocess 沙箱隔离 + 超时
        """
        skill = self.get_skill(skill_name)

        if skill.meta.type == "builtin":
            return self._run_builtin(skill, ctx)
        else:
            return self._run_user_sandboxed(skill, ctx)

    def _run_builtin(self, skill: SkillDef, ctx: SkillContext) -> Dict[str, Any]:
        """直接 import 执行内置 Skill。"""
        try:
            result = skill.module.run(ctx)
            if result is None:
                result = {}
            if not isinstance(result, dict):
                raise SkillExecutionError(
                    skill.meta.name,
                    ValueError(f"run(ctx) must return dict, got {type(result).__name__}"),
                )
            return result
        except SkillError:
            raise
        except Exception as e:
            raise SkillExecutionError(skill.meta.name, e)

    def _run_user_sandboxed(self, skill: SkillDef, ctx: SkillContext) -> Dict[str, Any]:
        """在 subprocess 中沙箱执行用户上传的 Skill。

        步骤：
        1. 将 SkillContext 序列化为 JSON（只传路径和参数，不传大数组）
        2. 生成 wrapper 脚本，在子进程中 import 并执行 run(ctx)
        3. 设置超时，捕获 stdout 作为结果
        """
        main_path = os.path.join(skill.skill_dir, "main.py")
        if not os.path.isfile(main_path):
            raise SkillExecutionError(
                skill.meta.name,
                FileNotFoundError(f"main.py not found in {skill.skill_dir}"),
            )

        # 构建可序列化的上下文数据（传路径而非大数组，子进程自己加载）
        ctx_data = {
            "case_id": ctx.case_id,
            "ct_nifti_path": ctx.ct_nifti_path,
            "mask_dir": ctx.mask_dir,
            "output_dir": ctx.output_dir,
            "params": ctx.params,
            "mask_paths": {},
        }
        # 传入掩码路径列表（子进程自己 lazy load）
        try:
            ctx_data["mask_paths"] = {
                name: ctx.get_mask_path(name) for name in ctx.list_masks()
            }
        except Exception:
            pass

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))

        # 生成 wrapper 脚本
        wrapper = f"""import sys, json, resource

# 沙箱资源限制
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))        # CPU time: 120s
resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))         # max child processes
resource.setrlimit(resource.RLIMIT_NOFILE, (512, 512))      # max open files

# 重建上下文
ctx_data = {json.dumps(ctx_data)}

class _SandboxCtx:
    \"\"\"沙箱中的 SkillContext 轻量实现（按需从磁盘加载数据）。\"\"\"

    def __init__(self, data):
        self.case_id = data["case_id"]
        self.ct_nifti_path = data["ct_nifti_path"]
        self.mask_dir = data["mask_dir"]
        self.output_dir = data["output_dir"]
        self.params = data.get("params", {{}})
        self._mask_paths = data.get("mask_paths", {{}})
        self._mask_cache = {{}}
        self._ct_array = None
        self._affine = None
        self._spacing = None

    def get_ct_array(self):
        if self._ct_array is None:
            import nibabel as nib
            import numpy as np
            nii = nib.load(self.ct_nifti_path)
            self._ct_array = np.array(nii.get_fdata())
            self._affine = np.array(nii.affine)
        return self._ct_array

    def get_affine(self):
        if self._affine is None:
            self.get_ct_array()
        return self._affine

    def get_voxel_spacing(self):
        if self._spacing is None:
            import nibabel as nib
            sp = nib.affines.voxel_sizes(self.get_affine())
            self._spacing = (float(sp[0]), float(sp[1]), float(sp[2]))
        return self._spacing

    def list_masks(self):
        return list(self._mask_paths.keys())

    def get_mask_path(self, name):
        path = self._mask_paths.get(name)
        if path is None:
            raise FileNotFoundError(f"Mask '{{name}}' not found")
        return path

    def get_mask(self, name):
        if name in self._mask_cache:
            return self._mask_cache[name]
        import nibabel as nib
        import numpy as np
        nii = nib.load(self.get_mask_path(name))
        mask = np.array(nii.get_fdata()) > 0
        self._mask_cache[name] = mask
        return mask

    def get_output_dir(self):
        return self.output_dir

    def log(self, msg):
        print(f"[SKILL:{{self.case_id}}] {{msg}}", file=sys.stderr, flush=True)

# 执行用户 Skill
sys.path.insert(0, {json.dumps(skill.skill_dir)})
from main import run

ctx = _SandboxCtx(ctx_data)
result = run(ctx)

# 输出结果（JSON 到 stdout）
if result is None:
    result = {{}}
if not isinstance(result, dict):
    raise TypeError(f"run(ctx) must return dict, got {{type(result).__name__}}")

# 确保可 JSON 序列化
import json as _json
_json.dumps(result)  # 预序列化校验
print("__SKILL_RESULT__:" + _json.dumps(result))
"""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=_USER_SKILL_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise SkillExecutionError(
                skill.meta.name,
                TimeoutError(f"Skill timed out after {_USER_SKILL_TIMEOUT}s"),
            )

        # 解析 stdout
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        # 查找结果标记
        marker = "__SKILL_RESULT__:"
        if marker in stdout:
            json_str = stdout.split(marker, 1)[1].strip()
            try:
                result = json.loads(json_str)
                if not isinstance(result, dict):
                    raise ValueError("result must be a dict")
                return result
            except json.JSONDecodeError as e:
                raise SkillExecutionError(
                    skill.meta.name,
                    ValueError(f"Skill output is not valid JSON: {e}"),
                )

        # 没有标记 → 出错了
        error_msg = stderr.strip() or stdout.strip() or "Unknown error"
        raise SkillExecutionError(
            skill.meta.name,
            RuntimeError(f"Sandbox execution failed: {error_msg[:500]}"),
        )
