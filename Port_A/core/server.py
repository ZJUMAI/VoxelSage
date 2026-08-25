"""
Port A — Qwen + Port B Skills Agent

Pipeline
1. Frontend uploads files through POST /api/upload.
2. Frontend sends WebSocket event=initial with the user's question and uploaded paths.
3. Port A validates .nii/.nii.gz/.dcm or a DICOM folder and calls Port B /api/process-lite.
4. Port A loads Port B /api/skills/list and passes the returned `tools` directly to Qwen.
5. Qwen returns zero, one, or multiple tool calls.
6. Port A validates and executes /api/skills/run calls, in parallel within a round.
7. Port A writes skill results into tool_store, rebuilds context via build_messages(),
8.   and repeats until Qwen returns final text.
9. Final text and generated resources are streamed to the frontend.

Removed from the previous server:
- missing-information questions / followup workflow
- editable case_record workflow
- generate_final approval gate
- old all-in-one /api/process dependency

Session storage is in-memory. Replace with Redis/DB for production.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

try:
    from jsonschema import Draft7Validator
except ImportError:  # optional, but recommended
    Draft7Validator = None

# P0 优化模块导入
from core.medical_knowledge_base import MedicalKnowledgeBase
from core.tool_optimizer import ToolOptimizer
from core.reflection_module import ReflectionEngine


# ============================================================================
# Custom exceptions
# ============================================================================
class ClientDisconnectedError(Exception):
    """客户端 WebSocket 已断开，workflow 应尽早终止。"""
    pass


# ============================================================================
# Configuration
# ============================================================================
SERVER_PUBLIC_HOST = os.getenv("SERVER_PUBLIC_HOST", "localhost")
PORT_A_PORT = int(os.getenv("PORT_A_PORT", "8900"))

PORT_B_INTERNAL = os.getenv("PORT_B_INTERNAL", "http://localhost:8765")
PORT_B_PUBLIC = os.getenv("PORT_B_PUBLIC", f"http://{SERVER_PUBLIC_HOST}:8765")
PORT_B_PROCESS_LITE_PATH = "/api/process-lite"
PORT_B_SKILLS_LIST_PATH = "/api/skills/list"
PORT_B_SKILLS_RUN_PATH = "/api/skills/run"

CACHE_ROOT = os.getenv("CACHE_ROOT", "./qwen_case_cache")
UPLOAD_DIR = os.path.join(CACHE_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── 会话持久化到磁盘 + 上传文件联动 ──
SESSIONS_DIR = os.path.join(CACHE_ROOT, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

def _session_path(session_id: str) -> str:
    """每个 session 一个文件夹: sessions/{session_id}/session.json"""
    folder = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "session.json")

def save_session(session: Dict[str, Any]) -> None:
    """把 session 写到磁盘，与 uploaded files 路径对应。"""
    path = _session_path(session["session_id"])
    data = json_clone(session)
    data.pop("workflow_task", None)  # asyncio.Task 不可序列化
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_all_sessions() -> Dict[str, Dict[str, Any]]:
    """服务启动时从磁盘恢复所有历史 session。"""
    restored: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(SESSIONS_DIR):
        return restored
    for name in os.listdir(SESSIONS_DIR):
        sess_path = os.path.join(SESSIONS_DIR, name, "session.json")
        if os.path.isfile(sess_path):
            try:
                with open(sess_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session_id = data.get("session_id", name)
                data["workflow_task"] = None
                restored[session_id] = data
                print(f"[RESTORE] restored session {session_id}")
            except Exception as e:
                print(f"[RESTORE] failed to restore {sess_path}: {e}")
    return restored

ALLOWED_DATA_ROOTS = [
    os.path.realpath(UPLOAD_DIR),
    os.path.realpath(os.getenv("DATA_ROOT", "/data")),
]

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://your-llm-endpoint.example.com/v1",
)
LLM_MODEL_NAME = (
    os.getenv("LLM_MODEL_NAME", "").strip()
    or os.getenv("QWEN_MODEL_NAME", "").strip()  # backward compatibility
)


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return not normalized or "your-" in normalized or "example.com" in normalized


def llm_configuration_errors() -> List[str]:
    errors = []
    if _is_placeholder(DASHSCOPE_API_KEY):
        errors.append("DASHSCOPE_API_KEY")
    if _is_placeholder(DASHSCOPE_BASE_URL):
        errors.append("DASHSCOPE_BASE_URL")
    if _is_placeholder(LLM_MODEL_NAME):
        errors.append("LLM_MODEL_NAME")
    return errors


def require_llm_configuration() -> None:
    missing = llm_configuration_errors()
    if missing:
        raise RuntimeError(
            "LLM configuration is incomplete. Set " + ", ".join(missing) + " in .env."
        )

PORT_B_PROCESS_TIMEOUT = float(os.getenv("PORT_B_PROCESS_TIMEOUT", "1200"))
PORT_B_SKILL_TIMEOUT = float(os.getenv("PORT_B_SKILL_TIMEOUT", "600"))  # 增加到600秒，因为liver_analysis可能需要230-290秒
PORT_B_LIST_TIMEOUT = float(os.getenv("PORT_B_LIST_TIMEOUT", "30"))

MAX_AGENT_ROUNDS = int(os.getenv("MAX_AGENT_ROUNDS", "10"))
MAX_SKILL_CALLS_PER_TURN = int(os.getenv("MAX_SKILL_CALLS_PER_TURN", "12"))
MAX_PARALLEL_SKILLS = int(os.getenv("MAX_PARALLEL_SKILLS", "3"))
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "30000"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "4"))
SKILLS_CACHE_TTL_SECONDS = float(os.getenv("SKILLS_CACHE_TTL_SECONDS", "60"))

if missing_llm_config := llm_configuration_errors():
    print(
        "[WARN] LLM configuration is incomplete: "
        + ", ".join(missing_llm_config)
        + ". Model calls will fail until .env is updated."
    )

llm_client = AsyncOpenAI(api_key=DASHSCOPE_API_KEY or "missing", base_url=DASHSCOPE_BASE_URL)


# ============================================================================
# FastAPI and state
# ============================================================================
app = FastAPI(title="Port A - Qwen Medical Skills Agent", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_STORE: Dict[str, Dict[str, Any]] = load_all_sessions()
print(f"[INIT] restored {len(SESSION_STORE)} historical sessions from disk")
WS_SEND_LOCKS: Dict[int, asyncio.Lock] = {}
SKILLS_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "payload": None}


# ============================================================================
# Logging and general helpers
# ============================================================================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(tag: str, message: str, **fields: Any) -> None:
    suffix = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    print(f"[{now_iso()}][{tag}] {message}" + (f" {suffix}" if suffix else ""))


def pydantic_dump(model: BaseModel) -> Dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def create_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"


def create_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def create_empty_session(session_id: str) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "workflow_state": "created",
        "status": "created",
        "original_question": "",
        "current_user_query": "",
        "conversation": [],       # 只存 system/user/assistant(final)
        "tool_store": {},         # tool_store[case_id][skill_name] = result
        "images": [],
        "medical_volumes": [],
        "case_ids": [],
        "segmentation_status": "pending",
        "process_lite_result": None,
        "available_skills": [],
        "available_tools": [],
        "agent_round": 0,
        "skill_call_history": [],
        "skill_result_cache": {},
        "final_answer": None,
        "errors": [],
        "outputs": {"html_url": None, "best_slices": []},
        "cancel_requested": False,
        "workflow_task": None,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }


# ============================================================================
# Frontend protocol
# ============================================================================
VolumeRole = Literal["raw_volume", "mask_volume", "enhanced_volume", "unknown"]


class ImageFile(BaseModel):
    file_id: str
    path: str
    name: Optional[str] = None


class MedicalVolumeFile(BaseModel):
    file_id: str
    path: str
    name: Optional[str] = None
    volume_role: VolumeRole = "unknown"


class FrontendInputs(BaseModel):
    images: List[ImageFile] = Field(default_factory=list)
    medical_volumes: List[MedicalVolumeFile] = Field(default_factory=list)


class FrontendOptions(BaseModel):
    device: str = "cuda:0"
    max_new_tokens: int = Field(default=4096, ge=256, le=16384)
    max_agent_rounds: int = Field(default=MAX_AGENT_ROUNDS, ge=1, le=12)


class WSInitialPayload(BaseModel):
    event: Literal["initial"]
    session_id: Optional[str] = None
    user_message: str = Field(min_length=1)
    inputs: FrontendInputs = Field(default_factory=FrontendInputs)
    options: FrontendOptions = Field(default_factory=FrontendOptions)


class WSChatMessagePayload(BaseModel):
    event: Literal["chat_message"]
    session_id: str
    user_message: str = Field(min_length=1)
    images: List[ImageFile] = Field(default_factory=list)
    medical_volumes: List[MedicalVolumeFile] = Field(default_factory=list)
    options: FrontendOptions = Field(default_factory=FrontendOptions)


class WSCancelPayload(BaseModel):
    event: Literal["cancel"]
    session_id: str


class FileUploadItem(BaseModel):
    file_id: str
    path: str
    name: str
    file_type: Literal["image", "medical_volume"]
    relative_path: Optional[str] = None


class FileUploadResponse(BaseModel):
    status: Literal["ok"] = "ok"
    files: List[FileUploadItem]
    session_id: Optional[str] = None
    message: str


# ============================================================================
# Port B response contracts
# ============================================================================
class ProcessLiteResponse(BaseModel):
    status: str
    case_id: str
    ct_nifti_path: str
    mask_dir: str
    output_dir: str
    mask_files: Dict[str, str] = Field(default_factory=dict)
    elapsed_seconds: Optional[float] = None


class SkillsListResponse(BaseModel):
    status: str
    skills: List[Dict[str, Any]] = Field(default_factory=list)
    tools: List[Dict[str, Any]] = Field(default_factory=list)


class SkillRunResponse(BaseModel):
    status: str
    result: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    message: Optional[str] = None
    error_code: Optional[str] = None
    retryable: Optional[bool] = None


# ============================================================================
# WebSocket utilities
# ============================================================================
async def ws_send(websocket: WebSocket, event: str, **payload: Any) -> None:
    key = id(websocket)
    lock = WS_SEND_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        try:
            await websocket.send_json({"event": event, **payload})
        except WebSocketDisconnect:
            # 标记 session 已断开，让 agent loop 尽早终止
            sid = payload.get("session_id")
            if sid and sid in SESSION_STORE:
                SESSION_STORE[sid]["_ws_disconnected"] = True
            log("WS", "client_disconnected_during_send", session_id=sid, event=event)
            return  # client disconnected, suppress
    if event != "answer_delta":
        log("WS->", event, session_id=payload.get("session_id"), stage=payload.get("stage"))


async def ws_progress(websocket: WebSocket, session_id: str, stage: str, message: str) -> None:
    await ws_send(websocket, "progress", session_id=session_id, stage=stage, message=message)


async def ws_error(
    websocket: WebSocket,
    session_id: Optional[str],
    stage: str,
    message: str,
    detail: Any = None,
) -> None:
    await ws_send(
        websocket,
        "error",
        session_id=session_id,
        stage=stage,
        message=message,
        detail=detail,
    )


def check_ws_connected(session: Dict[str, Any]) -> None:
    """检查 WebSocket 是否仍连接。若已断开则抛出 ClientDisconnectedError，
    让 agent loop 尽早终止，避免继续浪费 LLM 调用和算力。"""
    if session.get("_ws_disconnected"):
        raise ClientDisconnectedError(
            f"WebSocket disconnected for session {session.get('session_id', 'unknown')}"
        )


# ============================================================================
# File safety and upload
# ============================================================================
def is_allowed_path(path: str) -> bool:
    if not path:
        return False
    real = os.path.realpath(path)
    for root in ALLOWED_DATA_ROOTS:
        try:
            if os.path.commonpath([real, root]) == root:
                return True
        except ValueError:
            pass
    return False


def is_image_file(path: str) -> bool:
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))


def directory_contains_dicom(path: str, max_depth: int = 8) -> bool:
    if not os.path.isdir(path):
        return False
    base_depth = Path(path).resolve().parts
    try:
        for root, dirs, files in os.walk(path):
            depth = len(Path(root).resolve().parts) - len(base_depth)
            if depth >= max_depth:
                dirs.clear()
            if any(name.lower().endswith(".dcm") for name in files):
                return True
    except (PermissionError, OSError):
        return False
    return False


def is_medical_volume(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".nii", ".nii.gz", ".dcm")) or directory_contains_dicom(path)


def safe_relative_path(raw: str) -> str:
    raw = (raw or "unnamed").replace("\\", "/").lstrip("/")
    parts = [part for part in raw.split("/") if part not in ("", ".", "..")]
    if not parts:
        return f"unnamed_{uuid.uuid4().hex[:8]}"
    return "/".join(parts)


def safe_extract_zip(zip_path: str, target_dir: str) -> None:
    target = Path(target_dir).resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            destination = (target / member.filename).resolve()
            if destination != target and target not in destination.parents:
                raise ValueError(f"Unsafe zip member: {member.filename}")
        zf.extractall(target)


@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    folder_id: Optional[str] = Form(None),
) -> FileUploadResponse:
    """Upload images, NIfTI, DICOM files, DICOM folders, or DICOM zip archives.

    For browser folder upload, append every File with `webkitRelativePath` as the multipart filename.
    The server preserves relative paths, preventing duplicate DICOM filenames from overwriting each other.
    """
    batch_id = safe_relative_path(folder_id or uuid.uuid4().hex[:12]).replace("/", "_")
    save_dir = os.path.realpath(os.path.join(UPLOAD_DIR, batch_id))
    os.makedirs(save_dir, exist_ok=True)
    items: List[FileUploadItem] = []

    for upload in files:
        original_name = upload.filename or f"unnamed_{uuid.uuid4().hex[:8]}"
        relative = safe_relative_path(original_name)
        lower = relative.lower()
        destination = os.path.realpath(os.path.join(save_dir, relative))
        if os.path.commonpath([destination, save_dir]) != save_dir:
            raise HTTPException(status_code=400, detail="Unsafe upload path")
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        if lower.endswith(".zip"):
            with open(destination, "wb") as f:
                while chunk := await upload.read(1024 * 1024):
                    f.write(chunk)
            extract_dir = destination[:-4] + "_dicom"
            os.makedirs(extract_dir, exist_ok=True)
            try:
                safe_extract_zip(destination, extract_dir)
            except (zipfile.BadZipFile, ValueError) as exc:
                shutil.rmtree(extract_dir, ignore_errors=True)
                os.remove(destination)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            os.remove(destination)
            if directory_contains_dicom(extract_dir):
                items.append(FileUploadItem(
                    file_id=f"vol_{uuid.uuid4().hex[:8]}",
                    path=os.path.realpath(extract_dir),
                    name=os.path.basename(extract_dir),
                    file_type="medical_volume",
                    relative_path=relative,
                ))
            continue

        supported = lower.endswith((".nii", ".nii.gz", ".dcm", ".png", ".jpg", ".jpeg", ".webp"))
        if not supported:
            continue

        with open(destination, "wb") as f:
            while chunk := await upload.read(1024 * 1024):
                f.write(chunk)

        file_type: Literal["image", "medical_volume"] = (
            "image" if is_image_file(destination) else "medical_volume"
        )
        items.append(FileUploadItem(
            file_id=f"{'img' if file_type == 'image' else 'vol'}_{uuid.uuid4().hex[:8]}",
            path=destination,
            name=os.path.basename(relative),
            file_type=file_type,
            relative_path=relative,
        ))

    # If a browser uploaded a raw DICOM folder as many .dcm files, expose one folder volume.
    dcm_items = [item for item in items if item.path.lower().endswith(".dcm")]
    if dcm_items:
        # Find their shared root under this batch. Prefer first path's top-level folder.
        rel0 = dcm_items[0].relative_path or dcm_items[0].name
        top = rel0.split("/", 1)[0] if "/" in rel0 else ""
        folder_path = os.path.join(save_dir, top) if top else save_dir
        if directory_contains_dicom(folder_path):
            items = [item for item in items if not item.path.lower().endswith(".dcm")]
            items.append(FileUploadItem(
                file_id=f"vol_{uuid.uuid4().hex[:8]}",
                path=os.path.realpath(folder_path),
                name=top or f"dicom_{batch_id}",
                file_type="medical_volume",
                relative_path=top or None,
            ))

    if not items:
        raise HTTPException(
            status_code=400,
            detail="No supported files. Use nii/nii.gz/dcm, a DICOM folder/zip, or png/jpg/jpeg/webp.",
        )

    return FileUploadResponse(
        files=items,
        session_id=session_id,
        message=f"Uploaded {len(items)} input item(s).",
    )


# ============================================================================
# Input validation and session merge
# ============================================================================
def merge_initial(session: Dict[str, Any], payload: WSInitialPayload) -> None:
    clean_text, skills = parse_skill_hints(payload.user_message.strip())
    session["original_question"] = clean_text or payload.user_message.strip()
    session["current_user_query"] = clean_text or payload.user_message.strip()
    session["required_skills"] = skills
    session["images"] = [pydantic_dump(x) for x in payload.inputs.images]
    session["medical_volumes"] = [pydantic_dump(x) for x in payload.inputs.medical_volumes]
    # user message 不立即写入 conversation,等 agent loop 完成后与 answer 一并写入
    session["updated_at"] = now_iso()


def validate_session_inputs(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    if not session.get("current_user_query", "").strip():
        errors.append({"field": "user_message", "detail": "User question is required."})

    for item in session.get("images", []):
        path = item.get("path", "")
        if not os.path.isfile(path):
            errors.append({"file_id": item.get("file_id"), "path": path, "detail": "Image does not exist."})
        elif not is_allowed_path(path):
            errors.append({"file_id": item.get("file_id"), "path": path, "detail": "Image is outside allowed roots."})
        elif not is_image_file(path):
            errors.append({"file_id": item.get("file_id"), "path": path, "detail": "Unsupported image format."})

    for item in session.get("medical_volumes", []):
        path = item.get("path", "")
        if not (os.path.isfile(path) or os.path.isdir(path)):
            errors.append({"file_id": item.get("file_id"), "path": path, "detail": "Medical input does not exist."})
        elif not is_allowed_path(path):
            errors.append({"file_id": item.get("file_id"), "path": path, "detail": "Medical input is outside allowed roots."})
        elif not is_medical_volume(path):
            errors.append({"file_id": item.get("file_id"), "path": path, "detail": "Expected nii/nii.gz/dcm or DICOM folder."})
    return errors


def pick_primary_medical_volume(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    volumes = session.get("medical_volumes", [])
    for role in ("raw_volume", "enhanced_volume", "unknown"):
        for volume in volumes:
            if volume.get("volume_role", "unknown") == role and is_medical_volume(volume.get("path", "")):
                return volume
    return volumes[0] if volumes else None


def infer_case_name(path: str) -> str:
    name = os.path.basename(path.rstrip(os.sep)) or f"case_{uuid.uuid4().hex[:8]}"
    return re.sub(r"\.(nii\.gz|nii|dcm)$", "", name, flags=re.IGNORECASE)


def parse_skill_hints(text: str) -> Tuple[str, List[str]]:
    """从用户文本中提取 /+skill_name,返回(清理后的文本,技能名称列表)。"""
    if not text:
        return text, []
    skills: List[str] = []
    def _replacer(m: re.Match) -> str:
        skills.append(m.group(1))
        return ""
    cleaned = re.sub(r'/\+\s*([A-Za-z_][A-Za-z0-9_-]*)', _replacer, text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned, skills


# ============================================================================
# Port B client and strict communication contracts
# ============================================================================
async def port_b_process_lite(input_path: str, case_name: str, device: str) -> Dict[str, Any]:
    payload = {"input": input_path, "case_name": case_name, "device": device}
    async with httpx.AsyncClient(timeout=PORT_B_PROCESS_TIMEOUT) as client:
        response = await client.post(f"{PORT_B_INTERNAL}{PORT_B_PROCESS_LITE_PATH}", json=payload)
        if response.status_code >= 400:
            try:
                failure = response.json()
            except ValueError:
                failure = response.text[:2000]
            raise RuntimeError(
                f"Port B process-lite HTTP {response.status_code}: {failure}"
            )
        raw = response.json()
    parsed = ProcessLiteResponse(**raw)
    if parsed.status != "ok":
        raise RuntimeError(f"Port B process-lite returned status={parsed.status}: {raw}")
    return pydantic_dump(parsed)


async def port_b_list_skills(force_refresh: bool = False) -> Dict[str, Any]:
    now = time.monotonic()
    cached = SKILLS_CACHE.get("payload")
    if not force_refresh and cached and now - SKILLS_CACHE["loaded_at"] < SKILLS_CACHE_TTL_SECONDS:
        return json_clone(cached)

    async with httpx.AsyncClient(timeout=PORT_B_LIST_TIMEOUT) as client:
        response = await client.get(f"{PORT_B_INTERNAL}{PORT_B_SKILLS_LIST_PATH}")
        response.raise_for_status()
        raw = response.json()
    parsed = SkillsListResponse(**raw)
    if parsed.status != "ok" or not parsed.tools:
        raise RuntimeError(f"Invalid skills/list response: {raw}")
    validate_tools_contract(parsed.tools)
    payload = pydantic_dump(parsed)
    SKILLS_CACHE.update({"loaded_at": now, "payload": payload})
    return json_clone(payload)


async def port_b_run_skill(case_id: str, skill_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"skill_name": skill_name, "case_id": case_id, "params": params}
    async with httpx.AsyncClient(timeout=PORT_B_SKILL_TIMEOUT) as client:
        response = await client.post(f"{PORT_B_INTERNAL}{PORT_B_SKILLS_RUN_PATH}", json=payload)
        raw_text = response.text
        try:
            raw = response.json()
        except Exception as exc:
            raise RuntimeError(f"Port B returned non-JSON: {raw_text[:1000]}") from exc
        if response.status_code >= 400:
            return {
                "status": "error",
                "error_code": f"PORT_B_HTTP_{response.status_code}",
                "message": raw.get("detail") if isinstance(raw, dict) else raw_text[:1000],
                "retryable": response.status_code >= 500,
            }
    try:
        parsed = SkillRunResponse(**raw)
        return pydantic_dump(parsed)
    except ValidationError as exc:
        return {
            "status": "error",
            "error_code": "INVALID_PORT_B_SKILL_RESPONSE",
            "message": str(exc),
            "raw": raw,
            "retryable": False,
        }


def validate_tools_contract(tools: List[Dict[str, Any]]) -> None:
    names: set[str] = set()
    for index, tool in enumerate(tools):
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError(f"tools[{index}] is not OpenAI function-calling format")
        fn = tool["function"]
        name = fn.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
            raise ValueError(f"Invalid tool name at index {index}: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate tool name: {name}")
        names.add(name)
        parameters = fn.get("parameters", {})
        if not isinstance(parameters, dict) or parameters.get("type", "object") != "object":
            raise ValueError(f"Tool {name} parameters must be an object JSON Schema")


# ============================================================================
# Tool validation, cache, execution, and output normalization
# ============================================================================
def inject_target_case_id(tools: List[Dict[str, Any]], case_ids: List[str]) -> List[Dict[str, Any]]:
    """在所有工具参数末尾注入 target_case_id,让 LLM 可以选择对哪个 case 执行。"""
    if not case_ids:
        return tools
    target_param = {
        "type": "string",
        "description": "要执行此技能的病例 ID",
        "enum": case_ids,
    }
    out = []
    for tool in tools:
        t = json_clone(tool)
        props = t.setdefault("function", {}).setdefault("parameters", {}).setdefault("properties", {})
        props["target_case_id"] = target_param
        # 不加入 required，让 LLM 按需选择
        out.append(t)
    return out


def apply_tool_optimization(session: Dict[str, Any], available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """P0 优化：应用工具选择优化器，过滤冗余工具"""
    # 使用工具优化器过滤冗余工具
    filtered_tools, filter_reasons = ToolOptimizer.filter_redundant_tools(
        session,
        available_tools,
        case_id=None  # 检查所有 case
    )

    # 记录过滤信息
    if filter_reasons:
        summary = ToolOptimizer.generate_filter_summary(filter_reasons)
        log("TOOL_OPTIMIZER", f"过滤了 {len(filter_reasons)} 个冗余工具",
            session_id=session["session_id"],
            filtered_count=len(filter_reasons))
        session.setdefault("_tool_filter_log", []).append({
            "time": now_iso(),
            "filtered_count": len(filter_reasons),
            "reasons": filter_reasons
        })

    return filtered_tools

def available_tool_map(session: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {tool["function"]["name"]: tool for tool in session.get("available_tools", [])}


def parse_tool_arguments(raw: Any) -> Dict[str, Any]:
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Tool arguments must decode to a JSON object")
        return value
    raise ValueError("Unsupported tool arguments type")


def validate_tool_arguments(tool: Dict[str, Any], arguments: Dict[str, Any]) -> None:
    forbidden = {"ct_nifti_path", "mask_dir", "output_dir", "case_id", "input"}
    illegal = forbidden.intersection(arguments)
    if illegal:
        raise ValueError(f"Context parameters cannot be model-supplied: {sorted(illegal)}")
    schema = tool.get("function", {}).get("parameters", {"type": "object"})
    if Draft7Validator is not None:
        errors = sorted(Draft7Validator(schema).iter_errors(arguments), key=lambda e: list(e.path))
        if errors:
            raise ValueError("; ".join(error.message for error in errors[:5]))


def build_cache_key(case_id: str, skill_name: str, params: Dict[str, Any]) -> str:
    body = json.dumps(
        {"case_id": case_id, "skill_name": skill_name, "params": params},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def normalize_public_url(path_or_url: Optional[str]) -> Optional[str]:
    if not path_or_url:
        return None
    value = str(path_or_url).strip()
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("/output/") or value.startswith("/process-output/"):
        return f"{PORT_B_PUBLIC}{value}"
    if "/process_output/" in value:
        suffix = value.split("/process_output/", 1)[1]
        return f"{PORT_B_PUBLIC}/process-output/{suffix}"
    if "/output/" in value:
        suffix = value.split("/output/", 1)[1]
        return f"{PORT_B_PUBLIC}/output/{suffix}"
    return value


def normalize_result_paths(obj: Any) -> Any:
    """Recursively traverse tool result and convert all server paths to full URLs."""
    if isinstance(obj, str):
        result = normalize_public_url(obj)
        return result if result is not None else obj
    elif isinstance(obj, dict):
        return {k: normalize_result_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_result_paths(item) for item in obj]
    return obj


def summarize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "call_id": call["id"],
            "skill_name": call["function"]["name"],
            "params": parse_tool_arguments(call["function"].get("arguments", "{}")),
        }
        for call in tool_calls
    ]


# ============================================================================
# Conversation + Tool Store: 上下文管理层
# ============================================================================
def append_conversation(session: Dict[str, Any], role: str, content: str) -> None:
    """统一追加对话记录。只允许 user / assistant(final) 进入 conversation。"""
    if role not in ("user", "assistant"):
        return
    if not content:
        return
    # active_volumes 由调用方在 run_agent_loop 前设置，记录本轮前端传了哪些文件
    active = session.get("_active_volumes")
    active_cids = session.get("_active_case_ids", [])
    cids = session.get("case_ids", [])
    session["conversation"].append({
        "role": role,
        "content": content,
        "case_id": active_cids[0] if active_cids else (cids[-1] if cids else None),
        "case_ids": cids,
        "volume_name": _current_volume_name(session),
        "active_volumes": active,
        "time": now_iso(),
    })
    session["updated_at"] = now_iso()


def _current_volume_name(session: Dict[str, Any]) -> Optional[str]:
    """取当前活跃 case 对应的文件名，用于标注对话所属文件。"""
    active = session.get("_active_case_ids", [])
    case_id = active[0] if active else session.get("case_ids", [None])[-1]
    if not case_id:
        return None
    # 从 tool_store 的分割结果反查文件名
    case_data = session.get("tool_store", {}).get(case_id, {})
    seg = case_data.get("segmentation", {})
    path = seg.get("ct_nifti_path", "")
    if path:
        return os.path.basename(path)
    # 兜底：从 medical_volumes 里找
    for vol in session.get("medical_volumes", []):
        if vol.get("file_id") and vol.get("name"):
            return vol["name"]
    return None


def _resolve_active_case_ids(session: Dict[str, Any], payload_volumes: List[Dict[str, Any]]) -> List[str]:
    """从前端传来的 payload medical_volumes 中提取对应的 case_id 列表。"""
    tool_store = session.get("tool_store", {})
    if not tool_store:
        return []
    # 收集 payload 中的文件路径（去重）
    payload_paths = set()
    for vol in payload_volumes:
        p = vol.get("path") or vol.get("ct_nifti_path")
        if p:
            payload_paths.add(os.path.realpath(p))
    if not payload_paths:
        return list(tool_store.keys())  # 兜底：全部
    # 反向匹配：遍历 tool_store 中每个 case 的 segmentation.ct_nifti_path
    matched = []
    for cid, case_data in tool_store.items():
        seg = case_data.get("segmentation", {})
        sp = seg.get("ct_nifti_path", "")
        if sp and os.path.realpath(sp) in payload_paths:
            matched.append(cid)
    return matched or list(tool_store.keys())

def _case_volume_name(tool_store: Dict[str, Any], case_id: str) -> str:
    """从 tool_store 的分割结果反查文件名。"""
    case_data = tool_store.get(case_id, {})
    seg = case_data.get("segmentation", {})
    path = seg.get("ct_nifti_path", "") or seg.get("input_path", "")
    if path:
        return os.path.basename(path)
    return case_id


def format_tool_context(session: Dict[str, Any]) -> str:
    """读取 tool_store,整理成模型易读的文本（作为 system prompt 的一部分）。

    只展示本轮 _active_case_ids 中的 case。
    P0 优化：增加医学知识验证和临床建议
    """
    store = session.get("tool_store", {})
    active = session.get("_active_case_ids") or list(store.keys())
    if not store:
        return ""

    sections: list[str] = []
    sections.append("=" * 60)
    sections.append("【已有分析结果】以下数据已计算完成，直接使用，勿重复调用")
    sections.append("=" * 60)

    for case_id in active:
        case_data = store.get(case_id)
        if not isinstance(case_data, dict):
            continue

        vname = _case_volume_name(store, case_id)
        sections.append(f"\n病例: {case_id} / {vname}")
        sections.append("-" * 60)

        # liver_analysis 结果特殊处理 - 结构化展示
        if "liver_analysis" in case_data:
            result = case_data["liver_analysis"]

            # 检查是否是错误结果
            if result.get("_error"):
                sections.append("✗ liver_analysis 执行失败")
                sections.append(f"  错误: {result.get('message', 'Unknown error')}")
                sections.append("")
                continue

            sections.append("✓ liver_analysis 已完成:")

            # 肝脏体积 + P0 优化：医学验证
            if "liver_volume_cm3" in result:
                vol = result["liver_volume_cm3"]
                if isinstance(vol, (int, float)):
                    validation = MedicalKnowledgeBase.validate_measurement("liver_volume", vol)
                    status_icon = "⚠️" if validation.get("warning") else "✓"
                    sections.append(f"  {status_icon} 肝脏体积: {vol:.2f} cm³")
                    if validation.get("warning"):
                        sections.append(f"    [{validation['severity'].upper()}] {validation['warning']}")

            # 血管体积
            vessels = result.get("vessel_volumes", {})
            if vessels and isinstance(vessels, dict):
                sections.append(f"  • 血管:")
                for v_name, v_data in vessels.items():
                    if isinstance(v_data, dict) and "volume_cm3" in v_data:
                        vol = v_data["volume_cm3"]
                        if isinstance(vol, (int, float)):
                            sections.append(f"    - {v_name}: {vol:.2f} cm³")

            # 肿瘤结果 - 重点突出 + P0 优化：医学验证
            tumors = result.get("tumor_results", {})
            if isinstance(tumors, dict):
                num_tumors = len(tumors)
                sections.append(f"  • 肿瘤数量: {num_tumors}")

                if tumors:
                    sections.append(f"  • 肿瘤详情:")
                    for t_name, t_data in list(tumors.items())[:10]:  # 最多显示10个
                        if isinstance(t_data, dict):
                            diam = t_data.get("max_diameter_mm")
                            vol = t_data.get("volume_cm3")
                            parts = [f"{t_name}:"]
                            if isinstance(diam, (int, float)):
                                parts.append(f"{diam:.1f}mm")
                            if isinstance(vol, (int, float)):
                                parts.append(f"{vol:.2f}cm³")
                            sections.append(f"    - {' '.join(parts)}")

                    if len(tumors) > 10:
                        sections.append(f"    ... 还有 {len(tumors) - 10} 个肿瘤")

            # P0 优化：添加结果一致性检查
            consistency = MedicalKnowledgeBase.validate_result_consistency(result)
            if not consistency["consistent"]:
                sections.append(f"  ⚠️ 数据一致性警告:")
                for issue in consistency["issues"][:3]:  # 最多显示3个
                    sections.append(f"    - {issue['message']}")

            sections.append("")

        # 其他技能结果 - 结构化展示，让 LLM 可以直接看到数据
        for skill_name, result in case_data.items():
            if skill_name in ("liver_analysis", "segmentation"):
                continue
            if isinstance(result, dict):
                if result.get("_error"):
                    sections.append(f"✗ {skill_name} 执行失败")
                    sections.append("")
                    continue

                sections.append(f"✓ {skill_name} 已完成:")
                lines = _format_result_items(result, indent=4)
                if lines:
                    sections.extend(lines)
                else:
                    # 上文已经展示完（如 plan_resection 在 _format_plan_resection_result
                    # 中已经写入 sections），避免再输出无意义的空 ✓ 行
                    pass
            sections.append("")

    if sections[-1] == "":
        sections.pop()

    # P0 优化：添加临床建议部分
    for case_id in active:
        clinical_section = MedicalKnowledgeBase.generate_clinical_report_section(store, case_id)
        if clinical_section and "暂无特殊临床提示" not in clinical_section:
            sections.append("\n" + clinical_section)

    sections.append("\n" + "=" * 60)
    sections.append("⚠️ 回答问题时：")
    sections.append("  1. 直接从上述数据中提取答案")
    sections.append("  2. 病灶计数 = 肿瘤数量")
    sections.append("  3. 不要重复调用已完成的技能")
    sections.append("  4. 注意数据验证警告，必要时提醒用户")
    sections.append("=" * 60)

    return "\n".join(sections)


def _format_result_items(
    data: Dict[str, Any],
    indent: int = 2,
    max_str_len: int = 200,
    max_list_preview: int = 6,
) -> list[str]:
    """递归格式化 tool result，尽可能保留有用数据，不做过分过滤。"""
    lines: list[str] = []
    prefix = " " * indent
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, (int, float, bool)):
            lines.append(f"{prefix}{k}: {v}")
        elif isinstance(v, str):
            if not v:
                continue
            # 路径类: 只取文件名
            if v.startswith("/") or v.startswith("data:"):
                fname = v.rstrip("/").split("/")[-1]
                lines.append(f"{prefix}{k}: {fname}")
            elif len(v) > max_str_len:
                lines.append(f"{prefix}{k}: {v[:max_str_len]}...")
            else:
                lines.append(f"{prefix}{k}: {v}")
        elif isinstance(v, dict):
            sub = _format_result_items(v, indent=indent + 2, max_str_len=max_str_len, max_list_preview=max_list_preview)
            if sub:
                lines.append(f"{prefix}{k}:")
                lines.extend(sub)
            else:
                # 空 dict: 还是记一笔，表示这个 key 存在
                lines.append(f"{prefix}{k}: (empty)")
        elif isinstance(v, (list, tuple)):
            if not v:
                continue
            # 纯数值/短字符串列表: 内联显示
            if all(isinstance(x, (int, float, bool, str)) for x in v):
                preview = [str(x) for x in v[:max_list_preview]]
                suffix = f" ... (共 {len(v)} 项)" if len(v) > max_list_preview else ""
                lines.append(f"{prefix}{k}: {', '.join(preview)}{suffix}")
            else:
                # 对象列表: 逐个展开
                lines.append(f"{prefix}{k}:")
                for i, item in enumerate(v[:max_list_preview]):
                    if isinstance(item, dict):
                        sub = _format_result_items(item, indent=indent + 2)
                        if sub:
                            lines.append(f"{prefix}  [{i}]:")
                            lines.extend(sub)
                        else:
                            lines.append(f"{prefix}  [{i}]: {item}")
                    else:
                        lines.append(f"{prefix}  [{i}]: {item}")
                if len(v) > max_list_preview:
                    lines.append(f"{prefix}  ... (共 {len(v)} 项)")
    return lines


def merge_skill_outputs(session: Dict[str, Any], skill_name: str, result: Dict[str, Any]) -> None:
    """从 skill 结果中提取可视化产物 URL,合并到 session.outputs。"""
    if not isinstance(result, dict):
        return
    outputs = session.setdefault("outputs", {"html_url": None, "best_slices": []})
    # 提取 3D HTML 链接
    if result.get("html_url") and not outputs.get("html_url"):
        outputs["html_url"] = normalize_public_url(result["html_url"])
    # 合并最佳切片（去重）
    new_slices = result.get("best_slices") or result.get("slices") or []
    existing_paths = {s.get("path") for s in outputs.get("best_slices", []) if s.get("path")}
    for s in new_slices:
        if isinstance(s, dict) and s.get("path") not in existing_paths:
            s_copy = dict(s)
            if s_copy.get("path"):
                s_copy["url"] = normalize_public_url(s_copy.get("url") or s_copy.get("path"))
            outputs.setdefault("best_slices", []).append(s_copy)
            if s_copy.get("path"):
                existing_paths.add(s_copy["path"])


def build_current_user_content(session: Dict[str, Any]) -> Any:
    """构建当前用户消息内容（文本 + 可选图片）。

    返回 str（纯文本）或 list[dict]（多模态）。
    """
    query = session.get("current_user_query", "")
    # 追加用户指定的必须调用的 skills 指令
    required = session.get("required_skills", [])
    if required:
        query += (
            f"\n\n【用户指定必须调用】{', '.join(required)}。"
            f"请根据需要决定调用顺序，有依赖关系的技能分步调用。"
        )
    images = session.get("images", [])

    # 无图片 -> 纯文本
    if not images:
        return query or "请分析该病例。"

    # 有图片 -> 多模态内容
    content: list[dict] = [{"type": "text", "text": query or "请分析该病例。"}]
    count = 0
    for image in images:
        if count >= MAX_IMAGES_PER_REQUEST:
            break
        data_url = image_to_data_url(image.get("path", ""))
        if data_url:
            content.append({"type": "image_url", "image_url": {"url": data_url}})
            count += 1
    return content


def build_messages(session: Dict[str, Any]) -> list[Dict[str, Any]]:
    """统一构造发送给模型的 messages。

    结构:
        system (稳定 system prompt)
        system (tool context,动态变化)
        ↓
        conversation (user/assistant 历史)
        ↓
        current user (当前问题 + 图片,始终在最末)

    稳定内容在前 -> 自动命中 Prompt Cache。
    """
    messages: list[dict] = [{"role": "system", "content": build_system_prompt(session)}]

    # Tool context 作为独立 system message（每轮可能变化,但不影响前面 system 的缓存）
    tool_context = format_tool_context(session)
    if tool_context:
        messages.append({"role": "system", "content": tool_context})

    # Conversation history（只含已完成的 user/assistant 交换，标注所属文件）
    for entry in session.get("conversation", []):
        tag = ""
        cid = entry.get("case_id")
        vname = entry.get("volume_name")
        active = entry.get("active_volumes")
        if cid and vname:
            tag = f"[病例 {cid} / {vname}] "
        elif cid:
            tag = f"[病例 {cid}] "
        if active:
            tag += f"[文件: {', '.join(active)}] "
        content = tag + entry["content"] if tag else entry["content"]
        messages.append({"role": entry["role"], "content": content})

    # Current user question（始终最后,支持图片）
    messages.append({"role": "user", "content": build_current_user_content(session)})

    return messages


async def execute_one_tool_call(
    websocket: WebSocket,
    session: Dict[str, Any],
    round_index: int,
    call: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    call_id = call.get("id") or create_call_id()
    skill_name = call.get("function", {}).get("name", "")
    history_entry: Dict[str, Any] = {
        "call_id": call_id,
        "round": round_index,
        "skill_name": skill_name,
        "params": {},
        "status": "error",
        "result": {},
        "execution_time_ms": None,
        "cache_hit": False,
        "started_at": now_iso(),
        "finished_at": None,
    }

    # 预先取 tcid（fallback 到 case_ids[0]），保证异常分支也能用
    _pre_params = parse_tool_arguments(call.get("function", {}).get("arguments", "{}"))
    _pre_tcid = _pre_params.get("target_case_id") or (session.get("case_ids") or [None])[0]
    tcid = _pre_tcid or (session.get("case_ids") or [None])[0]

    try:
        # 如果客户端已断开，跳过 Port B 调用
        if session.get("_ws_disconnected"):
            raise ClientDisconnectedError("Client disconnected, skipping skill execution")

        tool = available_tool_map(session).get(skill_name)
        if tool is None:
            raise ValueError(f"Unknown or disabled skill: {skill_name}")
        params = parse_tool_arguments(call.get("function", {}).get("arguments", "{}"))
        validate_tool_arguments(tool, params)
        # 深拷贝 params 存日志，避免后续 pop 污染
        history_entry["params"] = json_clone(params)

        # 从 params 中提取 target_case_id，不给 Port B
        target_case_id = params.pop("target_case_id", None) or (session.get("case_ids") or [None])[0]
        tcid = target_case_id or (session.get("case_ids") or [None])[0]
        cache_key = build_cache_key(target_case_id or "none", skill_name, params)
        cached = session["skill_result_cache"].get(cache_key)
        if cached is not None:
            response = json_clone(cached)
            history_entry["cache_hit"] = True
            log("SKILL", f"[CACHE HIT] {skill_name}", session_id=session["session_id"], round=round_index)
        else:
            log("SKILL", f"[CALL] {skill_name}", session_id=session["session_id"], round=round_index, params=str(params)[:80])
            await ws_send(
                websocket,
                "skill_call_start",
                session_id=session["session_id"],
                round=round_index,
                call_id=call_id,
                skill_name=skill_name,
                params=params,
            )
            async with semaphore:
                response = await port_b_run_skill(target_case_id, skill_name, params)
            session["skill_result_cache"][cache_key] = json_clone(response)

        history_entry["status"] = response.get("status", "error")
        history_entry["result"] = response.get("result", {})
        history_entry["execution_time_ms"] = response.get("execution_time_ms")
        if response.get("status") == "ok":
            # 将 Skill 结果写入 tool_store（按 case_id 分类）
            if tcid:
                session.setdefault("tool_store", {}).setdefault(tcid, {})
                session["tool_store"][tcid][skill_name] = response.get("result", {})
            # 合并可视化产物到 session.outputs
            merge_skill_outputs(session, skill_name, response.get("result", {}))
            await ws_send(
                websocket,
                "skill_call_result",
                session_id=session["session_id"],
                round=round_index,
                call_id=call_id,
                skill_name=skill_name,
                status="ok",
                cache_hit=history_entry["cache_hit"],
                execution_time_ms=response.get("execution_time_ms"),
                result={"skill_outputs": list(response.get("result", {}).keys())},
            )
        else:
            # write error to tool_store so LLM knows it failed in next round
            if tcid:
                session.setdefault("tool_store", {}).setdefault(tcid, {})
                session["tool_store"][tcid][skill_name] = {
                    "_error": True,
                    "status": "error",
                    "error_code": response.get("error_code"),
                    "message": response.get("message", "Skill execution failed"),
                }
            await ws_send(
                websocket,
                "skill_call_error",
                session_id=session["session_id"],
                round=round_index,
                call_id=call_id,
                skill_name=skill_name,
                status="error",
                error_code=response.get("error_code"),
                message=response.get("message", "Skill execution failed"),
                retryable=response.get("retryable", False),
            )
    except Exception as exc:
        response = {
            "status": "error",
            "error_code": "PORT_A_TOOL_VALIDATION_OR_EXECUTION_ERROR",
            "message": str(exc),
            "retryable": False,
        }
        history_entry["error"] = response
        # 异常也写入 tool_store，防止 LLM 下一轮重试
        if tcid:
            session.setdefault("tool_store", {}).setdefault(tcid, {})
            session["tool_store"][tcid][skill_name] = {
                "_error": True,
                "status": "error",
                "error_code": response["error_code"],
                "message": response["message"],
            }
        await ws_send(
            websocket,
            "skill_call_error",
            session_id=session["session_id"],
            round=round_index,
            call_id=call_id,
            skill_name=skill_name,
            status="error",
            error_code=response["error_code"],
            message=response["message"],
            retryable=False,
        )

    history_entry["finished_at"] = now_iso()
    session["skill_call_history"].append(history_entry)
    session["updated_at"] = now_iso()
    return call, response


async def execute_tool_calls(
    websocket: WebSocket,
    session: Dict[str, Any],
    round_index: int,
    tool_calls: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    remaining = MAX_SKILL_CALLS_PER_TURN - len(session["skill_call_history"])
    if remaining <= 0:
        return [
            (call, {
                "status": "error",
                "error_code": "SKILL_CALL_LIMIT_REACHED",
                "message": "Maximum skill calls reached for this turn.",
                "retryable": False,
            })
            for call in tool_calls
        ]
    calls = tool_calls[:remaining]
    semaphore = asyncio.Semaphore(MAX_PARALLEL_SKILLS)

    # 展开没有 target_case_id 的调用：给每个活跃 case 生成一个
    expanded: List[Dict[str, Any]] = []
    active = session.get("_active_case_ids") or []
    for call in calls:
        try:
            args_str = call.get("function", {}).get("arguments", "{}")
            args = json.loads(args_str)
            if not args.get("target_case_id") and active:
                for cid in active:
                    new_call = json_clone(call)
                    new_args = json_clone(args)
                    new_args["target_case_id"] = cid
                    new_call["function"]["arguments"] = json.dumps(new_args, ensure_ascii=False)
                    expanded.append(new_call)
                continue
        except Exception:
            pass
        expanded.append(call)

    # 按 target_case_id 分组，同 case 内并行，不同 case 串行
    groups: Dict[str, list] = {}
    for call in expanded:
        try:
            cid = json.loads(call.get("function", {}).get("arguments", "{}")).get("target_case_id", "_unknown")
        except Exception:
            cid = "_unknown"
        groups.setdefault(cid, []).append(call)

    results = []
    for cid in sorted(groups.keys()):
        batch = await asyncio.gather(*[
            execute_one_tool_call(websocket, session, round_index, call, semaphore)
            for call in groups[cid]
        ])
        results.extend(batch)
    return results


# ============================================================================
# Qwen multimodal context and streaming function-calling loop
# ============================================================================
def image_to_data_url(path: str) -> Optional[str]:
    if not os.path.isfile(path) or not is_allowed_path(path):
        return None
    size = os.path.getsize(path)
    if size > MAX_IMAGE_BYTES:
        return None
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_system_prompt(session: Dict[str, Any]) -> str:
    """稳定的 System Prompt,不含动态内容。动态信息在 format_tool_context 中提供。"""
    return """
你是医学影像分析 Agent。你通过函数调用使用 Port B 的医学影像 Skills。

【核心技能能力】
- liver_analysis: 肝脏体积、血管体积、肿瘤直径、肿瘤-血管距离（聚合技能，涵盖下面3个）
- vessel_volume: 单独计算血管体积
- tumor_diameter: 单独计算肿瘤直径
- tumor_vessel_distance: 单独计算肿瘤与血管距离
- slice_selection: 选择最佳切片
- three_d_reconstruction: 3D可视化
- segmentation_modification: 分割编辑器
- plan_resection_sequence: 手术规划

【问题类型 → 技能映射】

1. 肝脏体积问题
   问题特征: "volume of the liver", "How many cubic centimeters", "size of the liver"
   调用技能: liver_analysis
   提取数据: liver_volume_cm3
   示例:
     Q: "Report the measured volume of the liver:"
     → 调用 liver_analysis()
     → 从返回的 liver_volume_cm3 中提取数值
     → 回答: "The liver volume is 1270.3 cm³."

2. 病灶计数问题
   问题特征: "How many lesion", "number of lesion", "instances of lesion"
   调用技能: liver_analysis
   提取数据: len(tumor_results)
   示例:
     Q: "How many instances of lesion appear in the liver?"
     → 调用 liver_analysis()
     → 统计 tumor_results 中的条目数
     → 如果 tumor_results = {"tumor_1": {...}, "tumor_2": {...}}
     → 回答: "There are 2 lesions in the liver."

3. 病灶存在性问题
   问题特征: "evidence of lesion", "free of lesion", "contain lesion"
   调用技能: liver_analysis
   提取数据: tumor_results 是否为空
   示例:
     Q: "Is there evidence of lesions in the liver?"
     → 调用 liver_analysis()
     → 检查 tumor_results 是否为空字典
     → 如果 tumor_results = {} → 回答: "No, there is no evidence of lesions."
     → 如果 tumor_results = {"tumor_1": {...}} → 回答: "Yes, there is evidence of lesions."

4. 病灶体积问题
   问题特征: "lesion volume", "tumor volume"
   调用技能: liver_analysis
   提取数据: tumor_results[tumor_name].volume_cm3
   示例:
     Q: "Indicate the liver lesion volume (cm³):"
     → 调用 liver_analysis()
     → 如果有多个肿瘤，计算总和或报告最大的
     → 回答: "The total lesion volume is 134.4 cm³."

【不支持的问题类型】
以下问题类型因技能限制无法回答，应明确拒绝：

1. HU值/密度问题
   特征: "HU", "Hounsfield", "density", "attenuation", "mean HU"
   原因: Port B 没有 HU 值计算功能
   回答模板: "抱歉，当前技能不支持计算 HU 值。可用的分析包括体积、直径和距离测量。"

2. 肝段分析问题
   特征: "segment 1", "segment 2", ..., "segment 8", "hepatic segment"
   原因: Port B 没有肝段分割功能（无 Couinaud 分类）
   回答模板: "抱歉，当前技能不支持肝段分割和分析。可以提供整体肝脏的病灶分布信息。"

3. 形态学/病理判断
   特征: "fatty liver", "cirrhosis", "steatosis", "enlarged"
   原因: 需要临床标准和病理判断
   回答模板: "这需要临床标准判断。我可以提供肝脏体积测量值供参考。"

【工作流程】
1. 理解问题类型（参考上面的映射）
2. 检查「已有分析结果」是否已包含所需数据
   - 如果已有 liver_analysis 结果 → 直接从中提取答案，**不要重复调用**
   - 如果缺少数据 → 调用相应技能
3. 调用技能（如需要）
4. 从返回结果中精确提取数据
5. 用简洁的语言回答，包含具体数值

【关键规则】
- 同一 case 的 liver_analysis 只调用一次，之后直接使用结果
- 不得编造工具未返回的数值
- 病灶计数 = len(tumor_results)，不是分割文件数量
- 体积单位统一为 cm³
- 工具失败时明确说明，不要猜测答案
- 工具参数中不得填写 case_id,CT 路径,mask 路径或输出目录；这些由 Port B 上下文自动注入
- 如果技能返回了可视化 URL，在最终回答中用 Markdown 格式嵌入
- 禁止在回复中使用任何 emoji 或表情符号（如 ✅❌⚠️🔴🟠🟡🟢✓✗ 等），全部使用纯文本表达

【输出格式要求】
你的最终回答必须使用 Markdown 格式，前端会将其渲染为富文本。严格遵循以下规范：

1. **标题层级**：使用 ## 作为主标题，### 作为子标题，组织报告结构
2. **关键数值加粗**：所有测量数值（体积、直径、距离等）必须用 **粗体** 包裹，例如 **1270.3 cm³**
3. **列表**：多项内容使用有序列表（1. 2. 3.）或无序列表（- ）呈现
4. **重点结论**：重要的诊断发现或异常结果使用 > 引用块突出显示
5. **表格**：需要对比多项数值时使用 Markdown 表格（| 列1 | 列2 |），表头与内容用 |---|---| 分隔
6. **分隔线**：不同分析维度之间使用 --- 分隔线
7. **可视化嵌入**：如有 3D 可视化 URL，用 Markdown 链接或图片语法嵌入：[查看3D可视化](URL)
8. **段落间距**：不同主题之间留空行，保持可读性

示例输出结构：
## 肝脏影像分析报告

### 基本信息
- 病例 ID: **case_001**
- 分析日期: **2026-08-05**

### 肝脏体积
> 肝脏总体积为 **1270.3 cm³**，在正常参考范围内。

### 病灶分析

| 病灶编号 | 最大直径 | 体积 |
|---------|---------|------|
| tumor_1 | **12.5 mm** | **134.4 cm³** |

### 总结
肝脏发现 **2** 个病灶，最大直径 **12.5 mm**，建议结合临床进一步评估。

---
[查看3D可视化](http://example.com/3d.html)
""".strip()


async def run_one_model_round(
    websocket: WebSocket,
    session: Dict[str, Any],
    messages: List[Dict[str, Any]],  # 由 build_messages() 统一构造
) -> Tuple[str, List[Dict[str, Any]]]:
    """Stream one model response.

    Returns (text, tool_calls). Text is streamed to the frontend only when the response contains no tool calls.
    """
    require_llm_configuration()
    stream = await llm_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=messages,
        tools=session["available_tools"],
        tool_choice="auto",
        parallel_tool_calls=True,
        stream=True,
        max_tokens=session.get("max_new_tokens", 4096),
    )

    text_parts: List[str] = []
    tool_acc: Dict[int, Dict[str, Any]] = {}
    answer_started = False

    async for chunk in stream:
        if session.get("cancel_requested"):
            raise asyncio.CancelledError()
        check_ws_connected(session)  # 流式接收中检查客户端是否已断开
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            text_parts.append(content)
        for tc in getattr(delta, "tool_calls", None) or []:
            index = getattr(tc, "index", 0)
            acc = tool_acc.setdefault(index, {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if getattr(tc, "id", None):
                acc["id"] += tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    acc["function"]["name"] += fn.name
                if getattr(fn, "arguments", None):
                    acc["function"]["arguments"] += fn.arguments

    tool_calls = [tool_acc[i] for i in sorted(tool_acc)]
    text = "".join(text_parts)

    # Strip Qwen think tags before streaming to frontend
    text = re.sub(r'</?think[^>]*>', '', text)

    if not tool_calls:
        check_ws_connected(session)  # 发送前检查连接
        await ws_send(websocket, "answer_start", session_id=session["session_id"])
        # The SDK stream has already been consumed to distinguish final text from tool calls.
        # Emit in moderate chunks to preserve the frontend streaming contract.
        for i in range(0, len(text), 80):
            if session.get("cancel_requested"):
                raise asyncio.CancelledError()
            check_ws_connected(session)  # 每个 chunk 前检查连接
            await ws_send(
                websocket,
                "answer_delta",
                session_id=session["session_id"],
                delta=text[i:i + 80],
            )
            await asyncio.sleep(0)
        check_ws_connected(session)
        await ws_send(websocket, "answer_end", session_id=session["session_id"])

    return text, tool_calls


async def run_agent_loop(
    websocket: WebSocket,
    session: Dict[str, Any],
    max_rounds: int,
) -> str:
    """Agent 循环:build_messages -> 模型推理 -> tool_store -> 循环直至最终回答。

    不再使用 role=tool message,所有工具结果写入 tool_store,
    通过 format_tool_context 在下轮 system prompt 中呈现。

    P0 优化：集成工具优化器和反思机制
    """
    session["workflow_state"] = "agent_running"
    current_q = session.get("current_user_query", "")

    # P0 优化：在开始前检查是否可以直接回答
    cache_check = ToolOptimizer.can_answer_from_cache(session, current_q)
    if cache_check["can_answer"]:
        log("CACHE_HIT", "可直接从缓存回答",
            session_id=session["session_id"],
            suggestion=cache_check["suggestion"])

    await ws_send(
        websocket,
        "agent_started",
        session_id=session["session_id"],
        question=current_q,
        available_skills=[s.get("name") for s in session.get("available_skills", [])],
    )

    for round_index in range(1, max_rounds + 1):
        session["agent_round"] = round_index

        # 检查客户端是否已断开
        check_ws_connected(session)

        # P0 优化：检查是否需要启用反思
        enable_reflection = ReflectionEngine.should_enable_reflection(session, round_index)

        await ws_send(
            websocket,
            "agent_round_start",
            session_id=session["session_id"],
            round=round_index,
            question=current_q,
            reflection_enabled=enable_reflection,
        )

        # 每轮统一构造 messages:system(tool_context) + conversation + current_user
        messages = build_messages(session)

        # P0 优化：如果启用反思，添加反思提示
        if enable_reflection:
            recent_failures = [
                (
                    {"function": {"name": call["skill_name"], "arguments": json.dumps(call["params"])}},
                    {"error_code": call.get("error", {}).get("error_code"), "message": call.get("error", {}).get("message", "")}
                )
                for call in session.get("skill_call_history", [])
                if call.get("round") == round_index - 1 and call.get("status") == "error"
            ]

            if recent_failures:
                reflection_prompt = ReflectionEngine.generate_reflection_prompt(
                    session, recent_failures, current_q
                )
                messages.append({
                    "role": "system",
                    "content": reflection_prompt
                })
                log("REFLECTION", "启用反思机制",
                    session_id=session["session_id"],
                    round=round_index,
                    failures=len(recent_failures))

        text, tool_calls = await run_one_model_round(websocket, session, messages)

        if not tool_calls:
            # 最终回答:将 user 问题和 assistant 回答写入 conversation
            append_conversation(session, "user", current_q)
            append_conversation(session, "assistant", text)
            return text

        # 有 tool calls:执行并将结果写入 tool_store（在 execute_one_tool_call 中完成）
        await ws_send(
            websocket,
            "execution_plan",
            session_id=session["session_id"],
            round=round_index,
            calls=summarize_tool_calls(tool_calls),
        )

        results = await execute_tool_calls(websocket, session, round_index, tool_calls)

        # 检查客户端是否在 skill 执行期间断开
        check_ws_connected(session)

        # P0 优化：检查是否有 skill 调用失败，应用反思机制
        failed_calls = [(call, resp) for call, resp in results if resp.get("status") == "error"]
        if failed_calls:
            # 分析失败原因
            analyses = []
            strategies = []

            for call, resp in failed_calls:
                skill_name = call.get("function", {}).get("name", "")
                error_code = resp.get("error_code")
                error_msg = resp.get("message", "")
                params = parse_tool_arguments(call.get("function", {}).get("arguments", "{}"))
                exec_time = resp.get("execution_time_ms")

                analysis = ReflectionEngine.analyze_failure(
                    skill_name, error_code, error_msg, params, exec_time
                )
                analyses.append(analysis)

                strategy = ReflectionEngine.suggest_recovery_strategy(
                    session, skill_name, analysis
                )
                strategies.append(strategy)

                log("REFLECTION_ANALYSIS", f"分析失败: {skill_name}",
                    session_id=session["session_id"],
                    error_code=error_code,
                    category=analysis.get("category"),
                    strategy=strategy.get("action"))

            # 创建反思摘要
            reflection_summary = ReflectionEngine.create_reflection_summary(analyses, strategies)
            session.setdefault("_reflection_log", []).append({
                "round": round_index,
                "summary": reflection_summary,
                "analyses": analyses,
                "strategies": strategies
            })

            # 发送反思结果给前端
            await ws_send(
                websocket,
                "reflection_result",
                session_id=session["session_id"],
                round=round_index,
                summary=reflection_summary,
                strategies=[{
                    "skill": s.get("reasoning"),
                    "action": s.get("action")
                } for s in strategies]
            )

            # 判断是否应该继续还是终止
            # 如果所有策略都是 skip，且无法基于已有数据回答，则终止
            all_skip = all(s.get("action") == "skip" for s in strategies)
            can_answer = any(
                ReflectionEngine._can_answer_without_skill(session, call.get("function", {}).get("name", ""))
                for call, _ in failed_calls
            )

            if all_skip and can_answer:
                # 可以基于已有数据回答
                log("REFLECTION_DECISION", "失败但有足够数据，继续生成答案",
                    session_id=session["session_id"])
                # 继续下一轮，让 LLM 基于已有数据回答
            elif all_skip and not can_answer:
                # 无法恢复，生成错误消息
                check_ws_connected(session)
                names = set()
                for call, resp in failed_calls:
                    sn = call.get("function", {}).get("name", "?")
                    names.add(f"{sn}({resp.get('error_code', '?')})")
                err_text = f"关键技能调用失败: {', '.join(names)}。无法提供完整分析，建议检查数据质量或联系技术支持。"

                await ws_send(websocket, "answer_start", session_id=session["session_id"])
                for i in range(0, len(err_text), 80):
                    await ws_send(websocket, "answer_delta", session_id=session["session_id"], delta=err_text[i:i + 80])
                await ws_send(websocket, "answer_end", session_id=session["session_id"])
                append_conversation(session, "user", current_q)
                append_conversation(session, "assistant", err_text)
                return err_text

        await ws_send(
            websocket,
            "agent_round_end",
            session_id=session["session_id"],
            round=round_index,
        )

    # 达到轮数上限:强制最终回答
    check_ws_connected(session)
    messages = build_messages(session)
    messages.append({
        "role": "user",
        "content": (
            "已达到工具调用轮数上限。请停止调用工具,严格基于现有结果回答当前用户问题,"
            "并清楚说明未解决部分和局限。"
        ),
    })
    require_llm_configuration()
    response = await llm_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=messages,
        stream=False,
        max_tokens=session.get("max_new_tokens", 4096),
    )
    text = response.choices[0].message.content or ""
    check_ws_connected(session)
    await ws_send(websocket, "answer_start", session_id=session["session_id"])
    for i in range(0, len(text), 80):
        await ws_send(websocket, "answer_delta", session_id=session["session_id"], delta=text[i:i + 80])
    await ws_send(websocket, "answer_end", session_id=session["session_id"])
    # 强制回答也记入 conversation
    append_conversation(session, "user", current_q)
    append_conversation(session, "assistant", text)
    return text


# ============================================================================
# Main workflow
# ============================================================================
async def _process_single_volume(session: Dict[str, Any], volume: Dict[str, Any], device: str) -> None:
    """只分割一个文件并写入 tool_store，不刷新 skills（用于后续文件的批量处理）。单个文件失败不阻断整体。"""
    try:
        path = volume["path"]
        case_name = infer_case_name(path)
        result = await port_b_process_lite(path, case_name, device)
        sid = result["case_id"]
        session.setdefault("case_ids", []).append(sid)
        session.setdefault("tool_store", {}).setdefault(sid, {})
        session["tool_store"][sid]["segmentation"] = result
        log("PROCESS", f"processed volume {volume.get('name','?')}", case_id=sid)
    except Exception as e:
        log("ERROR", f"failed to process volume {volume.get('name','?')}: {e}")


async def prepare_case_and_tools(
    websocket: WebSocket,
    session: Dict[str, Any],
    options: FrontendOptions,
    volume: Optional[Dict[str, Any]] = None,
) -> None:
    if volume is None:
        volume = pick_primary_medical_volume(session)
    if volume is None:
        raise ValueError("At least one .nii/.nii.gz/.dcm file or DICOM folder is required.")
    path = volume["path"]
    case_name = infer_case_name(path)

    session["workflow_state"] = "segmenting"
    session["segmentation_status"] = "running"
    await ws_send(
        websocket,
        "medical_input_detected",
        session_id=session["session_id"],
        path=path,
        input_type="dicom_folder" if os.path.isdir(path) else "medical_volume",
    )

    # 检查是否已存在分割结果
    output_dir = os.path.join(CACHE_ROOT, "segmentation_output", case_name)
    masks_dir = os.path.join(output_dir, "masks")

    if os.path.exists(masks_dir) and os.path.isdir(masks_dir):
        mask_files = os.listdir(masks_dir)
        if len(mask_files) > 0:
            # 复用已有分割结果
            await ws_send(
                websocket,
                "segmentation_start",
                session_id=session["session_id"],
                input=path,
                case_name=case_name,
                reused=True,
            )

            # 构造process_result
            ct_path = f"{output_dir}/ct.nii.gz"
            mask_dict = {f.replace('.nii.gz', ''): f"{masks_dir}/{f}" for f in mask_files if f.endswith('.nii.gz')}

            process_result = {
                "status": "ok",
                "case_id": case_name,
                "ct_nifti_path": ct_path,
                "mask_dir": masks_dir,
                "output_dir": output_dir,
                "mask_files": mask_dict,
                "elapsed_seconds": 0,
                "reused": True
            }
        else:
            # 分割目录存在但为空，重新分割
            await ws_send(
                websocket,
                "segmentation_start",
                session_id=session["session_id"],
                input=path,
                case_name=case_name,
            )
            process_result = await port_b_process_lite(path, case_name, options.device)
    else:
        # 分割目录不存在，执行分割
        await ws_send(
            websocket,
            "segmentation_start",
            session_id=session["session_id"],
            input=path,
            case_name=case_name,
        )
        process_result = await port_b_process_lite(path, case_name, options.device)

    sid = process_result["case_id"]
    if sid not in session.setdefault("case_ids", []):
        session["case_ids"].append(sid)
    session["process_lite_result"] = process_result
    # 将 segmentation 结果写入 tool_store（按 case_id 分类）
    session.setdefault("tool_store", {}).setdefault(sid, {})
    session["tool_store"][sid]["segmentation"] = process_result
    session["segmentation_status"] = "completed"
    session["workflow_state"] = "ready_for_agent"
    session["updated_at"] = now_iso()

    await ws_send(
        websocket,
        "segmentation_done",
        session_id=session["session_id"],
        case_id=sid,
        ct_nifti_path=process_result["ct_nifti_path"],
        mask_dir=process_result["mask_dir"],
        output_dir=process_result["output_dir"],
        mask_files=process_result.get("mask_files", {}),
        elapsed_seconds=process_result.get("elapsed_seconds"),
    )


async def run_initial_workflow(
    websocket: WebSocket,
    session: Dict[str, Any],
    options: FrontendOptions,
) -> None:
    session_id = session["session_id"]
    try:
        session["workflow_state"] = "validating"
        errors = validate_session_inputs(session)
        if errors:
            raise ValueError(json.dumps(errors, ensure_ascii=False))

        await prepare_case_and_tools(websocket, session, options)
        # 并行处理剩余的 medical_volumes（非 primary 文件）
        processed_paths = {session.get("process_lite_result", {}).get("ct_nifti_path", "")}
        remaining = [vol for vol in session.get("medical_volumes", []) if vol.get("path") not in processed_paths]
        if remaining:
            await asyncio.gather(*[_process_single_volume(session, vol, options.device) for vol in remaining])
        session["max_new_tokens"] = options.max_new_tokens
        session["_active_volumes"] = [v.get("name","") for v in session.get("medical_volumes", []) if v.get("name")]
        session["_active_case_ids"] = list(session.get("case_ids", []))
        # 所有文件分割完成后统一获取一次 skills
        if not session.get("available_tools"):
            skills_result = await port_b_list_skills()
            session["available_skills"] = skills_result["skills"]
            # P0 优化：应用工具优化器
            base_tools = inject_target_case_id(skills_result["tools"], session.get("_active_case_ids", session.get("case_ids", [])))
            session["available_tools"] = apply_tool_optimization(session, base_tools)
        # 校验用户指定的 /+skill 是否都在可用工具中
        required = session.get("required_skills", [])
        if required:
            tool_names = {t["function"]["name"] for t in session["available_tools"]}
            unknown = [s for s in required if s not in tool_names]
            if unknown:
                raise ValueError(
                    f"未知技能: {', '.join(unknown)}。"
                    f"可用技能: {', '.join(sorted(tool_names))}"
                )
        answer = await run_agent_loop(websocket, session, options.max_agent_rounds)

        session["final_answer"] = answer
        session["workflow_state"] = "done"
        session["status"] = "ok"
        session["updated_at"] = now_iso()
        save_session(session)

        succeeded = sum(1 for x in session["skill_call_history"] if x.get("status") == "ok")
        failed = len(session["skill_call_history"]) - succeeded
        await ws_send(
            websocket,
            "final",
            session_id=session_id,
            status="ok",
            answer={"text": answer, "format": "markdown"},
            case_id=(session.get("case_ids") or [None])[-1],
            skills_summary={
                "rounds": session.get("agent_round", 0),
                "called": len(session["skill_call_history"]),
                "succeeded": succeeded,
                "failed": failed,
            },
            skill_calls=session["skill_call_history"],
            outputs=session.get("outputs", {}),
            errors=session.get("errors", []),
        )
    except ClientDisconnectedError:
        session["workflow_state"] = "disconnected"
        session["status"] = "disconnected"
        save_session(session)
        log("WS", "workflow_aborted_client_disconnected", session_id=session_id)
        # 客户端已断开，不再尝试发送 WebSocket 消息
    except asyncio.CancelledError:
        session["workflow_state"] = "cancelled"
        session["status"] = "cancelled"
        save_session(session)
        await ws_send(websocket, "cancelled", session_id=session_id)
    except Exception as exc:
        session["workflow_state"] = "error"
        session["status"] = "error"
        error = {"stage": session.get("workflow_state"), "message": str(exc), "time": now_iso()}
        session["errors"].append(error)
        save_session(session)
        log("ERROR", "workflow failed", session_id=session_id, error=str(exc))
        await ws_error(websocket, session_id, "workflow", "Workflow failed.", error)


async def run_followup_turn(
    websocket: WebSocket,
    session: Dict[str, Any],
    payload: WSChatMessagePayload,
) -> None:
    """Multi-turn conversation.

    - If new medical_volumes provided -> re-run full pipeline (segment + skills + agent).
    - If no new medical_volumes -> reuse existing case_id/masks, just refresh skills,
      carry conversation history into agent loop.
    """
    try:
        session["cancel_requested"] = False
        clean_text, skills = parse_skill_hints(payload.user_message.strip())
        session["current_user_query"] = clean_text or payload.user_message.strip()
        session["required_skills"] = skills
        # 去重追加 images（按 file_id 判重）
        existing_img_ids = {img.get("file_id") for img in session["images"] if img.get("file_id")}
        for img in payload.images:
            img_dict = pydantic_dump(img)
            if img_dict.get("file_id") not in existing_img_ids:
                session["images"].append(img_dict)
                existing_img_ids.add(img_dict["file_id"])
        # user 问题不立即写入 conversation,agent loop 完成后一并写入

        # 清空 agent 运行状态（本轮将重新构建）
        session["agent_round"] = 0
        session["skill_call_history"] = []
        session["final_answer"] = None
        session["outputs"] = {"html_url": None, "best_slices": []}
        session["max_new_tokens"] = payload.options.max_new_tokens
        session["_active_volumes"] = [pydantic_dump(v).get("name","") for v in payload.medical_volumes]

        new_volumes = [pydantic_dump(x) for x in payload.medical_volumes]

        if new_volumes:
            existing_vol_ids = {vol.get("file_id") for vol in session["medical_volumes"] if vol.get("file_id")}
            truly_new = [vol for vol in new_volumes if vol.get("file_id") not in existing_vol_ids]

            if truly_new:
                for vol in truly_new:
                    session["medical_volumes"].append(vol)
                session["process_lite_result"] = None
                session["segmentation_status"] = "pending"
                session["case_ids"] = []
                session["available_skills"] = []
                session["available_tools"] = []
                # 第一个文件走完整流程（分割+刷 skills），后续文件并行分割
                await prepare_case_and_tools(websocket, session, payload.options, volume=truly_new[0])
                if len(truly_new) > 1:
                    await asyncio.gather(*[_process_single_volume(session, vol, payload.options.device) for vol in truly_new[1:]])
            else:
                if not session.get("case_ids"):
                    raise ValueError("当前会话没有已分割的影像。")
                skills_result = await port_b_list_skills()
                session["available_skills"] = skills_result["skills"]
                # P0 优化：应用工具优化器
                base_tools = inject_target_case_id(skills_result["tools"], session.get("_active_case_ids", session.get("case_ids", [])))
                session["available_tools"] = apply_tool_optimization(session, base_tools)
                session["workflow_state"] = "ready_for_agent"
                await ws_send(websocket, "progress", session_id=session["session_id"],
                              stage="reuse_segmentation",
                              message=f"文件已存在,复用已有数据 (case_ids={session.get('case_ids',[])})。")
        else:
            # ── 没有新影像:复用已有分割结果,仅刷新 skills ──
            if not session.get("case_ids"):
                raise ValueError("当前会话没有已分割的影像,请上传 .nii/.nii.gz/.dcm 文件。")
            skills_result = await port_b_list_skills()
            session["available_skills"] = skills_result["skills"]
            # P0 优化：应用工具优化器
            base_tools = inject_target_case_id(skills_result["tools"], session.get("_active_case_ids", session.get("case_ids", [])))
            session["available_tools"] = apply_tool_optimization(session, base_tools)
            session["workflow_state"] = "ready_for_agent"
            await ws_send(
                websocket,
                "progress",
                session_id=session["session_id"],
                stage="reuse_segmentation",
                message=f"复用已有分割结果,刷新 skills 完成。",
            )

        session["_active_case_ids"] = _resolve_active_case_ids(session, [pydantic_dump(x) for x in payload.medical_volumes])
        # 校验用户指定的 /+skill 是否都在可用工具中
        required = session.get("required_skills", [])
        if required and session.get("available_tools"):
            tool_names = {t["function"]["name"] for t in session["available_tools"]}
            unknown = [s for s in required if s not in tool_names]
            if unknown:
                raise ValueError(
                    f"未知技能: {', '.join(unknown)}。"
                    f"可用技能: {', '.join(sorted(tool_names))}"
                )
        answer = await run_agent_loop(websocket, session, payload.options.max_agent_rounds)

        session["final_answer"] = answer
        session["workflow_state"] = "done"
        session["updated_at"] = now_iso()
        save_session(session)
        succeeded = sum(1 for x in session["skill_call_history"] if x.get("status") == "ok")
        await ws_send(
            websocket,
            "final",
            session_id=session["session_id"],
            status="ok",
            answer={"text": answer, "format": "markdown"},
            case_id=(session.get("case_ids") or [None])[-1],
            skills_summary={
                "rounds": session.get("agent_round", 0),
                "called": len(session["skill_call_history"]),
                "succeeded": succeeded,
                "failed": len(session["skill_call_history"]) - succeeded,
            },
            skill_calls=session["skill_call_history"],
            outputs=session.get("outputs", {}),
            errors=session.get("errors", []),
        )
    except ClientDisconnectedError:
        session["workflow_state"] = "disconnected"
        session["status"] = "disconnected"
        save_session(session)
        log("WS", "followup_aborted_client_disconnected", session_id=session["session_id"])
        # 客户端已断开，不再尝试发送 WebSocket 消息
    except asyncio.CancelledError:
        session["workflow_state"] = "cancelled"
        save_session(session)
        await ws_send(websocket, "cancelled", session_id=session["session_id"])
    except Exception as exc:
        session["workflow_state"] = "error"
        error = {"stage": "chat_message", "message": str(exc), "time": now_iso()}
        session["errors"].append(error)
        save_session(session)
        log("ERROR", "followup turn failed", session_id=session["session_id"], error=str(exc))
        await ws_error(websocket, session["session_id"], "chat_message", "Follow-up turn failed.", error)


# ============================================================================
# WebSocket endpoint
# ============================================================================
@app.websocket("/ws/frontend/chat")
async def websocket_frontend_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    log("WS", "connected", client=websocket.client)
    current_session_id: Optional[str] = None

    try:
        while True:
            raw = await websocket.receive_json()
            event = raw.get("event")
            log("WS<-", event or "unknown", session_id=raw.get("session_id"))

            if event == "initial":
                try:
                    payload = WSInitialPayload(**raw)
                except ValidationError as exc:
                    await ws_error(websocket, raw.get("session_id"), "parse_initial", "Invalid initial payload.", exc.errors())
                    continue

                session_id = payload.session_id or create_session_id()
                current_session_id = session_id
                old = SESSION_STORE.get(session_id)
                if old and old.get("workflow_task") and not old["workflow_task"].done():
                    await ws_error(websocket, session_id, "initial", "A workflow is already running for this session.")
                    continue

                session = create_empty_session(session_id)
                merge_initial(session, payload)
                SESSION_STORE[session_id] = session
                save_session(session)
                await ws_send(websocket, "session_created", session_id=session_id)
                await ws_send(
                    websocket,
                    "workflow_started",
                    session_id=session_id,
                    question=session["current_user_query"],
                )
                task = asyncio.create_task(run_initial_workflow(websocket, session, payload.options))
                session["workflow_task"] = task

            elif event == "chat_message":
                try:
                    payload = WSChatMessagePayload(**raw)
                except ValidationError as exc:
                    await ws_error(websocket, raw.get("session_id"), "parse_chat_message", "Invalid chat_message payload.", exc.errors())
                    continue
                session = SESSION_STORE.get(payload.session_id)
                if not session:
                    await ws_error(websocket, payload.session_id, "chat_message", "Session not found.")
                    continue
                if session.get("workflow_task") and not session["workflow_task"].done():
                    await ws_error(websocket, payload.session_id, "chat_message", "A workflow is already running.")
                    continue
                task = asyncio.create_task(run_followup_turn(websocket, session, payload))
                session["workflow_task"] = task

            elif event == "cancel":
                try:
                    payload = WSCancelPayload(**raw)
                except ValidationError as exc:
                    await ws_error(websocket, raw.get("session_id"), "parse_cancel", "Invalid cancel payload.", exc.errors())
                    continue
                session = SESSION_STORE.get(payload.session_id)
                if not session:
                    await ws_error(websocket, payload.session_id, "cancel", "Session not found.")
                    continue
                session["cancel_requested"] = True
                task = session.get("workflow_task")
                if task and not task.done():
                    task.cancel()
                await ws_send(websocket, "cancel_requested", session_id=payload.session_id)

            else:
                await ws_error(
                    websocket,
                    raw.get("session_id") or current_session_id,
                    "dispatch",
                    f"Unsupported event: {event}. Supported: initial, chat_message, cancel.",
                )

    except WebSocketDisconnect:
        log("WS", "disconnected", session_id=current_session_id)
    finally:
        WS_SEND_LOCKS.pop(id(websocket), None)


# ============================================================================
# HTTP diagnostics
# ============================================================================
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "Port A Qwen Medical Skills Agent",
        "version": "2.0.0",
        "model": LLM_MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "upload": "/api/upload",
            "websocket": "/ws/frontend/chat",
            "session": "/api/sessions/{session_id}",
        },
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    port_b_ok = False
    port_b_detail: Any = None
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{PORT_B_INTERNAL}/health")
            port_b_detail = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text[:200]
            port_b_ok = (
                response.status_code < 400
                and isinstance(port_b_detail, dict)
                and port_b_detail.get("status") == "ok"
            )
    except Exception as exc:
        port_b_detail = str(exc)
    llm_errors = llm_configuration_errors()
    return {
        "status": "ok" if not llm_errors and port_b_ok else "degraded",
        "model_configured": not llm_errors,
        "model_configuration_errors": llm_errors,
        "model": LLM_MODEL_NAME or None,
        "port_b_ok": port_b_ok,
        "port_b_detail": port_b_detail,
        "sessions": len(SESSION_STORE),
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    session = SESSION_STORE.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    public = json_clone(session)
    public.pop("workflow_task", None)
    return public


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT_A_PORT)
