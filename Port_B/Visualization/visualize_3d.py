#!/usr/bin/env python3
"""
Interactive 3D Visualization of CT Segmentation Results.

Reads per-organ binary masks from a segmentation output directory,
extracts mesh surfaces using marching cubes, and renders them as an
interactive Three.js HTML file with different colors per organ/lesion.

Mesh extraction helpers used by API.py for the CRLM pipeline.
"""

import json
import os
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from skimage.measure import marching_cubes, mesh_surface_area

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Tool_Box.mask_resolution import scan_visualization_masks
# ---------------------------------------------------------------------------
# Timestamped log helper
# ---------------------------------------------------------------------------
import datetime as _dt
def _log(msg: str, end: str = "\n", flush: bool = False):
    """Print with ISO-8601 timestamp."""
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", end=end, flush=flush)



# ---------------------------------------------------------------------------
# Color scheme: organ name → color config
# ---------------------------------------------------------------------------
# Edit this dictionary to customize colors. Format: {"color": "#RRGGBB", "opacity": float}
ORGAN_COLORS = {
    # ------ Organs (semi-transparent, provides spatial context) ------
    "liver":            {"color": "#e8a58f", "opacity": 0.40},
    "spleen":           {"color": "#c4b0d9", "opacity": 0.45},
    "pancreas":         {"color": "#f9c08a", "opacity": 0.45},
    "colon":            {"color": "#a8d4e6", "opacity": 0.45},
    "left kidney":      {"color": "#a8e6cf", "opacity": 0.45},
    "right kidney":     {"color": "#7dceb0", "opacity": 0.45},
    "kidney":           {"color": "#a8e6cf", "opacity": 0.45},  # merged
    "adrenal gland":    {"color": "#b8e0d4", "opacity": 0.50},
    "right adrenal gland": {"color": "#b8e0d4", "opacity": 0.50},
    "left adrenal gland":  {"color": "#a0d4c4", "opacity": 0.50},
    "stomach":          {"color": "#f5c4a0", "opacity": 0.45},
    "duodenum":         {"color": "#f9d99a", "opacity": 0.50},
    "gallbladder":      {"color": "#f5e68c", "opacity": 0.50},
    "esophagus":        {"color": "#c9b8d4", "opacity": 0.50},
    "bladder":          {"color": "#80d4c0", "opacity": 0.45},
    "prostate":         {"color": "#f0a0c0", "opacity": 0.50},
    "lung":             {"color": "#c0c8d0", "opacity": 0.35},
    "left lung":        {"color": "#c0c8d0", "opacity": 0.35},
    "right lung":       {"color": "#b0b8c0", "opacity": 0.35},
    "heart":            {"color": "#e8a0a0", "opacity": 0.40},
    "bone":             {"color": "#e0dcc8", "opacity": 0.50},
    "rib":              {"color": "#d0ccc0", "opacity": 0.50},
    "vertebra":         {"color": "#d8d4c8", "opacity": 0.50},
    "muscle":           {"color": "#d4a8a0", "opacity": 0.40},
    "small bowel":      {"color": "#e0c880", "opacity": 0.40},
    "portal vein":      {"color": "#90b8d8", "opacity": 0.40},
    # ------ CRLM liver cancer structures (DICOM SEG naming) ------
    "liver_remnant":    {"color": "#d4a08a", "opacity": 0.35},
    "hepatic":          {"color": "#e85050", "opacity": 0.55},  # 肝静脉
    "portal":           {"color": "#5060e8", "opacity": 0.55},  # 门静脉
    # ------ Lesions / Abnormal (fully opaque, stands out) ------
    "pancreatic tumor":  {"color": "#e83030", "opacity": 1.0},
    "hepatic tumor":     {"color": "#e84040", "opacity": 1.0},
    "hepatic tumor2":    {"color": "#ff2020", "opacity": 1.0},
    "liver tumor":       {"color": "#e84040", "opacity": 1.0},
    "left kidney cyst":  {"color": "#f0c040", "opacity": 1.0},
    "right kidney cyst": {"color": "#e8b830", "opacity": 1.0},
    "kidney cyst":       {"color": "#f0c040", "opacity": 1.0},
    "kidney_cyst":       {"color": "#f0c040", "opacity": 1.0},
    "cyst":              {"color": "#f0c040", "opacity": 1.0},
    "lesion":            {"color": "#E91E63", "opacity": 1.0},
    "metastasis":        {"color": "#d04060", "opacity": 1.0},
    "nodule":            {"color": "#e85530", "opacity": 1.0},
    "tumor":             {"color": "#e83030", "opacity": 1.0},
    "tumor_1":           {"color": "#e84040", "opacity": 1.0},  # CRLM tumor pattern
    "tumor_2":           {"color": "#e84040", "opacity": 1.0},
    "tumor_3":           {"color": "#e84040", "opacity": 1.0},
    "tumor_4":           {"color": "#e84040", "opacity": 1.0},
    "tumor_5":           {"color": "#e84040", "opacity": 1.0},
}

# Display names (for legend and hover). Falls back to the key name.
ORGAN_DISPLAY_NAMES = {
    "liver":             "Liver",
    "spleen":            "Spleen",
    "pancreas":          "Pancreas",
    "colon":             "Colon",
    "left kidney":       "Left Kidney",
    "right kidney":      "Right Kidney",
    "kidney":            "Kidney",
    "adrenal gland":     "Adrenal Gland",
    "right adrenal gland": "Right Adrenal",
    "left adrenal gland":  "Left Adrenal",
    "stomach":           "Stomach",
    "duodenum":          "Duodenum",
    "gallbladder":       "Gallbladder",
    "esophagus":         "Esophagus",
    "bladder":           "Bladder",
    "aorta":             "Aorta",
    "inferior vena cava": "IVC",
    "prostate":          "Prostate",
    "pancreatic tumor":  "Pancreatic Tumor",
    "hepatic tumor":     "Hepatic Tumor #1",
    "hepatic tumor2":    "Hepatic Tumor #2",
    "liver tumor":       "Liver Tumor",
    "left kidney cyst":  "Left Kidney Cyst",
    "right kidney cyst": "Right Kidney Cyst",
    "kidney cyst":       "Kidney Cyst",
    "kidney_cyst":       "Kidney Cyst",
    "hepatic":           "Hepatic Veins",
    "portal":            "Portal Veins",
    "liver_remnant":     "Liver Remnant",
    "tumor_1":           "Tumor #1",
    "tumor_2":           "Tumor #2",
    "tumor_3":           "Tumor #3",
    "tumor_4":           "Tumor #4",
    "tumor_5":           "Tumor #5",
}


def get_display_name(name):
    """Return an English display name, including dynamic tumor_N fallbacks."""
    if name in ORGAN_DISPLAY_NAMES:
        return ORGAN_DISPLAY_NAMES[name]
    m = re.match(r"^tumor_(\d+)$", name)
    if m:
        return f"Tumor #{m.group(1)}"
    return name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def robust_load_nii(path: str, retries: int = 3):
    """Load a NIfTI file with retries (handles NFS/Blob transient errors)."""
    last_err = None
    for attempt in range(retries):
        try:
            return nib.load(path)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load {path} after {retries} retries: {last_err}")


def find_mask_files(case_dir: str):
    """
    Scan case_dir for individual organ mask files (*.nii.gz).
    Returns a dict: {organ_name: file_path}
    """
    case_dir = Path(case_dir)
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    masks = {}
    # Known organ names (any *.nii.gz file whose stem matches an organ name)
    known_organs = set(ORGAN_COLORS.keys())

    for logical_stem, resolved in scan_visualization_masks(case_dir).items():
        fpath = Path(resolved.path)
        stem = logical_stem
        # Skip merged/segment files
        if stem in ("all",):
            continue
        if re.match(r".+_segment_\d+$", stem):
            continue  # TotalSegmentator liver segments

        # Check if this organ name is known
        if stem in known_organs:
            masks[stem] = str(fpath)
            continue

        # Also check with space→underscore variants
        alt_stem = stem.replace("_", " ")
        if alt_stem in known_organs:
            masks[alt_stem] = str(fpath)
            continue

        # Fuzzy: check known organs that contain the stem or vice versa
        for known in known_organs:
            known_safe = known.replace(" ", "_")
            if stem == known_safe:
                masks[known] = str(fpath)
                break
            # Handle cases like "pancreatic_tumor" → "pancreatic tumor"
            stem_norm = stem.replace("-", " ").replace("_", " ")
            if stem_norm == known:
                masks[known] = str(fpath)
                break

        # CRLM pattern: tumor_N (tumor_1, tumor_2, ..., tumor_35)
        if re.match(r"^tumor_\d+$", stem):
            masks[stem] = str(fpath)
            continue

    if not masks:
        # Fallback: try all.nii.gz (merged label map)
        all_path = case_dir / "all.nii.gz"
        if all_path.exists():
            _log(f"[INFO] No individual mask files found, trying {all_path}")
            return _extract_from_all_nii(str(all_path))
        else:
            _log(f"[WARN] No mask files found in {case_dir}")

    return masks


def _extract_from_all_nii(all_path: str) -> dict:
    """
    Read merged label map (all.nii.gz) and split by label ID.
    Returns dict of {organ_name: file_path} but we need to load the data
    since we don't have separate files. We'll handle this lazily.
    """
    # AMOS label mapping (15 organs)
    AMOS_MAP = {
        1: "spleen", 2: "right kidney", 3: "left kidney",
        4: "gallbladder", 5: "esophagus", 6: "liver",
        7: "stomach", 8: "aorta", 9: "inferior vena cava",
        10: "pancreas", 11: "right adrenal gland", 12: "left adrenal gland",
        13: "duodenum", 14: "bladder", 15: "prostate",
    }

    try:
        nii = robust_load_nii(all_path)
        data = nii.get_fdata()
        affine = nii.affine
        header = nii.header
    except Exception as e:
        _log(f"[ERROR] Failed to load {all_path}: {e}")
        return {}

    unique_labels = np.unique(data).astype(int)
    masks = {}
    for label in unique_labels:
        if label == 0:
            continue  # background
        organ_name = AMOS_MAP.get(int(label), f"label_{int(label)}")
        # Create a temporary binary mask
        binary = (data == label).astype(np.uint8)
        # We'll return a special sentinel: the data and info needed
        masks[organ_name] = {
            "_data": binary,
            "_affine": affine,
            "_header": header,
        }
    return masks


def quantize_vertices(vertices: np.ndarray, precision_mm: float = 0.1) -> np.ndarray:
    """
    Quantize vertex coordinates to reduce JSON size without visible quality loss.

    Instead of ``123.456787109375`` (full float32, ~17 chars), produces
    ``123.5`` (1-decimal float, ~5 chars) in the serialized HTML.

    CT spatial resolution is 0.3-1 mm, so 0.1 mm precision is adequate.

    Args:
        vertices: (N, 3) float32 array of world coordinates in mm
        precision_mm: target precision in mm (default 0.1)

    Returns:
        (N, 3) float32 array, values rounded to ``precision_mm``
    """
    scale = 1.0 / precision_mm
    return np.round(vertices.astype(np.float64) * scale) / scale


def downsample_volume(volume: np.ndarray, factor: float) -> np.ndarray:
    """
    Downsample a 3D binary volume by the given factor.
    Uses nearest-neighbor interpolation to preserve binary values.
    """
    if factor <= 1.0:
        return volume
    new_shape = [max(1, round(s / factor)) for s in volume.shape]
    return zoom(volume.astype(np.float32), new_shape / np.array(volume.shape),
                order=0).astype(volume.dtype)


def _interp_z_1d(data: np.ndarray, zoom_z: float):
    """
    Fast 1D linear interpolation along the Z axis only.

    Instead of a full 3D spline (which is 10-50× slower), this linearly
    interpolates each (H,W) slice stack independently.  The result is a
    smooth continuous gradient at mask edges, letting marching cubes
    find an anti-aliased surface at level=0.5.

    Args:
        data: 3D array in (D, H, W) format
        zoom_z: Z-axis zoom factor (> 1 = interpolate, < 1 = decimate)

    Returns:
        Interpolated array in (new_D, H, W), float32, values in [0, 1]
    """
    D, H, W = data.shape
    new_D = max(1, int(round(D * zoom_z)))
    if new_D == D:
        return data.astype(np.float32, copy=False)

    result = np.empty((new_D, H, W), dtype=np.float32)
    z_in = np.arange(D, dtype=np.float32)
    z_out = np.linspace(0.0, D - 1.0, new_D)

    for z_idx in range(new_D):
        z = z_out[z_idx]
        z0 = int(np.floor(z))
        z1 = min(z0 + 1, D - 1)
        w = z - z0                         # fractional weight
        result[z_idx] = data[z0] * (1.0 - w) + data[z1] * w

    return result


def resample_for_smoothing(mask_data: np.ndarray, affine: np.ndarray,
                           target_max_spacing: float = 1.0,
                           max_zoom: float = 3.0):
    """
    Resample a binary mask to a smoother resolution by linearly
    interpolating along axes coarser than ``target_max_spacing``.

    Marching cubes on a thick-slice binary mask produces staircase
    artifacts.  This function creates a continuous gradient at mask
    boundaries, so ``marching_cubes(level=0.5)`` finds an anti-aliased
    surface.

    Only the Z axis is interpolated by default — XY in-plane resolution
    is usually already fine.  The affine is updated to reflect the new
    voxel grid.

    Args:
        mask_data: binary mask (uint8, values 0/1), any axis order
        affine: 4×4 voxel→world affine matrix
        target_max_spacing: target max voxel spacing in mm (default 1.0)
        max_zoom: max zoom factor (prevents memory explosion, default 3×)

    Returns:
        (resampled_data, new_affine) — float32, values in [0, 1]
    """
    orig_spacing = np.linalg.norm(affine[:3, :3], axis=0)

    # Only zoom axes coarser than the target; cap to max_zoom
    zoom_factors = np.ones(3)
    for i in range(3):
        if orig_spacing[i] > target_max_spacing:
            zoom_factors[i] = min(orig_spacing[i] / target_max_spacing, max_zoom)

    if np.allclose(zoom_factors, 1.0):
        return mask_data.astype(np.float32, copy=False), affine

    # Ensure data is in (D, H, W) — we always interpolate axis 0
    # The caller may pass (H, W, D); we use the affine to decide.
    # But for safety, detect shape: the most-anisotropic dim → axis 0
    if mask_data.ndim == 3:
        # Assume mask_data is (H, W, D) — typical nibabel load
        data = np.ascontiguousarray(mask_data.transpose(2, 0, 1).astype(np.float32))
    else:
        data = np.ascontiguousarray(mask_data.astype(np.float32))

    # Interpolate along Z (axis 0)
    resampled = _interp_z_1d(data, zoom_factors[2])

    # Transpose back to (H, W, D) to match nibabel convention
    resampled = np.ascontiguousarray(resampled.transpose(1, 2, 0))

    # Update affine: new spacing = old / zoom_factor
    new_affine = affine.copy().astype(np.float64)
    for i in range(3):
        if zoom_factors[i] != 1.0:
            new_affine[:3, i] = affine[:3, i] / zoom_factors[i]

    return resampled, new_affine


def binarize_mask(mask_data: np.ndarray, prob_threshold: float = 0.5) -> np.ndarray:
    """
    Convert mask data to binary (0/1 uint8).

    Handles two cases:
    1. Probability maps (0-1000 scale from TotalSegmentator, or 0.0-1.0 float)
    2. Already binary (0/1 or 0/255)

    Args:
        mask_data: input mask/probability array
        prob_threshold: probability threshold (0-1), applied after normalizing
                        the data to [0, 1] range

    Returns:
        uint8 binary mask (0 or 1)
    """
    data = np.asarray(mask_data)

    # Already binary (only 0 and 1)?
    uniq = np.unique(data)
    if len(uniq) <= 2 and np.array_equal(uniq, [0, 1]):
        return data.astype(np.uint8)
    if len(uniq) == 1 and uniq[0] in (0, 1):
        return data.astype(np.uint8)

    dmin, dmax = float(data.min()), float(data.max())

    # Case 1: TotalSegmentator-style probability (0-1000 integer scale)
    if dmax > 1.0 and dmax <= 1000:
        # Normalize to [0, 1]
        normalized = data.astype(np.float64) / dmax
        threshold = prob_threshold
        _log(f"[prob map 0-1000] thresh={threshold:.2f}, max={int(dmax)}", end=" ")
        binary = (normalized > threshold).astype(np.uint8)
        return binary

    # Case 2: Float probability map (0.0-1.0) with values NOT just 0/1
    if dmax <= 1.0 and dmin >= 0.0:
        normalized = data.astype(np.float64)
        threshold = prob_threshold
        _log(f"[prob map float] thresh={threshold:.2f}", end=" ")
        binary = (normalized > threshold).astype(np.uint8)
        return binary

    # Case 3: Generic — normalize to [0, 1] and threshold
    if dmax > dmin:
        normalized = (data.astype(np.float64) - dmin) / (dmax - dmin)
    else:
        normalized = data.astype(np.float64)
    threshold = prob_threshold
    _log(f"[generic values {dmin:.1f}-{dmax:.1f}] thresh={threshold:.2f}", end=" ")
    binary = (normalized > threshold).astype(np.uint8)
    return binary


def extract_mesh(mask_data: np.ndarray, spacing: tuple, step_size: int,
                 smooth: bool = True):
    """
    Extract mesh surface from a binary mask using marching cubes.

    Args:
        mask_data: 3D binary array (H, W, D)
        spacing: voxel spacing in mm (from affine)
        step_size: marching cubes step size (1=finest)
        smooth: whether to apply Laplacian smoothing

    Returns:
        (vertices, faces) or (None, None) if mask is empty
    """
    if mask_data.sum() < 4:
        return None, None  # too few voxels for a mesh

    try:
        verts, faces, _, _ = marching_cubes(
            mask_data,
            level=0.5,
            spacing=spacing,
            step_size=step_size,
            allow_degenerate=False,
        )
    except (RuntimeError, ValueError) as e:
        _log(f"    [WARN] Marching cubes failed: {e}")
        return None, None

    if len(verts) == 0 or len(faces) == 0:
        return None, None

    # -- Optional Laplacian smoothing --
    if smooth and len(verts) > 100:
        verts = laplacian_smooth(verts, faces, iterations=10, lam=0.5)

    return verts, faces


def laplacian_smooth(vertices: np.ndarray, faces: np.ndarray,
                     iterations: int = 10, lam: float = 0.5):
    """
    Simple Taubin-style Laplacian smoothing.
    Each vertex moves toward the average of its neighbors.
    """
    # Build adjacency
    adj = {i: set() for i in range(len(vertices))}
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            adj[a].add(b)
            adj[b].add(a)

    verts = vertices.copy()
    for _ in range(iterations):
        new_verts = verts.copy()
        for i in range(len(verts)):
            nbors = adj.get(i)
            if not nbors or len(nbors) < 2:
                continue
            avg = np.mean([verts[j] for j in nbors], axis=0)
            new_verts[i] = verts[i] + lam * (avg - verts[i])
        verts = new_verts

    return verts


def voxel_to_world(vertices_voxel: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Transform vertices from voxel coordinates to world coordinates (mm).

    marching_cubes returns (row, col, slice) = (i, j, k).
    affine maps (i, j, k, 1) → (x, y, z, 1).
    """
    if vertices_voxel.size == 0:
        return vertices_voxel
    # Add homogeneous coordinate
    ones = np.ones((vertices_voxel.shape[0], 1))
    vox_h = np.hstack([vertices_voxel, ones])
    # Transform
    world = (affine @ vox_h.T).T
    return world[:, :3]


# ---------------------------------------------------------------------------
# Organ discovery from label_dict.json
# ---------------------------------------------------------------------------

def get_known_organs_from_seg_backend(seg_backend: str) -> dict:
    """Try to read label_dict.json from the segmentation backend."""
    project_root = Path(__file__).resolve().parents[1]
    try:
        if seg_backend.lower() in ("biomedparse",):
            path = project_root / "SegAgent" / "BiomedParse" / "label_dict.json"
        elif seg_backend.lower() in ("vista3d", "vista3d"):
            path = project_root / "SegAgent" / "VISTA3d" / "label_dict.json"
        else:
            return {}
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Three.js renderer — proper transparency with depthWrite
# ---------------------------------------------------------------------------

def _read_bezier_surface_source(three_dir=None):
    """
    Read BezierSurface.js from the three/ directory.

    Returns the file content as string, or None if not found.
    """
    if three_dir is None:
        three_dir = Path(__file__).resolve().parent / "three"
    three_dir = Path(three_dir)
    bs_path = three_dir / "BezierSurface.js"
    if not bs_path.exists():
        return None
    try:
        return bs_path.read_text(encoding="utf-8")
    except Exception:
        return None


def _read_threejs_sources(three_dir=None):
    """
    Read Three.js source files from local 'three/' directory.

    Returns a dict {'three': str, 'orbit_controls': str, 'drag_controls': str}
    or None to fall back to CDN.
    """
    if three_dir is None:
        three_dir = Path(__file__).resolve().parent / "three"
    three_dir = Path(three_dir)

    three_main = three_dir / "three.module.min.js"
    orbit_ctrl = three_dir / "OrbitControls.js"
    drag_ctrl = three_dir / "DragControls.js"

    if not (three_main.exists() and orbit_ctrl.exists() and drag_ctrl.exists()):
        _log(f"  [INFO] Local Three.js not found in {three_dir} — using CDN")
        return None

    try:
        three_code = three_main.read_text(encoding="utf-8")
        oc_code = orbit_ctrl.read_text(encoding="utf-8")
        dc_code = drag_ctrl.read_text(encoding="utf-8")
        _log(f"  [INFO] Using local Three.js ({len(three_code)//1024} KB + {len(oc_code)//1024} KB + {len(dc_code)//1024} KB)")
        # Escape </script> just in case
        three_code = three_code.replace("</script>", "<\\/script>")
        oc_code = oc_code.replace("</script>", "<\\/script>")
        dc_code = dc_code.replace("</script>", "<\\/script>")
        return {"three": three_code, "orbit_controls": oc_code, "drag_controls": dc_code}
    except Exception as e:
        _log(f"  [WARN] Failed to read Three.js files: {e} — using CDN")
        return None


# ---- JS module template for Three.js viewer ----
# (built outside f-string to avoid brace escaping issues)
_JS_MODULE_TEMPLATE = """

// ================================================================
//  0. Mesh data
// ================================================================
const DATA_URL = ##DATA_URL##;
const meshes = [];
let planeObjects = [];
let currentPlanes = [];
let activePlane = 0;
let resectionSequence = null;
let resectionPathObjects = null;
let resectionCellObjects = [];
let resectionCellPlaneIndex = -1;
let resectionPathStep = 0;
let resectionPathTimer = null;
let sequenceControlsWired = false;
let selectedStartCell = null;
let pickStartMode = false;
let cellPickingWired = false;
let pathPlanningBusy = false;
const startCellPreviewCache = new Map();
let resectionPlanAbortController = null;

// ================================================================
//  1. Scene setup
// ================================================================
const scene = new THREE.Scene();
scene.background = new THREE.Color(##BG_COLOR##);

const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 10000);
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.sortObjects = true;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

// Append to 3D container instead of body
const view3d = document.getElementById('view-3d');
view3d.appendChild(renderer.domElement);

function resizeRendererToView() {
  const rect = view3d.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function getModelSafeAspect() {
  const rect = view3d.getBoundingClientRect();
  const rootStyle = getComputedStyle(document.documentElement);
  const safeLeft = parseFloat(rootStyle.getPropertyValue('--model-safe-left')) || 0;
  const safeRight = parseFloat(rootStyle.getPropertyValue('--model-safe-right')) || 0;
  const safeWidth = Math.max(rect.width * 0.28, rect.width - safeLeft - safeRight);
  return safeWidth / Math.max(1, rect.height);
}

function fitCameraToScene() {
  const safeAspect = getModelSafeAspect();
  const aspectFit = safeAspect < 1 ? 1 / safeAspect : 1;
  const compactCollapsed = document.body.classList.contains('structures-collapsed')
    && window.matchMedia('(max-width: 620px)').matches;
  const safePadding = compactCollapsed ? 0.68 : (safeAspect < 1 ? 1.14 : 1.04);
  const dist = maxDim * 1.36 * aspectFit * safePadding;
  INITIAL_CAM_POS = new THREE.Vector3(dist * 0.35, dist * 0.50, dist * 0.78);
  camera.position.copy(INITIAL_CAM_POS);
  camera.lookAt(0, 0, 0);
  controls.update();
}
resizeRendererToView();

// ================================================================
//  2. Controls
// ================================================================
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 1.0;
controls.zoomSpeed = 1.2;
controls.target.set(0, 0, 0);
controls.update();

// ================================================================
//  3. Lighting
// ================================================================
scene.add(new THREE.AmbientLight(0x8888aa, 0.8));
scene.add(new THREE.HemisphereLight(0xaaaaff, 0xffeedd, 0.8));

const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(300, 400, 500);
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(0xccddff, 0.5);
fillLight.position.set(-200, -100, -200);
scene.add(fillLight);

const backLight = new THREE.DirectionalLight(0xffffff, 0.6);
backLight.position.set(-300, -400, -500);
scene.add(backLight);

scene.add(new THREE.DirectionalLight(0xffffff, 0.3));
scene.children[scene.children.length - 1].position.set(-50, 150, -100);

// Fallback colors — used by mesh building
const FALLBACK_COLORS = [
  '#E74C3C','#9B59B6','#E67E22','#2980B9','#2ECC71','#27AE60',
  '#1ABC9C','#D35400','#F39C12','#F1C40F','#8E44AD','#16A085',
  '#7F8C8D','#C0392B','#E91E63','#FF5722',
];

// ================================================================
//  4. Load data from JSON — fetch then build scene + organ tree
// ================================================================
let MESH_DATA = [];
let maxDim = 300;
let INITIAL_CAM_POS = null;

fetch(DATA_URL)
  .then(r => {
    if (!r.ok) throw new Error('Failed to load 3D data: ' + DATA_URL);
    return r.json();
  })
  .then(jsonData => {
    MESH_DATA = jsonData.meshes || [];
    const resectionPlanes = jsonData.resection_planes || [];
    currentPlanes = resectionPlanes;
    document.body.classList.toggle('has-resection-workspace', resectionPlanes.length > 0);
    requestAnimationFrame(resizeRendererToView);
    const centerOffset = jsonData.center_offset || null;

    // Show/hide resection toggle based on whether planes data exists
    const toggleContainer = document.getElementById('resection-toggle');
    if (toggleContainer) {
      toggleContainer.style.display = 'flex';
      document.getElementById('toggle-resection-plane').checked = currentPlanes.length > 0;
    }

    // 4a. Build organ meshes
    MESH_DATA.forEach((data, idx) => {
      const verts = new Float32Array(data.verts);
      const faces = new Uint32Array(data.faces);
      const color = data.color || FALLBACK_COLORS[idx % FALLBACK_COLORS.length];
      const opacity = data.opacity != null ? data.opacity : 0.85;
      const name = data.name || `Organ ${idx + 1}`;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(verts, 3));
      geometry.setIndex(new THREE.BufferAttribute(faces, 1));
      geometry.computeVertexNormals();

      const isOpaque = opacity >= 1.0;
      const material = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(color),
        transparent: !isOpaque,
        opacity: opacity,
        depthWrite: isOpaque,
        depthTest: true,
        side: THREE.DoubleSide,
        metalness: 0.02,
        roughness: 0.5,
        clearcoat: 0.0,
        envMapIntensity: 0.3,
        premultipliedAlpha: !isOpaque,
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = isOpaque ? 0 : 1;
      mesh.userData = { name, color, opacity, idx };
      scene.add(mesh);
      meshes.push(mesh);
    });

    // 4b. Build resection planes (Bezier surfaces) if present
    if (resectionPlanes.length > 0) {
      Promise.all([
        import('BezierSurface'),
        import('three/addons/controls/DragControls.js'),
      ]).then(([BS, { DragControls }]) => {
        const box = new THREE.Box3().setFromObject(scene);
        const size = box.getSize(new THREE.Vector3());
        const dim = Math.max(size.x, size.y, size.z) || 100;
        const tumorCloudData = jsonData.tumor_cloud || [];
        const tumorCloud = new Float32Array(tumorCloudData);

        // Render all planes, but only show the first one
        planeObjects = [];
        const savedPlaneIndex = Number.isInteger(jsonData.selected_resection_plane_index)
          ? jsonData.selected_resection_plane_index : 0;
        activePlane = (savedPlaneIndex >= 0 && savedPlaneIndex < resectionPlanes.length)
          ? savedPlaneIndex : 0;
        let isEditMode = false;
        let activeDragControls = null;

        resectionPlanes.forEach((planeData, pi) => {
          if (centerOffset && !planeData.center_offset) {
            planeData.center_offset = centerOffset;
          }
          const result = BS.buildBezierSurface(planeData, dim);
          scene.add(result.mesh);
          scene.add(result.line);
          if (result.gridHelper) scene.add(result.gridHelper);
          if (result.label) scene.add(result.label);

          // Build control points (hidden by default)
          const cpGroup = BS.buildControlPoints(planeData, dim);
          scene.add(cpGroup);
          cpGroup.visible = false;

          // Hide all but the first plane
          const visible = pi === activePlane;
          result.mesh.visible = visible;
          result.line.visible = visible;
          if (result.gridHelper) result.gridHelper.visible = visible;
          if (result.label) result.label.visible = visible;

          result.mesh.userData = {
            name: resectionPlanes[pi].candidate_name || ('Resection Plan ' + (pi + 1)),
            type: 'resection_plane',
            planeIndex: pi,
          };
          planeObjects.push({ ...result, cpGroup, planeData, hasCP: !!planeData.control_points_3d });
        });

        // --- Load and display the saved sequence path, if it exists ---
        const sequenceUrl = DATA_URL.replace(/_3d\\.json(?:\\?.*)?$/, '_resection_sequence.json');
        function loadSequence(sequence) {
          const sequencePlaneIndex = Number.isInteger(sequence.saved_plane_index)
            ? sequence.saved_plane_index : activePlane;
          const sequencePlane = resectionPlanes[sequencePlaneIndex];
          if (sequence.saved_at && (!sequencePlane || !sequencePlane.saved_at
              || sequence.saved_at !== sequencePlane.saved_at)) {
            throw new Error('Sequence belongs to an older saved resection plane');
          }
          clearSequenceCells();
          resectionSequence = sequence;
          selectedStartCell = Number.isInteger(sequence.start_cell) ? sequence.start_cell : selectedStartCell;
          const algorithmSelect = document.getElementById('path-algorithm');
          if (algorithmSelect && sequence.algorithm) algorithmSelect.value = sequence.algorithm;
          const sequenceParams = sequence.parameters || {};
          if (sequenceParams.vascular_safe_distance_mm != null) document.getElementById('path-vascular-distance').value = sequenceParams.vascular_safe_distance_mm;
          if (sequenceParams.liver_intersection_min_samples != null) document.getElementById('path-liver-samples').value = sequenceParams.liver_intersection_min_samples;
          if (sequenceParams.step_time_seconds != null) document.getElementById('path-step-time').value = sequenceParams.step_time_seconds;
          const idx = sequencePlaneIndex;
          if (idx >= 0 && idx < planeObjects.length && idx !== activePlane) {
            activePlane = idx;
            planeObjects.forEach((po, i) => {
              const visible = i === activePlane;
              po.mesh.visible = visible;
              po.line.visible = visible;
              if (po.gridHelper) po.gridHelper.visible = visible;
              if (po.label) po.label.visible = visible;
            });
          }
          // Replace the original reference grid with the exact planner grid.
          // This also handles deliberately requested coarse experimental grids.
          const seqResolution = sequence.grid && sequence.grid.vertex_resolution;
          if (seqResolution && planeObjects[activePlane]) {
            const po = planeObjects[activePlane];
            if (po.gridHelper) {
              scene.remove(po.gridHelper);
              po.gridHelper.geometry.dispose();
              po.gridHelper.material.dispose();
            }
            po.gridHelper = BS.buildSurfaceGridFromCP(
              po.planeData.control_points_3d,
              seqResolution[0], seqResolution[1]
            );
            po.gridHelper.renderOrder = 6;
            po.gridHelper.visible = po.mesh.visible;
            scene.add(po.gridHelper);
          }
          createSequenceControls();
          drawSequenceStep(0);
        }

        function fetchSequence(cacheBust) {
          const url = cacheBust ? `${sequenceUrl}?t=${Date.now()}` : sequenceUrl;
          return fetch(url).then(r => {
            if (!r.ok) throw new Error('No saved sequence');
            return r.json();
          }).then(loadSequence).catch(() => {
            const panel = document.getElementById('path-planning-panel');
            if (panel) {
              panel.dataset.available = 'false';
              panel.style.display = 'none';
            }
          });
        }
        if (jsonData.resection_sequence_available === true) {
          fetchSequence(false);
        }

        function sequenceCellPosition(planeData, cell) {
          const grid = resectionSequence && resectionSequence.grid;
          const nU = grid ? grid.vertex_resolution[0] : (planeData.surface_resolution || [20, 20])[0];
          const nV = grid ? grid.vertex_resolution[1] : (planeData.surface_resolution || [20, 20])[1];
          const cols = nV - 1;
          const i = cell.grid_ij ? cell.grid_ij[0] : Math.floor(cell.cell / cols);
          const j = cell.grid_ij ? cell.grid_ij[1] : cell.cell % cols;
          const cp = planeData.control_points_3d;
          const positions = BS.computeSurfacePositions(cp, nU, nV);
          const index = (u, v) => (u * nV + v) * 3;
          const a = index(i, j), b = index(i + 1, j), c = index(i, j + 1), d = index(i + 1, j + 1);
          return new THREE.Vector3(
            (positions[a] + positions[b] + positions[c] + positions[d]) / 4,
            (positions[a + 1] + positions[b + 1] + positions[c + 1] + positions[d + 1]) / 4,
            (positions[a + 2] + positions[b + 2] + positions[c + 2] + positions[d + 2]) / 4,
          );
        }

        function sequenceCellCorners(planeData, cell) {
          const grid = resectionSequence && resectionSequence.grid;
          const nU = grid ? grid.vertex_resolution[0] : (planeData.surface_resolution || [20, 20])[0];
          const nV = grid ? grid.vertex_resolution[1] : (planeData.surface_resolution || [20, 20])[1];
          const cols = nV - 1;
          const i = cell.grid_ij ? cell.grid_ij[0] : Math.floor(cell.cell / cols);
          const j = cell.grid_ij ? cell.grid_ij[1] : cell.cell % cols;
          const positions = BS.computeSurfacePositions(planeData.control_points_3d, nU, nV);
          const point = (u, v) => {
            const k = (u * nV + v) * 3;
            return new THREE.Vector3(positions[k], positions[k + 1], positions[k + 2]);
          };
          return [point(i, j), point(i, j + 1), point(i + 1, j + 1), point(i + 1, j)];
        }

        function clearSequenceCells() {
          resectionCellObjects.forEach(mesh => {
            scene.remove(mesh);
            mesh.geometry.dispose();
            mesh.material.dispose();
          });
          resectionCellObjects = [];
          resectionCellPlaneIndex = -1;
        }

        function ensureSequenceCells(planeData) {
          if (resectionCellPlaneIndex === activePlane && resectionCellObjects.length) return;
          clearSequenceCells();
          const unique = new Map();
          const grid = resectionSequence && resectionSequence.grid;
          const nU = grid ? grid.vertex_resolution[0] : (planeData.surface_resolution || [20, 20])[0];
          const nV = grid ? grid.vertex_resolution[1] : (planeData.surface_resolution || [20, 20])[1];
          const cells = resectionSequence && resectionSequence.cell_states
            ? resectionSequence.cell_states
            : Array.from({length: (nU - 1) * (nV - 1)}, (_, cell) => ({
                cell, grid_ij: [Math.floor(cell / (nV - 1)), cell % (nV - 1)], state: 'candidate'
              }));
          cells.forEach(step => unique.set(step.cell, step));
          unique.forEach(step => {
            const corners = sequenceCellCorners(planeData, step);
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(
              corners.flatMap(p => [p.x, p.y, p.z]), 3));
            geometry.setIndex([0, 1, 2, 0, 2, 3]);
            geometry.computeVertexNormals();
            const material = new THREE.MeshBasicMaterial({
              color: 0x8b949e,
              transparent: true,
              opacity: 0.72,
              side: THREE.DoubleSide,
              depthTest: false,
              depthWrite: false,
            });
            const mesh = new THREE.Mesh(geometry, material);
            mesh.renderOrder = 18;
            mesh.userData.sequenceCell = step.cell;
            mesh.userData.sequenceState = step.state || 'candidate';
            scene.add(mesh);
            resectionCellObjects.push(mesh);
          });
          resectionCellPlaneIndex = activePlane;
          wireCellPicking();
        }

        function updateCellColors() {
          const visited = resectionSequence && resectionSequence.path
            ? new Set(resectionSequence.path.slice(0, resectionPathStep + 1).map(s => s.cell))
            : new Set();
          const current = resectionSequence && resectionSequence.path && resectionSequence.path.length
            ? resectionSequence.path[resectionPathStep].cell : null;
          resectionCellObjects.forEach(mesh => {
            const state = mesh.userData.sequenceState;
            const cell = mesh.userData.sequenceCell;
            if (cell === current) {
              mesh.material.color.setHex(0xff1744);
              mesh.material.opacity = 0.88;
            } else if (cell === selectedStartCell) {
              mesh.material.color.setHex(0x1976d2);
              mesh.material.opacity = 0.9;
            } else if (visited.has(cell)) {
              mesh.material.color.setHex(0x808892);
              mesh.material.opacity = 0.72;
            } else if (state === 'outside_liver') {
              mesh.material.color.setHex(0x4e342e);
              mesh.material.opacity = 0.32;
            } else if (state === 'vascular_risk') {
              mesh.material.color.setHex(0xff8f00);
              mesh.material.opacity = 0.62;
            } else if (state === 'unreachable') {
              mesh.material.color.setHex(0x8e24aa);
              mesh.material.opacity = 0.62;
            } else {
              mesh.material.color.setHex(state === 'selectable' ? 0x43a047 : (state === 'candidate' ? 0x90caf9 : 0xb8c0c8));
              mesh.material.opacity = state === 'selectable' ? 0.52 : (state === 'candidate' ? 0.22 : 0.18);
            }
          });
        }

        function wireCellPicking() {
          if (cellPickingWired) return;
          cellPickingWired = true;
          const raycaster = new THREE.Raycaster();
          renderer.domElement.addEventListener('pointerup', event => {
            if (!pickStartMode || pathPlanningBusy || !resectionCellObjects.length) return;
            const rect = renderer.domElement.getBoundingClientRect();
            const pointer = new THREE.Vector2(
              ((event.clientX - rect.left) / rect.width) * 2 - 1,
              -((event.clientY - rect.top) / rect.height) * 2 + 1
            );
            raycaster.setFromCamera(pointer, camera);
            const hits = raycaster.intersectObjects(resectionCellObjects, false);
            if (!hits.length || hits[0].object.userData.sequenceState !== 'selectable') return;
            selectedStartCell = hits[0].object.userData.sequenceCell;
            pickStartMode = false;
            updateCellColors();
            createSequenceControls(`Start cell ${selectedStartCell} selected.`);
          });
        }

        function requestStartCellPreview() {
          const cd = document.getElementById('case-data');
          const po = planeObjects[activePlane];
          if (!cd || !po || !po.planeData || !po.planeData.user_saved) {
            createSequenceControls('Save the edited plane before choosing a start.');
            return;
          }
          const vascularDistance = Number(document.getElementById('path-vascular-distance').value);
          const liverSamples = Number(document.getElementById('path-liver-samples').value);
          const previewKey = [
            activePlane,
            po.planeData.saved_at || '',
            vascularDistance,
            liverSamples,
            JSON.stringify(po.planeData.control_points_3d || []),
          ].join('|');
          const activatePreview = preview => {
            clearDisplayedSequence('Preparing valid start cells…');
            resectionSequence = preview;
            selectedStartCell = null;
            pickStartMode = true;
            ensureSequenceCells(po.planeData);
            updateCellColors();
            createSequenceControls('Green boundary cells are valid starting points.');
          };
          const cachedPreview = startCellPreviewCache.get(previewKey);
          if (cachedPreview) {
            activatePreview(cachedPreview);
            return;
          }
          pathPlanningBusy = true;
          clearDisplayedSequence('Finding valid start cells…');
          fetch('/api/skills/run', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              skill_name: 'plan_resection_sequence',
              case_id: cd.dataset.caseName || '',
              params: {
                preview_only: true,
                vascular_safe_distance_mm: vascularDistance,
                liver_intersection_min_samples: liverSamples
              }
            })
          }).then(r => r.json()).then(data => {
            if (data.status !== 'ok' || !data.result || data.result.status !== 'preview') {
              throw new Error(data.message || data.detail || 'Could not identify valid start cells');
            }
            startCellPreviewCache.set(previewKey, data.result);
            activatePreview(data.result);
          }).catch(err => {
            createSequenceControls('Start-cell preview failed: ' + err.message);
          }).finally(() => {
            pathPlanningBusy = false;
            createSequenceControls();
          });
        }

        function drawSequenceStep(step) {
          if (!resectionSequence || !resectionSequence.path || !planeObjects.length) return;
          const steps = resectionSequence.path;
          resectionPathStep = Math.max(0, Math.min(step, steps.length - 1));
          const plane = planeObjects[activePlane].planeData;
          ensureSequenceCells(plane);
          updateCellColors();
          const points = steps.map(s => sequenceCellPosition(plane, s));
          if (resectionPathObjects) {
            scene.remove(resectionPathObjects.completed);
            scene.remove(resectionPathObjects.remaining);
            scene.remove(resectionPathObjects.current);
            resectionPathObjects.completed.geometry.dispose();
            resectionPathObjects.remaining.geometry.dispose();
            resectionPathObjects.current.geometry.dispose();
          }
          const completedPoints = points.slice(0, resectionPathStep + 1);
          const remainingPoints = points.slice(resectionPathStep);
          const makeLine = (arr, color, width) => {
            const geometry = new THREE.BufferGeometry().setFromPoints(arr.length > 1 ? arr : points.slice(0, 1));
            const material = new THREE.LineBasicMaterial({color, linewidth: width, depthTest: false, transparent: true, opacity: 0.95});
            const line = new THREE.Line(geometry, material);
            line.renderOrder = 20;
            return line;
          };
          const completed = makeLine(completedPoints, 0x20c997, 3);
          const remaining = makeLine(remainingPoints, 0xffc107, 2);
          const current = new THREE.Mesh(
            new THREE.SphereGeometry(Math.max(maxDim * 0.012, 1), 16, 12),
            new THREE.MeshBasicMaterial({color: 0xff1744, depthTest: false})
          );
          current.position.copy(points[resectionPathStep]);
          current.renderOrder = 21;
          scene.add(completed); scene.add(remaining); scene.add(current);
          resectionPathObjects = {completed, remaining, current};
          const label = document.getElementById('path-status');
          if (label) {
            const coverageText = `Step ${resectionPathStep + 1}/${steps.length} · ${steps[resectionPathStep].action} · Coverage ${((new Set(steps.slice(0, resectionPathStep + 1).map(s => s.cell)).size / (resectionSequence.target_cell_count || 1)) * 100).toFixed(1)}%`;
            label.textContent = resectionSequence.status === 'partial'
              ? `${coverageText} · ${resectionSequence.failure_reason || 'Partial coverage'}`
              : coverageText;
          }
          const slider = document.getElementById('path-slider');
          if (slider) slider.value = resectionPathStep;
        }

        function createSequenceControls(message) {
          const panel = document.getElementById('path-planning-panel');
          if (!panel) return;
          panel.dataset.available = 'true';
          const pathToggle = document.getElementById('toggle-path-planning');
          const planeToggle = document.getElementById('toggle-resection-plane');
          panel.style.display = pathToggle && pathToggle.checked
            && planeToggle && planeToggle.checked ? 'flex' : 'none';
          const po = planeObjects[activePlane];
          const slider = document.getElementById('path-slider');
          const hasPath = !!(resectionSequence && resectionSequence.path && resectionSequence.path.length);
          slider.disabled = !hasPath;
          slider.max = hasPath ? Math.max(0, resectionSequence.path.length - 1) : 0;
          document.getElementById('path-play').disabled = !hasPath;
          document.getElementById('path-prev').disabled = !hasPath;
          document.getElementById('path-next').disabled = !hasPath;
          const replanButton = document.getElementById('path-replan');
          if (replanButton) replanButton.disabled = pathPlanningBusy;
          const pickButton = document.getElementById('path-pick-start');
          pickButton.disabled = !po || !po.planeData.user_saved || pathPlanningBusy;
          pickButton.textContent = pickStartMode ? 'Click a cell' : 'Pick Start';
          if (message) document.getElementById('path-status').textContent = message;
          else if (selectedStartCell != null) document.getElementById('path-status').textContent = `Start cell ${selectedStartCell}`;
          if (!resectionCellObjects.length && po && resectionSequence) ensureSequenceCells(po.planeData);
          if (sequenceControlsWired) return;
          sequenceControlsWired = true;
          slider.addEventListener('input', () => { stopSequence(); drawSequenceStep(Number(slider.value)); });
          document.getElementById('path-play').addEventListener('click', () => {
            if (!resectionSequence || !resectionSequence.path.length) return;
            if (resectionPathTimer) { stopSequence(); return; }
            resectionPathTimer = setInterval(() => {
              if (resectionPathStep >= resectionSequence.path.length - 1) { stopSequence(); return; }
              drawSequenceStep(resectionPathStep + 1);
            }, 120);
            document.getElementById('path-play').textContent = 'Pause';
          });
          document.getElementById('path-prev').addEventListener('click', () => { stopSequence(); drawSequenceStep(resectionPathStep - 1); });
          document.getElementById('path-next').addEventListener('click', () => { stopSequence(); drawSequenceStep(resectionPathStep + 1); });
          pickButton.addEventListener('click', () => {
            if (pickStartMode) {
              pickStartMode = false;
              createSequenceControls('Start selection cancelled.');
              return;
            }
            requestStartCellPreview();
          });
          document.getElementById('path-replan').addEventListener('click', requestSequencePlan);
        }

        function stopSequence() {
          if (resectionPathTimer) clearInterval(resectionPathTimer);
          resectionPathTimer = null;
          const btn = document.getElementById('path-play');
          if (btn) btn.textContent = 'Play';
        }

        function clearDisplayedSequence(message, revealPanel = true) {
          stopSequence();
          if (resectionPathObjects) {
            scene.remove(resectionPathObjects.completed);
            scene.remove(resectionPathObjects.remaining);
            scene.remove(resectionPathObjects.current);
            resectionPathObjects.completed.geometry.dispose();
            resectionPathObjects.remaining.geometry.dispose();
            resectionPathObjects.current.geometry.dispose();
            resectionPathObjects = null;
          }
          clearSequenceCells();
          resectionSequence = null;
          selectedStartCell = null;
          pickStartMode = false;
          if (revealPanel) {
            createSequenceControls(message || 'Save the plane, then replan.');
          } else {
            const panel = document.getElementById('path-planning-panel');
            if (panel) {
              panel.dataset.available = 'false';
              panel.style.display = 'none';
            }
            if (message) document.getElementById('path-status').textContent = message;
          }
        }

        function requestSequencePlan() {
          const cd = document.getElementById('case-data');
          const po = planeObjects[activePlane];
          if (!cd || !po || !po.planeData || !po.planeData.user_saved) {
            createSequenceControls('Save the edited plane first.');
            return;
          }
          if (selectedStartCell == null) {
            createSequenceControls('Pick a start cell on the surface first.');
            return;
          }
          const button = document.getElementById('path-replan');
          button.disabled = true;
          pathPlanningBusy = true;
          button.textContent = 'Planning…';
          createSequenceControls('Planning from the saved plane…');
          fetch('/api/skills/run', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              skill_name: 'plan_resection_sequence',
              case_id: cd.dataset.caseName || '',
              params: {
                algorithm: document.getElementById('path-algorithm').value,
                start_cell: selectedStartCell,
                vascular_safe_distance_mm: Number(document.getElementById('path-vascular-distance').value),
                liver_intersection_min_samples: Number(document.getElementById('path-liver-samples').value),
                step_time_seconds: Number(document.getElementById('path-step-time').value)
              }
            })
          }).then(r => r.json()).then(data => {
            if (data.status !== 'ok' || !data.result || !['ok', 'partial'].includes(data.result.status)) {
              throw new Error(data.message || data.detail || 'Path planning failed');
            }
            return fetchSequence(true);
          }).then(() => {
            if (!resectionSequence) throw new Error('Could not load the new path result');
          }).catch(err => {
            createSequenceControls('Path planning failed: ' + err.message);
          }).finally(() => {
            pathPlanningBusy = false;
            button.disabled = false;
            button.textContent = 'Replan';
            createSequenceControls();
          });
        }

        function handlePathPlanningOpenRequest(event) {
          const requestedOpen = !event || !event.detail || event.detail.open !== false;
          const panel = document.getElementById('path-planning-panel');
          if (!requestedOpen) {
            stopSequence();
            if (panel) panel.style.display = 'none';
            return;
          }
          const po = planeObjects[activePlane];
          const message = po && po.planeData && po.planeData.user_saved
            ? (resectionSequence ? undefined : 'Pick a start cell, then generate the optimal path.')
            : 'Save the selected resection plane before planning a surgical path.';
          createSequenceControls(message);
        }
        document.addEventListener('path-planning-open-request', handlePathPlanningOpenRequest);
        const pathToggle = document.getElementById('toggle-path-planning');
        if (pathToggle && pathToggle.checked) {
          handlePathPlanningOpenRequest();
        }

        // --- Edit mode controls ---
        function enterEditMode() {
          if (isEditMode) return;
          const po = planeObjects[activePlane];
          if (!po.hasCP) return; // no control points data

          isEditMode = true;
          // Editing is single-plane mode: hide every other candidate and all
          // overlays so the user sees only the surface being modified.
          planeObjects.forEach((other, index) => {
            const visible = index === activePlane;
            other.mesh.visible = visible;
            other.line.visible = visible;
            if (other.gridHelper) other.gridHelper.visible = false;
            if (other.label) other.label.visible = false;
            other.cpGroup.visible = false;
          });
          po.cpGroup.visible = true;
          po.mesh.visible = true;
          po.line.visible = true;
          if (po.gridHelper) po.gridHelper.visible = false;
          if (po.label) po.label.visible = false;

          const cpSpheres = po.cpGroup.children;
          activeDragControls = new DragControls(cpSpheres, camera, renderer.domElement);
          activeDragControls.addEventListener('dragstart', () => { controls.enabled = false; });
          activeDragControls.addEventListener('dragstart', () => {
            // Any geometry change invalidates the previous saved-plane state.
            po.planeData.user_saved = false;
            po.planeData.unsaved_changes = true;
            delete po.planeData.saved_at;
            startCellPreviewCache.clear();
            clearDisplayedSequence('Plane edited. Save it, then replan.', false);
            const cd = document.getElementById('case-data');
            if (cd) {
              fetch('/api/resection-plane/invalidate', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                  output_dir: cd.dataset.outputDir || '',
                  json_file: cd.dataset.jsonFile || '',
                  plane_index: activePlane
                })
              }).catch(() => {});
            }
          });
          activeDragControls.addEventListener('drag', () => {
            // Extract updated 4x4 control points from sphere positions
            const cp3d = [];
            for (let ci = 0; ci < 4; ci++) {
              cp3d[ci] = [];
              for (let cj = 0; cj < 4; cj++) {
                const sphere = cpSpheres[ci * 4 + cj];
                cp3d[ci][cj] = [sphere.position.x, sphere.position.y, sphere.position.z];
              }
            }
            po.planeData.control_points_3d = cp3d;
            // Rebuild surface geometry with new positions + colors
            const nU = (po.planeData.surface_resolution && po.planeData.surface_resolution[0]) || 20;
            const nV = (po.planeData.surface_resolution && po.planeData.surface_resolution[1]) || 20;
            const newPos = BS.computeSurfacePositions(cp3d, nU, nV);
            const dists = BS.computeVertexDistances(newPos, tumorCloud);
            console.log('[drag] dists min=', Math.min(...dists).toFixed(3), 'max=', Math.max(...dists).toFixed(3));
            const cols = BS.distancesToColors(dists);
            BS.updateSurfaceGeometry(po.mesh.geometry, newPos, cols);
            // Update label with current margin
            const marginMin = Math.min(...dists);
            if (po.label) {
              scene.remove(po.label);
              po.planeData.margin_min_mm = marginMin;
              po.planeData.margin_p05_mm = 0;
              po.label = BS.buildMarginLabel(po.planeData, dim);
              if (po.label) scene.add(po.label);
            }
            // Rebuild boundary line
            scene.remove(po.line);
            po.line = BS.buildBoundaryLineFromCP(cp3d, nU, nV);
            scene.add(po.line);
          });
          activeDragControls.addEventListener('dragend', () => {
            controls.enabled = true;
            // The old grid is sampled from the pre-edit surface. Replace it
            // only after dragging ends, so exiting edit mode never reveals a
            // second, stale plane behind the edited mesh.
            const nU = (po.planeData.surface_resolution && po.planeData.surface_resolution[0]) || 20;
            const nV = (po.planeData.surface_resolution && po.planeData.surface_resolution[1]) || 20;
            if (po.gridHelper) {
              scene.remove(po.gridHelper);
              po.gridHelper.geometry.dispose();
              po.gridHelper.material.dispose();
            }
            po.gridHelper = BS.buildSurfaceGridFromCP(po.planeData.control_points_3d, nU, nV);
            po.gridHelper.renderOrder = 6;
            po.gridHelper.visible = false;
            scene.add(po.gridHelper);
          });
        }

        function exitEditMode() {
          if (!isEditMode) return;
          isEditMode = false;
          const po = planeObjects[activePlane];
          po.cpGroup.visible = false;
          if (po.gridHelper) po.gridHelper.visible = po.mesh.visible;
          if (po.label) po.label.visible = po.mesh.visible;
          if (activeDragControls) {
            activeDragControls.dispose();
            activeDragControls = null;
          }
        }

        // Add plane selector and persistence UI
        if (resectionPlanes.length > 0) {
          const selector = document.createElement('div');
          selector.id = 'plane-selector';
          selector.style.cssText = 'position:absolute;bottom:55px;left:50%;transform:translateX(-50%);z-index:25;'
            + 'gap:8px;align-items:center;'
            + 'background:rgba(255,255,255,0.9);padding:6px 14px;border-radius:8px;'
            + 'border:1px solid rgba(0,0,0,0.12);font-size:13px;color:#333;';

          const prevBtn = document.createElement('button');
          prevBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m14.5 6-6 6 6 6"/></svg>';
          prevBtn.setAttribute('aria-label', 'Previous plane');
          prevBtn.style.cssText = 'border:none;background:transparent;cursor:pointer;font-size:16px;padding:2px 8px;';
          const nextBtn = document.createElement('button');
          nextBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9.5 6 6 6-6 6"/></svg>';
          nextBtn.setAttribute('aria-label', 'Next plane');
          nextBtn.style.cssText = 'border:none;background:transparent;cursor:pointer;font-size:16px;padding:2px 8px;';
          const label = document.createElement('span');
          label.id = 'plane-label';
          label.style.cssText = 'min-width:120px;text-align:center;font-weight:500;';

          // Edit mode toggle
          const editBtn = document.createElement('button');
          editBtn.innerHTML = '<svg viewBox="0 0 1024 1024" width="17" height="17" aria-hidden="true" fill="currentColor"><path d="M257.536 744.448 307.2 595.2 717.824 184.576a42.667 42.667 0 0 1 60.33 0l61.27 61.269a42.667 42.667 0 0 1 0 60.33L428.8 716.8l-149.248 49.664a17.067 17.067 0 0 1-22.016-22.016zM348.16 646.4l47.104 47.104 49.664-16.554-80.214-80.214L348.16 646.4z"></path></svg><span>Edit</span>';
          editBtn.title = 'Edit control points';
          editBtn.style.cssText = 'border:none;background:transparent;color:#333;cursor:pointer;font-size:11px;padding:4px 8px;border-radius:4px;display:flex;align-items:center;gap:4px;transition:all 0.15s;';
          const editLabel = editBtn.querySelector('span');
          editLabel.style.cssText = 'font-size:11px;line-height:17px;';

          const saveBtn = document.createElement('button');
          saveBtn.innerHTML = '<svg viewBox="0 0 1024 1024" width="18" height="18" aria-hidden="true" fill="currentColor"><path d="M170.666667 128h597.333333l115.498667 115.498667a42.666667 42.666667 0 0 1 12.501333 30.165333V853.333333a42.666667 42.666667 0 0 1-42.666667 42.666667H170.666667a42.666667 42.666667 0 0 1-42.666667-42.666667V170.666667a42.666667 42.666667 0 0 1 42.666667-42.666667z m128 42.666667v213.333333h384V170.666667H298.666667z m-42.666667 341.333333v298.666667h512v-298.666667H256z m298.666667-298.666667h85.333333v128h-85.333333V213.333333z"></path></svg>';
          saveBtn.title = 'Save current plane';
          saveBtn.innerHTML += '<span>Save</span>';
          saveBtn.style.cssText = 'border:none;background:transparent;color:#333;cursor:pointer;font-size:11px;padding:4px 8px;border-radius:4px;display:flex;align-items:center;gap:4px;transition:all 0.15s;';
          const saveLabel = saveBtn.querySelector('span');
          saveLabel.style.cssText = 'font-size:11px;line-height:18px;';

          saveBtn.addEventListener('click', () => {
            const po = planeObjects[activePlane];
            const cd = document.getElementById('case-data');
            if (!po || !po.planeData || !cd) return;
            saveBtn.disabled = true;
            saveLabel.textContent = '...';
            fetch('/api/resection-plane/save', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                output_dir: cd.dataset.outputDir || '',
                json_file: cd.dataset.jsonFile || '',
                plane_index: activePlane,
                control_points_3d: po.planeData.control_points_3d,
                candidate_name: po.planeData.candidate_name || '',
                source: 'three_d_editor'
              })
            }).then(r => r.json()).then(data => {
              if (data.status !== 'ok') throw new Error(data.message || 'Save failed');
              po.planeData.user_saved = true;
              po.planeData.saved_at = data.saved_at || new Date().toISOString();
              startCellPreviewCache.clear();
              saveLabel.textContent = 'Saved';
              saveBtn.style.background = '#34a853';
              saveBtn.style.color = '#fff';
              clearDisplayedSequence('Plane saved. Path planning is now available.', false);
            }).catch(err => {
              saveLabel.textContent = 'Save';
              alert('Plane save failed: ' + err.message);
            }).finally(() => { saveBtn.disabled = false; });
          });

          editBtn.addEventListener('click', () => {
            const po = planeObjects[activePlane];
            if (!po.hasCP) return;
            if (!isEditMode) {
              enterEditMode();
              editBtn.style.background = '#4285f4';
              editBtn.style.color = '#fff';
              editLabel.textContent = 'Done';
            } else {
              exitEditMode();
              editBtn.style.background = 'transparent';
              editBtn.style.color = '#333';
              editLabel.textContent = 'Edit';
            }
          });

          function updatePlaneView(idx) {
            if (isEditMode) { exitEditMode(); editBtn.style.background = 'transparent'; editBtn.style.color = '#333'; editLabel.textContent = 'Edit'; }
            planeObjects.forEach((po, i) => {
              const v = i === idx;
              po.mesh.visible = v;
              po.line.visible = v;
              if (po.gridHelper) po.gridHelper.visible = v;
              if (po.label) po.label.visible = v;
              po.cpGroup.visible = false;
            });
            const p = resectionPlanes[idx];
            const marginStr = p.margin_success ? 'SAFE' : 'RISK';
            label.textContent = (idx + 1) + '/' + resectionPlanes.length + ' '
              + marginStr + ' · ' + (p.margin_min_mm || 0).toFixed(1) + ' mm';
          }

          prevBtn.addEventListener('click', function() {
            activePlane = (activePlane - 1 + resectionPlanes.length) % resectionPlanes.length;
            updatePlaneView(activePlane);
          });
          nextBtn.addEventListener('click', function() {
            activePlane = (activePlane + 1) % resectionPlanes.length;
            updatePlaneView(activePlane);
          });

          updatePlaneView(0);
          const planeHeading = document.createElement('div');
          planeHeading.className = 'panel-heading plane-panel-heading';
          planeHeading.dataset.dragHandle = 'true';
          planeHeading.innerHTML = '<div><div class="panel-title">Resection Plane</div>'
            + '<div class="panel-subtitle">Candidate and safety margin</div></div>'
            + '<button id="plane-panel-toggle" type="button" aria-label="Collapse resection plane panel" aria-expanded="true">'
            + '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m15 6-6 6 6 6"></path></svg>'
            + '</button>';
          const planeControls = document.createElement('div');
          planeControls.className = 'plane-selector-controls';
          planeControls.appendChild(prevBtn);
          planeControls.appendChild(label);
          planeControls.appendChild(nextBtn);
          planeControls.appendChild(editBtn);
          planeControls.appendChild(saveBtn);
          selector.appendChild(planeHeading);
          selector.appendChild(planeControls);
          const rightWorkspace = document.getElementById('right-workspace');
          const sequencePanel = document.getElementById('path-planning-panel');
          if (rightWorkspace) rightWorkspace.insertBefore(selector, sequencePanel);
          else document.body.appendChild(selector);
        }

        // --- Color legend ---
        const legend = document.createElement('div');
        legend.id = 'color-legend';
        legend.style.cssText = 'position:absolute;bottom:80px;right:20px;z-index:25;'
          + 'background:rgba(255,255,255,0.92);border-radius:8px;padding:10px 14px;'
          + 'border:1px solid rgba(0,0,0,0.12);font-size:12px;color:#333;'
          + 'box-shadow:0 2px 8px rgba(0,0,0,0.08);';
        legend.innerHTML = '<div class="panel-heading distance-panel-heading" data-drag-handle="true">'
          + '<div><div class="panel-title">Distance to Tumor</div>'
          + '<div class="panel-subtitle">Safety margin bands</div></div>'
          + '<button id="distance-panel-toggle" type="button" aria-label="Collapse distance panel" aria-expanded="true">'
          + '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m15 6-6 6 6 6"></path></svg>'
          + '</button></div>'
          + '<div class="distance-legend-body">'
          + '<div class="distance-legend-row"><i class="distance-legend-swatch" style="background:#2878b8"></i><span>≥ 20 mm · Ample</span></div>'
          + '<div class="distance-legend-row"><i class="distance-legend-swatch" style="background:#2f8f6b"></i><span>10–20 mm · Safe</span></div>'
          + '<div class="distance-legend-row"><i class="distance-legend-swatch" style="background:#d89221"></i><span>5–10 mm · Caution</span></div>'
          + '<div class="distance-legend-row"><i class="distance-legend-swatch" style="background:#c43d3d"></i><span>&lt; 5 mm · Below margin</span></div>'
          + '</div>';
        const rightWorkspace = document.getElementById('right-workspace');
        const planeSelector = document.getElementById('plane-selector');
        const sequencePanel = document.getElementById('path-planning-panel');
        if (rightWorkspace) rightWorkspace.insertBefore(legend, planeSelector || sequencePanel);
        else document.body.appendChild(legend);
      });
    }

    // 4c. Auto-fit camera
    const box = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3());
    maxDim = Math.max(size.x, size.y, size.z) || 100;
    fitCameraToScene();

    // 4d. Build organ tree
    buildOrganTree();
  })
  .catch(err => {
    document.getElementById('info').textContent = 'Data error · ' + (err.message || err);
    console.error('3D data load error:', err);
  });

// ================================================================
//  5. Organ Tree Panel — built after data is loaded
// ================================================================
function buildOrganTree() {
  const treeEl = document.getElementById('organ-tree');
  const structureCount = document.getElementById('structure-count');
  if (structureCount) structureCount.textContent = String(MESH_DATA.length);

  const LESION_KEYS = ['tumor','cancer','metastasis','cyst','lesion','nodule'];
  const VESSEL_KEYS = ['hepatic','portal','vein','artery','vascular'];

  function classify(orig) {
    const n = (orig || '').toLowerCase();
    if (LESION_KEYS.some(k => n.includes(k))) return 'lesion';
    if (VESSEL_KEYS.some(k => n.includes(k))) return 'vessel';
    return 'organ';
  }

  const groups = { organ: [], lesion: [], vessel: [], plan: [] };
  const groupNames = { organ: 'Organs', lesion: 'Lesions', vessel: 'Vessels', plan: 'Plan' };
  const groupLabels = { organ: '', lesion: '', vessel: '', plan: '' };

  MESH_DATA.forEach((data, idx) => {
    const cat = classify(data.original_name);
    groups[cat].push({ ...data, idx });
  });

  // Add resection planes to a dedicated group
  meshes.forEach((m, idx) => {
    if (m.userData.type === 'resection_plane') {
      groups.plan.push({
        name: m.userData.name,
        color: '#44aaff',
        opacity: 0.45,
        volume_cm3: 0,
        idx: idx,
        original_name: 'resection_plane',
      });
    }
  });

  let totalVolume = 0;

  for (const [key, items] of Object.entries(groups)) {
    if (items.length === 0) continue;
    const groupDiv = document.createElement('div');
    groupDiv.className = 'tree-group';

    const header = document.createElement('div');
    header.className = 'tree-group-header';
    header.innerHTML = `<span class="arrow open" aria-hidden="true"></span>${groupLabels[key]} ${groupNames[key]} (${items.length})`;
    const body = document.createElement('div');
    body.className = 'tree-group-body';

    header.addEventListener('click', () => {
      body.classList.toggle('collapsed');
      header.querySelector('.arrow').classList.toggle('open');
    });

    items.forEach(d => {
      totalVolume += d.volume_cm3 || 0;
      const item = document.createElement('div');
      item.className = 'tree-item';
      item.dataset.idx = d.idx;

      const eye = document.createElement('span');
      eye.className = 'eye';
      eye.textContent = '';
      eye.title = 'Toggle visibility';
      eye.addEventListener('click', (e) => {
        e.stopPropagation();
        const m = meshes[d.idx];
        if (m) { m.visible = !m.visible; }
        eye.classList.toggle('hidden');
        item.classList.toggle('mesh-hidden');
      });

      const swatch = document.createElement('span');
      swatch.className = 'swatch';
      swatch.style.background = d.color;

      const label = document.createElement('span');
      label.className = 'label';
      label.textContent = d.name;
      label.title = d.name;

      const vol = document.createElement('span');
      vol.className = 'volume';
      const v = d.volume_cm3 || 0;
      if (v >= 100) vol.textContent = `${v.toFixed(0)} mL`;
      else if (v >= 1) vol.textContent = `${v.toFixed(1)} mL`;
      else if (v > 0) vol.textContent = `${v.toFixed(2)} mL`;
      else vol.textContent = '';

      const opacityDot = document.createElement('span');
      opacityDot.className = 'opacity-dot full';
      opacityDot.title = 'Toggle opacity';
      opacityDot.addEventListener('click', (e) => {
        e.stopPropagation();
        const m = meshes[d.idx];
        if (!m) return;
        const cur = m.material.opacity;
        if (cur >= 1) { m.material.opacity = 0.4; m.material.transparent = true; m.renderOrder = 1;
          opacityDot.className = 'opacity-dot half'; }
        else if (cur > 0) { m.material.opacity = 0; m.material.transparent = true;
          opacityDot.className = 'opacity-dot transp'; }
        else { m.material.opacity = m.userData.opacity;
          const isOpaque = m.userData.opacity >= 1;
          m.material.transparent = !isOpaque; m.renderOrder = isOpaque ? 0 : 1;
          opacityDot.className = 'opacity-dot full'; }
        m.material.needsUpdate = true;
      });

      item.appendChild(eye);
      item.appendChild(swatch);
      item.appendChild(label);
      item.appendChild(vol);
      item.appendChild(opacityDot);
      body.appendChild(item);
    });

    groupDiv.appendChild(header);
    groupDiv.appendChild(body);
    treeEl.appendChild(groupDiv);
  }

  if (totalVolume > 0) {
    const footer = document.createElement('div');
    footer.className = 'tree-total';
  footer.innerHTML = `<span>Total</span><span>${totalVolume >= 100 ? totalVolume.toFixed(0) : totalVolume.toFixed(1)} mL</span>`;
    treeEl.appendChild(footer);
  }
}

// ================================================================
//  6. Organ Tree Horizontal Resize
// ================================================================
const ORGAN_TREE_MIN_WIDTH = 190;
const ORGAN_TREE_MAX_WIDTH = 480;
const organTree = document.getElementById('organ-tree');
const organTreeResizeHandle = document.getElementById('organ-tree-resize-handle');

organTreeResizeHandle.addEventListener('pointerdown', (event) => {
  event.preventDefault();
  event.stopPropagation();
  organTreeResizeHandle.setPointerCapture(event.pointerId);
  organTree.classList.add('resizing');

  const startX = event.clientX;
  const startWidth = organTree.getBoundingClientRect().width;

  const onPointerMove = (moveEvent) => {
    const viewportMaximum = Math.max(
      0,
      window.innerWidth - organTree.offsetLeft - 24
    );
    const minimum = Math.min(ORGAN_TREE_MIN_WIDTH, viewportMaximum);
    const maximum = Math.min(ORGAN_TREE_MAX_WIDTH, viewportMaximum);
    const nextWidth = Math.min(
      maximum,
      Math.max(minimum, startWidth + moveEvent.clientX - startX)
    );
    organTree.style.width = `${nextWidth}px`;
  };

  const finishResize = () => {
    organTree.classList.remove('resizing');
    organTreeResizeHandle.removeEventListener('pointermove', onPointerMove);
    organTreeResizeHandle.removeEventListener('pointerup', finishResize);
    organTreeResizeHandle.removeEventListener('pointercancel', finishResize);
  };

  organTreeResizeHandle.addEventListener('pointermove', onPointerMove);
  organTreeResizeHandle.addEventListener('pointerup', finishResize);
  organTreeResizeHandle.addEventListener('pointercancel', finishResize);
});
// Organ tree resize end

// ================================================================
//  7. Toolbar actions
// ================================================================

// 7a. Reset View
document.getElementById('btn-reset').addEventListener('click', () => {
  camera.position.copy(INITIAL_CAM_POS);
  controls.target.set(0, 0, 0);
  controls.update();
});

// 7a-bis. Zoom in / out
function zoomCamera(factor) {
  const dir = new THREE.Vector3().subVectors(controls.target, camera.position).normalize();
  const dist = camera.position.distanceTo(controls.target);
  const newDist = dist * factor;
  camera.position.copy(controls.target).addScaledVector(dir, -newDist);
  controls.update();
}
document.getElementById('btn-zoom-in').addEventListener('click', () => zoomCamera(0.75));
document.getElementById('btn-zoom-out').addEventListener('click', () => zoomCamera(1.35));

// 7b. Auto-rotate
let isRotating = false;
const btnRotate = document.getElementById('btn-rotate');
btnRotate.addEventListener('click', () => {
  isRotating = !isRotating;
  controls.autoRotate = isRotating;
  controls.autoRotateSpeed = 2.0;
  btnRotate.classList.toggle('active', isRotating);
  btnRotate.title = isRotating ? 'Stop rotation' : 'Auto rotate';
  btnRotate.setAttribute('aria-label', btnRotate.title);
  btnRotate.setAttribute('aria-pressed', String(isRotating));
  const tooltip = btnRotate.querySelector('.tooltip');
  if (tooltip) tooltip.textContent = btnRotate.title;
});

// 7c. Background toggle
let isDarkBg = false;
const btnBg = document.getElementById('btn-bg');
btnBg.addEventListener('click', () => {
  isDarkBg = !isDarkBg;
  scene.background = new THREE.Color(isDarkBg ? '#1a1a2e' : ##BG_COLOR##);
  document.body.classList.toggle('dark-mode', isDarkBg);
  btnBg.title = isDarkBg ? 'Switch to light background' : 'Switch to dark background';
  btnBg.setAttribute('aria-label', btnBg.title);
  btnBg.setAttribute('aria-pressed', String(isDarkBg));
  const tooltip = btnBg.querySelector('.tooltip');
  if (tooltip) tooltip.textContent = btnBg.title;
});

// 7d. Screenshot
document.getElementById('btn-screenshot').addEventListener('click', () => {
  const oldRatio = renderer.getPixelRatio();
  // Render at 2x for high-res screenshot
  renderer.setPixelRatio(2);
  renderer.render(scene, camera);
  const link = document.createElement('a');
  link.download = `screenshot_3d_${Date.now()}.png`;
  link.href = renderer.domElement.toDataURL('image/png');
  link.click();
  renderer.setPixelRatio(oldRatio);
});

// ================================================================
//  8. Measurement Tool
// ================================================================
const measureManager = {
  active: false,
  points: [],
  markers: [],
  lines: [],
  labels: [],
  group: new THREE.Group(),
  pointerDown: { x: 0, y: 0 },

  init() {
    scene.add(this.group);
    this.bindEvents();
  },

  bindEvents() {
    const el = renderer.domElement;
    el.addEventListener('pointerdown', (e) => {
      this.pointerDown.x = e.clientX;
      this.pointerDown.y = e.clientY;
    });
    el.addEventListener('pointerup', (e) => {
      if (!this.active) return;
      const dx = e.clientX - this.pointerDown.x;
      const dy = e.clientY - this.pointerDown.y;
      if (Math.sqrt(dx*dx + dy*dy) > 5) return;
      this.handleClick(e);
    });
    el.addEventListener('contextmenu', (e) => {
      if (this.active) { e.preventDefault(); this.cancelPending(); }
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.active) this.cancelPending();
    });
  },

  handleClick(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera);
    const visibleMeshes = meshes.filter(m => m.visible);
    const intersects = raycaster.intersectObjects(visibleMeshes);
    if (intersects.length === 0) return;

    const point = intersects[0].point.clone();
    this.placePoint(point);
  },

  placePoint(point) {
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(maxDim * 0.004, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffdd44 })
    );
    marker.position.copy(point);
    this.group.add(marker);
    this.markers.push(marker);
    this.points.push(point);

    const statusEl = document.getElementById('measure-status');

    if (this.points.length === 1) {
      statusEl.textContent = 'MEASURE · Select the second point';
      statusEl.style.display = 'block';
    } else if (this.points.length === 2) {
      this.finalizeMeasurement(this.points[0], this.points[1]);
      this.points = [];
      statusEl.textContent = 'MEASURE · Select the first point to continue';
      document.getElementById('measure-clear-btn').style.display = 'block';
    }
  },

  finalizeMeasurement(pA, pB) {
    const dist = pA.distanceTo(pB);

    // Draw line
    const lineGeo = new THREE.BufferGeometry().setFromPoints([pA, pB]);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xffdd44 });
    const line = new THREE.Line(lineGeo, lineMat);
    this.group.add(line);
    this.lines.push(line);

    // Distance label (Sprite with canvas texture)
    const mid = new THREE.Vector3().addVectors(pA, pB).multiplyScalar(0.5);
    const distText = dist >= 100 ? `${(dist/10).toFixed(1)} cm` : `${dist.toFixed(1)} mm`;
    const label = this.makeTextSprite(distText);
    label.position.copy(mid);
    this.group.add(label);
    this.labels.push(label);
  },

  makeTextSprite(message) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 512;
    canvas.height = 96;
    ctx.clearRect(0, 0, 512, 96);
    // Background rect (fully opaque)
    ctx.fillStyle = 'rgba(0,0,0,0.88)';
    ctx.beginPath();
    ctx.roundRect(10, 8, 492, 54, 10);
    ctx.fill();
    // Text (bigger font)
    ctx.font = 'Bold 30px Arial';
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(message, 256, 38);
    // Texture
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, depthTest: false, sizeAttenuation: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(maxDim * 0.12, maxDim * 0.035, 1);
    return sprite;
  },

  cancelPending() {
    this.points = [];
    if (this.markers.length > 0) {
      const lastMarker = this.markers.pop();
      this.group.remove(lastMarker);
      lastMarker.geometry.dispose();
      lastMarker.material.dispose();
    }
    document.getElementById('measure-status').style.display = 'none';
  },

  clearAll() {
    while (this.markers.length) {
      const m = this.markers.pop();
      this.group.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
    while (this.lines.length) {
      const l = this.lines.pop();
      this.group.remove(l);
      l.geometry.dispose();
      l.material.dispose();
    }
    while (this.labels.length) {
      const l = this.labels.pop();
      this.group.remove(l);
      l.material.map?.dispose();
      l.material.dispose();
    }
    this.points = [];
    document.getElementById('measure-status').style.display = 'none';
    document.getElementById('measure-clear-btn').style.display = 'none';
  },

  meshHasMeasurement(mesh) { return false; },

  removeMeasurementsForMesh(mesh) {},
};

measureManager.init();

// Mount measurement clear button
document.getElementById('measure-clear-btn').addEventListener('click', () => {
  measureManager.clearAll();
});

// Toggle measurement mode
const btnMeasure = document.getElementById('btn-measure');
let isMeasureMode = false;
btnMeasure.addEventListener('click', () => {
  isMeasureMode = !isMeasureMode;
  measureManager.active = isMeasureMode;
  btnMeasure.classList.toggle('measuring', isMeasureMode);
  btnMeasure.setAttribute('aria-pressed', String(isMeasureMode));
  renderer.domElement.style.cursor = isMeasureMode ? 'crosshair' : 'default';
  if (isMeasureMode) {
    document.getElementById('measure-status').textContent = 'MEASURE · Select the first point';
    document.getElementById('measure-status').style.display = 'block';
  } else {
    measureManager.cancelPending();
    document.getElementById('measure-status').style.display = 'none';
  }
});

// ================================================================
//  9a. Annotation Manager
// ================================================================
let annotateIdCounter = 0;
const annotationManager = {
  active: false,
  annotations: [],
  group: new THREE.Group(),
  pendingPoint: null,

  init() {
    scene.add(this.group);
    this.bindEvents();
  },

  bindEvents() {
    const el = renderer.domElement;
    el.addEventListener('pointerup', (e) => {
      if (!this.active) return;
      const dx = e.clientX - measureManager.pointerDown.x;
      const dy = e.clientY - measureManager.pointerDown.y;
      if (Math.sqrt(dx*dx + dy*dy) > 5) return;
      this.handleClick(e);
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.active && !this.pendingPoint) {
        this.deactivate();
      }
    });
  },

  handleClick(event) {
    if (this.pendingPoint) return; // already awaiting text input
    const rect = renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera);
    const visibleMeshes = meshes.filter(m => m.visible);
    const intersects = raycaster.intersectObjects(visibleMeshes);
    if (intersects.length === 0) return;

    const point = intersects[0].point.clone();
    this.showInput(point);
  },

  showInput(point) {
    this.pendingPoint = point;
    document.getElementById('annotation-input-wrap').style.display = 'block';
    const textarea = document.getElementById('annotation-text');
    textarea.value = '';
    textarea.focus();
    document.getElementById('annotation-status').style.display = 'none';
  },

  confirmAnnotation() {
    const text = document.getElementById('annotation-text').value.trim();
    if (!text || !this.pendingPoint) {
      this.hideInput();
      return;
    }
    annotateIdCounter++;
    const point = this.pendingPoint;
    this.pendingPoint = null;

    // Marker sphere (red)
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(maxDim * 0.010, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xff4444 })
    );
    marker.position.copy(point);
    this.group.add(marker);

    // Label sprite
    const labelText = `#${annotateIdCounter} ${text}`;
    const label = this.makeTextSprite(labelText);
    label.position.copy(point.clone().add(new THREE.Vector3(0, maxDim * 0.025, 0)));
    this.group.add(label);

    this.annotations.push({
      id: annotateIdCounter,
      position: point.clone(),
      text: text,
      marker: marker,
      label: label,
    });

    this.hideInput();
    document.getElementById('annotation-status').textContent = `ANNOTATE · Note ${annotateIdCounter} placed; select another surface`;
    document.getElementById('annotation-status').style.display = 'block';
    document.getElementById('annotation-clear-btn').style.display = 'block';
  },

  cancelAnnotation() {
    this.pendingPoint = null;
    this.hideInput();
    if (this.annotations.length === 0) {
      document.getElementById('annotation-clear-btn').style.display = 'none';
    }
    if (this.active) {
      document.getElementById('annotation-status').textContent = 'ANNOTATE · Select an organ surface';
      document.getElementById('annotation-status').style.display = 'block';
    }
  },

  hideInput() {
    document.getElementById('annotation-input-wrap').style.display = 'none';
  },

  makeTextSprite(message) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 640;
    canvas.height = 120;
    ctx.clearRect(0, 0, 640, 120);

    // Measure text width
    ctx.font = 'Bold 30px Arial';
    const tw = ctx.measureText(message).width;
    const bw = Math.max(tw + 40, 100);
    const bx = 8;

    // Background (fully opaque red)
    ctx.fillStyle = 'rgba(200,50,50,1.0)';
    ctx.beginPath();
    ctx.roundRect(bx, 10, bw + 14, 54, 10);
    ctx.fill();

    // Text (bigger font)
    ctx.font = 'Bold 30px Arial';
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(message, bx + 20, 42);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, depthTest: false, sizeAttenuation: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(maxDim * 0.28, maxDim * 0.045, 1);
    return sprite;
  },

  clearAll() {
    while (this.annotations.length) {
      const a = this.annotations.pop();
      this.group.remove(a.marker);
      a.marker.geometry.dispose();
      a.marker.material.dispose();
      this.group.remove(a.label);
      a.label.material.map?.dispose();
      a.label.material.dispose();
    }
    document.getElementById('annotation-clear-btn').style.display = 'none';
    document.getElementById('annotation-status').style.display = 'none';
  },

  deactivate() {
    this.active = false;
    this.pendingPoint = null;
    this.hideInput();
    document.getElementById('annotation-status').style.display = 'none';
    btnAnnotate.classList.remove('active');
    btnAnnotate.setAttribute('aria-pressed', 'false');
    renderer.domElement.style.cursor = 'default';
  },

  activate() {
    this.active = true;
    btnAnnotate.classList.add('active');
    btnAnnotate.setAttribute('aria-pressed', 'true');
    renderer.domElement.style.cursor = 'crosshair';
    document.getElementById('annotation-status').textContent = 'ANNOTATE · Select an organ surface';
    document.getElementById('annotation-status').style.display = 'block';
  }
};

annotationManager.init();

// Annotation UI buttons
document.getElementById('annotation-confirm').addEventListener('click', () => annotationManager.confirmAnnotation());
document.getElementById('annotation-cancel').addEventListener('click', () => annotationManager.cancelAnnotation());
document.getElementById('annotation-text').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); annotationManager.confirmAnnotation(); }
  if (e.key === 'Escape') { annotationManager.cancelAnnotation(); }
});
document.getElementById('annotation-clear-btn').addEventListener('click', () => annotationManager.clearAll());

// Toggle annotation mode
const btnAnnotate = document.getElementById('btn-annotate');
btnAnnotate.addEventListener('click', () => {
  if (annotationManager.active) {
    annotationManager.deactivate();
  } else {
    // Deactivate measure mode if active
    if (isMeasureMode) btnMeasure.click();
    annotationManager.activate();
  }
});

// ================================================================
//  9b. Resize handler
// ================================================================
window.addEventListener('resize', () => {
  resizeRendererToView();
});
// ================================================================
//  10. Render loop
// ================================================================
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ================================================================
//  11. Resection plane toggle — show/hide or compute on demand
// ================================================================
document.getElementById('toggle-resection-plane').addEventListener('change', function () {
  const loadingEl = document.getElementById('resection-loading');
  const toggle = this;
  if (this.checked) {
    if (planeObjects.length > 0) {
      // Planes already rendered — show only active plane
      planeObjects.forEach(function (po, i) {
        var v = i === activePlane;
        po.mesh.visible = v;
        po.line.visible = v;
        if (po.gridHelper) po.gridHelper.visible = v;
        if (po.label) po.label.visible = v;
        po.cpGroup.visible = false;
      });
      var sel = document.getElementById('plane-selector');
      if (sel) sel.style.display = 'block';
      var leg = document.getElementById('color-legend');
      if (leg) leg.style.display = 'block';
      var pathPanel = document.getElementById('path-planning-panel');
      var pathToggle = document.getElementById('toggle-path-planning');
      if (pathPanel) {
        pathPanel.style.display = pathToggle && pathToggle.checked
          && pathPanel.dataset.available === 'true' ? 'flex' : 'none';
      }
      document.dispatchEvent(new CustomEvent('resection-workspace-open-request', {
        detail: {open: true}
      }));
    } else {
      // No planes yet — call API to compute on demand
      if (loadingEl) loadingEl.style.display = 'flex';
      toggle.disabled = true;
      const controller = new AbortController();
      resectionPlanAbortController = controller;
      var cd = document.getElementById('case-data');
      fetch('/api/plan-resection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          case_dir: cd ? cd.dataset.maskDir : '',
          output_dir: cd ? cd.dataset.outputDir : '',
          case_name: cd ? cd.dataset.caseName : '',
          tumor_margin_mm: 5.0
        })
      })
      .then(function (r) {
        if (!r.ok) {
          return r.text().then(function (bodyText) {
            var detail = bodyText || ('HTTP ' + r.status);
            try { detail = JSON.parse(bodyText).message || detail; } catch (e) { /* keep text */ }
            throw new Error('请求失败 (' + r.status + '): ' + detail);
          });
        }
        return r.json();
      })
      .then(function (data) {
        if (resectionPlanAbortController !== controller) return;
        if (data.json_updated || data.resection_plane_count > 0) {
          // Computation succeeded — reload to show new planes
          window.location.reload();
        } else {
          if (loadingEl) loadingEl.style.display = 'none';
          toggle.checked = false;
          alert('Resection plane computation failed: ' + (data.message || 'Unknown error'));
        }
      })
      .catch(function (err) {
        if (err.name === 'AbortError') return;
        if (resectionPlanAbortController !== controller) return;
        if (loadingEl) loadingEl.style.display = 'none';
        toggle.checked = false;
        alert('Network request failed: ' + err.message);
      })
      .finally(function () {
        if (resectionPlanAbortController === controller) {
          resectionPlanAbortController = null;
          toggle.disabled = false;
        }
      });
    }
  } else {
    // Hide all planes
    planeObjects.forEach(function (po) {
      po.mesh.visible = false;
      po.line.visible = false;
      if (po.gridHelper) po.gridHelper.visible = false;
      if (po.label) po.label.visible = false;
      po.cpGroup.visible = false;
    });
    var sel = document.getElementById('plane-selector');
    if (sel) sel.style.display = 'none';
    var leg = document.getElementById('color-legend');
    if (leg) leg.style.display = 'none';
    var pathPanel = document.getElementById('path-planning-panel');
    if (pathPanel) pathPanel.style.display = 'none';
    var pathToggle = document.getElementById('toggle-path-planning');
    if (pathToggle) {
      pathToggle.checked = false;
    }
  }
});

document.getElementById('cancel-resection-computation').addEventListener('click', function () {
  if (!resectionPlanAbortController) return;
  resectionPlanAbortController.abort();
  resectionPlanAbortController = null;
  const loadingEl = document.getElementById('resection-loading');
  if (loadingEl) loadingEl.style.display = 'none';
  const toggle = document.getElementById('toggle-resection-plane');
  toggle.checked = false;
  toggle.disabled = false;
});"""

def _make_threejs_html(title="3D Segmentation Visualization",
                        bg_color="#f5f5f8", three_sources=None,
                        json_url=None, bezier_source=None,
                        case_name="", mask_dir="", output_dir=""):
    """
    Generate Three.js HTML that fetches mesh data from a JSON file.

    Args:
        title: page title
        bg_color: background color hex
        three_sources: dict from _read_threejs_sources(), or None for CDN
        json_url: URL for the JSON data file (relative or absolute)
            Example: "P026_3d.json" (co-located with HTML)
        bezier_source: string content of BezierSurface.js, or None to skip

    Returns:
        HTML string
    """
    import json

    if json_url is None:
        json_url = "data_3d.json"

    ui_css_path = Path(__file__).resolve().parent / "three" / "viewer-ui.css"
    ui_css = ui_css_path.read_text(encoding="utf-8") if ui_css_path.exists() else ""

    # ---- Build import section: local inline or CDN fallback ----
    if three_sources:
        ts_src = three_sources["three"]
        oc_src = three_sources["orbit_controls"]
        dc_src = three_sources.get("drag_controls", "")
        # BezierSurface.js is always embedded as Blob URL
        bs_src = bezier_source or ""
        bezier_blob_script = (
            f'<script type="text/plain" id="bs-source">\n'
            f'{bs_src}\n'
            f'</script>\n'
        ) if bs_src else ""

        blob_bezier_var = (
            "var sB=document.getElementById('bs-source').textContent;"
            "var uB=URL.createObjectURL(new Blob([sB],{type:'application/javascript'}));"
        ) if bs_src else ""

        blob_drag_var = (
            "var sD=document.getElementById('dc-source').textContent;"
            "var uD=URL.createObjectURL(new Blob([sD],{type:'application/javascript'}));"
        ) if dc_src else ""

        # Build extra import map entries
        extra_parts = []
        if bs_src:
            extra_parts.append('"BezierSurface":uB')
        if dc_src:
            extra_parts.append('"three/addons/controls/DragControls.js":uD')
        extra_imports_blob = ", " + ", ".join(extra_parts) if extra_parts else ""

        dc_source_block = (f'<script type="text/plain" id="dc-source">\n{dc_src}\n</script>\n') if dc_src else ""
        import_section = (
            f'<script type="text/plain" id="ts-three-source">\n'
            f'{ts_src}\n'
            f'</script>\n'
            f'<script type="text/plain" id="ts-orbit-source">\n'
            f'{oc_src}\n'
            f'</script>\n'
            f'{dc_source_block}'
            f'{bezier_blob_script}'
            f'<script>\n'
            f'(function(){{\n'
            f'  var s3 = document.getElementById("ts-three-source").textContent;\n'
            f'  var sO = document.getElementById("ts-orbit-source").textContent;\n'
            f'  var u3 = URL.createObjectURL(new Blob([s3], {{type:"application/javascript"}}));\n'
            f'  var uO = URL.createObjectURL(new Blob([sO], {{type:"application/javascript"}}));\n'
            f'{blob_drag_var}'
            f'{blob_bezier_var}'
            f'  var im = document.createElement("script");\n'
            f'  im.type = "importmap";\n'
            f'  im.textContent = JSON.stringify({{imports:{{"three":u3,"three/addons/controls/OrbitControls.js":uO{extra_imports_blob}}}}});\n'
            f'  document.currentScript.after(im);\n'
            f'}})();\n'
            f'</script>\n'
        )
    else:
        # CDN mode: "three/addons/" wildcard already covers DragControls
        extra_imports_cdn = ', "BezierSurface": "./BezierSurface.js"' if bezier_source else ""
        import_section = (
            '<script type="importmap">\n'
            '{\n'
            '  "imports": {\n'
            '    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",\n'
            '    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"'
            f'{extra_imports_cdn}\n'
            '  }\n'
            '}\n'
            '</script>\n'
        )
    # ---- Build JS module code (separate from f-string) ----
    bg_color_q = json.dumps(bg_color)
    js_url_q = json.dumps(json_url)
    js_module_code = _JS_MODULE_TEMPLATE.replace("##BG_COLOR##", bg_color_q).replace("##DATA_URL##", js_url_q)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: {bg_color}; overflow: hidden; font-family: 'Segoe UI', Arial, sans-serif;
          transition: background 0.3s; }}

  /* ===== Info bar ===== */
  #info {{
    position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
    color: rgba(0,0,0,0.35); font-size: 13px; z-index: 20;
    pointer-events: none; user-select: none;
    background: rgba(255,255,255,0.7); padding: 4px 16px; border-radius: 4px;
    white-space: nowrap; transition: color 0.3s, background 0.3s;
  }}

  /* ===== Toolbar (top center, below info) ===== */
  #toolbar {{
    position: absolute; top: 44px; left: 50%; transform: translateX(-50%);
    display: flex; gap: 6px; z-index: 20; align-items: center;
    background: rgba(255,255,255,0.88); padding: 4px 10px; border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.1); transition: background 0.3s, border-color 0.3s;
  }}
  #toolbar .sep {{ width: 1px; height: 24px; background: rgba(0,0,0,0.12); margin: 0 4px; }}
  .tb-btn {{
    width: 34px; height: 34px; border-radius: 6px; border: 1px solid transparent;
    background: transparent; cursor: pointer; font-size: 18px; line-height: 1;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.15s; color: #444; position: relative;
  }}
  .tb-btn:hover {{ background: rgba(0,0,0,0.06); border-color: rgba(0,0,0,0.15); }}
  .tb-btn:active {{ background: rgba(0,0,0,0.10); }}
  .tb-btn.active {{ background: rgba(66,133,244,0.15); border-color: #4285f4; color: #4285f4; }}
  .tb-btn.measuring {{ background: rgba(234,67,53,0.15); border-color: #ea4335; color: #ea4335; }}
  .tb-btn .tooltip {{
    display: none; position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,0.8); color: #fff; font-size: 11px; padding: 2px 8px;
    border-radius: 4px; white-space: nowrap; margin-top: 4px; pointer-events: none; z-index: 30;
  }}
  .tb-btn:hover .tooltip {{ display: block; }}

  /* ===== Organ Tree Panel (left) ===== */
  #organ-tree {{
    position: absolute; top: 60px; left: 12px; width: 236px;
    min-width: min(190px, calc(100vw - 36px));
    max-width: min(480px, calc(100vw - 36px));
    max-height: calc(100vh - 80px); z-index: 15;
    background: rgba(255,255,255,0.92); border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.1); padding: 6px 0;
    overflow-y: auto; user-select: none;
    font-size: 13px; color: #2a3f5f; transition: background 0.3s, border-color 0.3s, color 0.3s;
  }}
  #organ-tree::-webkit-scrollbar {{ width: 4px; }}
  #organ-tree::-webkit-scrollbar-thumb {{ background: rgba(0,0,0,0.15); border-radius: 2px; }}
  #organ-tree-resize-handle {{
    position: absolute; top: 0; right: 0; width: 8px; height: 100%;
    z-index: 2; cursor: ew-resize; touch-action: none;
  }}
  #organ-tree-resize-handle::after {{
    content: ''; position: absolute; top: 10px; bottom: 10px; left: 3px;
    width: 2px; border-radius: 2px; background: transparent;
    transition: background 0.15s;
  }}
  #organ-tree:hover #organ-tree-resize-handle::after,
  #organ-tree.resizing #organ-tree-resize-handle::after {{
    background: rgba(66,133,244,0.45);
  }}
  #organ-tree.resizing {{ cursor: ew-resize; }}
  .tree-group {{ margin: 2px 0; }}
  .tree-group-header {{
    display: flex; align-items: center; gap: 4px; padding: 4px 10px;
    font-weight: 600; font-size: 12px; color: #888; cursor: pointer;
    letter-spacing: 0.5px;
  }}
  .tree-group-header:hover {{ color: #555; }}
  .tree-group-header .arrow {{ font-size: 10px; transition: transform 0.2s; display: inline-block; }}
  .tree-group-header .arrow.open {{ transform: rotate(90deg); }}
  .tree-group-body {{ }}
  .tree-group-body.collapsed {{ display: none; }}

  .tree-item {{
    display: flex; align-items: center; gap: 6px; padding: 3px 10px 3px 18px;
    cursor: pointer; border-radius: 3px; transition: background 0.15s;
  }}
  .tree-item:hover {{ background: rgba(0,0,0,0.04); }}
  .tree-item .swatch {{
    width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0;
    border: 1px solid rgba(0,0,0,0.15); transition: opacity 0.2s;
  }}
  .tree-item .eye {{
    font-size: 14px; flex-shrink: 0; width: 18px; text-align: center;
    opacity: 0.5; transition: opacity 0.15s; cursor: pointer;
  }}
  .tree-item .eye:hover {{ opacity: 1; }}
  .tree-item .eye.hidden {{ opacity: 0.2; }}
  .tree-item .label {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .tree-item .volume {{
    font-size: 11px; color: #999; flex-shrink: 0; margin-left: auto;
    font-variant-numeric: tabular-nums;
  }}
  .tree-item .opacity-dot {{
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; cursor: pointer;
    border: 1px solid rgba(0,0,0,0.2); transition: all 0.2s;
  }}
  .tree-item .opacity-dot.full {{ background: #444; }}
  .tree-item .opacity-dot.half {{ background: #aaa; }}
  .tree-item .opacity-dot.transp {{ background: transparent; }}
  .tree-item.mesh-hidden .swatch {{ opacity: 0.25; }}
  .tree-item.mesh-hidden .label {{ opacity: 0.4; text-decoration: line-through; }}

  .tree-total {{
    border-top: 1px solid rgba(0,0,0,0.08); margin-top: 4px; padding: 6px 10px 2px;
    display: flex; justify-content: space-between; font-size: 12px; color: #666; font-weight: 500;
  }}

  /* ===== Measurement UI ===== */
  #measure-status {{
    position: absolute; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,0.7); color: #fff; font-size: 13px; padding: 6px 18px;
    border-radius: 20px; z-index: 20; display: none; pointer-events: none;
    white-space: nowrap;
  }}
  #measure-clear-btn {{
    position: absolute; bottom: 24px; right: 24px; z-index: 20;
    background: rgba(255,255,255,0.9); border: 1px solid rgba(0,0,0,0.15);
    border-radius: 6px; padding: 6px 14px; font-size: 12px; cursor: pointer;
    display: none; color: #444;
  }}
  #measure-clear-btn:hover {{ background: #fff; }}

  /* ===== Annotation UI ===== */
  #annotation-input-wrap {{
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    z-index: 30; background: rgba(255,255,255,0.95); border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.15); padding: 14px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    width: 280px;
  }}
  #annotation-input-box {{ display: flex; flex-direction: column; gap: 8px; }}
  #annotation-text {{
    width: 100%; border: 1px solid rgba(0,0,0,0.2); border-radius: 6px;
    padding: 8px 10px; font-size: 13px; font-family: inherit; resize: none;
    outline: none; background: #fff; color: #333;
  }}
  #annotation-text:focus {{ border-color: #4285f4; }}
  #annotation-input-actions {{ display: flex; gap: 8px; justify-content: flex-end; }}
  #annotation-input-actions button {{
    padding: 5px 16px; border-radius: 6px; border: 1px solid rgba(0,0,0,0.15);
    font-size: 12px; cursor: pointer; background: #f5f5f5; color: #333;
  }}
  #annotation-input-actions button:hover {{ background: #e8e8e8; }}
  #annotation-input-actions #annotation-confirm {{ background: #4285f4; color: #fff; border-color: #4285f4; }}
  #annotation-input-actions #annotation-confirm:hover {{ background: #3367d6; }}
  #annotation-status {{
    position: absolute; bottom: 70px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,0.7); color: #fff; font-size: 13px; padding: 6px 18px;
    border-radius: 20px; z-index: 20; display: none; pointer-events: none;
    white-space: nowrap;
  }}
  #annotation-clear-btn {{
    position: absolute; bottom: 24px; right: 100px; z-index: 20;
    background: rgba(255,255,255,0.9); border: 1px solid rgba(0,0,0,0.15);
    border-radius: 6px; padding: 6px 14px; font-size: 12px; cursor: pointer;
    display: none; color: #444;
  }}
  #annotation-clear-btn:hover {{ background: #fff; }}

  /* ===== Dark mode overrides for annotation ===== */
  body.dark-mode #annotation-input-wrap {{ background: rgba(35,35,55,0.95); border-color: rgba(255,255,255,0.12); }}
  body.dark-mode #annotation-text {{ background: #2a2a45; color: #e0e0e0; border-color: rgba(255,255,255,0.15); }}
  body.dark-mode #annotation-input-actions button {{ background: #3a3a55; color: #ccc; border-color: rgba(255,255,255,0.1); }}
  body.dark-mode #annotation-clear-btn {{ background: rgba(40,40,60,0.9); border-color: rgba(255,255,255,0.15); color: #ccc; }}
  body.dark-mode #annotation-clear-btn:hover {{ background: rgba(60,60,80,0.95); }}
  body.dark-mode {{ background: #1a1a2e; }}
  body.dark-mode #info {{ color: rgba(255,255,255,0.4); background: rgba(0,0,0,0.5); }}
  body.dark-mode #toolbar {{ background: rgba(30,30,50,0.88); border-color: rgba(255,255,255,0.1); }}
  body.dark-mode .tb-btn {{ color: #ccc; }}
  body.dark-mode .tb-btn:hover {{ background: rgba(255,255,255,0.08); }}
  body.dark-mode .tb-btn.active {{ background: rgba(66,133,244,0.25); }}
  body.dark-mode .tb-btn.measuring {{ background: rgba(234,67,53,0.25); }}
  body.dark-mode #organ-tree {{ background: rgba(25,25,45,0.94); border-color: rgba(255,255,255,0.08); color: #d0d0e0; }}
  body.dark-mode .tree-group-header {{ color: #888; }}
  body.dark-mode .tree-group-header:hover {{ color: #bbb; }}
  body.dark-mode .tree-item:hover {{ background: rgba(255,255,255,0.05); }}
  body.dark-mode .tree-item .volume {{ color: #777; }}
  body.dark-mode .tree-total {{ border-top-color: rgba(255,255,255,0.08); color: #999; }}
  body.dark-mode #measure-clear-btn {{ background: rgba(40,40,60,0.9); border-color: rgba(255,255,255,0.15); color: #ccc; }}
  body.dark-mode #measure-clear-btn:hover {{ background: rgba(60,60,80,0.95); }}
  body.dark-mode #color-legend {{ background: rgba(35,35,55,0.92); border-color: rgba(255,255,255,0.12); color: #d0d0e0; }}
  body.dark-mode #plane-selector {{ background: rgba(30,30,50,0.88); border-color: rgba(255,255,255,0.1); color: #d0d0e0; }}

  /* ===== Resection plane toggle (top-right) ===== */
  #resection-toggle {{
    position: absolute; top: 12px; right: 12px; z-index: 25;
    background: rgba(255,255,255,0.9); border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.12); padding: 6px 14px;
    font-size: 13px; color: #333; display: none;
    cursor: pointer; user-select: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: background 0.3s, color 0.3s, border-color 0.3s;
    align-items: center; gap: 6px;
  }}
  #resection-toggle input[type="checkbox"] {{
    width: 16px; height: 16px; cursor: pointer; vertical-align: middle;
    accent-color: #4285f4;
  }}
  #resection-toggle label {{ cursor: pointer; display: flex; align-items: center; gap: 6px; }}
  body.dark-mode #resection-toggle {{
    background: rgba(30,30,50,0.88); border-color: rgba(255,255,255,0.1); color: #d0d0e0;
  }}
  /* ===== Loading overlay ===== */
  #resection-loading {{
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    z-index: 100; background: rgba(0,0,0,0.55);
    display: none; flex-direction: column; align-items: center;
    justify-content: center; color: #fff; font-size: 18px;
    font-family: 'Segoe UI', Arial, sans-serif;
  }}
  #resection-loading .spinner {{
    width: 48px; height: 48px; border: 4px solid rgba(255,255,255,0.3);
    border-top-color: #4285f4; border-radius: 50%;
    animation: spin 0.8s linear infinite; margin-bottom: 20px;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  #resection-loading .loading-text {{ margin-bottom: 8px; font-weight: 500; }}
  #resection-loading .loading-sub {{ font-size: 14px; opacity: 0.7; }}
  #cancel-resection-computation {{
    margin-top: 18px; padding: 8px 20px; border: 1px solid rgba(255,255,255,0.8);
    border-radius: 6px; background: rgba(255,255,255,0.16); color: #fff;
    cursor: pointer; font-size: 14px;
  }}
  #cancel-resection-computation:hover {{ background: rgba(255,255,255,0.28); }}

  /* ===== Layout containers ===== */
  #view-3d, #view-slice, #view-combined {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: none; }}
  #view-3d.active, #view-slice.active, #view-combined.active {{ display: block; }}
  #view-slice {{ display: none; flex-direction: column; align-items: center; justify-content: center; }}
  #view-slice.active {{ display: flex; }}
  #slice-viewer {{ display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 20px; }}

  /* ===== Compact layout for 1/3 screen (narrow viewport) ===== */
  @media (max-width: 800px) {{
    #info {{ font-size: 11px; padding: 2px 10px; top: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 70vw; }}
    #toolbar {{ top: 32px; padding: 2px 5px; gap: 2px; }}
    #toolbar .sep {{ height: 16px; margin: 0 2px; }}
    .tb-btn {{ width: 24px; height: 24px; font-size: 12px; }}
    .tb-btn .tooltip {{ display: none !important; }}

    #organ-tree {{ top: 48px; left: 6px; width: 130px; font-size: 11px; }}
    #organ-tree .tree-group-header {{ padding: 3px 6px; font-size: 11px; }}
    #organ-tree .tree-item {{ padding: 2px 6px 2px 10px; gap: 4px; }}
    #organ-tree .tree-item .swatch {{ width: 10px; height: 10px; }}
    #organ-tree .tree-item .eye {{ font-size: 12px; width: 14px; }}
    #organ-tree .tree-item .volume {{ font-size: 10px; }}

    #plane-selector {{ font-size: 12px; padding: 3px 8px; bottom: 8px; }}
    #plane-selector button {{ font-size: 13px !important; padding: 1px 5px !important; }}

    #color-legend {{ padding: 5px 8px; font-size: 11px; bottom: 50px; right: 8px; }}
    #color-legend div {{ margin: 2px 0 !important; }}

    #measure-status, #annotation-status {{ font-size: 11px; padding: 3px 10px; bottom: 14px; }}
    #measure-clear-btn, #annotation-clear-btn {{ padding: 3px 8px; font-size: 11px; bottom: 14px; right: 10px; }}
    #annotation-clear-btn {{ right: 80px; }}
  }}

  /* ===== Extra compact for very narrow (e.g. ~500px) ===== */
  @media (max-width: 550px) {{
    #organ-tree {{ width: 110px; font-size: 10px; left: 4px; top: 42px; }}
    #organ-tree .tree-item {{ padding: 1px 4px 1px 8px; }}
    #organ-tree .tree-item .swatch {{ width: 8px; height: 8px; }}
    #organ-tree .tree-item .eye {{ display: none; }}
    #color-legend {{ display: none; }}
    #plane-selector {{ font-size: 11px; padding: 2px 6px; }}
    .tb-btn {{ width: 22px; height: 22px; font-size: 11px; }}
  }}
  #slice-viewer img {{ max-width: 90vw; max-height: 70vh; border-radius: 4px; border: 1px solid rgba(0,0,0,0.15); }}
  #slice-controls {{ display: flex; align-items: center; gap: 12px; font-size: 14px; color: #555; }}
  #slice-controls button {{ width: 36px; height: 36px; border-radius: 6px; border: 1px solid rgba(0,0,0,0.12);
    background: rgba(255,255,255,0.9); cursor: pointer; font-size: 16px; }}
  #slice-controls button:hover {{ background: #fff; }}
  #view-combined {{ display: none; }}
  #view-combined.active {{ display: flex; flex-direction: row; }}
  #combined-3d {{ flex: 1; position: relative; }}
  #combined-slice {{ flex: 1; display: flex; align-items: center; justify-content: center; padding: 20px; }}
  #combined-slice img {{ max-width: 100%; max-height: 90vh; border-radius: 4px; border: 1px solid rgba(0,0,0,0.15); }}

  #path-planning-panel {{
    position: absolute; left: 16px; bottom: 72px; z-index: 25;
    display: none; flex-direction: column; align-items: stretch; gap: 8px;
    width: min(276px, calc(100vw - 32px)); max-height: min(40vh, 330px);
    overflow-y: auto; box-sizing: border-box;
    padding: 12px; color: #243247; font-size: 12px;
    background: rgba(248, 251, 255, 0.96); border: 1px solid #d8e2ef;
    border-radius: 12px; box-shadow: 0 8px 28px rgba(30, 52, 80, 0.18);
    backdrop-filter: blur(10px);
  }}
  #path-planning-panel .path-panel-title {{ display: flex; align-items: center; justify-content: space-between; font-size: 13px; font-weight: 700; color: #163a63; cursor: grab; user-select: none; touch-action: none; }}
  #path-planning-panel.dragging .path-panel-title {{ cursor: grabbing; }}
  #path-planning-panel .path-panel-collapse {{ min-height: 22px; width: 24px; padding: 0; border: none; background: transparent; color: #52657b; font-size: 16px; line-height: 20px; }}
  #path-planning-panel .path-panel-body {{ display: flex; flex-direction: column; gap: 8px; }}
  #path-planning-panel.collapsed .path-panel-body {{ display: none; }}
  #path-advanced {{ border-top: 1px solid #e0e8f1; padding-top: 6px; }}
  #path-advanced summary {{ color: #52657b; cursor: pointer; font-weight: 600; user-select: none; }}
  #path-advanced .path-panel-section {{ margin-top: 7px; }}
  #path-planning-panel .path-panel-section {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
  #path-planning-panel label {{ display: flex; align-items: center; justify-content: space-between; gap: 6px; color: #52657b; }}
  #path-planning-panel button, #path-planning-panel select, #path-planning-panel input {{ min-height: 28px; box-sizing: border-box; border: 1px solid #c9d6e5; border-radius: 6px; background: #fff; color: #243247; font-size: 12px; }}
  #path-planning-panel #path-vascular-distance, #path-planning-panel #path-step-time {{ width: 44px; flex: 0 0 44px; padding: 3px 4px; }}
  #path-planning-panel #path-liver-samples {{ width: 42px; flex: 0 0 42px; padding: 3px 4px; }}
  #path-planning-panel button {{ padding: 4px 8px; cursor: pointer; font-weight: 600; }}
  #path-planning-panel button:hover:not(:disabled) {{ background: #edf5ff; border-color: #6ea8df; }}
  #path-planning-panel button:disabled {{ cursor: not-allowed; opacity: .48; }}
  #path-pick-start, #path-replan {{ width: 100%; }}
  #path-replan {{ background: #2166a5 !important; border-color: #2166a5 !important; color: #fff !important; }}
  #path-replan:hover:not(:disabled) {{ background: #174c7d !important; }}
  #path-planning-panel .path-playback {{ display: grid; grid-template-columns: auto auto auto 1fr; align-items: center; gap: 5px; }}
  #path-planning-panel #path-slider {{ width: 100%; accent-color: #2166a5; }}
  #path-status {{ min-height: 16px; color: #52657b; line-height: 1.35; }}
  #path-legend {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5px 10px; padding-top: 2px; border-top: 1px solid #e0e8f1; color: #52657b; }}
  #path-legend .legend-item {{ display: flex; align-items: center; gap: 6px; white-space: nowrap; }}
  #path-legend .legend-swatch {{ width: 11px; height: 11px; border-radius: 3px; border: 1px solid rgba(0,0,0,.12); flex: 0 0 auto; }}
  @media (max-width: 550px) {{
    #path-planning-panel {{ left: 8px; bottom: 48px; width: min(250px, calc(100vw - 16px)); padding: 9px; gap: 6px; max-height: min(38vh, 290px); }}
  }}
{ui_css}
</style>
</head>
<body>
<div id="info">Anatomy &amp; resection planning</div>

<!-- Toolbar -->
<div id="toolbar">
  <button class="tb-btn" id="btn-zoom-in" type="button" title="Zoom in" aria-label="Zoom in"><span class="tooltip">Zoom in</span></button>
  <button class="tb-btn" id="btn-zoom-out" type="button" title="Zoom out" aria-label="Zoom out"><span class="tooltip">Zoom out</span></button>
  <div class="sep"></div>
  <button class="tb-btn" id="btn-reset" title="Reset view"><span class="tooltip">Reset view</span></button>
  <div class="sep"></div>
  <button class="tb-btn" id="btn-rotate" title="Auto rotate"><span class="tooltip">Auto rotate</span></button>
  <button class="tb-btn" id="btn-bg" title="Toggle background"><span class="tooltip">Toggle background</span></button>
  <button class="tb-btn" id="btn-screenshot" title="截图导出 (Screenshot)"><svg viewBox="0 0 1024 1024" width="18" height="18" fill="currentColor"><path d="M677.888 494.592q0 28.672-10.752 53.76t-29.184 43.52-43.008 29.184-53.248 10.752-53.248-10.752-43.008-29.184-29.184-43.52-10.752-53.76q0-27.648 10.752-52.736t29.184-43.52 43.008-29.184 53.248-10.752 53.248 10.752 43.008 29.184 29.184 43.52 10.752 52.736zM171.008 766.976q-28.672 0-51.2-5.12t-37.888-17.408-23.552-33.28-8.192-52.736l0-346.112q0-57.344 27.136-79.872t85.504-22.528l172.032 0q16.384 0 27.136-6.144t17.408-16.384 11.776-24.064 11.264-28.16q10.24-26.624 35.84-46.08t58.368-19.456l95.232 0q37.888 0 61.952 20.992t32.256 44.544q11.264 30.72 29.696 52.736t38.912 22.016l130.048 0q45.056-1.024 71.68 24.576t26.624 74.752l0 351.232q0 52.224-27.648 79.36t-73.728 27.136l-710.656 0zM539.648 280.576q-45.056 0-83.968 16.896t-67.584 46.08-45.568 68.096-16.896 82.944q0 45.056 16.896 83.968t45.568 67.584 67.584 45.568 83.968 16.896q44.032 0 82.944-16.896t67.584-45.568 45.568-67.584 16.896-83.968q0-44.032-16.896-82.944t-45.568-68.096-67.584-46.08-82.944-16.896zM611.328 169.984q0-16.384-1.536-25.6t-20.992-9.216l-84.992 0q-19.456-1.024-20.992 8.192t-1.536 26.624q-1.024 19.456 2.048 27.648t20.48 8.192l84.992 0q19.456 0 20.992-9.216t1.536-26.624z"/></svg><span class="tooltip">截图导出</span></button>
  <div class="sep"></div>
  <button class="tb-btn" id="btn-measure" title="测量距离 (Measure)"><svg viewBox="0 0 1024 1024" width="18" height="18" fill="currentColor"><path d="M921.1 62.6H101.6c-10.1 0-18.3 8.2-18.3 18.3v263.8c0 10.1 8.2 18.3 18.3 18.3h819.5c10.1 0 18.3-8.2 18.3-18.3V80.9c0-10.1-8.2-18.3-18.3-18.3z m-18.3 263.8h-72.4v-59.1c0-10.1-8.2-18.3-18.3-18.3-10.1 0-18.3 8.2-18.3 18.3v59.1H682.6v-94.3c0-10.1-8.2-18.3-18.3-18.3-10.1 0-18.3 8.2-18.3 18.3v94.3H534.9v-59.1c0-10.1-8.2-18.3-18.3-18.3-10.1 0-18.3 8.2-18.3 18.3v59.1H387.2v-87.2c0-10.1-8.2-18.3-18.3-18.3s-18.3 8.2-18.3 18.3v87.2H239.5v-59.1c0-10.1-8.2-18.3-18.3-18.3s-18.3 8.2-18.3 18.3v59.1h-83V99.1h782.9v227.3zM919.3 730.8c-10.1 0-18.3 8.2-18.3 18.3v95.1H136.3v-95.1c0-10.1-8.2-18.3-18.3-18.3s-18.3 8.2-18.3 18.3v190.3c0 10.1 8.2 18.3 18.3 18.3s18.3-8.2 18.3-18.3v-58.5H901v58.5c0 10.1 8.2 18.3 18.3 18.3 10.1 0 18.3-8.2 18.3-18.3V749.1c0-10.1-8.2-18.3-18.3-18.3z"/></svg><span class="tooltip">测量距离</span></button>
  <button class="tb-btn" id="btn-annotate" title="标注 (Annotate)"><svg viewBox="0 0 1024 1024" width="18" height="18" fill="currentColor"><path d="M832 64 192 64C121.6 64 64 121.6 64 192l0 512c0 70.4 57.6 128 128 128l128 0 132.096 120.448C459.072 957.632 466.88 960 474.432 960 493.824 960 512 944.704 512 922.496L512 832l320 0c70.4 0 128-57.6 128-128L960 192C960 121.6 902.4 64 832 64zM896 704c0 35.328-28.672 64-64 64L512 768c-16.96 0-33.28 6.72-45.248 18.752S448 815.04 448 832l0 30.08-84.864-77.376C351.296 773.952 335.936 768 320 768L192 768c-35.328 0-64-28.672-64-64L128 192c0-35.328 28.672-64 64-64l640 0c35.328 0 64 28.672 64 64L896 704zM736 320l-448 0C270.336 320 256 334.336 256 352S270.336 384 288 384l448 0C753.664 384 768 369.664 768 352S753.664 320 736 320zM736 512l-448 0C270.336 512 256 526.336 256 544S270.336 576 288 576l448 0C753.664 576 768 561.664 768 544S753.664 512 736 512z"/></svg><span class="tooltip">标注</span></button>
  <div class="sep"></div>
  <button class="tb-btn" id="btn-help" type="button" title="Navigation help" aria-controls="interaction-help" aria-expanded="false"><span class="tooltip">Navigation help</span></button>
</div>

<!-- Organ Tree Panel -->
<div id="organ-tree">
  <div class="panel-heading" data-drag-handle="true">
    <div>
      <div class="panel-title">Structures</div>
      <div class="panel-subtitle">Visibility and opacity</div>
    </div>
    <div class="panel-heading-actions">
      <span class="panel-badge" id="structure-count">0</span>
      <button id="structure-panel-toggle" type="button" aria-label="Collapse structures panel" aria-expanded="true">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m15 6-6 6 6 6"></path></svg>
      </button>
    </div>
  </div>
  <div id="organ-tree-resize-handle" role="separator" aria-orientation="vertical" aria-label="Resize organ panel"></div>
</div>

<div id="interaction-help" role="dialog" aria-label="3D navigation instructions" aria-hidden="true">
  <div class="interaction-help-title">
    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><path d="M12 11v5"></path><path d="M12 8h.01"></path></svg>
    <span>Navigation</span>
  </div>
  <div class="interaction-help-row"><kbd>Drag</kbd><span>Rotate model</span></div>
  <div class="interaction-help-row"><kbd>Scroll</kbd><span>Zoom view</span></div>
  <div class="interaction-help-row"><kbd>Right-drag</kbd><span>Pan view</span></div>
</div>

<!-- Case metadata for on-demand resection computation -->
<div id="case-data" data-case-name="{case_name}" data-mask-dir="{mask_dir}" data-output-dir="{output_dir}" data-json-file="{json_url}" style="display:none"></div>

<!-- Planning toggles (top-right) -->
<div id="planning-toggles" aria-label="Surgical planning layers">
  <div id="path-planning-toggle" class="planning-toggle" title="最佳手术路径规划 (Optimal surgical path)">
    <label>
      <input type="checkbox" id="toggle-path-planning" aria-label="Optimal path" aria-controls="path-planning-panel">
      <span>Optimal path</span>
    </label>
  </div>
  <div id="resection-toggle" class="planning-toggle">
    <label>
      <input type="checkbox" id="toggle-resection-plane" aria-label="Resection plane">
      <span>Resection plane</span>
    </label>
  </div>
</div>

<!-- Saved sequence path playback -->
<aside id="right-workspace" aria-label="Resection planning tools">
<div id="path-planning-panel">
  <div class="path-panel-title" data-drag-handle="true"><span>Resection Sequence</span><button class="path-panel-collapse" id="path-panel-collapse" type="button" aria-label="Collapse panel"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg></button></div>
  <div class="path-panel-body">
    <button id="path-pick-start" type="button">Pick Start Cell</button>
    <button id="path-replan" type="button">Replan</button>
    <div class="path-playback"><button id="path-prev" type="button" aria-label="Previous step"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m14.5 6-6 6 6 6"/></svg></button><button id="path-play" type="button">Play</button><button id="path-next" type="button" aria-label="Next step"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9.5 6 6 6-6 6"/></svg></button><input id="path-slider" type="range" min="0" value="0"></div>
    <span id="path-status">Loading…</span>
    <details id="path-advanced">
      <summary>Advanced settings</summary>
      <div class="path-panel-section">
        <label title="Planning algorithm">Algorithm <select id="path-algorithm"><option value="nearest">Nearest Neighbor</option><option value="dfs">Depth First</option><option value="spanning_tree">Spanning Tree</option></select></label>
        <label title="Vessel exclusion distance">Vessel <input id="path-vascular-distance" type="number" min="0" step="0.5" value="5"> <span>mm</span></label>
        <label title="Minimum Liver intersection samples">Liver samples <input id="path-liver-samples" type="number" min="1" max="5" step="1" value="1"></label>
        <label title="Estimated time per step">Step time <input id="path-step-time" type="number" min="0.01" step="0.1" value="1"> <span>s</span></label>
      </div>
    </details>
    <div id="path-legend" aria-label="Grid state legend">
    <span class="legend-item"><i class="legend-swatch" style="background:#808892"></i>Visited</span>
    <span class="legend-item"><i class="legend-swatch" style="background:#ff1744"></i>Current</span>
    <span class="legend-item"><i class="legend-swatch" style="background:#1976d2"></i>Start</span>
    <span class="legend-item"><i class="legend-swatch" style="background:#4e342e"></i>Outside liver</span>
    <span class="legend-item"><i class="legend-swatch" style="background:#ff8f00"></i>Vessel risk</span>
    <span class="legend-item"><i class="legend-swatch" style="background:#8e24aa"></i>Unreachable</span>
    </div>
  </div>
</div>
</aside>

<!-- Loading overlay for on-demand resection computation -->
<div id="resection-loading">
  <div class="spinner"></div>
  <div class="loading-text">Computing the optimal resection plane…</div>
  <div class="loading-sub">Bézier surface optimization · about 30 seconds</div>
  <button id="cancel-resection-computation" type="button">取消计算</button>
</div>

<!-- View layout containers -->
<div id="view-3d" class="view-container active"></div>
<div id="view-slice" class="view-container">
  <div id="slice-viewer">
    <img id="slice-img" src="" alt="CT Slice" />
    <div id="slice-controls">
      <button id="slice-prev">\u25c0</button>
      <span id="slice-label">No slices</span>
      <button id="slice-next">\u25b6</button>
    </div>
  </div>
</div>
<div id="view-combined" class="view-container">
  <div id="combined-3d"></div>
  <div id="combined-slice">
    <img id="combined-slice-img" src="" alt="CT Slice" />
  </div>
</div>

<!-- Measurement status bar -->
<div id="measure-status">MEASURE · Select the first point</div>
<button id="measure-clear-btn">Clear measurements</button>

<!-- Annotation UI -->
<div id="annotation-input-wrap" style="display:none;">
  <div id="annotation-input-box">
    <textarea id="annotation-text" rows="2" placeholder="Enter annotation..." maxlength="200"></textarea>
    <div id="annotation-input-actions">
      <button id="annotation-confirm">Confirm</button>
      <button id="annotation-cancel">Cancel</button>
    </div>
  </div>
</div>
<div id="annotation-status">ANNOTATE · Select an organ surface</div>
<button id="annotation-clear-btn">Clear annotations</button>

<script>
  // Normalize toolbar icons while preserving the existing button actions.
  const toolbarIcons = {{
    'btn-zoom-in': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>',
    'btn-zoom-out': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M5 12h14"/></svg>',
    'btn-reset': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 11a8 8 0 1 1 2.34 5.66"/><path d="M4 5v6h6"/></svg>',
    'btn-rotate': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 12a8 8 0 1 1-2.34-5.66"/><path d="M20 5v7h-7"/></svg>',
    'btn-bg': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a9 9 0 1 0 9 9 7 7 0 0 1-9-9Z"/></svg>',
    'btn-screenshot': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h3l1.4-2h7.2L17 7h3v12H4V7Z"/><circle cx="12" cy="13" r="3.5"/></svg>',
    'btn-measure': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m5 16 11-11 3 3L8 19H5v-3Z"/><path d="m12 9 3 3M9 12l3 3"/></svg>',
    'btn-annotate': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21v-8"/><path d="M8 4h8l-1 6 3 3H6l3-3-1-6Z"/></svg>',
    'btn-help': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.8 9a2.3 2.3 0 1 1 3.5 2c-.8.5-1.3 1-1.3 2"/><path d="M12 17h.01"/></svg>'
  }};
  const toolbarLabels = {{
    'btn-zoom-in': 'Zoom in',
    'btn-zoom-out': 'Zoom out',
    'btn-reset': 'Reset view',
    'btn-rotate': 'Auto rotate',
    'btn-bg': 'Toggle background',
    'btn-screenshot': 'Screenshot',
    'btn-measure': 'Measure distance',
    'btn-annotate': 'Annotate',
    'btn-help': 'Navigation help'
  }};
  Object.entries(toolbarIcons).forEach(([id, icon]) => {{
    const button = document.getElementById(id);
    if (!button) return;
    const label = toolbarLabels[id];
    button.innerHTML = icon
      + `<span class="button-label">${{label}}</span>`
      + `<span class="tooltip">${{label}}</span>`;
    button.title = label;
    button.setAttribute('aria-label', label);
    if (['btn-rotate', 'btn-bg', 'btn-measure', 'btn-annotate'].includes(id)) {{
      button.setAttribute('aria-pressed', 'false');
    }}
  }});

  // Keep legacy toolbar labels consistent with the English interface.
  const englishToolbarLabels = {{
    'btn-screenshot': ['Screenshot', 'Screenshot'],
    'btn-measure': ['Measure distance', 'Measure distance'],
    'btn-annotate': ['Annotate', 'Annotate']
  }};
  Object.entries(englishToolbarLabels).forEach(([id, labels]) => {{
    const button = document.getElementById(id);
    if (button) {{
      button.title = labels[0];
      const tooltip = button.querySelector('.tooltip');
      if (tooltip) tooltip.textContent = labels[1];
    }}
  }});

  const helpButton = document.getElementById('btn-help');
  const interactionHelp = document.getElementById('interaction-help');
  const setHelpOpen = open => {{
    if (!helpButton || !interactionHelp) return;
    interactionHelp.classList.toggle('open', open);
    interactionHelp.setAttribute('aria-hidden', String(!open));
    helpButton.setAttribute('aria-expanded', String(open));
    helpButton.classList.toggle('active', open);
  }};
  helpButton?.addEventListener('click', event => {{
    event.stopPropagation();
    setHelpOpen(!interactionHelp.classList.contains('open'));
  }});
  document.addEventListener('pointerdown', event => {{
    if (interactionHelp?.classList.contains('open')
        && !interactionHelp.contains(event.target)
        && !helpButton.contains(event.target)) {{
      setHelpOpen(false);
    }}
  }});
  document.addEventListener('keydown', event => {{
    if (event.key === 'Escape') setHelpOpen(false);
  }});

  const structurePanelToggle = document.getElementById('structure-panel-toggle');
  const setStructuresCollapsed = collapsed => {{
    const structurePanel = document.getElementById('organ-tree');
    const before = structurePanel?.getBoundingClientRect();
    document.body.classList.toggle('structures-collapsed', collapsed);
    structurePanelToggle?.setAttribute('aria-expanded', String(!collapsed));
    structurePanelToggle?.setAttribute(
      'aria-label', collapsed ? 'Expand structures panel' : 'Collapse structures panel');
    // Collapsing changes the panel dimensions and responsive transforms. Offset
    // that layout delta so the user-selected top-left anchor stays unchanged.
    if (structurePanel && before) {{
      const after = structurePanel.getBoundingClientRect();
      const dragX = Number(structurePanel.dataset.dragX || 0) + before.left - after.left;
      const dragY = Number(structurePanel.dataset.dragY || 0) + before.top - after.top;
      structurePanel.dataset.dragX = String(dragX);
      structurePanel.dataset.dragY = String(dragY);
      structurePanel.style.translate = `${{dragX}}px ${{dragY}}px`;
    }}
  }};
  if (window.matchMedia('(max-width: 620px)').matches) {{
    setStructuresCollapsed(true);
  }}
  structurePanelToggle?.addEventListener('click', event => {{
    event.preventDefault();
    event.stopPropagation();
    setStructuresCollapsed(!document.body.classList.contains('structures-collapsed'));
  }});

  const pathPlanningToggle = document.getElementById('toggle-path-planning');
  const setPathPlanningRequested = open => {{
    if (!pathPlanningToggle) return;
    if (open && !document.body.classList.contains('has-resection-workspace')) {{
      pathPlanningToggle.checked = false;
      const planeToggle = document.getElementById('toggle-resection-plane');
      if (planeToggle && !planeToggle.checked && !planeToggle.disabled) {{
        planeToggle.checked = true;
        planeToggle.dispatchEvent(new Event('change', {{bubbles: true}}));
      }}
      return;
    }}
    pathPlanningToggle.checked = open;
    const planeToggle = document.getElementById('toggle-resection-plane');
    if (open && planeToggle && !planeToggle.checked) {{
      planeToggle.checked = true;
      planeToggle.dispatchEvent(new Event('change', {{bubbles: true}}));
    }}
    if (open) {{
      // Let the sequence panel occupy the right rail without covering the
      // bottom toolbar. Users can still reopen either supporting card.
      requestAnimationFrame(() => {{
        setDistanceCollapsed(true);
        setPlaneCollapsed(true);
      }});
    }}
    document.dispatchEvent(new CustomEvent('path-planning-open-request', {{detail: {{open}}}}));
  }};
  pathPlanningToggle?.addEventListener('change', event => {{
    event.stopPropagation();
    setPathPlanningRequested(pathPlanningToggle.checked);
  }});

  const widePanelMedia = window.matchMedia('(min-width: 821px)');
  const draggablePanelIds = [
    'organ-tree', 'color-legend', 'plane-selector', 'path-planning-panel'
  ];
  const resetCompactPanelPositions = () => {{
    draggablePanelIds.forEach(id => {{
      const panel = document.getElementById(id);
      if (!panel) return;
      panel.style.translate = '';
      delete panel.dataset.dragX;
      delete panel.dataset.dragY;
    }});
  }};

  const setDistanceCollapsed = collapsed => {{
    const legend = document.getElementById('color-legend');
    const toggle = document.getElementById('distance-panel-toggle');
    if (!legend || !toggle) return;
    const wasDragged = legend.dataset.dragX !== undefined
      || legend.dataset.dragY !== undefined;
    const before = legend.getBoundingClientRect();
    legend.classList.toggle('collapsed', collapsed);
    toggle.setAttribute('aria-expanded', String(!collapsed));
    toggle.setAttribute(
      'aria-label', collapsed ? 'Expand distance panel' : 'Collapse distance panel');
    // Preserve the dragged top-left anchor while the legend body changes size.
    if (wasDragged) {{
      const after = legend.getBoundingClientRect();
      const dragX = Number(legend.dataset.dragX || 0) + before.left - after.left;
      const dragY = Number(legend.dataset.dragY || 0) + before.top - after.top;
      legend.dataset.dragX = String(dragX);
      legend.dataset.dragY = String(dragY);
      legend.style.translate = `${{dragX}}px ${{dragY}}px`;
    }}
  }};

  const setPlaneCollapsed = collapsed => {{
    const selector = document.getElementById('plane-selector');
    const toggle = document.getElementById('plane-panel-toggle');
    if (!selector || !toggle) return;
    const wasDragged = selector.dataset.dragX !== undefined
      || selector.dataset.dragY !== undefined;
    const before = selector.getBoundingClientRect();
    selector.classList.toggle('collapsed', collapsed);
    toggle.setAttribute('aria-expanded', String(!collapsed));
    toggle.setAttribute(
      'aria-label', collapsed
        ? 'Expand resection plane panel'
        : 'Collapse resection plane panel');
    // Preserve the dragged top-left anchor while the controls change height.
    if (wasDragged) {{
      const after = selector.getBoundingClientRect();
      const dragX = Number(selector.dataset.dragX || 0) + before.left - after.left;
      const dragY = Number(selector.dataset.dragY || 0) + before.top - after.top;
      selector.dataset.dragX = String(dragX);
      selector.dataset.dragY = String(dragY);
      selector.style.translate = `${{dragX}}px ${{dragY}}px`;
    }}
  }};

  document.addEventListener('resection-workspace-open-request', event => {{
    if (event.detail && event.detail.open === false) return;
    requestAnimationFrame(() => {{
      setDistanceCollapsed(false);
      setPlaneCollapsed(false);
    }});
  }});

  const initializeCompactPanel = panel => {{
    if (!panel || panel.dataset.compactDragReady === 'true') return;
    const handle = panel.querySelector('[data-drag-handle="true"]');
    if (!handle) return;
    panel.dataset.compactDragReady = 'true';

    if (panel.id === 'color-legend') {{
      const distanceToggle = panel.querySelector('#distance-panel-toggle');
      distanceToggle?.addEventListener('click', event => {{
        event.stopPropagation();
        setDistanceCollapsed(!panel.classList.contains('collapsed'));
      }});
    }}
    if (panel.id === 'plane-selector') {{
      const planePanelToggle = panel.querySelector('#plane-panel-toggle');
      planePanelToggle?.addEventListener('click', event => {{
        event.stopPropagation();
        setPlaneCollapsed(!panel.classList.contains('collapsed'));
      }});
    }}

    handle.addEventListener('pointerdown', event => {{
      if (event.button !== 0
          || event.target.closest('button, input, select, textarea, a')) return;
      event.preventDefault();
      event.stopPropagation();

      const startRect = panel.getBoundingClientRect();
      const startX = event.clientX;
      const startY = event.clientY;
      const startDragX = Number(panel.dataset.dragX || 0);
      const startDragY = Number(panel.dataset.dragY || 0);
      const viewportPadding = 6;
      const headerHeight = parseFloat(
        getComputedStyle(document.documentElement).getPropertyValue('--header-height')) || 0;

      panel.classList.add('panel-dragging');

      const movePanel = moveEvent => {{
        const requestedLeft = startRect.left + moveEvent.clientX - startX;
        const requestedTop = startRect.top + moveEvent.clientY - startY;
        const minTop = headerHeight + viewportPadding;
        const maxLeft = Math.max(viewportPadding, window.innerWidth - startRect.width - viewportPadding);
        const maxTop = Math.max(minTop, window.innerHeight - startRect.height - viewportPadding);
        const clampedLeft = Math.min(maxLeft, Math.max(viewportPadding, requestedLeft));
        const clampedTop = Math.min(maxTop, Math.max(minTop, requestedTop));
        const nextX = startDragX + clampedLeft - startRect.left;
        const nextY = startDragY + clampedTop - startRect.top;
        panel.dataset.dragX = String(nextX);
        panel.dataset.dragY = String(nextY);
        panel.style.translate = `${{nextX}}px ${{nextY}}px`;
      }};

      const finishPanelDrag = () => {{
        panel.classList.remove('panel-dragging');
        document.removeEventListener('pointermove', movePanel);
        document.removeEventListener('pointerup', finishPanelDrag);
        document.removeEventListener('pointercancel', finishPanelDrag);
      }};

      document.addEventListener('pointermove', movePanel);
      document.addEventListener('pointerup', finishPanelDrag);
      document.addEventListener('pointercancel', finishPanelDrag);
    }});
  }};

  const initializeCompactPanels = () => {{
    draggablePanelIds.forEach(id => initializeCompactPanel(document.getElementById(id)));
  }};
  initializeCompactPanels();
  const compactPanelObserver = new MutationObserver(initializeCompactPanels);
  compactPanelObserver.observe(document.body, {{childList: true, subtree: true}});
  // Returning from full screen restores the centered preview layout; entering
  // it restores the clean top-aligned layout before any new user drag.
  widePanelMedia.addEventListener('change', resetCompactPanelPositions);

  const pathPanel = document.getElementById('path-planning-panel');
  const collapseButton = document.getElementById('path-panel-collapse');
  if (collapseButton && pathPanel) {{
    collapseButton.addEventListener('click', event => {{
      event.stopPropagation();
      pathPanel.classList.toggle('collapsed');
      const collapsed = pathPanel.classList.contains('collapsed');
      collapseButton.innerHTML = collapsed
        ? '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 15 6-6 6 6"/></svg>'
        : '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>';
      collapseButton.setAttribute('aria-label', collapsed ? 'Expand panel' : 'Collapse panel');
    }});
  }}
</script>

{import_section}

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
{js_module_code}
</script>
</body>
</html>"""
    return html



def resolve_case_dir(image_id: str, data_root: str, seg_backend: str) -> str:
    """
    Try to find the case directory by scanning dataset subdirectories.

    Expected structure:
        {data_root}/Subset-v3/segmentations/{seg_backend}/{dataset}/{image_id}/
    """
    base = Path(data_root) / "Subset-v3" / "segmentations" / seg_backend
    if not base.exists():
        raise FileNotFoundError(f"Segmentation base directory not found: {base}")

    # Scan dataset dirs
    for dataset_dir in base.iterdir():
        if not dataset_dir.is_dir():
            continue
        candidate = dataset_dir / image_id
        if candidate.exists() and candidate.is_dir():
            return str(candidate)

    raise FileNotFoundError(
        f"Could not find case directory for image_id='{image_id}' "
        f"under {base}. Use --case-dir to specify directly."
    )
