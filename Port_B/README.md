
# Port B: Medical Imaging Analysis Services

Experimental research software for abdominal CT segmentation orchestration,
quantitative analysis, structured summaries, key-slice selection, and
interactive 3D visualization. It is **not a medical device** and must not be
used for clinical diagnosis or treatment decisions.

## What is included

This directory is the public-release candidate. It contains the FastAPI
service, reusable analysis utilities, visualization code, and built-in Skills.
It deliberately excludes patient data, model weights, experimental evaluation
datasets, internal documents, training code, generated outputs, and vendored
third-party model source trees.

## Installation

For a complete local installation from the repository root, use the shared
setup script:

```bash
./scripts/setup.sh
```

It installs VISTA3D as the default backend and does not install
TotalSegmentator. Start the complete four-service application with
`./scripts/start.sh`; see the root [README](../README.md#quick-start).

To run only Port B manually after setup, use two terminals:

```bash
cd Port_B
../.venv/bin/python API.py server --port 8765
# In another terminal:
cd Port_B
../.venv/bin/python file_proxy.py --port 8898
```

Set `PUBLIC_BASE_URL` when the proxy is served from another host or port. The
default is `http://127.0.0.1:8898`. Set `VOXELSAGE_OUTPUT_DIR` to place runtime
outputs outside the source tree. The variables in `.env.example` are examples;
copy the settings you need into the root `.env` for `scripts/start.sh`, or
export them before a manual Port B startup.

## Segmentation backends

`SEGMENTATION_BACKEND=vista3d` is the server default. The API request field
`seg_backend` and the pipeline option `--seg-backend` can override it for one
case.

### VISTA3D (default)

`./scripts/setup.sh` clones the official `Project-MONAI/VISTA` repository to
`third_party/VISTA`. VoxelSage automatically resolves that path, uses
`SegAgent/VISTA3d/configs/infer.yaml`, and stores its model files under
`Port_B/models/vista3d` by default. On first inference, the upstream loader
downloads `model_monai1.3.pt` from `nvidia/NV-Segment-CT` on Hugging Face; later
runs reuse the cache.

Override a manually managed installation with `VISTA3D_ROOT`, the inference
configuration with `VISTA3D_CONFIG`, or the model directory with
`VISTA3D_MODEL_DIR`.

VISTA3D requires a CUDA-capable NVIDIA GPU. Check the driver, PyTorch CUDA
build, compatibility-sensitive packages, and an optional input volume before
starting the services:

```bash
./scripts/doctor.py
./scripts/doctor.py /absolute/path/to/ct.nii.gz
```

VoxelSage validates NIfTI geometry before loading VISTA3D, preserves the full
MONAI exception chain, and retries only explicit CUDA memory/resource errors.
Invalid geometry returns HTTP `422`, unavailable CUDA returns `503`, and other
inference failures return `500`. A known-incompatible VISTA3D environment also
returns `503` instead of reporting a successful HTTP response with an error body.

### TotalSegmentator (optional)

TotalSegmentator is not part of the default dependency set. Install it only
when required:

```bash
./scripts/setup.sh --with-totalsegmentator
```

Then set `SEGMENTATION_BACKEND=totalsegmentator` in `.env`, pass
`"seg_backend": "totalsegmentator"` to `/api/process` or `/api/process-lite`,
or run:

```bash
cd Port_B
../.venv/bin/python API.py pipeline /absolute/path/to/ct.nii.gz \
  --seg-backend totalsegmentator
```

Do not publish downloaded or fine-tuned weights unless you have confirmed the
rights to distribute both the base model and any derived weights.

## Data handling

Only use data that you are authorized to process. Remove DICOM identifiers and
other protected health information before any demonstration, issue attachment,
log upload, or public release. `data/`, `output/`, `models/`, and common
medical-image formats are excluded by `.gitignore`.

## Licence and contributions

The repository's Apache License 2.0 applies to code authored for VoxelSage; see
the root-level `LICENSE` and `NOTICE`. It does not relicense external software,
model weights, or datasets. Contributions must be compatible with Apache 2.0
and must not include protected health information, credentials, or material
that cannot be publicly redistributed.

Direct dependency, method-reference, and provenance notices are maintained at
the repository root in `THIRD_PARTY_NOTICES.md`.
