
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

Use Python 3.10+ in a clean virtual environment:

```bash
pip install -r requirements.txt
```

Start the API and optional output proxy in separate terminals:

```bash
python API.py server --port 8765
python file_proxy.py --port 8898
```

Set `PUBLIC_BASE_URL` when the proxy is served from another host or port. The
default is `http://127.0.0.1:8898`.

## Segmentation backends

TotalSegmentator is the self-contained default dependency listed in
`requirements.txt`; its weights are downloaded and licensed by that project.

VISTA3D support is optional. This repository does not redistribute VISTA3D
source code or weights. Install them separately under the upstream licence,
then set `VISTA3D_ROOT` to the upstream `vista3d` source directory and either
place model assets under `./models` or set `VISTA3D_CONFIG` to a local inference
configuration. Do not publish fine-tuned weights unless you have confirmed the
rights to distribute both the base model and the derived weights.

## Data handling

Only use data that you are authorized to process. Remove DICOM identifiers and
other protected health information before any demonstration, issue attachment,
log upload, or public release. `data/`, `output/`, `models/`, and common
medical-image formats are excluded by `.gitignore`.

## Licence and contributions

The repository's Apache License 2.0 applies to code authored for GeoSurge; see
the root-level `LICENSE` and `NOTICE`. It does not relicense external software,
model weights, or datasets. Contributions must be compatible with Apache 2.0
and must not include protected health information, credentials, or material
that cannot be publicly redistributed.

Direct dependency, method-reference, and provenance notices are maintained at
the repository root in `THIRD_PARTY_NOTICES.md`.
