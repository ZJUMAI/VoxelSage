# Public release audit

Audit date: 2026-08-10

## Included

- FastAPI orchestration (`API.py`) and output proxy.
- Segmentation adapters, analysis utilities, visualization code, and Skills.
- The VISTA3D label map and a generic configuration template; VISTA3D itself
  is an optional external dependency.

## Excluded

- All patient/medical image files, generated masks, reports, and outputs.
- Data roots, model weights, checkpoints, caches, and runtime logs.
- Research/evaluation pipelines and CSV datasets.
- Papers, reference PDFs, internal planning material, training scripts, and
  vendored VISTA source/assets.
- Bundled Three.js files; visualization falls back to the configured CDN.

## Remediations applied

- Replaced internal IP addresses and absolute cache/data paths with loopback,
  project-relative paths, or environment variables.
- Removed an example that named an internal dataset path.
- Added a release-specific ignore policy, dependency manifest, and example
  environment file.

## Mandatory review before push

1. Scan this directory and the intended public Git history with a secret
   scanner; rotate any credential that ever appears in history.
2. Confirm no DICOM/NIfTI, screenshots, reports, issue attachments, CI logs,
   or Git LFS objects contain protected health information.
3. Confirm third-party attribution requirements before each release.
4. Run installation and smoke tests in a clean environment with synthetic or
   fully authorized de-identified data.
5. Keep public releases free of private-repository history and verify that
   release artifacts contain no protected health information.
