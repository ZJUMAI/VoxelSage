# VoxelSage

VoxelSage is an experimental research project for medical-imaging analysis and
related interactive tools. It is not a medical device and must not be used for
clinical diagnosis or treatment decisions.

## Repository layout

- `Port_A/`: application-side components.
- `Port_B/`: medical imaging-analysis service and Skills.
- `Frontend/voxelsage-public/`: user-facing React application.

## Local deployment

Prerequisites are Python 3.10+, Node.js 20+, npm, and enough system resources
for the selected segmentation backend. From a clean clone:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Port_B/requirements.txt
pip install -r Port_A/requirements.txt

cd Frontend/voxelsage-public
npm ci
cp .env.example .env.local
```

Start the three services in separate terminals from the repository root:

```bash
cd Port_B
../.venv/bin/python API.py server --port 8765
```

```bash
cd Port_B
../.venv/bin/python file_proxy.py --port 8898
```

```bash
export DASHSCOPE_API_KEY="your-api-key"
export DASHSCOPE_BASE_URL="https://your-llm-endpoint.example.com/v1"
cd Port_A
../.venv/bin/python -m core.server
```

```bash
cd Frontend/voxelsage-public
npm run dev
```

Open `http://localhost:3000`. For non-local or HTTPS deployments, copy
`Frontend/voxelsage-public/.env.example` to `.env.local`, set the three
`VITE_*` backend URLs before building, and serve `dist/` from your static web
server after `npm run build`.

TotalSegmentator may download its own model assets. VISTA3D remains optional
and requires a separately licensed checkout/configuration as described in
`Port_B/README.md`; no model weights or patient data are included here.

## Verification

```bash
cd Port_B && ../.venv/bin/python -m pytest -q
cd ../Frontend/voxelsage-public && npm run lint && npm run build
```

Source code authored for this repository is licensed under Apache License 2.0;
see `LICENSE` and `NOTICE`. External software, model weights, and datasets keep
their original licenses and are not relicensed by this repository.

## Acknowledgments and third-party notices

Port B was informed by the 3DMedAgent project and by published work on
Bézier-surface liver-resection planning. Direct dependency, reference, and
provenance information is recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
This project is not affiliated with or endorsed by those upstream authors.
