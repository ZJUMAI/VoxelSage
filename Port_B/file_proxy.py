#!/usr/bin/env python3
"""
VoxelSage Port B 文件代理服务器
=============================
仅做一件事：把 /output/ 和 /process-output/ 请求转发到 API.py (:8765)。

用法:
    python file_proxy.py --port 8898

将 ``PUBLIC_BASE_URL`` 设为此代理的公开地址后，API 返回的可视化 URL
即可经由该代理访问。
"""

import argparse
import sys
from pathlib import Path

try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import Response
    import httpx
except ImportError as e:
    print(f"[ERROR] 缺少依赖: {e}")
    print("请运行: pip install fastapi uvicorn httpx")
    sys.exit(1)

VIZ_API = "http://127.0.0.1:8765"
app = FastAPI(title="File Proxy")

async def _proxy(path: str, prefix: str):
    url = f"{VIZ_API}/{prefix}/{path}"
    # VIZ_API is a loopback service. Ignore http_proxy/HTTP_PROXY inherited
    # from the shell so local forwarding cannot be diverted to an external
    # proxy.
    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            resp = await client.get(url, timeout=1200.0)
            ct = resp.headers.get("content-type", "application/octet-stream")
            return Response(content=resp.content, media_type=ct, status_code=resp.status_code)
        except httpx.RequestError as e:
            return Response(content=f"Proxy error: {e}", status_code=502)

@app.get("/output/{path:path}")
async def proxy_output(path: str):
    return await _proxy(path, "output")

@app.get("/process-output/{path:path}")
async def proxy_process(path: str):
    return await _proxy(path, "process-output")

@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_api(path: str, request: Request):
    """代理 /api/* 请求到 API.py (8765)。"""
    url = f"{VIZ_API}/api/{path}"
    body = await request.body() if request.method in ("POST", "PUT") else None
    headers = {"Content-Type": request.headers.get("content-type", "application/json")} if body else {}
    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            if request.method == "GET":
                resp = await client.get(url, timeout=1200.0)
            elif request.method == "DELETE":
                resp = await client.delete(url, timeout=1200.0)
            elif request.method == "POST":
                resp = await client.post(url, content=body, headers=headers, timeout=1200.0)
            elif request.method == "PUT":
                resp = await client.put(url, content=body, headers=headers, timeout=1200.0)
            else:
                return Response(content=f"Method {request.method} not supported", status_code=405)
            ct = resp.headers.get("content-type", "application/octet-stream")
            return Response(content=resp.content, media_type=ct, status_code=resp.status_code)
        except httpx.RequestError as e:
            return Response(content=f"Proxy error: {e}", status_code=502)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "file-proxy"}

def main():
    p = argparse.ArgumentParser(description="文件代理服务器")
    p.add_argument("--port", type=int, default=8898)
    p.add_argument("--host", type=str, default="0.0.0.0")
    args = p.parse_args()

    print(f"文件代理服务器已启动: http://0.0.0.0:{args.port}")
    print(f"  /output/*        → {VIZ_API}/output/*")
    print(f"  /process-output/* → {VIZ_API}/process-output/*")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
