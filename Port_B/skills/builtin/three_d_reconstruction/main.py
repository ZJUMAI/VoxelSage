"""three_d_reconstruction: 3D 交互式可视化生成 + 可选手术最优切除剖面。

调用 API.py 的 generate_visualization（底层是 visualize_3d.py 的
Marching Cubes → Three.js 管线）生成可交互的 3D HTML。

如果 generate_resection_plane=True，还会调用 plan_resection 计算
Bézier 切除剖面并追加到 3D JSON 文件中。
"""

import os
from pathlib import Path


def run(ctx):
    from API import generate_visualization

    case_name = ctx.params.get("case_name", ctx.case_id)
    step_size = ctx.params.get("step_size", 2)
    downsample = ctx.params.get("downsample_factor", 1.0)
    generate_resection = ctx.params.get("generate_resection_plane", False)

    ctx.log(f"Generating 3D visualization for {case_name} (step_size={step_size})...")

    result = generate_visualization(
        case_dir=ctx.mask_dir,
        output_dir=ctx.output_dir,
        output_filename=f"{case_name}_3d",
        step_size=step_size,
        downsample_factor=downsample,
        title=f"三维医学影像可视化 — {case_name}",
    )

    if result.get("status") == "error":
        raise RuntimeError(f"3D generation failed: {result.get('message')}")

    html_path = result.get("file_path", "") or None
    html_url = result.get("url", "") or None

    # ---- 可选：计算手术最优切除剖面 ----
    if generate_resection:
        ctx.log(f"Computing optimal resection plane for {case_name}...")
        margin_mm = float(ctx.params.get("tumor_margin_mm", 5.0))
        ctx.params["tumor_margin_mm"] = margin_mm

        from skills.builtin.plan_resection.main import run as run_plan_resection
        plane_result = run_plan_resection(ctx)

        result["resection"] = plane_result
        ctx.log(
            f"Resection plane done: margin_min={plane_result.get('margin_min_mm', 'N/A')}mm, "
            f"success={plane_result.get('margin_success', False)}"
        )

    return {
        "html_path": html_url or html_path,
        "html_url": html_url or html_path,
        "filename": os.path.basename(html_path) if html_path else "",
        "organ_count": len(result.get("organs", [])),
        "organs": result.get("organs", []),
    }
