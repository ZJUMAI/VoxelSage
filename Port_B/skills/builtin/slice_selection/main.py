"""slice_selection: 选取信息量最大的 K 张切片。

调用 API.py 的 select_top_slices 进行评分排序，
再调用 save_slice_images 保存为 PNG（原始+轮廓叠加）。
返回的路径转换为经 8898 代理的 URL。

注意：从 API.py import 函数不会导致循环 import，因为 API.py 只在模块级别
import skills.engine，而技能模块此时尚未加载（SkillEngine 运行时懒加载）。
若未来需完全消除对 API.py 的依赖，可将 select_top_slices / _load_ct 等
共享函数迁移至 Tool_Box/ 下的独立模块。
"""

import os
from pathlib import Path
from API import (
    select_top_slices,
    save_slice_images,
    _load_ct,
    _load_ct_raw,
    _CRLM_DEFAULT_ORGANS,
    PUBLIC_BASE_URL,
    DEFAULT_OUTPUT_DIR,
)
from skills.utils import convert_numpy


def run(ctx):
    case_name = ctx.params.get("case_name", ctx.case_id)
    top_k = ctx.params.get("top_k", 3)
    scoring_mode = ctx.params.get("scoring_mode", "crlm")

    ctx.log(f"Selecting top-{top_k} slices (mode: {scoring_mode})...")

    # 加载 CT
    ct_volume, affine = _load_ct(ctx.ct_nifti_path)

    # 加载原始 HU（CFLT/CRLM 模式需要）
    ct_raw = None
    if scoring_mode in ("cflt", "crlm"):
        try:
            ct_raw, _ = _load_ct_raw(ctx.ct_nifti_path)
        except Exception:
            ctx.log("Could not load raw HU, falling back to composite mode")
            scoring_mode = "composite"

    # 确定器官列表
    # 扫描 mask_dir 获取实际存在的器官。CRLM 后处理会把 VISTA3D 原名重命名/拆分
    # （hepatic vessel → hepatic，portal vein and splenic vein → portal，
    #  hepatic tumor → tumor_1..tumor_N），因此不能用 VISTA3D 原名精确匹配，
    # 否则肿瘤/血管 mask 全部漏掉，crlm 评分奖励与 overlay 都会失效。
    from Tool_Box.mask_resolution import scan_logical_masks
    available = list(scan_logical_masks(ctx.mask_dir).keys())
    if not available:
        available = _CRLM_DEFAULT_ORGANS

    # 执行切片评分与选取
    indices, label_masks, scores = select_top_slices(
        ct_volume=ct_volume,
        mask_dir=ctx.mask_dir,
        organ_list=available,
        top_k=top_k,
        scoring_mode=scoring_mode,
        ct_volume_raw=ct_raw,
        ct_affine=affine,
    )

    ctx.log(f"Top slice indices: {indices}, scores: {[round(s, 4) for s in scores]}")

    # 保存为 PNG
    slice_results = save_slice_images(
        ct_volume=ct_volume,
        slice_indices=indices,
        label_masks=label_masks,
        output_dir=ctx.output_dir,
        case_name=case_name,
        organ_names=available,
        slice_scores=list(scores),
        affine=affine,
    )

    # 将文件系统绝对路径转换为 8898 代理 URL
    output_root = Path(DEFAULT_OUTPUT_DIR).resolve()
    for slc in slice_results:
        if png_path := slc.get("png_path"):
            try:
                rel = str(Path(png_path).resolve().relative_to(output_root))
                slc["png_url"] = f"{PUBLIC_BASE_URL}/output/{rel}"
            except ValueError:
                slc["png_url"] = f"{PUBLIC_BASE_URL}/output/{Path(png_path).name}"
        if overlay_path := slc.get("overlay_path"):
            try:
                rel = str(Path(overlay_path).resolve().relative_to(output_root))
                slc["overlay_url"] = f"{PUBLIC_BASE_URL}/output/{rel}"
            except ValueError:
                slc["overlay_url"] = f"{PUBLIC_BASE_URL}/output/{Path(overlay_path).name}"

    return convert_numpy({
        "slices": slice_results,
        "top_k": top_k,
        "scoring_mode": scoring_mode,
    })
