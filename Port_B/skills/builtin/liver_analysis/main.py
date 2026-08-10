"""liver_analysis: 综合肝脏分析。

调用 Tool_Box.liver_analysis.analyze_liver_case 计算所有指标，
再调用 generate_liver_report 生成格式化报告。
"""

from Tool_Box.liver_analysis import analyze_liver_case, generate_liver_report
from skills.utils import convert_numpy


def run(ctx):
    # 构建 mask_paths 字典
    mask_paths = {}
    for name in ctx.list_masks():
        mask_paths[name] = ctx.get_mask_path(name)

    ctx.log(f"Running full liver analysis with {len(mask_paths)} masks...")

    # 执行综合分析
    analysis = analyze_liver_case(
        ct_nifti_path=ctx.ct_nifti_path,
        mask_paths=mask_paths,
    )

    # 生成格式化报告
    report_text = generate_liver_report(analysis)

    ctx.log("Liver analysis complete")

    # 去掉 largest_mask: 该三维数组尺寸 = CT 全分辨率 (H,W,D),
    # 里面 99.99% 是 0，序列化为 JSON 会膨胀到数千万个嵌套列表。
    # 这些 mask 只在 compute_largest_cc → diameter/distance 中间计算时需要，
    # 最终输出只需要统计量（largest_voxels, largest_ratio 等）。
    tumors = analysis.get("tumor_results", {})
    for tr in tumors.values():
        cc = tr.get("largest_cc", {})
        if "largest_mask" in cc:
            del cc["largest_mask"]

    return convert_numpy({
        "liver_volume_cm3": analysis.get("liver_volume_cm3"),
        "vessel_volumes": analysis.get("vessel_volumes", {}),
        "tumor_results": tumors,
        "report_text": report_text,
    })
