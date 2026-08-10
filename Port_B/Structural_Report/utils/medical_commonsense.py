"""
medical_commonsense.py — 医学常识校验模块

用于验证器官/病变体积测量结果是否符合人体解剖学常识。
当分割算法计算出明显不符合实际情况的器官体积时，在报告中添加
警告标记，帮助下游模型判断测量结果的可靠性。

参考来源:
  - 成人腹部 CT 器官体积正常参考值（放射学教材及 CT volumetry 文献）
  - Taylor et al. AJR 1991 (spleen)
  - Geraghty et al. RadioGraphics 2004 (kidney)
  - Kipp et al. Abdom Radiol 2016 (pancreas)
"""

# ======================================================================
# 器官体积参考范围（单位: mL）—— 成人正常值
# ======================================================================
# 当计算体积超出 absurd 界限时，说明分割结果极可能错误
# 当超出 normal 范围时，说明体积异常但可能反映真实病理

ORGAN_VOLUME_RANGES = {
    "liver": {
        "normal_min": 800.0,     # 正常下限
        "normal_max": 2500.0,    # 正常上限（>2500 提示肝肿大）
        "absurd_min": 300.0,     # 无论如何不可能小于此值
        "absurd_max": 6000.0,    # 无论如何不可能大于此值
        "reference": "正常成人肝脏体积约 1200-1600 mL，肝肿大 >2500 mL",
    },
    "spleen": {
        "normal_min": 50.0,
        "normal_max": 350.0,
        "absurd_min": 15.0,
        "absurd_max": 2000.0,
        "reference": "正常成人脾脏体积约 150-250 mL，脾肿大 >350 mL，巨脾 >430 mL",
    },
    "pancreas": {
        "normal_min": 30.0,
        "normal_max": 120.0,
        "absurd_min": 8.0,
        "absurd_max": 500.0,
        "reference": "正常成人胰腺体积约 50-100 mL",
    },
    "kidney_right": {
        "normal_min": 80.0,
        "normal_max": 250.0,
        "absurd_min": 20.0,
        "absurd_max": 500.0,
        "reference": "正常成人右肾体积约 120-180 mL",
    },
    "kidney_left": {
        "normal_min": 80.0,
        "normal_max": 250.0,
        "absurd_min": 20.0,
        "absurd_max": 500.0,
        "reference": "正常成人左肾体积约 120-180 mL",
    },
    "kidney_total": {
        "normal_min": 160.0,
        "normal_max": 500.0,
        "absurd_min": 50.0,
        "absurd_max": 1000.0,
        "reference": "正常成人双肾总体积约 240-360 mL",
    },
    "colon": {
        "normal_min": 200.0,
        "normal_max": 1500.0,
        "absurd_min": 50.0,
        "absurd_max": 6000.0,
        "reference": "结肠体积个体差异大，通常 300-1000 mL（受内容物影响）",
    },
}

# 病变相对器官的最大合理体积比
# 如果病变体积超过器官体积的 80%，极可能是分割错误
MAX_LESION_ORGAN_RATIO = 0.8

# 双肾对称性：较小侧 / 较大侧 不应低于此比例
KIDNEY_SYMMETRY_MIN_RATIO = 0.5


def validate_organ_volume(organ_key: str, volume_cm3: float) -> tuple:
    """
    校验器官体积是否在医学合理范围内。

    参数:
        organ_key: 器官标识（如 'liver', 'spleen', 'kidney_right' 等）
        volume_cm3: 器官体积（立方厘米）

    返回:
        (is_plausible: bool, warning: str or None)
        - is_plausible: True 表示体积合理或仅轻微异常
        - warning: None 表示无警告，非 None 表示需附加到报告中的警告文本
    """
    if volume_cm3 is None:
        return True, None

    if organ_key not in ORGAN_VOLUME_RANGES:
        return True, None

    ranges = ORGAN_VOLUME_RANGES[organ_key]

    # ---- "荒谬"界限：超出此范围说明分割极可能错误 ----
    if volume_cm3 < ranges["absurd_min"]:
        return False, (
            f"[医学警告] {organ_key} 的计算体积 ({volume_cm3:.1f} mL) "
            f"低于解剖学下限 ({ranges['absurd_min']} mL)，"
            f"分割结果可能严重错误（如仅分割到部分器官）。"
        )

    if volume_cm3 > ranges["absurd_max"]:
        return False, (
            f"[医学警告] {organ_key} 的计算体积 ({volume_cm3:.1f} mL) "
            f"超过解剖学上限 ({ranges['absurd_max']} mL)，"
            f"分割结果可能严重错误（如包含了周围组织）。"
        )

    # ---- "正常"范围之外但尚合理：可能是真实病理 ----
    if volume_cm3 < ranges["normal_min"]:
        return True, (
            f"[提示] {organ_key} 体积偏小 ({volume_cm3:.1f} mL，"
            f"参考范围 {ranges['normal_min']}–{ranges['normal_max']} mL)，"
            f"可能反映真实萎缩或发育变异。"
        )

    if volume_cm3 > ranges["normal_max"]:
        return True, (
            f"[提示] {organ_key} 体积偏大 ({volume_cm3:.1f} mL，"
            f"参考范围 {ranges['normal_min']}–{ranges['normal_max']} mL)，"
            f"可能反映器官肿大。"
        )

    return True, None


def check_kidney_symmetry(vol_left_cm3: float, vol_right_cm3: float) -> tuple:
    """
    校验双肾体积是否对称。

    正常肾脏双侧体积大致对称，一侧不应显著小于另一侧。

    返回:
        (is_symmetric: bool, warning: str or None)
    """
    if vol_left_cm3 is None or vol_right_cm3 is None:
        return True, None

    if vol_left_cm3 <= 0 or vol_right_cm3 <= 0:
        return True, None

    smaller = min(vol_left_cm3, vol_right_cm3)
    larger = max(vol_left_cm3, vol_right_cm3)
    ratio = smaller / larger

    if ratio < KIDNEY_SYMMETRY_MIN_RATIO:
        side_smaller = "左" if vol_left_cm3 < vol_right_cm3 else "右"
        larger_val = max(vol_left_cm3, vol_right_cm3)
        smaller_val = min(vol_left_cm3, vol_right_cm3)
        return False, (
            f"[医学警告] 双肾体积不对称（{side_smaller}肾 {smaller_val:.1f} mL "
            f"vs 对侧 {larger_val:.1f} mL，比值 {ratio:.2f}），"
            f"可能反映一侧分割不完整或真实萎缩。"
        )

    return True, None


def validate_lesion_volume(
    lesion_voxels: int,
    organ_voxels: int,
    lesion_desc: str = "病灶",
) -> tuple:
    """
    校验病灶体积是否超过所属器官体积的合理比例。

    参数:
        lesion_voxels: 病灶体素数（1mm 各向同性空间）
        organ_voxels:  所属器官体素数
        lesion_desc:   病灶描述（如 "肝肿瘤"、"肾囊肿"）

    返回:
        (is_plausible: bool, warning: str or None)
    """
    if organ_voxels <= 0 or lesion_voxels <= 0:
        return True, None

    ratio = lesion_voxels / organ_voxels
    if ratio > MAX_LESION_ORGAN_RATIO:
        lesion_cm3 = lesion_voxels / 1000.0
        organ_cm3 = organ_voxels / 1000.0
        return False, (
            f"[医学警告] {lesion_desc} 体积 ({lesion_cm3:.1f} mL) "
            f"占所属器官体积的 {ratio:.0%}（器官 {organ_cm3:.1f} mL），"
            f"超过合理上限 {MAX_LESION_ORGAN_RATIO:.0%}，分割结果可能错误。"
        )

    return True, None


def get_organ_reference(organ_key: str) -> str:
    """获取器官体积参考范围描述。"""
    if organ_key in ORGAN_VOLUME_RANGES:
        r = ORGAN_VOLUME_RANGES[organ_key]
        return (
            f"{organ_key}: 正常范围 {r['normal_min']}–{r['normal_max']} mL, "
            f"合理界限 [{r['absurd_min']}, {r['absurd_max']}] mL. "
            f"({r['reference']})"
        )
    return ""
