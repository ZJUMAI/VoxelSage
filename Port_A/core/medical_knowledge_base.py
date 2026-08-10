"""
医学知识库模块 - P0 优化
提供医学参考范围、临床规则和结果验证功能
"""
from typing import Dict, List, Optional, Any
import re


class MedicalKnowledgeBase:
    """医学知识库：提供参考范围和临床规则"""

    # 医学测量参考范围
    REFERENCE_RANGES = {
        "liver_volume": {
            "min": 500,
            "max": 2500,
            "unit": "cm³",
            "typical_range": (1000, 1800),
            "description": "正常成人肝脏体积"
        },
        "tumor_diameter": {
            "min": 0.5,
            "max": 200,
            "unit": "mm",
            "description": "肿瘤最大直径"
        },
        "tumor_volume": {
            "min": 0.1,
            "max": 500,
            "unit": "cm³",
            "description": "单个肿瘤体积"
        },
        "vessel_volume": {
            "min": 1,
            "max": 100,
            "unit": "cm³",
            "description": "肝内血管体积"
        },
        "tumor_vessel_distance": {
            "min": 0,
            "max": 150,
            "unit": "mm",
            "description": "肿瘤与血管最短距离"
        }
    }

    # 临床判断规则
    CLINICAL_RULES = [
        {
            "id": "multiple_lesions",
            "condition": lambda data: data.get("tumor_count", 0) > 3,
            "implication": "多发性病灶（>3个），建议进一步分期评估",
            "urgency": "medium",
            "recommendations": ["完善增强CT", "评估肿瘤分期", "考虑穿刺活检"]
        },
        {
            "id": "large_tumor",
            "condition": lambda data: data.get("max_tumor_diameter", 0) > 50,
            "implication": "发现大病灶（>5cm），需重点关注",
            "urgency": "high",
            "recommendations": ["MDT讨论", "评估手术可行性", "排查血管侵犯"]
        },
        {
            "id": "vessel_involvement",
            "condition": lambda data: data.get("min_tumor_vessel_distance", 999) < 5,
            "implication": "肿瘤邻近主要血管（<5mm），手术风险较高",
            "urgency": "high",
            "recommendations": ["血管重建评估", "介入治疗评估", "术前规划"]
        },
        {
            "id": "enlarged_liver",
            "condition": lambda data: data.get("liver_volume", 0) > 2000,
            "implication": "肝脏体积增大，可能存在肝肿大或脂肪肝",
            "urgency": "low",
            "recommendations": ["肝功能检查", "排查代谢性疾病"]
        },
        {
            "id": "small_liver",
            "condition": lambda data: data.get("liver_volume", 9999) < 800,
            "implication": "肝脏体积偏小，注意肝硬化可能",
            "urgency": "medium",
            "recommendations": ["肝纤维化评估", "Child-Pugh分级", "肝储备功能评估"]
        }
    ]

    @classmethod
    def validate_measurement(cls, metric: str, value: float) -> Dict[str, Any]:
        """
        验证测量值是否在医学合理范围内

        Args:
            metric: 测量指标名称（如 'liver_volume'）
            value: 测量值

        Returns:
            验证结果字典
        """
        ref = cls.REFERENCE_RANGES.get(metric)
        if not ref:
            return {
                "valid": True,
                "warning": None,
                "severity": "none"
            }

        # 检查是否在绝对范围内
        if not (ref["min"] <= value <= ref["max"]):
            return {
                "valid": False,
                "warning": f"{metric} 值 {value:.2f} {ref['unit']} 超出合理范围 [{ref['min']}, {ref['max']}]",
                "severity": "critical",
                "action": "建议人工复核，可能存在测量错误或极端病例",
                "reference": ref
            }

        # 检查是否在典型范围内（如果定义了）
        if "typical_range" in ref:
            typical_min, typical_max = ref["typical_range"]
            if not (typical_min <= value <= typical_max):
                return {
                    "valid": True,
                    "warning": f"{metric} 值 {value:.2f} {ref['unit']} 在合理范围内，但偏离典型值 [{typical_min}, {typical_max}]",
                    "severity": "info",
                    "action": "数值偏高或偏低，建议关注",
                    "reference": ref
                }

        return {
            "valid": True,
            "warning": None,
            "severity": "none",
            "reference": ref
        }

    @classmethod
    def validate_result_consistency(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证结果的内部一致性

        Args:
            result: liver_analysis 返回的结果

        Returns:
            一致性检查结果
        """
        issues = []

        # 检查肿瘤数量与 tumor_results 是否一致
        tumor_results = result.get("tumor_results", {})
        actual_tumor_count = len(tumor_results) if isinstance(tumor_results, dict) else 0

        # 检查各个肿瘤的数据完整性
        for tumor_name, tumor_data in tumor_results.items():
            if not isinstance(tumor_data, dict):
                issues.append({
                    "type": "data_integrity",
                    "severity": "warning",
                    "message": f"{tumor_name} 数据格式异常"
                })
                continue

            # 检查必需字段
            required_fields = ["volume_cm3", "max_diameter_mm"]
            missing = [f for f in required_fields if f not in tumor_data]
            if missing:
                issues.append({
                    "type": "missing_data",
                    "severity": "warning",
                    "message": f"{tumor_name} 缺少字段: {', '.join(missing)}"
                })

            # 检查体积与直径的合理性（粗略估计：球体）
            vol = tumor_data.get("volume_cm3")
            diam = tumor_data.get("max_diameter_mm")
            if vol and diam:
                # 球体体积公式: V = 4/3 * π * r³，r = d/2
                # 将 mm 转为 cm: diam_cm = diam / 10
                expected_vol_approx = (4/3) * 3.14159 * ((diam / 10 / 2) ** 3)
                # 允许 5 倍偏差（因为肿瘤不规则）
                if abs(vol - expected_vol_approx) > expected_vol_approx * 5:
                    issues.append({
                        "type": "measurement_inconsistency",
                        "severity": "info",
                        "message": f"{tumor_name} 体积({vol:.2f}cm³)与直径({diam:.1f}mm)可能不匹配（预期约{expected_vol_approx:.2f}cm³）"
                    })

        return {
            "consistent": len(issues) == 0,
            "issues": issues,
            "tumor_count": actual_tumor_count
        }

    @classmethod
    def apply_clinical_rules(cls, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        应用临床规则，生成临床建议

        Args:
            analysis_data: 综合分析数据

        Returns:
            触发的临床规则列表
        """
        triggered_rules = []

        for rule in cls.CLINICAL_RULES:
            try:
                if rule["condition"](analysis_data):
                    triggered_rules.append({
                        "rule_id": rule["id"],
                        "implication": rule["implication"],
                        "urgency": rule["urgency"],
                        "recommendations": rule["recommendations"]
                    })
            except Exception:
                # 规则评估失败，跳过
                continue

        # 按紧急程度排序
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        triggered_rules.sort(key=lambda r: urgency_order.get(r["urgency"], 99))

        return triggered_rules

    @classmethod
    def extract_analysis_summary(cls, tool_store: Dict[str, Any], case_id: str) -> Dict[str, Any]:
        """
        从 tool_store 提取关键指标用于临床规则评估

        Args:
            tool_store: 会话的 tool_store
            case_id: 病例 ID

        Returns:
            提取的关键指标
        """
        case_data = tool_store.get(case_id, {})
        liver_result = case_data.get("liver_analysis", {})

        summary = {
            "case_id": case_id,
            "liver_volume": liver_result.get("liver_volume_cm3"),
            "tumor_count": 0,
            "max_tumor_diameter": 0,
            "total_tumor_volume": 0,
            "min_tumor_vessel_distance": 999
        }

        # 提取肿瘤信息
        tumor_results = liver_result.get("tumor_results", {})
        if isinstance(tumor_results, dict):
            summary["tumor_count"] = len(tumor_results)

            for tumor_data in tumor_results.values():
                if not isinstance(tumor_data, dict):
                    continue

                # 最大直径
                diam = tumor_data.get("max_diameter_mm", 0)
                if diam > summary["max_tumor_diameter"]:
                    summary["max_tumor_diameter"] = diam

                # 总体积
                vol = tumor_data.get("volume_cm3", 0)
                summary["total_tumor_volume"] += vol

                # 最小距离
                dist = tumor_data.get("min_distance_to_vessel_mm", 999)
                if dist < summary["min_tumor_vessel_distance"]:
                    summary["min_tumor_vessel_distance"] = dist

        return summary

    @classmethod
    def generate_clinical_report_section(cls, tool_store: Dict[str, Any], case_id: str) -> str:
        """
        生成临床意义和建议部分

        Args:
            tool_store: 会话的 tool_store
            case_id: 病例 ID

        Returns:
            格式化的临床报告文本
        """
        summary = cls.extract_analysis_summary(tool_store, case_id)
        rules = cls.apply_clinical_rules(summary)

        if not rules:
            return "测量值均在正常范围内，暂无特殊临床提示。"

        sections = ["## 临床意义与建议\n"]

        for i, rule in enumerate(rules, 1):
            urgency_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢"
            }
            emoji = urgency_emoji.get(rule["urgency"], "ℹ️")

            sections.append(f"{emoji} **{rule['implication']}**")
            if rule["recommendations"]:
                sections.append("   建议：" + "、".join(rule["recommendations"]))
            sections.append("")

        return "\n".join(sections)
