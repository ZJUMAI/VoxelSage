"""
工具选择优化器 - P0 优化
基于已有结果智能过滤冗余工具调用，提高效率
"""
from typing import Dict, List, Any, Set, Optional
import json


class ToolOptimizer:
    """工具选择优化器：避免冗余调用，提高效率"""

    # 技能依赖关系：如果父技能已完成，子技能可以跳过
    SKILL_HIERARCHY = {
        "liver_analysis": {
            "includes": ["vessel_volume", "tumor_diameter", "tumor_vessel_distance"],
            "description": "liver_analysis 已包含血管体积、肿瘤直径和肿瘤-血管距离分析"
        }
    }

    # 技能互斥关系：某些技能不应同时调用
    SKILL_CONFLICTS = [
        {
            "skills": ["liver_analysis", "vessel_volume"],
            "reason": "liver_analysis 已包含 vessel_volume 的功能"
        },
        {
            "skills": ["liver_analysis", "tumor_diameter"],
            "reason": "liver_analysis 已包含 tumor_diameter 的功能"
        },
        {
            "skills": ["liver_analysis", "tumor_vessel_distance"],
            "reason": "liver_analysis 已包含 tumor_vessel_distance 的功能"
        }
    ]

    # 技能数据映射：定义每个技能提供的数据
    SKILL_DATA_MAPPING = {
        "liver_analysis": {
            "provides": [
                "liver_volume_cm3",
                "vessel_volumes",
                "tumor_results",
                "tumor_count",
                "tumor_diameters",
                "tumor_vessel_distances"
            ]
        },
        "vessel_volume": {
            "provides": ["vessel_volumes"]
        },
        "tumor_diameter": {
            "provides": ["tumor_diameters"]
        },
        "tumor_vessel_distance": {
            "provides": ["tumor_vessel_distances"]
        },
        "slice_selection": {
            "provides": ["best_slices", "slice_images"]
        },
        "three_d_reconstruction": {
            "provides": ["html_url", "3d_visualization"]
        }
    }

    @classmethod
    def filter_redundant_tools(
        cls,
        session: Dict[str, Any],
        available_tools: List[Dict[str, Any]],
        case_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """
        基于 tool_store 中已有结果，过滤掉冗余工具

        Args:
            session: 当前会话
            available_tools: 可用工具列表
            case_id: 目标病例 ID（如果为 None，检查所有 case）

        Returns:
            (filtered_tools, filter_reasons) 元组
        """
        tool_store = session.get("tool_store", {})
        filtered_tools = []
        filter_reasons = []

        # 确定要检查的 case_ids
        if case_id:
            check_cases = [case_id]
        else:
            check_cases = list(tool_store.keys())

        # 收集已完成的技能
        completed_skills: Set[str] = set()
        for cid in check_cases:
            case_data = tool_store.get(cid, {})
            for skill_name, result in case_data.items():
                if skill_name == "segmentation":
                    continue
                # 检查是否成功完成（没有 _error 标记）
                if isinstance(result, dict) and not result.get("_error"):
                    completed_skills.add(skill_name)

        # 过滤工具
        for tool in available_tools:
            tool_name = tool.get("function", {}).get("name", "")

            # 检查是否被父技能包含
            should_filter = False
            for parent_skill, hierarchy in cls.SKILL_HIERARCHY.items():
                if parent_skill in completed_skills:
                    if tool_name in hierarchy["includes"]:
                        filter_reasons.append({
                            "filtered_tool": tool_name,
                            "reason": f"已被 {parent_skill} 包含",
                            "detail": hierarchy["description"]
                        })
                        should_filter = True
                        break

            if not should_filter:
                # 检查是否已经完成
                if tool_name in completed_skills:
                    # 特殊情况：可视化类技能允许重复调用（用户可能想要不同视角）
                    visualization_tools = {"slice_selection", "three_d_reconstruction", "segmentation_modification"}
                    if tool_name not in visualization_tools:
                        filter_reasons.append({
                            "filtered_tool": tool_name,
                            "reason": "该技能已完成",
                            "detail": f"{tool_name} 的结果已在 tool_store 中"
                        })
                        should_filter = True

            if not should_filter:
                filtered_tools.append(tool)

        return filtered_tools, filter_reasons

    @classmethod
    def suggest_next_skills(
        cls,
        session: Dict[str, Any],
        user_query: str,
        case_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        基于用户问题和已有数据，智能推荐下一步应该调用的技能

        Args:
            session: 当前会话
            user_query: 用户问题
            case_id: 目标病例 ID

        Returns:
            推荐技能列表，按优先级排序
        """
        tool_store = session.get("tool_store", {})
        suggestions = []

        # 确定要检查的 case
        if case_id:
            check_cases = [case_id]
        else:
            check_cases = list(tool_store.keys())

        # 收集已完成的技能
        completed_skills: Set[str] = set()
        for cid in check_cases:
            case_data = tool_store.get(cid, {})
            for skill_name, result in case_data.items():
                if isinstance(result, dict) and not result.get("_error"):
                    completed_skills.add(skill_name)

        query_lower = user_query.lower()

        # 规则1: 体积相关问题 → liver_analysis
        if any(kw in query_lower for kw in ["volume", "体积", "大小", "cm³", "cubic"]):
            if "liver_analysis" not in completed_skills:
                suggestions.append({
                    "skill": "liver_analysis",
                    "priority": 1,
                    "reason": "问题涉及体积测量，liver_analysis 提供完整的肝脏和肿瘤体积数据"
                })

        # 规则2: 病灶计数问题 → liver_analysis
        if any(kw in query_lower for kw in ["how many", "lesion", "tumor", "病灶", "肿瘤", "个数", "数量"]):
            if "liver_analysis" not in completed_skills:
                suggestions.append({
                    "skill": "liver_analysis",
                    "priority": 1,
                    "reason": "问题涉及病灶计数，liver_analysis 提供 tumor_results 统计"
                })

        # 规则3: 直径相关问题 → liver_analysis
        if any(kw in query_lower for kw in ["diameter", "直径", "size", "mm", "millimeter"]):
            if "liver_analysis" not in completed_skills:
                suggestions.append({
                    "skill": "liver_analysis",
                    "priority": 1,
                    "reason": "问题涉及直径测量，liver_analysis 提供肿瘤最大直径"
                })

        # 规则4: 血管相关问题 → liver_analysis
        if any(kw in query_lower for kw in ["vessel", "blood", "vascular", "血管", "距离", "distance"]):
            if "liver_analysis" not in completed_skills:
                suggestions.append({
                    "skill": "liver_analysis",
                    "priority": 1,
                    "reason": "问题涉及血管分析，liver_analysis 提供血管体积和肿瘤-血管距离"
                })

        # 规则5: 可视化需求 → slice_selection 或 3D 重建
        if any(kw in query_lower for kw in ["show", "visualize", "slice", "image", "显示", "切片", "图像", "可视化"]):
            if "slice_selection" not in completed_skills:
                suggestions.append({
                    "skill": "slice_selection",
                    "priority": 2,
                    "reason": "问题需要可视化，slice_selection 可提供最佳切片"
                })
            if "three_d_reconstruction" not in completed_skills:
                suggestions.append({
                    "skill": "three_d_reconstruction",
                    "priority": 2,
                    "reason": "问题需要可视化，3D 重建可提供立体视图"
                })

        # 去重并排序
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s["skill"] not in seen:
                seen.add(s["skill"])
                unique_suggestions.append(s)

        unique_suggestions.sort(key=lambda x: x["priority"])
        return unique_suggestions

    @classmethod
    def can_answer_from_cache(
        cls,
        session: Dict[str, Any],
        user_query: str,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        判断是否可以直接从 tool_store 回答问题，无需新的技能调用

        Args:
            session: 当前会话
            user_query: 用户问题
            case_id: 目标病例 ID

        Returns:
            {"can_answer": bool, "required_data": [], "missing_data": [], "suggestion": str}
        """
        tool_store = session.get("tool_store", {})
        query_lower = user_query.lower()

        # 确定要检查的 case
        if case_id:
            check_cases = [case_id]
        else:
            check_cases = list(tool_store.keys())

        if not check_cases:
            return {
                "can_answer": False,
                "required_data": [],
                "missing_data": ["基础数据（需要先运行 segmentation）"],
                "suggestion": "需要先上传并分析影像数据"
            }

        # 分析问题需要什么数据
        required_data = []
        if any(kw in query_lower for kw in ["liver", "volume", "肝脏", "体积"]):
            required_data.append("liver_volume_cm3")

        if any(kw in query_lower for kw in ["lesion", "tumor", "how many", "病灶", "肿瘤", "个数"]):
            required_data.extend(["tumor_results", "tumor_count"])

        if any(kw in query_lower for kw in ["diameter", "直径", "size"]):
            required_data.append("tumor_diameters")

        if any(kw in query_lower for kw in ["vessel", "血管"]):
            required_data.extend(["vessel_volumes", "tumor_vessel_distances"])

        # 检查数据是否已存在
        available_data = set()
        for cid in check_cases:
            case_data = tool_store.get(cid, {})
            liver_result = case_data.get("liver_analysis", {})

            if liver_result and not liver_result.get("_error"):
                if "liver_volume_cm3" in liver_result:
                    available_data.add("liver_volume_cm3")
                if "tumor_results" in liver_result:
                    available_data.add("tumor_results")
                    available_data.add("tumor_count")
                    available_data.add("tumor_diameters")
                    available_data.add("tumor_vessel_distances")
                if "vessel_volumes" in liver_result:
                    available_data.add("vessel_volumes")

        # 计算缺失数据
        missing_data = [d for d in required_data if d not in available_data]

        can_answer = len(missing_data) == 0 and len(required_data) > 0

        suggestion = ""
        if can_answer:
            suggestion = "所需数据已完整，可直接从 tool_store 提取答案"
        elif missing_data:
            suggestion = f"缺少数据: {', '.join(missing_data)}，建议调用 liver_analysis"
        elif not required_data:
            suggestion = "无法从问题中识别所需数据类型，建议让 LLM 自行判断"

        return {
            "can_answer": can_answer,
            "required_data": required_data,
            "missing_data": missing_data,
            "available_data": list(available_data),
            "suggestion": suggestion
        }

    @classmethod
    def generate_filter_summary(cls, filter_reasons: List[Dict[str, str]]) -> str:
        """
        生成工具过滤摘要，用于日志或提示信息

        Args:
            filter_reasons: 过滤原因列表

        Returns:
            格式化的摘要文本
        """
        if not filter_reasons:
            return "未过滤任何工具"

        summary_lines = [f"已过滤 {len(filter_reasons)} 个冗余工具："]
        for item in filter_reasons:
            summary_lines.append(f"  - {item['filtered_tool']}: {item['reason']}")

        return "\n".join(summary_lines)
