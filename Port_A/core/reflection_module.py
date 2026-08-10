"""
反思机制模块 - P0 优化
分析技能调用失败原因，提供补救策略和智能重试
"""
from typing import Dict, List, Any, Optional, Tuple
import json
import re


class ReflectionEngine:
    """反思引擎：分析失败原因并生成补救策略"""

    # 错误类型分类
    ERROR_CATEGORIES = {
        "PORT_B_HTTP_500": {
            "type": "temporary",
            "severity": "medium",
            "retryable": True,
            "description": "Port B 服务器内部错误",
            "common_causes": ["服务过载", "临时故障", "内存不足"],
            "strategies": ["等待后重试", "降低并发数", "检查 Port B 日志"]
        },
        "PORT_B_HTTP_504": {
            "type": "timeout",
            "severity": "medium",
            "retryable": True,
            "description": "Port B 请求超时",
            "common_causes": ["数据量过大", "计算复杂度高", "网络延迟"],
            "strategies": ["增加超时时间", "分解任务", "优化参数"]
        },
        "PORT_A_TOOL_VALIDATION_OR_EXECUTION_ERROR": {
            "type": "validation",
            "severity": "high",
            "retryable": False,
            "description": "工具参数验证失败",
            "common_causes": ["参数类型错误", "缺少必需参数", "参数值超出范围"],
            "strategies": ["检查参数格式", "查看工具 schema", "使用默认参数"]
        },
        "INVALID_PORT_B_SKILL_RESPONSE": {
            "type": "protocol",
            "severity": "high",
            "retryable": False,
            "description": "Port B 返回格式异常",
            "common_causes": ["Port B 版本不匹配", "响应被截断", "序列化错误"],
            "strategies": ["检查 Port B 版本", "查看完整响应", "联系 Port B 维护者"]
        },
        "SKILL_CALL_LIMIT_REACHED": {
            "type": "resource",
            "severity": "low",
            "retryable": False,
            "description": "达到技能调用次数上限",
            "common_causes": ["agent 陷入循环", "任务过于复杂"],
            "strategies": ["优化 agent 策略", "提高调用上限", "简化任务"]
        }
    }

    # 技能特定的失败模式
    SKILL_SPECIFIC_FAILURES = {
        "liver_analysis": {
            "timeout_likely": True,
            "typical_time": 250.0,
            "timeout_strategy": "这是计算密集型任务，考虑增加超时时间到 600 秒",
            "prerequisites": ["segmentation 必须成功完成", "mask 文件必须存在"],
            "common_issues": [
                "mask 文件路径错误",
                "CT 数据质量差",
                "肿瘤过多导致计算超时"
            ]
        },
        "slice_selection": {
            "timeout_likely": False,
            "typical_time": 30.0,
            "prerequisites": ["segmentation 必须完成"],
            "common_issues": ["输出目录权限问题", "图像格式不支持"]
        },
        "three_d_reconstruction": {
            "timeout_likely": False,
            "typical_time": 60.0,
            "prerequisites": ["segmentation 必须完成"],
            "common_issues": ["WebGL 渲染失败", "文件过大"]
        }
    }

    @classmethod
    def analyze_failure(
        cls,
        skill_name: str,
        error_code: Optional[str],
        error_message: str,
        params: Dict[str, Any],
        execution_time_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        深度分析技能调用失败原因

        Args:
            skill_name: 失败的技能名称
            error_code: 错误代码
            error_message: 错误消息
            params: 调用参数
            execution_time_ms: 执行时间（毫秒）

        Returns:
            详细的失败分析报告
        """
        analysis = {
            "skill_name": skill_name,
            "error_code": error_code,
            "error_message": error_message,
            "category": "unknown",
            "severity": "unknown",
            "retryable": False,
            "root_cause": [],
            "impact": "",
            "strategies": []
        }

        # 查找错误类型
        error_info = cls.ERROR_CATEGORIES.get(error_code or "UNKNOWN")
        if error_info:
            analysis["category"] = error_info["type"]
            analysis["severity"] = error_info["severity"]
            analysis["retryable"] = error_info["retryable"]
            analysis["root_cause"] = error_info["common_causes"]
            analysis["strategies"] = error_info["strategies"].copy()

        # 技能特定分析
        skill_info = cls.SKILL_SPECIFIC_FAILURES.get(skill_name)
        if skill_info:
            # 检查是否超时
            if execution_time_ms and skill_info.get("timeout_likely"):
                typical_time_ms = skill_info["typical_time"] * 1000
                if execution_time_ms > typical_time_ms * 1.5:
                    analysis["root_cause"].append("执行时间超过预期")
                    if skill_info.get("timeout_strategy"):
                        analysis["strategies"].append(skill_info["timeout_strategy"])

            # 添加前置条件检查
            if "prerequisites" in skill_info:
                analysis["prerequisites"] = skill_info["prerequisites"]

            # 常见问题
            if "common_issues" in skill_info:
                analysis["common_issues"] = skill_info["common_issues"]

        # 参数分析
        param_issues = cls._analyze_parameters(skill_name, params)
        if param_issues:
            analysis["parameter_issues"] = param_issues
            analysis["strategies"].extend([f"修正参数: {issue}" for issue in param_issues])

        # 评估影响
        analysis["impact"] = cls._assess_impact(skill_name, error_code)

        return analysis

    @classmethod
    def _analyze_parameters(cls, skill_name: str, params: Dict[str, Any]) -> List[str]:
        """分析参数是否可能导致失败"""
        issues = []

        # 通用参数检查
        if not params:
            issues.append("参数为空，可能缺少必需参数")

        # 检查是否包含不应该由 LLM 提供的上下文参数
        forbidden_params = {"ct_nifti_path", "mask_dir", "output_dir", "case_id", "input"}
        found_forbidden = forbidden_params.intersection(params.keys())
        if found_forbidden:
            issues.append(f"不应包含上下文参数: {', '.join(found_forbidden)}")

        return issues

    @classmethod
    def _assess_impact(cls, skill_name: str, error_code: Optional[str]) -> str:
        """评估失败的影响程度"""
        # liver_analysis 失败影响大
        if skill_name == "liver_analysis":
            return "严重：liver_analysis 是核心分析技能，失败将导致无法回答大多数问题"

        # 可视化失败影响小
        if skill_name in ["slice_selection", "three_d_reconstruction"]:
            return "轻微：可视化失败不影响数据分析，但用户体验下降"

        # 超时类错误影响中等
        if error_code and "TIMEOUT" in error_code.upper():
            return "中等：超时可能导致部分数据缺失，但可以重试"

        return "中等：技能失败可能影响部分功能"

    @classmethod
    def suggest_recovery_strategy(
        cls,
        session: Dict[str, Any],
        failed_skill: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        基于失败分析，建议恢复策略

        Args:
            session: 当前会话
            failed_skill: 失败的技能名称
            analysis: 失败分析结果

        Returns:
            恢复策略
        """
        strategy = {
            "action": "skip",  # skip | retry | alternative | ask_user
            "reasoning": "",
            "alternative_skills": [],
            "retry_params": None,
            "user_prompt": None
        }

        tool_store = session.get("tool_store", {})
        error_code = analysis.get("error_code")

        # 策略1: 如果是临时性错误且可重试，建议重试
        if analysis.get("retryable") and analysis.get("category") == "temporary":
            strategy["action"] = "retry"
            strategy["reasoning"] = "临时性错误，建议重试"
            strategy["retry_params"] = {
                "max_retries": 2,
                "backoff_seconds": 5
            }
            return strategy

        # 策略2: 如果是超时错误，且是 liver_analysis，建议增加超时后重试
        if "TIMEOUT" in (error_code or "").upper() and failed_skill == "liver_analysis":
            strategy["action"] = "retry"
            strategy["reasoning"] = "liver_analysis 超时，建议增加超时时间后重试"
            strategy["retry_params"] = {
                "max_retries": 1,
                "timeout_multiplier": 2.0
            }
            return strategy

        # 策略3: 如果 liver_analysis 失败，检查是否可以用子技能替代
        if failed_skill == "liver_analysis":
            # 检查哪些子技能已经完成
            available_alternatives = []
            for case_data in tool_store.values():
                if case_data.get("vessel_volume") and not case_data["vessel_volume"].get("_error"):
                    available_alternatives.append("vessel_volume")
                if case_data.get("tumor_diameter") and not case_data["tumor_diameter"].get("_error"):
                    available_alternatives.append("tumor_diameter")

            if not available_alternatives:
                strategy["action"] = "alternative"
                strategy["reasoning"] = "liver_analysis 失败，尝试调用单独的子技能"
                strategy["alternative_skills"] = ["vessel_volume", "tumor_diameter", "tumor_vessel_distance"]
                return strategy

        # 策略4: 如果是参数验证错误，建议询问用户或使用默认参数
        if analysis.get("category") == "validation":
            strategy["action"] = "skip"
            strategy["reasoning"] = "参数验证失败，无法自动修复，建议基于已有数据回答"
            return strategy

        # 策略5: 可视化失败，影响小，直接跳过
        if failed_skill in ["slice_selection", "three_d_reconstruction", "segmentation_modification"]:
            strategy["action"] = "skip"
            strategy["reasoning"] = "可视化失败不影响核心分析，跳过该步骤"
            return strategy

        # 策略6: 检查是否可以基于已有数据回答问题
        if cls._can_answer_without_skill(session, failed_skill):
            strategy["action"] = "skip"
            strategy["reasoning"] = f"{failed_skill} 失败，但已有足够数据回答问题"
            return strategy

        # 默认：跳过
        strategy["action"] = "skip"
        strategy["reasoning"] = "无法自动恢复，建议基于已有数据尽力回答"
        return strategy

    @classmethod
    def _can_answer_without_skill(cls, session: Dict[str, Any], failed_skill: str) -> bool:
        """判断即使某个技能失败，是否仍能回答问题"""
        tool_store = session.get("tool_store", {})

        # 检查是否有任何 case 完成了基础分析
        for case_data in tool_store.values():
            if case_data.get("liver_analysis") and not case_data["liver_analysis"].get("_error"):
                # 有完整的 liver_analysis 结果
                return True

            # 或者有足够的子技能结果
            has_volume = case_data.get("vessel_volume") and not case_data["vessel_volume"].get("_error")
            has_diameter = case_data.get("tumor_diameter") and not case_data["tumor_diameter"].get("_error")
            if has_volume and has_diameter:
                return True

        return False

    @classmethod
    def generate_reflection_prompt(
        cls,
        session: Dict[str, Any],
        failed_calls: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        user_query: str
    ) -> str:
        """
        生成反思提示，引导 LLM 基于失败情况调整策略

        Args:
            session: 当前会话
            failed_calls: 失败的调用列表 [(call, response)]
            user_query: 用户原始问题

        Returns:
            反思提示文本
        """
        prompt_parts = [
            "## 技能调用失败分析\n",
            f"原始问题: {user_query}\n"
        ]

        # 分析每个失败
        for i, (call, response) in enumerate(failed_calls, 1):
            skill_name = call.get("function", {}).get("name", "unknown")
            error_code = response.get("error_code", "UNKNOWN")
            error_msg = response.get("message", "")

            analysis = cls.analyze_failure(
                skill_name,
                error_code,
                error_msg,
                {},
                response.get("execution_time_ms")
            )

            prompt_parts.append(f"\n### 失败 {i}: {skill_name}")
            prompt_parts.append(f"- 错误类型: {analysis['category']}")
            prompt_parts.append(f"- 影响: {analysis['impact']}")
            prompt_parts.append(f"- 可能原因: {', '.join(analysis['root_cause'])}")

            # 建议策略
            strategy = cls.suggest_recovery_strategy(session, skill_name, analysis)
            prompt_parts.append(f"- 建议策略: {strategy['action']}")
            prompt_parts.append(f"- 理由: {strategy['reasoning']}")

            if strategy.get("alternative_skills"):
                prompt_parts.append(f"- 替代方案: {', '.join(strategy['alternative_skills'])}")

        # 检查已有数据
        tool_store = session.get("tool_store", {})
        available_data = []
        for case_id, case_data in tool_store.items():
            for skill_name, result in case_data.items():
                if skill_name != "segmentation" and isinstance(result, dict) and not result.get("_error"):
                    available_data.append(f"{case_id}/{skill_name}")

        if available_data:
            prompt_parts.append("\n## 已有可用数据")
            prompt_parts.append("以下技能已成功完成，可以直接使用其结果：")
            for data in available_data:
                prompt_parts.append(f"- {data}")

        # 指导方向
        prompt_parts.append("\n## 建议行动")
        prompt_parts.append("请根据以上分析：")
        prompt_parts.append("1. 如果已有数据足够回答问题，直接基于现有结果作答")
        prompt_parts.append("2. 如果建议重试，可以再次尝试调用相同技能")
        prompt_parts.append("3. 如果建议使用替代方案，调用推荐的替代技能")
        prompt_parts.append("4. 如果无法完整回答，明确说明缺少哪些数据和原因")

        return "\n".join(prompt_parts)

    @classmethod
    def should_enable_reflection(
        cls,
        session: Dict[str, Any],
        current_round: int
    ) -> bool:
        """
        判断是否应该启用反思机制

        Args:
            session: 当前会话
            current_round: 当前轮数

        Returns:
            是否启用反思
        """
        # 第一轮不需要反思
        if current_round <= 1:
            return False

        # 检查最近一轮是否有失败
        recent_failures = [
            call for call in session.get("skill_call_history", [])
            if call.get("round") == current_round - 1 and call.get("status") == "error"
        ]

        # 有失败则启用反思
        return len(recent_failures) > 0

    @classmethod
    def create_reflection_summary(
        cls,
        analyses: List[Dict[str, Any]],
        strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        创建反思摘要，用于日志和监控

        Args:
            analyses: 失败分析列表
            strategies: 恢复策略列表

        Returns:
            摘要信息
        """
        summary = {
            "total_failures": len(analyses),
            "by_category": {},
            "by_severity": {},
            "retryable_count": 0,
            "strategies_distribution": {},
            "critical_issues": []
        }

        for analysis in analyses:
            # 按类型统计
            category = analysis.get("category", "unknown")
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1

            # 按严重程度统计
            severity = analysis.get("severity", "unknown")
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1

            # 可重试数量
            if analysis.get("retryable"):
                summary["retryable_count"] += 1

            # 收集严重问题
            if severity == "high":
                summary["critical_issues"].append({
                    "skill": analysis["skill_name"],
                    "error": analysis["error_code"],
                    "impact": analysis.get("impact")
                })

        # 策略分布
        for strategy in strategies:
            action = strategy.get("action", "unknown")
            summary["strategies_distribution"][action] = summary["strategies_distribution"].get(action, 0) + 1

        return summary
