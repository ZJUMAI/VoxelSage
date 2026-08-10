"""
P0 优化集成测试脚本
测试医学知识库、工具优化器和反思机制的功能
"""
import sys
from pathlib import Path
# 确保能找到 core/ 模块（无论从项目根目录还是 tests/ 目录运行）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
from core.medical_knowledge_base import MedicalKnowledgeBase
from core.tool_optimizer import ToolOptimizer
from core.reflection_module import ReflectionEngine


def test_medical_knowledge_base():
    """测试医学知识库功能"""
    print("=" * 60)
    print("测试 1: 医学知识库")
    print("=" * 60)

    # 测试体积验证
    print("\n1.1 测试肝脏体积验证:")
    test_cases = [
        ("正常体积", 1270.3),
        ("偏大体积", 2200.0),
        ("异常大", 3000.0),
        ("偏小体积", 700.0),
        ("异常小", 400.0)
    ]

    for name, volume in test_cases:
        result = MedicalKnowledgeBase.validate_measurement("liver_volume", volume)
        print(f"  {name} ({volume} cm³): {result['severity']} - {result.get('warning', '正常')}")

    # 测试结果一致性
    print("\n1.2 测试结果一致性验证:")
    mock_result = {
        "liver_volume_cm3": 1500.0,
        "tumor_results": {
            "tumor_1": {
                "volume_cm3": 10.5,
                "max_diameter_mm": 35.2
            },
            "tumor_2": {
                "volume_cm3": 5.2,
                "max_diameter_mm": 28.0
            }
        }
    }
    consistency = MedicalKnowledgeBase.validate_result_consistency(mock_result)
    print(f"  一致性: {'通过' if consistency['consistent'] else '不一致'}")
    print(f"  肿瘤数量: {consistency['tumor_count']}")
    if consistency['issues']:
        print(f"  问题: {len(consistency['issues'])} 个")
        for issue in consistency['issues'][:3]:
            print(f"    - {issue['message']}")

    # 测试临床规则
    print("\n1.3 测试临床规则:")
    analysis_data = {
        "liver_volume": 2100.0,
        "tumor_count": 4,
        "max_tumor_diameter": 55.0,
        "min_tumor_vessel_distance": 3.5
    }
    rules = MedicalKnowledgeBase.apply_clinical_rules(analysis_data)
    print(f"  触发规则: {len(rules)} 条")
    for rule in rules:
        print(f"    [{rule['urgency'].upper()}] {rule['implication']}")
        print(f"      建议: {', '.join(rule['recommendations'][:2])}")

    print("\n✓ 医学知识库测试完成\n")


def test_tool_optimizer():
    """测试工具优化器功能"""
    print("=" * 60)
    print("测试 2: 工具优化器")
    print("=" * 60)

    # 模拟 session 和工具
    mock_session = {
        "session_id": "test_session",
        "tool_store": {
            "case_001": {
                "liver_analysis": {
                    "liver_volume_cm3": 1500.0,
                    "tumor_results": {
                        "tumor_1": {"volume_cm3": 10.5, "max_diameter_mm": 35.2}
                    }
                }
            }
        }
    }

    mock_tools = [
        {"function": {"name": "liver_analysis"}},
        {"function": {"name": "vessel_volume"}},
        {"function": {"name": "tumor_diameter"}},
        {"function": {"name": "slice_selection"}},
        {"function": {"name": "three_d_reconstruction"}}
    ]

    # 测试工具过滤
    print("\n2.1 测试冗余工具过滤:")
    filtered, reasons = ToolOptimizer.filter_redundant_tools(mock_session, mock_tools)
    print(f"  原始工具: {len(mock_tools)} 个")
    print(f"  过滤后: {len(filtered)} 个")
    print(f"  过滤原因: {len(reasons)} 条")
    for reason in reasons:
        print(f"    - {reason['filtered_tool']}: {reason['reason']}")

    # 测试缓存检查
    print("\n2.2 测试缓存回答能力:")
    test_queries = [
        "Report the measured volume of the liver",
        "How many lesions are in the liver?",
        "Show me the 3D visualization"
    ]

    for query in test_queries:
        result = ToolOptimizer.can_answer_from_cache(mock_session, query)
        print(f"  问题: {query}")
        print(f"    可直接回答: {'是' if result['can_answer'] else '否'}")
        print(f"    建议: {result['suggestion']}")

    # 测试技能推荐
    print("\n2.3 测试技能推荐:")
    mock_session["tool_store"] = {}  # 清空，测试推荐
    query = "What is the liver volume and how many tumors are there?"
    suggestions = ToolOptimizer.suggest_next_skills(mock_session, query)
    print(f"  问题: {query}")
    print(f"  推荐技能: {len(suggestions)} 个")
    for s in suggestions:
        print(f"    - {s['skill']} (优先级 {s['priority']}): {s['reason']}")

    print("\n✓ 工具优化器测试完成\n")


def test_reflection_engine():
    """测试反思机制功能"""
    print("=" * 60)
    print("测试 3: 反思引擎")
    print("=" * 60)

    # 测试失败分析
    print("\n3.1 测试失败原因分析:")
    test_failures = [
        ("liver_analysis", "PORT_B_HTTP_504", "Gateway Timeout", {}, 5000.0),
        ("slice_selection", "PORT_A_TOOL_VALIDATION_OR_EXECUTION_ERROR", "Invalid parameters", {"invalid_param": "test"}, None),
        ("vessel_volume", "PORT_B_HTTP_500", "Internal Server Error", {}, 1000.0)
    ]

    for skill, code, msg, params, exec_time in test_failures:
        analysis = ReflectionEngine.analyze_failure(skill, code, msg, params, exec_time)
        print(f"  失败: {skill} - {code}")
        print(f"    类型: {analysis['category']}")
        print(f"    严重程度: {analysis['severity']}")
        print(f"    可重试: {'是' if analysis['retryable'] else '否'}")
        print(f"    影响: {analysis['impact']}")
        print(f"    策略: {', '.join(analysis['strategies'][:2])}")

    # 测试恢复策略
    print("\n3.2 测试恢复策略建议:")
    mock_session = {
        "tool_store": {
            "case_001": {
                "segmentation": {"status": "ok"}
            }
        }
    }

    for skill, code, msg, params, exec_time in test_failures[:2]:
        analysis = ReflectionEngine.analyze_failure(skill, code, msg, params, exec_time)
        strategy = ReflectionEngine.suggest_recovery_strategy(mock_session, skill, analysis)
        print(f"  技能: {skill}")
        print(f"    建议操作: {strategy['action']}")
        print(f"    理由: {strategy['reasoning']}")
        if strategy.get('alternative_skills'):
            print(f"    替代技能: {', '.join(strategy['alternative_skills'])}")

    # 测试反思提示生成
    print("\n3.3 测试反思提示生成:")
    failed_calls = [
        (
            {"function": {"name": "liver_analysis", "arguments": "{}"}},
            {"status": "error", "error_code": "PORT_B_HTTP_504", "message": "Timeout"}
        )
    ]
    user_query = "What is the liver volume?"
    prompt = ReflectionEngine.generate_reflection_prompt(mock_session, failed_calls, user_query)
    print(f"  生成的反思提示长度: {len(prompt)} 字符")
    print(f"  预览:\n{prompt[:300]}...")

    # 测试反思摘要
    print("\n3.4 测试反思摘要:")
    analyses = [
        ReflectionEngine.analyze_failure(skill, code, msg, params, exec_time)
        for skill, code, msg, params, exec_time in test_failures
    ]
    strategies = [
        ReflectionEngine.suggest_recovery_strategy(mock_session, a["skill_name"], a)
        for a in analyses
    ]
    summary = ReflectionEngine.create_reflection_summary(analyses, strategies)
    print(f"  总失败数: {summary['total_failures']}")
    print(f"  按类型: {summary['by_category']}")
    print(f"  按严重程度: {summary['by_severity']}")
    print(f"  可重试数: {summary['retryable_count']}")
    print(f"  策略分布: {summary['strategies_distribution']}")

    print("\n✓ 反思引擎测试完成\n")


def test_integration():
    """集成测试：模拟完整流程"""
    print("=" * 60)
    print("测试 4: 集成测试")
    print("=" * 60)

    # 模拟一个完整的会话流程
    print("\n4.1 模拟会话流程:")

    # 初始状态：空 tool_store
    session = {
        "session_id": "integration_test",
        "tool_store": {},
        "current_user_query": "What is the liver volume?"
    }

    tools = [
        {"function": {"name": "liver_analysis"}},
        {"function": {"name": "vessel_volume"}},
        {"function": {"name": "tumor_diameter"}},
        {"function": {"name": "slice_selection"}}
    ]

    print("  步骤 1: 检查是否可以从缓存回答")
    cache_check = ToolOptimizer.can_answer_from_cache(session, session["current_user_query"])
    print(f"    可直接回答: {cache_check['can_answer']}")
    print(f"    建议: {cache_check['suggestion']}")

    print("\n  步骤 2: 推荐技能")
    suggestions = ToolOptimizer.suggest_next_skills(session, session["current_user_query"])
    print(f"    推荐: {[s['skill'] for s in suggestions]}")

    print("\n  步骤 3: 过滤工具（第一轮，无过滤）")
    filtered, reasons = ToolOptimizer.filter_redundant_tools(session, tools)
    print(f"    可用工具: {len(filtered)} 个")

    print("\n  步骤 4: 模拟 liver_analysis 成功")
    session["tool_store"]["case_001"] = {
        "liver_analysis": {
            "liver_volume_cm3": 1500.0,
            "tumor_results": {
                "tumor_1": {"volume_cm3": 10.5, "max_diameter_mm": 35.2}
            },
            "vessel_volumes": {
                "portal_vein": {"volume_cm3": 15.2}
            }
        }
    }

    print("\n  步骤 5: 医学验证")
    validation = MedicalKnowledgeBase.validate_measurement("liver_volume", 1500.0)
    print(f"    体积验证: {validation['severity']} - {validation.get('warning', '正常')}")

    consistency = MedicalKnowledgeBase.validate_result_consistency(
        session["tool_store"]["case_001"]["liver_analysis"]
    )
    print(f"    一致性: {'通过' if consistency['consistent'] else '不一致'}")

    print("\n  步骤 6: 临床规则")
    analysis_data = MedicalKnowledgeBase.extract_analysis_summary(
        session["tool_store"], "case_001"
    )
    rules = MedicalKnowledgeBase.apply_clinical_rules(analysis_data)
    print(f"    触发规则: {len(rules)} 条")

    print("\n  步骤 7: 再次检查缓存")
    cache_check = ToolOptimizer.can_answer_from_cache(session, session["current_user_query"])
    print(f"    可直接回答: {cache_check['can_answer']}")

    print("\n  步骤 8: 过滤工具（第二轮，应该过滤冗余）")
    filtered, reasons = ToolOptimizer.filter_redundant_tools(session, tools)
    print(f"    可用工具: {len(filtered)} 个")
    print(f"    过滤了: {[r['filtered_tool'] for r in reasons]}")

    print("\n✓ 集成测试完成\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("P0 优化模块测试套件")
    print("=" * 60 + "\n")

    try:
        test_medical_knowledge_base()
        test_tool_optimizer()
        test_reflection_engine()
        test_integration()

        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
