# Port A — Medical Imaging Agent

基于 LLM 的医学影像分析 Agent 服务，通过 Agent Loop 自动调用影像分析技能，实现 CT 影像的智能问答。

## 项目结构

```
qwen_vl_demo/
├── core/
│   ├── server.py                  # FastAPI 主服务器 (WebSocket / Agent Loop)
│   ├── medical_knowledge_base.py  # 医学知识库：测量值校验、临床规则检查
│   ├── tool_optimizer.py          # 工具优化器：过滤冗余技能调用
│   └── reflection_module.py       # 反思引擎：失败诊断与自动恢复
├── tests/
│   └── test_p0_optimizations.py   # 单元测试
├── docs/
│   └── ARCHITECTURE.md            # 架构文档
├── restart.sh                     # 服务重启脚本
└── README.md
```

## 快速开始

### 环境变量

```bash
export DASHSCOPE_API_KEY="your-api-key"
export DASHSCOPE_BASE_URL="https://your-llm-endpoint/v1"
export PORT_B_INTERNAL="http://localhost:8765"
export CACHE_ROOT="./case_cache"
```

### 启动服务

```bash
python -m core.server
```

或：

```bash
bash restart.sh
```

### 运行测试

```bash
python tests/test_p0_optimizations.py
```

## 架构

- **Port A**（本服务）：WebSocket 服务器，管理用户会话、Agent 循环、工具调用编排
- **Port B**（外部）：医学影像分割和分析引擎，提供 skills API
- **Agent Loop**：LLM 与 Port B 技能之间的迭代调用循环，直到生成最终回答

### 核心模块

| 模块 | 职责 |
|------|------|
| **MedicalKnowledgeBase** | 测量值参考范围校验、临床规则检查、异常标记 |
| **ToolOptimizer** | 技能层级过滤（父技能完成后自动禁用子技能）、缓存回答判断 |
| **ReflectionEngine** | 错误分类、恢复策略生成（重试/降级/跳过）、反思 prompt 注入 |

详细架构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
