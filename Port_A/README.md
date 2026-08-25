# Port A — Medical Imaging Agent

基于 LLM 的医学影像分析 Agent 服务，通过 Agent Loop 自动调用影像分析技能，实现 CT 影像的智能问答。

## 项目结构

```
Port_A/
├── core/
│   ├── server.py                  # FastAPI 主服务器 (WebSocket / Agent Loop)
│   ├── medical_knowledge_base.py  # 医学知识库：测量值校验、临床规则检查
│   ├── tool_optimizer.py          # 工具优化器：过滤冗余技能调用
│   └── reflection_module.py       # 反思引擎：失败诊断与自动恢复
├── tests/
│   └── test_p0_optimizations.py   # 单元测试
├── docs/
│   └── ARCHITECTURE.md            # 架构文档
├── requirements.txt               # Python 运行依赖
└── README.md
```

## 快速开始

在仓库根目录创建 Python 3.10、3.11 或 3.12 虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Port_A/requirements.txt
```

### 环境变量

```bash
export DASHSCOPE_API_KEY="your-api-key"
export DASHSCOPE_BASE_URL="https://your-llm-endpoint/v1"
export LLM_MODEL_NAME="your-endpoint-model-name"
export PORT_B_INTERNAL="http://localhost:8765"
export CACHE_ROOT="./case_cache"
```

`LLM_MODEL_NAME` 必须与兼容 OpenAI API 的服务实际暴露的模型 ID 完全一致。
旧变量 `QWEN_MODEL_NAME` 仍可使用，但建议迁移到通用名称 `LLM_MODEL_NAME`。

### 启动服务

完整应用建议在仓库根目录通过 `./scripts/start.sh` 一键启动。仅调试 Port A 时：

```bash
cd Port_A
../.venv/bin/python -m core.server
```

服务默认监听 `0.0.0.0:8900`。启动前请先运行 Port B，并确保
`PORT_B_INTERNAL` 指向其可访问地址。

### 运行测试

```bash
cd Port_A
python -m pytest -q
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
