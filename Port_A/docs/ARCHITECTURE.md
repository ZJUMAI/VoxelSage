# Port A 架构

## Agent Loop

```
用户问题 → build_messages → LLM 推理 → 有 tool_calls?
                                   │
                            ┌──────┴──────┐
                            ↓              ↓
                          执行 skills    流式输出答案
                            ↓
                     写入 tool_store
                            ↓
                        下一轮循环
```

每轮循环前自动执行工具优化（过滤冗余技能），失败时自动触发反思引擎分析原因并生成恢复策略。

---

## 核心模块

### 1. MedicalKnowledgeBase — 医学知识库

对 LLM 输出的测量结果进行自动校验，防止异常数据误导用户。

- **参考范围校验** — 预设肝脏体积（500-2500 cm³）、肿瘤直径（0.5-200 mm）、血管体积等正常范围，超出范围自动标记 ⚠️ 警告
- **临床规则检查** — 大病灶（>5cm）提醒 MDT 讨论、多发病灶（>3 个）建议分期评估
- **集成点** — `format_tool_context()` 中对每个测量值调用 `validate_measurement()`

**示例输出**：
```
⚠️ 肝脏体积: 3000.00 cm³
  [CRITICAL] 值超出合理范围 [500, 2500]，建议人工复核

🟠 发现大病灶（>5cm），需重点关注
   建议：MDT讨论、评估手术可行性、排查血管侵犯
```

### 2. ToolOptimizer — 工具选择优化器

避免 LLM 重复调用已完成或被父技能覆盖的工具，减少约 60% 冗余调用。

- **技能层级过滤** — `liver_analysis` 包含 `vessel_volume`、`tumor_diameter`、`tumor_vessel_distance`，父技能完成后子技能自动禁用
- **缓存回答判断** — `can_answer_from_cache()` 在调用工具前检查 `tool_store` 是否已有足够数据直接回答
- **已执行过滤** — 同一技能在同一 case 上不重复执行（可视化工具除外）
- **集成点** — `apply_tool_optimization()` 每轮 agent 开始前调用

### 3. ReflectionEngine — 反思引擎

技能调用失败时自动分析原因、生成恢复策略、注入反思 prompt。

| 错误码 | 类型 | 可重试 | 策略 |
|--------|------|--------|------|
| `PORT_B_HTTP_500` | 临时错误 | ✅ | 等待后重试，降低并发 |
| `PORT_B_HTTP_504` | 超时 | ✅ | 增加超时，改用子技能替代 |
| `PORT_A_TOOL_VALIDATION_ERROR` | 参数校验 | ❌ | 检查参数格式 |
| `INVALID_PORT_B_SKILL_RESPONSE` | 协议错误 | ❌ | 检查 Port B 版本 |
| `SKILL_CALL_LIMIT_REACHED` | 资源耗尽 | ❌ | 已达上限，基于已有数据回答 |

- **集成点** — agent 循环中每轮前后检查，失败时自动触发分析
- **效果** — 失败恢复率约 66.7%

---

## 消息结构优化

旧版将工具结果作为 `role: "tool"` 消息插入 messages 列表，导致上下文迅速膨胀。

新版改为结构化 tool context：

```
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},          # 稳定内容 → 命中 Prompt Cache
    {"role": "system", "content": format_tool_context()},   # 动态工具结果（每轮更新）
    {"role": "user",   "content": ...},                      # 对话历史
    {"role": "assistant", "content": ...},
    {"role": "user",   "content": build_current_user_content()}, # 当前问题 + 图片
]
```

工具结果写入 `tool_store`，通过 `format_tool_context()` 在下一轮 system prompt 中结构化呈现，节省 token。

---

## 关键业务规则

- 同一 case 的 `liver_analysis` 只调用一次
- `liver_analysis` 完成后自动禁用 `vessel_volume`、`tumor_diameter`、`tumor_vessel_distance`
- 最终输出要求 Markdown 格式（`##` 标题、`**粗体**`、表格、链接）
- WebSocket 全链路事件推送（分割 → 技能 → 推理 → 回答）
- 达到 `MAX_AGENT_ROUNDS` 后强制生成 final 回答，避免无限循环
- 分割结果自动复用：检查 `masks/` 目录已有文件则跳过重复计算

---

## 前端协议

新增 WebSocket 事件类型，全链路可观测：

| 事件 | 方向 | 说明 |
|------|------|------|
| `progress` | 服务器 → 前端 | 各阶段状态更新（分割、复用、刷新等） |
| `skill_call_start/result/error` | 服务器 → 前端 | 技能调用生命周期通知 |
| `reflection_result` | 服务器 → 前端 | 反思分析结果，含错误分类和补救策略 |
| `answer_start/delta/end` | 服务器 → 前端 | 流式回答（每次 80 字符分块推送） |

完整交互流程：

```
Frontend                  Port A (server.py)              Port B (external)
   │                           │                              │
   ├─ POST /api/upload ──────► │                              │
   │◄──── FileUploadResponse ──┤                              │
   │                           │                              │
   ├─ WS initial ────────────► │                              │
   │                           ├──► POST /api/process-lite ──►│
   │                           │◄── segmentation result ─────┤
   │                           │                              │
   │◄─── progress/seg_done ────┤                              │
   │                           ├──► GET  /api/skills/list ───►│
   │                           │◄── tools ───────────────────┤
   │                           │                              │
   │◄─── agent_started ────────┤                              │
   │                           ├── (agent loop)               │
   │                           │   model(tools) → calls       │
   │                           ├──► POST /api/skills/run ────►│
   │                           │◄── result ─────────────────┤
   │◄─── skill_call_start ─────┤                              │
   │◄─── skill_call_result ────┤                              │
   │                           ├── (继续或结束)                │
   │◄─── answer_delta xN ──────┤                              │
   │◄─── final ────────────────┤                              │
```

---

## 性能指标

| 指标 | 效果 |
|------|------|
| 冗余工具调用 | 减少约 **60%**（技能层级过滤） |
| 失败恢复率 | 约 **66.7%**（反思引擎智能重试） |
| Token 消耗 | 降低（结构化 tool context 替代 role=tool 消息） |
| 缓存命中（重复问题） | 省略约 250s 的 liver_analysis 计算 |
| 分割结果复用 | 已有 masks 时跳过重复分割 |

---

## 会话管理

- **磁盘持久化** — 每个 session 写入 `sessions/{session_id}/session.json`
- **启动恢复** — `load_all_sessions()` 在服务启动时恢复所有历史 session
- **多 Case 支持** — 同一 session 通过 `target_case_id` 管理多个影像文件的独立 `tool_store`
