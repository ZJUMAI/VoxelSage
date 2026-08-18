# v10.6 出处说明（v10.7 发表前确认实验）

> 生成日期：2026-08-16
> 目的：v10.7 发表前确认实验引用 v10.6 冻结结果前，必须记录的 v10.6 数据出处
> 与已知边界。本文件只说明 v10.6，不改变 v10.6 任何结果。

## 1. v10.6 Test/Stress 缓存命中异常场景

在最终 `evaluation/test.json` 中，有 **4 个场景**表现为
`shield_record_cache_hits == macro_action_count` 且 `shield_record_cache_misses == 0`；
在 `evaluation/stress.json` 中有 **19 个场景**同样如此。

这是各场景在执行过程中对精确盾 record cache 的命中模式，属正常运行时状态；
它不影响动作、T、B 或 Gate。详见第 2 节缓存等价性审计。

## 2. 缓存等价性审计

`results/clinical_window_v10_6_shielded_learning/audit/shield_record_cache_equivalence.json`
判定为 **PASS**：uncached / fill / hit 三种缓存状态逐场景比较，
completion、failure_reason、elapsed_minutes、realized_episode_B_ml、budget_ml、
selected_max_B_total_ml、all_candidates_max_B_total_ml、shield_intervention_count、
safety_invariant_violations、macro_action_count、action_sequence_hash、delta_T_min、
delta_B_ml 全部相等。因此缓存不改变动作、T、B 或任何 Gate。

## 3. v10.6 Test 延迟仍不满足部署门

v10.6 Test 的 wall p50/p95 为 **205.768 / 458.773 s**（正式 `test.json` 数值），
远大于 60 s 部署阈值。v10.6 最终结论为 `research GO; not deployable-latency`。
本问题不改变 v10.6 研究 GO，也不属于 v10.7 要解决的延迟问题（延迟工程应另开版本）。

## 4. test_gate_l5_result.md 与 test.json 的延迟差异

`report/test_gate_l5_result.md`（早期摘要）中记载 Test wall p50/p95 为
**199.6 / 452.5 s**，与最终 `evaluation/test.json` 的 **205.768 / 458.773 s** 略有差异。
差异来源：`test_gate_l5_result.md` 写于批量评估完成时对单场景 `wall_seconds`
分位数的一次即时打印，而 `test.json` 是持久化的最终 summary（含
`batch_wall_seconds` 等完整字段）。正式数值以 **`test.json` 的
p50/p95 = 205.768 / 458.773 s** 为准。

## 5. 不重测 v10.6

v10.6 Test/Stress 均为一次性评估，已按冻结顺序完成且 GO。本 v10.7 实验
**不得重测 v10.6 Test/Stress**，不得用 v10.7 结果反向修改 v10.6 的
checkpoint、scales、margin 或任何产物。
