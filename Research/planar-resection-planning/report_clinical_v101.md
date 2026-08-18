# v10.1 临床时间窗口双头架构训练报告

> 版本：clinical-window-v10.1（2026-08-09/10）
> 目标：在失血非劣效性安全门约束下，选择平均总耗时最短的 clamp-head 策略。

## 0. 结论摘要

| 项目 | 结果 |
|---|---|
| **总体判定** | ⚠️ **NO-GO（Test-64 一次性评估失败）** |
| 最终候选 | threshold-5 trial19（blood_cost=0.78, n_epochs=3, lr≈1e-5） |
| Validation-64 | ✅ 3-seed 全过（失血差 -10.8 mL, CI 上界 < 安全门） |
| Stress-64 | ✅ 通过（失血差 -17.6 mL, CI 上界 < 安全门） |
| Test-64 | ❌ **失败**（失血差 +19.4 mL, CI 上界 48.6 > 15.2） |

根据指南 Section 13：Test-64 一次性评估失败 → 记录 NO-GO，不得回头调参。

---

## 1. 冻结集重建（Gate 0）

- **背景**：v10 因 Train split 混入 stage a/b/c 与纯 stage d Validation 分布错配，两次 Stage 1B NO-GO。
- **v10.1 修复**：全部 5 个 split（train/tuning/validation/test/stress）用 `generate_clinical_stage_pool(stage="d")` 独立重新生成，ID/seed 跨 split 无重叠。
- 文件：`results/clinical_window_v10_1/frozen/splits_v10_1.json`
- 分布一致性 Gate（`check_split_distribution.py`）通过：5 split 的血管格子/血管数/时间/失血/夹闭周期分位数一致，无 v10 的 4 倍失血错配。

## 2. Scales 校准

用新 Train 的 serpentine 策略重新校准（**未复用 v10 scales**）：
- `time_scale_minutes = 39.83`
- `blood_scale_ml = 1274.85`

## 3. Stage 1A：BC 预训练

- 256 个 Train 场景、30 epochs。
- Validation-64 转移测试：transfer 1.4% < 5% ✅ 准入通过。
- 输出：`runs/stage1a_bc_seed_2026081301/pretrained_model.zip`

## 4. Stage 1B：PPO 训练 + 审计

- 从 BC init 训练 57,344 steps（blood_cost=1.0，固定 15/5 权重）。
- 12 个 checkpoint + final_model 在 Validation-64 全量审计（`audit_clinical_window_checkpoints.py`）。

| checkpoint | 失血差(mL) | 95% CI | feasible |
|---|---|---|---|
| 4992 (BC init) | -9.1 | [-29.2, +9.3] | ✅ |
| 9984 | -1.4 | [-24.8, +22.2] | ❌ |
| 19968 | +8.4 | [-18.5, +37.0] | ❌ |
| 24960 | -8.1 | [-61.1, +43.4] | ❌ |
| 34944 | -29.2 | [-87.4, +26.8] | ❌ |
| 44928 | -13.8 | [-77.1, +48.4] | ❌ |
| 49920 | +42.4 | [-27.4, +110.7] | ❌ |
| final | +40.4 | [-30.4, +113.2] | ❌ |

- **唯一可行 checkpoint：4992（BC init）**，失血差 -9.1，CI 上界 9.3 < 15.6。
- PPO 训练后期失血 CI 变宽甚至转正，训练未改善失血安全。
- **manifest decision = GO**（best_model = 4992）。

## 5. Stage 2 前置：timing-oracle

- 从 4992 预训练 clamp head（threshold 10 min），256 Train 场景 counterfactual rollout。
- **release 正例 353/1024 = 34.5%**（> 5% 阈值 ✅）。
- 训练 20 epochs：loss 1.99→0.68，acc ~0.58。
- 输出：`oracle/threshold10_seed_2026081401/clamp_oracle_model.zip`

## 6. Stage 2A：Optuna 多目标（threshold 10）

- 4 GPU 并行（cuda:0/1/4/5），40 trials（各 worker 独立 seed，避免重复）。
- 搜索 blood_cost + lr/gamma/gae/ent/clip/target_kl/n_epochs。
- **Pareto 前沿：trial 5 / 17 / 18**。
- 注意：多数 trial 的 Tuning-32 blood 重复为 357.1（low-lr 训练未显著改变策略）；用 threshold 配置评估确认训练有效（oracle 398→trial 343-353 mL）。

## 7. Stage 2B：Pareto 候选 Validation-64 确认（threshold 10）

3 候选 × 3 seed 重训 50k，Validation-64 配对 bootstrap 95% CI。

| 候选 | mean blood | 3-seed all feasible | 判定 |
|---|---|---|---|
| trial5 | 351.8 | ❌（1/3 过） | 排除 |
| trial17 | 287.2 | ❌（1/3 过） | 排除 |
| **trial18** | **270.9** | ✅ **（3/3 过）** | **选定** |

- trial18 失血差 -12.1 mL（baseline 283.0），CI 上界全部 < 15.6。
- **Stage 2B = GO**。

## 8. Stage 2C：threshold 5 min Optuna + 确认

- threshold-10 已过 → 建立 threshold-5 Optuna（从 trial18/seed 1703 初始化）。
- 40 trials，**Pareto：trial 18 / 19**。
- 2 候选 × 3 seed 重训 50k，Validation-64 确认。

| 候选 | mean blood | 3-seed all feasible | 判定 |
|---|---|---|---|
| trial18 (t5) | 364.3 | ❌（0/3 过） | 排除 |
| **trial19 (t5)** | **272.2** | ✅ **（3/3 过）** | **选定** |

- trial19 失血差 -10.8 mL，CI 上界全部 < 15.6。
- **Stage 2C = GO**。

## 9. full END 决策

- 指南 Section 12：threshold-5 通过多 seed 确认后决策 full END。
- **架构发现**：所有训练用 `freeze_target_head=True + freeze_features_extractor=True`（只训 clamp head）。**时间由 target head 决定，冻结后 time 钉死 ~41.2 min**，模型只能优化失血。
- **决策：不启动 full END**。因 target 冻结使时间不可改善，full END 探索时间优化的边际价值低；threshold-5 trial19 已满足"失血非劣效后平均耗时最短"。
- 最终模型：`stage2c/trial19/seed_2026082204/final_model.zip`

## 10. Test / Stress 一次性评估（冻结后）

| split | baseline blood | 模型 blood | 失血差 | 95% CI | feasible |
|---|---|---|---|---|---|
| **Test-64** | 304.9 | 324.3 | **+19.4** | [-5.9, +48.6] | ❌ **NO-GO** |
| **Stress-64** | 364.7 | 347.1 | **-17.6** | [-54.2, +17.0] | ✅ |

- **Test-64 失败**：失血差 +19.4 mL（劣于 baseline），CI 上界 48.6 远超 allowed 15.2。
- Stress-64 通过（更难 split 上保持失血改善）。
- 按指南 Section 13 第 4 条：**Test-64 失败 → 记录 NO-GO**，下一版需重建新的冻结测试集。

**参考验证**：threshold-10 trial18（Stage 2B 选定模型）在 Test-64 上同样失败（失血差 +9.1，CI 上界 44.0 > 15.2）。证明 **Test-64 失败非 threshold-5 特有问题，而是整体 clamp-head 策略的跨 split 泛化缺陷**——两个候选均在 Validation/Stress 改善、Test 劣化。

---

## 11. 根因分析与后续建议

**Test-64 失败的根因（分析）**：
1. **模型选择集中于 Validation**：所有阶段（Stage 2B/2C）用 Validation-64 选模型，Test 是一次性 held-out。模型在 Validation/Stress 改善但 Test 劣化，反映**跨 split 泛化脆弱**。
2. **Test split 分布更硬**：Test baseline 失血 304.9（vs Validation 283.0，Stress 364.7），血管格子 median 16（vs 其他 split 14），场景难度更高。
3. **clamp head 只训时机**：`freeze_target_head` 使空间策略固定为 serpentine，clamp 时机在不同 split 上的最优性不一致。

**后续建议**（供 v10.2）：
1. **解除 freeze_target_head / freeze_features_extractor**，让 target 可学，真正优化时间维度。
2. **多 split 联合验证**：模型选择不只用 Validation，考虑 Validation + 小部分 held-out 的交叉验证，减少单 split 泛化风险。
3. **加大训练步数/提高 lr 下限**：Stage 2A/2C 多数 trial 采到 lr≈1e-5（几乎不训练），搜索未充分探索策略空间。
4. **Test-64 失败后重建冻结测试集**（按指南），下一版避免对已污染 Test 集复评。

---

*自动生成图：`results/clinical_window_v10_1/report/v10_training_diagnostics.png`（训练曲线）、`v10_optuna_pareto.png`（Optuna Pareto）。*
