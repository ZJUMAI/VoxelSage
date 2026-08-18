# 临床时间窗口 v10 双头架构 + Optuna 代理训练指南

> **历史指南，已停用。** v10/v10.1 已完成并以 Test NO-GO 结束。新的代理不得继续执行本文；
> 请使用 `临床时间窗口v10.2目标条件松夹代理训练指南.md`。

> 版本：v10-hierarchical-optuna-handoff-v1（2026-08-08）
> 执行目录：`Research/planar-resection-planning`
> 结果目录：`results/clinical_window_v10_hierarchical/`
> 当前状态：代码和单元测试已就绪，**尚未运行 v10 正式训练**
> 边界：该二维环境和失血模型仅供研究，不是临床决策系统

## 1. v9 证据与 v10 目标

v9 已证明“高层选目标 + 底层最短路转移”能稳定完成任务，但存在：

- Stage 1B 失血改善仅 `-4.66 mL`，D-16 配对 bootstrap 95% CI 为
  `[-83.8, +70.4] mL`，不能证明优于 baseline；
- Stage 2C 被选 checkpoint 失血比 baseline 多 `322.7 mL`，95% CI
  `[+156.8, +505.2] mL`，明确恶化；
- 旧规则按耗时先于失血选 checkpoint，选中了 Stage 2C 失血最高的早期模型；
- END 是单独的零时间 RL step，与 1200 个空间目标混在同一动作头中；
- Stage 2 期间 transfer overhead 持续上升，表明学 END 时空间策略也在漂移。

v10 只对必要部分做局部重构：

```text
共享空间编码器
├── clamp head：continue / release-now
└── target head：1200 个 padded 前沿格
                 ↓
环境在一个宏 step 中执行：可选松夹 → 自动转移 → 切割/封闭
```

不得恢复五方向导航。

## 2. v10 已实现的接口

| 文件 | 用途 |
| --- | --- |
| `clinical_hierarchical_environment.py` | `MultiDiscrete([2,1200])` 联合动作，松夹与目标执行同步计费 |
| `clinical_hierarchical_policy.py` | clamp/target 双头策略；clamp head 同时使用 mean/max pooling |
| `train_clamp_timing_oracle.py` | 比较“现在松夹”和“继续夹闭”反事实 rollout，预训练 clamp head |
| `optimize_clinical_v10_optuna.py` | Optuna 多目标搜索时间、失血和 transfer overhead |
| `prepare_clinical_v10_splits.py` | 冻结 Train/Tuning/Validation/Test/Stress，防止 D-16 泄漏 |
| `audit_clinical_window_checkpoints.py` | 完成率硬门、失血安全门和配对 bootstrap CI |
| `report_clinical_v10.py` | 生成 PNG 训练曲线、Optuna Pareto 图和 Markdown 报告 |

v10 安全 mask 只编码确定规则：**存在暴露未封闭血管时禁止 release**。
它不会把最佳松夹时机硬编码给 RL。

## 3. 不可变更项

- 最长夹闭 15 min，开放固定 5 min；
- 单格 4 mm，大于 1 格的血管分量处理时间为 3 倍；
- 使用 Train-only `scales_v5.json`，不在 Tuning/Validation/Test 重算；
- 底层自动转移逐格计入时间、mL 和 15/5 时相边界；
- baseline 为机械 S 形 + 机械 15/5 + 相同底层转移器；
- `bleeding_probability=1.0` 保留为当前保守研究设置，报告必须说明尚未临床校准；
- v10 首轮 `mechanics_update_interval=0`，张力权重全为 0；END 稳定后再做独立张力实验。

## 4. 环境与依赖预检

```bash
python -c "import torch, optuna; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), optuna.__version__)"
python -m py_compile \
  clinical_hierarchical_environment.py clinical_hierarchical_policy.py \
  train_clamp_timing_oracle.py optimize_clinical_v10_optuna.py \
  prepare_clinical_v10_splits.py report_clinical_v10.py \
  train_clinical_window_ppo.py clinical_window_evaluation.py \
  audit_clinical_window_checkpoints.py
MPLCONFIGDIR=/tmp/mpl-clinical-v10 \
  python -m unittest discover -s tests -p 'test_clinical*.py' -q
```

任一失败都禁止启动训练。代理应记录 CUDA 和 Optuna 版本。

## 5. Gate 0：冻结独立数据

已反复使用的 v9 D-16 不得再作为最终 Validation/Test。

```bash
python prepare_clinical_v10_splits.py \
  --train-source results/clinical_window_v5/frozen/splits_curriculum_d_v5.json \
  --output results/clinical_window_v10_hierarchical/frozen/splits_v10.json \
  --tuning-count 32 --validation-count 64 --test-count 64 --stress-count 64
cp results/clinical_window_v5/frozen/scales_v5.json \
  results/clinical_window_v10_hierarchical/frozen/scales_v10.json
sha256sum results/clinical_window_v10_hierarchical/frozen/* \
  > results/clinical_window_v10_hierarchical/frozen/SHA256SUMS
```

用途必须锁定：

| split | 用途 |
| --- | --- |
| Train | BC、timing oracle、PPO |
| Tuning-32 | Optuna 唯一调参集 |
| Validation-64 | Pareto 候选的多 seed 确认与模型选择 |
| Test-64 | 权重完全冻结后只评估一次 |
| Stress-64 | 最后稳健性评估，不反向调参 |

## 6. Gate A：v10 双头环境 baseline

```bash
python clinical_window_evaluation.py \
  --control-mode hierarchical --algorithm serpentine \
  --splits results/clinical_window_v10_hierarchical/frozen/splits_v10.json \
  --split validation --limit 64 \
  --scales results/clinical_window_v10_hierarchical/frozen/scales_v10.json \
  --early-end-mode disabled \
  --output results/clinical_window_v10_hierarchical/evaluation/baseline_hierarchical_validation64.json
```

必须 64/64 完成、覆盖率 1、合法率 1、END 0、无停滞。

## 7. Stage 1A：重新训练 v10 target head

v9 是 Discrete(1201)，v10 是 MultiDiscrete([2,1200])，不能直接加载 v9 checkpoint。

```bash
python train_clinical_window_ppo.py \
  --control-mode hierarchical \
  --splits results/clinical_window_v10_hierarchical/frozen/splits_v10.json \
  --scales results/clinical_window_v10_hierarchical/frozen/scales_v10.json \
  --output-dir results/clinical_window_v10_hierarchical/runs/stage1a_bc_seed_2026081301 \
  --early-end-mode disabled --end-action-initial-bias -4 \
  --bc-scenarios 256 --bc-epochs 30 --bc-batch-size 512 \
  --bc-learning-rate 1e-3 --bc-margin 2 --bc-v-weight 0 \
  --timesteps 16 --n-envs 1 --n-steps 16 --batch-size 16 --n-epochs 1 \
  --learning-rate 3e-4 --gamma 0.9999 --gae-lambda 0.98 \
  --ent-coef 0.001 --clip-range 0.2 --target-kl 0.03 \
  --seed 2026081301 --device cuda:0 \
  --time-cost 1 --blood-cost 1 --progress-bonus 5 --seal-progress-bonus 2 \
  --front-tension-cost 0 --organ-energy-cost 0 --vessel-strain-cost 0
```

使用 `pretrained_model.zip` 做 Validation-64 评估。准入要求：64/64 完成、停滞 0、
END 0、transfer overhead 不高于 baseline 5%。

```bash
python clinical_window_evaluation.py \
  --control-mode hierarchical --algorithm ppo \
  --model results/clinical_window_v10_hierarchical/runs/stage1a_bc_seed_2026081301/pretrained_model.zip \
  --splits results/clinical_window_v10_hierarchical/frozen/splits_v10.json \
  --split validation --limit 64 \
  --scales results/clinical_window_v10_hierarchical/frozen/scales_v10.json \
  --early-end-mode disabled \
  --output results/clinical_window_v10_hierarchical/evaluation/stage1a_bc_validation64.json
```

## 8. Stage 1B：固定 15/5 目标格 PPO

首轮使用 v10 BC checkpoint，END 继续屏蔽。Stage 1B 可保留少量完成信号，但宏动作已
保证进度，因此临床优化阶段设为：

```text
progress_bonus = 0
seal_progress_bonus = 0
completion_bonus = 5
failure_penalty = 10
time_cost = 1
blood_cost = 1
```

训练 50k，审计所有 checkpoint。审计时必须加：

```bash
python train_clinical_window_ppo.py \
  --control-mode hierarchical \
  --splits results/clinical_window_v10_hierarchical/frozen/splits_v10.json \
  --scales results/clinical_window_v10_hierarchical/frozen/scales_v10.json \
  --init-model results/clinical_window_v10_hierarchical/runs/stage1a_bc_seed_2026081301/pretrained_model.zip \
  --output-dir results/clinical_window_v10_hierarchical/runs/stage1b_fixed_seed_2026081302 \
  --early-end-mode disabled --timesteps 50000 \
  --n-envs 16 --n-steps 512 --batch-size 512 --n-epochs 5 \
  --learning-rate 3e-4 --gamma 0.9999 --gae-lambda 0.98 \
  --ent-coef 0.001 --clip-range 0.2 --target-kl 0.03 \
  --bc-scenarios 0 --bc-epochs 0 --rl-margin-coef 0 \
  --seed 2026081302 --device cuda:0 \
  --time-cost 1 --blood-cost 1 --progress-bonus 0 --seal-progress-bonus 0 \
  --completion-bonus 5 --failure-penalty 10 \
  --front-tension-cost 0 --organ-energy-cost 0 --vessel-strain-cost 0
```

然后审计所有 checkpoint：

```bash
python audit_clinical_window_checkpoints.py \
  --run-dir <stage1b-run> \
  --splits results/clinical_window_v10_hierarchical/frozen/splits_v10.json \
  --scales results/clinical_window_v10_hierarchical/frozen/scales_v10.json \
  --limit 64 \
  --baseline-evaluation results/clinical_window_v10_hierarchical/evaluation/baseline_hierarchical_validation64.json \
  --blood-safety-ratio 1.05 --bootstrap-samples 10000 \
  --output-dir results/clinical_window_v10_hierarchical/evaluation/stage1b_checkpoint_audit_validation64
```

`manifest.json` 的 `best_model` 若为 `null`，必须 NO-GO，不得人工选一个继续。

## 9. Stage 2 前置：timing-oracle 预训练 clamp head

从 Stage 1B 安全 checkpoint 开始，对同一状态比较：

```text
A: release-now + 同一 target + 后续机械 S target
B: continue    + 同一 target + 后续机械 S target
```

标签只由完整后续耗时和失血决定，不给 END 固定奖励。

```bash
python train_clamp_timing_oracle.py \
  --model <stage1b-safe-model.zip> \
  --splits results/clinical_window_v10_hierarchical/frozen/splits_v10.json \
  --split train --scenario-limit 256 \
  --scales results/clinical_window_v10_hierarchical/frozen/scales_v10.json \
  --output-dir results/clinical_window_v10_hierarchical/oracle/threshold10_seed_2026081401 \
  --early-end-mode threshold --early-end-minutes 10 \
  --max-examples 1024 --sample-every 8 \
  --epochs 20 --batch-size 128 --learning-rate 3e-4 \
  --time-cost 1 --blood-cost 1 --seed 2026081401 --device cuda:0
```

强制审计：

- 数据必须同时包含 continue/release；
- 汇报正负例比、loss、accuracy 和 release advantage 分布；
- 抽查至少 20 对 counterfactual replay；
- 若 release 正例少于 5%，不得盲目 PPO，先检查场景是否存在可优化时窗。

## 10. Stage 2A：Optuna 多目标 Pilot

仅搜索算法/权重，不搜索 15/5、血管定义或安全 mask。

```bash
python optimize_clinical_v10_optuna.py \
  --splits results/clinical_window_v10_hierarchical/frozen/splits_v10.json \
  --scales results/clinical_window_v10_hierarchical/frozen/scales_v10.json \
  --init-model results/clinical_window_v10_hierarchical/oracle/threshold10_seed_2026081401/clamp_oracle_model.zip \
  --baseline-evaluation results/clinical_window_v10_hierarchical/evaluation/baseline_hierarchical_tuning32.json \
  --output-dir results/clinical_window_v10_hierarchical/optuna/stage2a_threshold10 \
  --storage sqlite:///results/clinical_window_v10_hierarchical/optuna/stage2a_threshold10.db \
  --study-name clinical-v10-stage2a-threshold10 \
  --trials 40 --timesteps 25000 --n-envs 8 --n-steps 256 --batch-size 256 \
  --tuning-limit 32 --blood-safety-ratio 1.05 \
  --early-end-minutes 10 --device cuda:0 --seed 2026081501
```

注意：要先在 Tuning-32 上生成机械 S baseline JSON，不得用 Validation baseline 代替。

Optuna 搜索：

- `blood_cost`；
- learning rate、`gamma`、`gae_lambda`、`ent_coef`、`clip_range`、`target_kl`、`n_epochs`。

三目标同时最小化：原始分钟、原始 mL、transfer overhead。约束为：

- 完成率 1；
- 合法率 1；
- 停滞/两格失败 0；
- 平均失血不超 baseline 5%。

Optuna Pareto 前沿不是最终模型。

## 11. Pareto 候选的 Validation-64 确认

从可行 Pareto 前沿选 5–10 组，每组用 3 个新 seed 从同一 oracle checkpoint 重训
50k，不得直接把 Tuning-32 最优 trial 称为最终模型。

每组必须在 Validation-64 计算：

- 3 seed 均值和最差 seed；
- 逐场景时间/失血配对差；
- bootstrap 95% CI；
- END 前的暴露血管面积和预计出血率；
- END 后 5 min 失血；
- 开放总分钟、夹闭周期、transfer overhead；
- target head 相对父 checkpoint 的一致率。

安全门：失血配对差 95% CI 上界不得超 baseline 的 5%。在所有可行模型中，
再选平均耗时最短者。这实现“失血安全约束下最小化总时间”。

## 12. Stage 2B / full END

- 仅当 threshold 10 min 在 Validation-64 的 3 seed 全部通过，才可建立新 Optuna study
  搜索 threshold 5 min；
- threshold 5 min 仍从安全的 threshold-10 checkpoint 初始化；
- full END 不得自动启动。需 threshold-5 在 Validation-64 多 seed 全部通过后再决策；
- 即使 full END，“暴露未封闭血管时禁止 release”仍保留。

## 13. Test / Stress 一次性评估

当架构、权重、checkpoint 和选择规则全部冻结后：

1. 在 Test-64 评估一次；
2. 在 Stress-64 评估一次；
3. 无论结果如何，不得回到同一 Test/Stress 调参；
4. 若失败，记录 NO-GO，下一版重建新的冻结测试集。

## 14. 强制输出训练图表和报告

每次阶段性交付前执行：

```bash
MPLCONFIGDIR=/tmp/mpl-clinical-v10 \
python report_clinical_v10.py \
  --results-dir results/clinical_window_v10_hierarchical \
  --output-dir results/clinical_window_v10_hierarchical/report
```

必须产生：

```text
report/
├── report_clinical_v10.md
├── v10_training_diagnostics.png
└── v10_optuna_pareto.png
```

报告中必须显式嵌入两张 PNG，并在正文补充：

- BC loss 和 timing-oracle loss/accuracy；
- PPO 训练池滚动失血和 transfer overhead；
- Optuna 可行/不可行 trial 与 Pareto 前沿；
- Validation/Test 的原始分钟、mL、END、开放时间和配对 CI；
- 训练池曲线不能代替冻结 Validation 的声明；
- 张力本轮关闭、失血系数未临床校准的限制。

## 15. 立即 NO-GO 条件

任一条触发即停止当前路线：

- v10 baseline 或 BC 不能 100% 完成；
- 存在暴露未封闭血管时 release；
- timing-oracle 只生成单一类标签；
- Optuna 直接使用 Validation/Test 调参；
- 没有可行 Pareto trial 仍人工选模型；
- 失血安全门未通过，却因耗时较短而选中；
- 冻结 target head 阶段 transfer overhead 仍大幅上升；
- 未输出训练图表、冻结评估和配对 CI；
- 未经独立设计就同时加入张力 reward。

## 16. 代理最终交付清单

```text
results/clinical_window_v10_hierarchical/
├── frozen/
│   ├── splits_v10.json
│   ├── scales_v10.json
│   └── SHA256SUMS
├── runs/
├── oracle/
│   └── .../clamp_oracle_report.json
├── optuna/
│   ├── *.db
│   └── .../optuna_summary.json
├── evaluation/
│   ├── baseline_*.json
│   ├── *_checkpoint_audit_*/manifest.json
│   └── final_{validation,test,stress}*.json
└── report/
    ├── report_clinical_v10.md
    ├── v10_training_diagnostics.png
    └── v10_optuna_pareto.png
```

代理必须把每个 Gate 的 GO/NO-GO 和原因写入报告，不得只提供 checkpoint。
