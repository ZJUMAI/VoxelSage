# 可变尺寸 RL 训练交接文档

> 更新日期：2026-07-31  
> 工作目录：`Research/planar-resection-planning`  
> 当前状态：阶段 A 已完成三个训练种子；阶段 B 已完成单种子迁移训练、三方冻结评估及
> 阶段 A 冻结集上的遗忘检查；评价脚本已补齐风险指标与基线并行评估；
> 阶段 C 已完成三个独立训练种子（2026073301/02/03，均 75,776 步）并在 A/B/C 三冻结集
> 全部通过工程晋级门槛（完成/合法率 100%，相对 S 形 transfer 99.7%–105.5% < 110%，
> 风险指标零越线），三种子汇总见第 6.5 节、风险面对比见第 6.6 节；
> C 教师缓存 192/192 完成（`teacher_cache_stage_c_seed2026073301_192_isolated.npz`）；
> 阶段 D 已开始：纯尺寸 pilot 完成（第 7.2 节），256 混合场景 D 教师缓存生成中。

## 1. 目标与边界

最终二维研究目标是处理任意较小病例至最大 $30\times40$ 的不规则有效边界。每个网格单元
固定对应一次约 4 mm 的超声刀切除动作，因此最大离散范围约为
$120\,\mathrm{mm}\times160\,\mathrm{mm}$。

当前工作只涉及研究模拟器中的二维 RL。所有张力、器官能量和血管应变均为未经临床标定的
仿真代理量，不能解释为临床安全性、诊断或疗效结论。

旧的 `toy7_vessel_spatial_v3` 仍是模拟器登记的默认 7×7 策略。本文记录的新可变尺寸模型
尚未注册为默认策略，也不应在完成阶段 C/D、跨尺寸回归和最终评价门之前替换默认策略。

## 2. 一句话进度

可变尺寸环境、30×40 填充观测、1,200 动作的全卷积 Maskable PPO、课程场景生成、筛选教师、
行为克隆、阶段迁移和冻结评估已经实现；阶段 A 已完成三个种子，阶段 B 的单种子工程验证通过，
且已完成阶段 B 模型在阶段 A 冻结集上的遗忘检查（通过，相对 S 形 transfer 100.41% < 110%）；
评价脚本已补齐风险指标（CVaR/累计应变/阈值步数比例/最大峰值）与 S 形/规划器的进程级并行。
阶段 C 三个独立种子（2026073301/02/03，均 75,776 步）全部完成，在 A/B/C 三冻结集通过工程
门槛（完成/合法率 100%，相对 S 形 transfer 99.7%–105.5% < 110%，风险指标零越线，
跨种子方差极小，见第 6.5 节）。下一步是补齐独立 Validation/Stress 分离与逐场景配对
bootstrap 后评估阶段 D 预算。

## 3. 阅读顺序

接手 Agent 建议按以下顺序读取：

1. 本文；
2. `../机器学习准备总文档.md` 的第 7.4、7.5、12 节；
3. `environment.py` 中 `variable_grid_*` 和 `VariableGridScenarioPoolEnv`；
4. `variable_scenarios.py`、`variable_policy.py`；
5. `variable_teacher.py`、`cache_variable_teachers.py`；
6. `train_variable_masked_ppo.py`、`evaluate_variable_policy.py`；
7. `results/阶段A冻结评估报告.md` 与 `results/阶段B冻结评估报告.md`。

不要只看旧的 `泛化训练准备说明.md`；它描述的是固定 7×7 Spatial-v3，不是当前 30×40
可变尺寸主线。

## 4. 已实现架构

### 4.1 环境与动作语义

- 底层继续使用 `PlanarResectionEnv` 的 50×50 规范画布、`cut`、自动 `transfer`、
  自动 `release`、力学状态和 JSON replay。
- `variable_grid_observation()` 输出固定形状 `(18, 30, 40)`：
  原 15 个状态通道，加行坐标、列坐标和 `transfer_distance`。
- `variable_grid_action_masks()` 输出长度为 1,200 的布尔 mask。
- 可变动作编号为 `row * 40 + col`；`variable_to_canvas_action()` 将其转换为底层
  `row * 50 + col`。
- `VariableGridScenarioPoolEnv` 对所有实际尺寸进行零填充，不缩放长宽比；填充区动作始终非法。
- `PlanarResectionEnv._info()` 已加入 `peak_vessel_strain`、`peak_front_tension`、
  `peak_organ_energy`，供逐步冻结评估采集（仅扩展 info 字段，不影响观测与奖励）。

### 4.2 策略

`variable_policy.py` 中：

- `PaddedSpatialExtractor` 使用两层 3×3 卷积，保留每格空间结构；
- `SpatialActionHead` 使用共享卷积输出 30×40 的逐格 logits；
- `VariableSpatialPolicy` 继承 `MaskableActorCriticPolicy`；
- 当前模型约 6.2 万参数，动作头不再依赖固定 49 维分类器。

观测和动作空间始终固定为 30×40 的上限，因此同一模型可处理 A–D 的所有尺寸；实际合法区域
由状态通道和动作 mask 决定。

### 4.3 课程场景

`variable_scenarios.py` 定义：

| 阶段 | 行范围 | 列范围 | 最大障碍格数 |
| --- | ---: | ---: | ---: |
| A | 5–8 | 5–8 | 1 |
| B | 9–16 | 9–16 | 3 |
| C | 17–24 | 17–32 | 6 |
| D | 20–30 | 24–40 | 10 |

`generate_stage_pool()` 只生成指定阶段，供冻结评估使用。

`generate_curriculum_train_pool()` 会轮流混入当前阶段及所有以前阶段，避免遗忘：

- B 训练池按 A/B 轮换；
- C 训练池按 A/B/C 轮换；
- D 训练池按 A/B/C/D 轮换。

注意：`cache_variable_teachers.py --stage c --scenarios 4` 同样调用混合课程生成器，因此四个
场景实际是 A、B、C、A，只包含一个 C 场景。它不能作为纯阶段 C 的可靠耗时基准。

### 4.4 教师与行为克隆

`variable_teacher.py` 对每个场景分别回放：

1. S 形优先策略；
2. 规则规划器。

只有当规划器同时满足“完成、transfer 不高于 S 形、平均血管应变不高于 S 形”时才采用规划器
轨迹，否则采用 S 形轨迹。

`cache_variable_teachers.py --isolated` 为每个场景启动独立子进程，避免力学原生库退出导致整个
缓存失败；每个场景超时默认 120 秒，并写出 `.failures.json`。该模式支持：

- `--workers N`：并发回放子进程数（默认 1，即原顺序行为）；
- `--timeout S`：每场景子进程预算（阶段 C 纯场景双回放实测最大约 181 秒，需 `--timeout 300`）；
- `--shard-dir DIR`：持久分片目录，重启时跳过已完成的 `teacher_*.npz` 实现断点续跑
  （实测 2 场景完成后重跑仅约 2.4 秒）；
- `--current-stage-only`：用 `generate_stage_pool(split="train")` 生成纯当前阶段场景，
  供耗时 pilot 使用（**不要**用于正式 C/D 缓存，正式缓存需混合课程）。

训练脚本先执行 10 epoch 行为克隆，再运行 Maskable PPO。阶段迁移通过 `--init-model` 加载
上一阶段模型，然后在新阶段混合教师上继续行为克隆和 PPO。

### 4.5 冻结评估

`evaluate_variable_policy.py` 支持：

- `ppo`：8 个子进程并行力学环境，模型推理留在父进程；
- `serpentine` / `planner`：均支持 `--workers` 进程级场景并行
  （`ProcessPoolExecutor` + fork，`executor.map` 保持输入顺序），
  `--workers 1` 等价于旧单进程路径。

每个 episode 记录（PPO 与规则基线使用同一规范字段）包括：

- 完成率、合法动作率、`transfer_count / cut_count`；
- `mean_vessel_strain`、`cumulative_vessel_strain`（累计应变）、
  `worst_10pct_vessel_strain`（最差 10% 步均值，即 CVaR）、
  `fraction_steps_above_safe` / `fraction_steps_above_tear`（超过力学
  `safe=0.12` / `tear=0.25` 阈值的步数比例）、`max_vessel_strain`；
- `mean_front_tension` / `worst_10pct_front_tension`、
  `mean_organ_energy` / `worst_10pct_organ_energy`、`max_risk_peak`。

风险指标只用于外部评价，未写入 reward。`evaluation.evaluate_policy` 与
`evaluate_variable_policy._risk_metrics` 共享同一套计算。

三种方法使用同一 seed 生成相同顺序的冻结场景，可做逐场景配对 bootstrap。

## 5. 代码与文件职责

| 文件 | 作用 | 当前状态 |
| --- | --- | --- |
| `environment.py` | 底层环境及可变尺寸适配器 | 已实现并完成小规模 smoke |
| `variable_policy.py` | 30×40 全卷积逐格策略 | 已用于 A/B 正式训练 |
| `variable_scenarios.py` | A–D 场景与混合课程 | 已用于 A/B |
| `variable_teacher.py` | S 形/规划器教师筛选 | 已用于 A/B |
| `cache_variable_teachers.py` | 可审计、隔离式教师缓存 CLI，支持并行/断点续跑/可配置超时 | A/B 已用；C 已用（2026-07-31） |
| `train_variable_masked_ppo.py` | BC + Maskable PPO + 阶段初始化 | A/B 已验证；C 训练中 |
| `evaluate_variable_policy.py` | PPO/S 形/规划器冻结评估，含风险指标与基线 `--workers` 并行 | A/B 已验证（新旧数值逐字节一致） |
| `evaluation.py` | 外部指标与基线策略，含 CVaR/累计应变/阈值步数比例 | 已扩展 |
| `pilot_stage_c.py` | 纯当前阶段耗时 pilot（逐场景 S 形/规划器计时、教师选择、障碍组件数） | 已完成 C（2026-07-31） |
| `run_stage_a_replications.py` | 阶段 A 三种子可恢复编排 | 已完成 |
| `results/阶段A冻结评估报告.md` | A 单种子与基线的配对结果 | 已完成 |
| `results/阶段B冻结评估报告.md` | B 单种子与基线的配对结果 | 已完成 |

## 6. 已完成实验

### 6.1 阶段 A：5×5–8×8

配置：

- 训练场景：每个种子 128；
- 教师缓存：隔离模式，128/128 成功；
- PPO 请求步数：25,000，实际因 8 环境 rollout 对齐为 26,624；
- `n_envs=8`、`n_steps=256`、`batch_size=256`、`bc_epochs=10`；
- `transfer_cost=2.0`；
- GPU：当时使用 `CUDA_VISIBLE_DEVICES=6`。

三个最终模型：

- `results/variable_spatial_stage_a_seed2026073102_25k/final_model.zip`
- `results/variable_spatial_stage_a_seed2026073103_25k/final_model.zip`
- `results/variable_spatial_stage_a_seed2026073104_25k/final_model.zip`

冻结集：128 个 A 场景，seed `2026074201`。

| 训练 seed | 完成率 | 合法率 | transfer overhead | 平均血管应变 |
| --- | ---: | ---: | ---: | ---: |
| 2026073102 | 1.000 | 1.000 | 0.25878 | 0.02651 |
| 2026073103 | 1.000 | 1.000 | 0.25850 | 0.02630 |
| 2026073104 | 1.000 | 1.000 | 0.26319 | 0.02644 |
| 三种子平均 | 1.000 | 1.000 | 0.26016 | 0.02642 |

三种子汇总：
`results/variable_eval_stage_a_replications.json`。

阶段 A 的单种子基线比较详见 `results/阶段A冻结评估报告.md`。注意该报告基于 seed
`2026073103`，三种子汇总并未重新计算对基线的跨种子层级置信区间。

### 6.2 阶段 B：9×9–16×16

配置：

- 教师缓存：`results/teacher_cache_stage_b_seed2026073201_128_isolated.npz`，
  128/128 成功；
- 初始化：
  `results/variable_spatial_stage_a_seed2026073103_25k/final_model.zip`；
- 训练池：128 个 A/B 轮换场景；
- PPO 请求步数：50,000；
- 其余主要参数与阶段 A 相同。

最终模型：

`results/variable_spatial_stage_b_from_a3103_seed2026073201_50k/final_model.zip`

冻结集：128 个 B 场景，seed `2026074202`。

| 方法 | 完成率 | 合法率 | transfer overhead | 平均血管应变 |
| --- | ---: | ---: | ---: | ---: |
| PPO | 1.000 | 1.000 | 0.35899 | 0.02843 |
| S 形 | 1.000 | 1.000 | 0.34761 | 0.02850 |
| 规则规划器 | 1.000 | 1.000 | 0.53618 | 0.04022 |

配对结果：

- 对 S 形，PPO transfer 高 3.27%，均值差 0.01138，95% CI
  `[-0.00190, 0.02497]`；
- 对 S 形，PPO 平均血管应变低 0.23%，均值差 -0.000067，95% CI
  `[-0.000264, 0.000133]`；
- 对规划器，PPO transfer 低 33.05%，95% CI
  `[-0.21178, -0.14294]`；
- 对规划器，PPO 平均血管应变低 29.32%，95% CI
  `[-0.01315, -0.01043]`。

完整报告：`results/阶段B冻结评估报告.md`。

阶段 B 已通过“完成/合法率 100%，且相对 S 形 transfer 退化小于 10%”的工程课程门槛。
它尚未通过本文第 8 节列出的完整科研评价门。

### 6.3 阶段 B 模型在阶段 A 冻结集上的遗忘检查（2026-07-31 通过）

用阶段 B 最终模型重新评估 A 冻结集（128 场景，seed `2026074201`）：

```bash
evaluate_variable_policy.py --method ppo --stage a --count 128 --seed 2026074201 --workers 8 \
  --model-path results/variable_spatial_stage_b_from_a3103_seed2026073201_50k/final_model.zip \
  --output results/variable_eval_stage_b_model_on_stage_a.json
```

| 指标 | B 模型在 A 集 | A 三种子平均 | S 形基线 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 完成率 | 1.000 | 1.000 | 1.000 | 通过 |
| 合法率 | 1.000 | 1.000 | 1.000 | 通过 |
| transfer overhead | 0.26296 | 0.26016 | 0.26188 | 相对 S 形 100.41%，低于 110% 门槛 |
| 平均血管应变 | 0.02633 | 0.02642 | 0.02637 | 不低于 A 三种子均值 |

结论：阶段 B 模型没有小尺寸遗忘。允许继续推进阶段 C。

### 6.4 阶段 C 首个候选种子（2026-07-31，seed 2026073301）

从阶段 B 最终模型初始化，192 混合训练场景（A/B/C 各 64），BC 初始化 10 epochs
（192/192 episodes，loss ≈ 0）后训练 75k Maskable PPO，实际 75,776 步。模型：
`results/variable_spatial_stage_c_from_b_seed2026073301_75k/final_model.zip`；
中间 checkpoint 4 个（18,744 / 37,488 / 56,232 / 74,976 步，CheckpointCallback 的
`save_freq` 语义为每 `env.step()` 一次调用，实际约 4 个而非 32 个）。

三冻结集（各 128 场景，seed 2026074201/02/03）与基线配对评估：

| 冻结集 | C 模型 完成/合法 | C 模型 transfer | S 形 transfer | 规划器 transfer | 相对 S 形 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 1.000 / 1.000 | 0.261 | 0.262 | 0.245 | 99.8% ✅ |
| B | 1.000 / 1.000 | 0.359 | 0.348 | 0.536 | 103.3% ✅ |
| C | 1.000 / 1.000 | 0.277 | 0.269 | 0.808 | 103.1% ✅ |

外源风险指标（C 模型，评估期统计，不进入 reward）：

| 冻结集 | 平均应变 | 累计应变 | >safe(0.12) | >tear(0.25) | max_peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.0264 | 1.06 | 0.0 | 0.0 | 0.055 |
| B | 0.0285 | 2.35 | 0.0 | 0.0 | 0.059 |
| C | 0.0309 | 8.39 | 0.0 | 0.0 | 0.065 |

评估文件：`results/variable_eval_stage_c_model_on_stage_{a,b,c}.json`、
`results/variable_eval_stage_c_serpentine.json`、`results/variable_eval_stage_c_planner.json`。
A/B 集的 S 形/规划器基线复用 `variable_eval_stage_{a,b}_{serpentine,planner}.json`。

结论：首个 C 候选通过工程晋级门槛——三集完成/合法率 100%，相对 S 形 transfer 全部
<110%（99.8%–103.3%），无小尺寸遗忘（A/B 集仍 100%），无任何步超过 safe/tear 阈值。
规划器在 C 尺度明显退化（transfer 0.808，为 C 模型的 2.9×），验证课程式 PPO 的价值
与教师筛选的回退逻辑。

评估命令示例（A 集）：

```bash
evaluate_variable_policy.py --method ppo --stage a --count 128 --seed 2026074201 --workers 8 \
  --model-path results/variable_spatial_stage_c_from_b_seed2026073301_75k/final_model.zip \
  --output results/variable_eval_stage_c_model_on_stage_a.json
```

### 6.5 阶段 C 三种子汇总（2026-07-31）

三个独立训练 seed（2026073301 / 2026073302 / 2026073303，均 75,776 步，从阶段 B 最终模型
初始化）在 A/B/C 冻结集（各 128 场景，seed 2026074201/02/03）上的结果。

| 冻结集 | 完成率 | 合法率 | transfer 均值（min–max） | 相对 S 形均值 | 累计应变 | 最差 10% | >safe | >tear | max_peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1.000 | 1.000 | 0.261（0.261–0.261） | 99.7% | 1.06 | 0.041 | 0.00 | 0.00 | 0.055 |
| B | 1.000 | 1.000 | 0.363（0.359–0.367） | 104.3% | 2.35 | 0.042 | 0.00 | 0.00 | 0.059 |
| C | 1.000 | 1.000 | 0.278（0.276–0.281） | 103.2% | 8.39 | 0.042 | 0.00 | 0.00 | 0.065 |

评估文件：`variable_eval_stage_c_{model,seed2026073302,seed2026073303}_on_stage_{a,b,c}.json`。

结论：
- 三种子全部通过工程晋级门槛（完成/合法率 100%，相对 S 形 transfer 99.7%–105.5% < 110%）；
- 跨种子方差极小（同冻结集 transfer 波动 ≤ 0.008，B 集 0.359–0.367 最大），训练稳定；
- A 集三种子 transfer 完全一致（0.261），C 集一致（0.276–0.281），B 集为三种子差异最大处；
- 风险指标与单候选一致：全部步低于 safe(0.12)/tear(0.25)，max_peak 随尺寸上升
  （A 0.055 → B 0.059 → C 0.065）；
- 规划器在 B/C 尺度退化（transfer 0.536 / 0.808），C 模型显著更优，验证课程式 PPO 价值。

阶段 C 三 seed 稳定性结论已成立（区别于阶段 B 的单 seed）。仍待补齐的科研评价项见第 8.1 节
（独立 Validation/Stress 分离、逐场景配对 bootstrap 置信区间）。

### 6.6 风险面与基线对比（2026-07-31）

C 模型（三种子均值）在 A/B/C 冻结集上与 S 形、规划器基线的外源风险指标对比。
A/B 的 S 形/规划器基线用新代码重跑补全风险字段，文件
`results/variable_eval_stage_{a,b}_{serpentine,planner}_full.json`；C 集基线为原文件。

| 冻结集 | 方法 | transfer | 平均应变 | 累计应变 | CVaR@10% | >safe | >tear | max_peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | C 模型 | 0.261 | 0.0264 | 1.062 | 0.0413 | 0.00 | 0.00 | 0.0549 |
| A | S 形 | 0.262 | 0.0264 | 1.062 | 0.0413 | 0.00 | 0.00 | 0.0549 |
| A | 规划器 | 0.245 | 0.0390 | 1.566 | 0.0417 | 0.00 | 0.00 | 0.0536 |
| B | C 模型 | 0.363 | 0.0285 | 2.351 | 0.0419 | 0.00 | 0.00 | 0.0585 |
| B | S 形 | 0.348 | 0.0285 | 2.350 | 0.0419 | 0.00 | 0.00 | 0.0585 |
| B | 规划器 | 0.536 | 0.0402 | 3.364 | 0.0422 | 0.00 | 0.00 | 0.0581 |
| C | C 模型 | 0.278 | 0.0309 | 8.393 | 0.0423 | 0.00 | 0.00 | 0.0653 |
| C | S 形 | 0.269 | 0.0309 | 8.387 | 0.0423 | 0.00 | 0.00 | 0.0653 |
| C | 规划器 | 0.808 | 0.0414 | 11.113 | 0.0424 | 0.00 | 0.00 | 0.0650 |

关键结论：
- **C 模型风险面 = S 形**：三集累计应变仅差 <0.1%，即 C 模型在不牺牲安全性的前提下
  学会近似 S 形的高效路径（transfer 仅高 ~3%）；
- **规划器风险面显著更差**：平均/累计应变比 C 模型高 34–48%（A 集 cum 1.566 vs 1.062），
  与它高出 1.9–2.9× 的 transfer 开销对应——每次额外转移累积应变；
- **区分度观察**：三种方法最差 10% 步应变（CVaR）都落在 0.041–0.042、峰值均远低于
  safe(0.12)，差异体现在累积/平均应变而非尖峰。若希望 >safe 阈值产生区分度，
  需调低阈值或改变应变聚集方式（后续可选工作，不影响工程晋级结论）。

## 7. 实测耗时

以下时间来自 2026-07-30 当前服务器，GPU 6 当时存在其他任务，不能直接外推为空闲 GPU：

| 任务 | 实测或近似实测 |
| --- | ---: |
| A 单种子 25k PPO | 约 7–9 分钟 |
| A 三种子补齐、缓存及评估编排 | 约 33 分钟 |
| B 128 条隔离混合教师缓存 | 约 22 分钟 |
| B 50k 迁移 PPO | 约 28 分钟 |
| B PPO 128 场景并行评估 | 约 10 分钟 |
| B S 形 128 场景单进程评估 | 约 16 分钟 |
| B 规划器 128 场景单进程评估 | 约 16 分钟 |
| B 规划器 `--workers 8`（16 场景实测） | 17.6 秒，相对单进程 63.7 秒约 3.6× 加速 |
| B 模型在 A 冻结集遗忘检查（PPO，workers 8） | 约 3–4 分钟 |

阶段 B 从缓存到三方评估约 1.5 小时。评价脚本并行化后，S 形/规划器 128 场景评估可由
单进程约 16 分钟降至约 4–5 分钟。

阶段 C（2026-07-31）实测：

| 任务 | 实测 |
| --- | ---: |
| C 192 混合场景教师缓存（workers 12，含一次 24×32 >300 s 超时重试后 192/192） | 约 45–60 分钟 |
| C 单种子 75k PPO（含 BC，GPU 7） | 约 70 分钟（fps ≈ 18） |
| C PPO 128 场景冻结评估（workers 8，单集） | 约 15–20 分钟 |
| C S 形 128 场景并行评估 | 约 5–10 分钟 |
| C 规划器 128 场景并行评估 | 约 15–30 分钟 |
| C 三集 PPO + C 双基线并行（5 任务，48 核） | 约 30–40 分钟 |

### 7.1 阶段 C 纯尺寸 pilot 实测（2026-07-31，seed 2026073301，8 场景）

`results/stage_c_pilot_seed2026073301.json`：

| 指标 | 数值 |
| --- | ---: |
| 完成率 | 1.000 |
| 尺寸分布 | 18–24 行 × 17–29 列 |
| 障碍组件数分布 | 1–6 |
| S 形单场景耗时 | 均值 52.0 s，最大 88.7 s |
| 规划器单场景耗时 | 均值 52.8 s，最大 92.1 s |
| 双回放（S 形+规划器）合计 | 均值 104.7 s，最大 180.9 s |
| 演示步数 | ≈ 有效域格数，均值 272，最大 361 |
| 规划器被选中率 | 0.000（C 尺度下规划器 transfer 是 S 形的 3–5 倍，筛选正确回退 S 形） |

pilot 同时验证了**规划器在 C 尺度能完成**（completion 1.0），但 transfer overhead
0.84–1.20 远高于 S 形的 0.21–0.32——这是贪心基线的尺度退化，不是框架故障。

按 192 混合场景（64 A/B/C）预算：64 个 C 场景双回放 ≈ 6,720 s，A/B 约 1,000 s；
单进程约 2 小时，`--workers 8-12` 约 20–30 分钟。

阶段 C/D 的 episode 更长，当前不能用面积简单线性外推。必须先做小规模纯当前阶段 benchmark，
再给出完整缓存和训练时间。保守预留：

| 阶段 | 单个候选的初步预算 |
| --- | ---: |
| C：缓存 + 75k 训练 + 多尺寸评估 | 约 3–8 小时，须由 pilot 修正 |
| D：缓存 + 100k 训练 + 全尺度评估 | 约 6–16 小时，须由 pilot 修正 |
| 每阶段三个训练 seed | 上述训练和 PPO 评估部分约乘 3；固定规则基线可复用 |

按 7.2 节 D pilot 修正：D 缓存（256 混合，workers 12）约 1 小时，100k PPO 约 2–3 小时，
全尺度评估约 30–60 分钟，即 D 单候选约 **4–5 小时**（比保守预留 6–16 小时乐观）。

### 7.2 阶段 D 纯尺寸 pilot 实测（2026-07-31，seed 2026074401，4 场景）

`results/stage_d_pilot_seed2026074401.json`：

| 指标 | S 形回放 | 规划器回放 | 双回放合计 |
| --- | ---: | ---: | ---: |
| 单场景均值 | 177.9 s | 184.3 s | 362.1 s |
| 单场景最大 | 249.4 s | 257.0 s | 506.4 s |

- 场景 22–30 × 28–33，有效域格 364–584，障碍组件 3–8；
- 完成率 1.0；演示步数均值 490、最大 583；
- 规划器选中率 0.000（D 尺度全部回退 S 形，教师筛选继续正确）；
- 据此缓存超时设为 `--timeout 900`。

D 缓存命令（已启动，见 8.1 节状态）：

```bash
cache_variable_teachers.py --output results/teacher_cache_stage_d_seed2026074401_256_isolated.npz \
  --stage d --scenarios 256 --seed 2026074401 --isolated --workers 12 --timeout 900 \
  --shard-dir results/shards_stage_d_seed2026074401_256
```

## 8. 尚未完成与已知风险

### 8.1 进入阶段 C 前必须补齐

1. ✅ **小尺寸遗忘检查已完成（2026-07-31）**：阶段 B 模型在 A 冻结集
   `seed 2026074201` 上回归通过，完成/合法率 1.000，相对 S 形 transfer 100.41% < 110%。
   结果见 `results/variable_eval_stage_b_model_on_stage_a.json` 与第 6.3 节。
2. **阶段 B 只有一个训练 seed**：可用于工程晋级 pilot，不能用于稳定性结论；正式报告至少还需
   两个独立 B seed。
3. ✅ **最终风险指标已实现**：`evaluate_variable_policy.py` 与 `evaluation.py` 现在逐 episode
   报告 CVaR（`worst_10pct_vessel_strain`）、累计应变、`safe`/`tear` 阈值以上步数比例、
   最大峰值及前沿张力/器官能量的均值与最差 10%。已在 A/B 冻结集上验证新旧 transfer 与平均
   应变逐字节一致。
4. **缺少 Validation/Stress 分离**：当前可变尺寸流程只有训练池和冻结 `split="frozen"`；
   还没有独立 Validation 选 checkpoint，也没有专门 Stress 集。
5. ✅ **阶段 C 纯尺寸 pilot 已完成（2026-07-31）**：结果见第 7.1 节与
   `results/stage_c_pilot_seed2026073301.json`。此前中断的
   `teacher_cache_stage_c_pilot4_*` 无产物，无需重跑；正式 C 缓存（192 混合场景）已启动。
6. ✅ **阶段 C 三种子已完成（2026-07-31）**：种子 2026073301/02/03 全部训练完成并通过
   A/B/C 三冻结集工程门槛（汇总见第 6.5 节）。中途一次为脱离 SSH 会话而重启训练，
   原两段半程目录已改名保留：`results/variable_spatial_stage_c_from_b_seed202607330{2,3}_75k_interrupted_partial`。
   C 的独立 Validation/Stress 集分离沿用第 4 项待办。
7. 🔄 **阶段 D 进行中（2026-07-31）**：D 纯尺寸 pilot 已完成（见第 7.2 节，
   `results/stage_d_pilot_seed2026074401.json`）；256 混合场景 D 教师缓存生成中
   （`--workers 12 --timeout 900 --shard-dir results/shards_stage_d_seed2026074401_256`，
   约 1 小时）。缓存完成后从 C 模型初始化训练 100k PPO（步骤 5）。

### 8.2 性能与实现注意事项

- ✅ **S 形/规划器冻结评估已支持 `--workers` 进程级并行**：`executor.map` 保持输出顺序与
  场景 ID 可配对；`--workers 1` 与旧路径逐字节一致。实测 B 规划器约 3.6× 加速。
- ✅ **隔离缓存已支持并行与断点续跑**：`--workers` 并发回放、`--shard-dir` 持久分片目录、
  `--timeout` 可配置。阶段 C 缓存（192 混合场景）已用此能力完成；阶段 D 缓存（256 混合）
  正使用此能力生成。
- `cache_variable_teachers.py --isolated` 只在全部场景结束后合并最终 `.npz`。中断时临时分片随
  `TemporaryDirectory` 消失，不支持断点续跑。
- `--init-model` 会加载上一模型保存的 PPO 超参数。当前 A→B 的参数相同，所以没有冲突；如果
  C/D 想改变学习率、`n_steps` 或 `n_epochs`，必须显式确认 SB3 load 后是否覆盖成功。
- `--init-model` 路径虽然写入 metadata，但加载后随机数是否完全按新 seed 重置尚未做严格审计。
- `run_metadata.json` 记录请求参数和场景，但没有在训练结束后写入实际步数、结束时间或硬件快照。
- 当前场景 `obstacle_cells` 是血管代理格；障碍格数增加不等同于真实多组件血管复杂性，需要检查
  组件数分布。
- 结果目录和新增脚本当前多为 Git 未跟踪状态。不要执行 `git clean`、删除 `results/` 或清理
  checkpoint；这些是本轮训练证据。
- 仓库没有 remote。除非用户明确要求，不要添加 remote、push 或发布。

## 9. 下一步执行计划

### 步骤 0：接手审计（约 15–30 分钟）

1. 确认没有遗留进程：

```bash
ps -eo pid,etime,cmd | rg \
  '[c]ache_variable_teachers.py|[t]rain_variable_masked_ppo.py|[e]valuate_variable_policy.py'
```

2. 确认阶段 A/B 的最终模型和 JSON 存在。
3. 运行现有环境、力学和训练 smoke tests；不要先重写环境。
4. 记录 Python、PyTorch、SB3、sb3-contrib、CUDA 和 GPU 状态。

### 步骤 1：阶段 B 模型的小尺寸回归（✅ 已完成 2026-07-31）

用阶段 B 模型重新评估 A 冻结集：

```bash
cd Research/planar-resection-planning
MPLCONFIGDIR=/tmp/mpl_codex \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u evaluate_variable_policy.py \
  --method ppo --stage a --count 128 --seed 2026074201 --workers 8 \
  --model-path results/variable_spatial_stage_b_from_a3103_seed2026073201_50k/final_model.zip \
  --output results/variable_eval_stage_b_model_on_stage_a.json
```

与 `variable_eval_stage_a_serpentine.json`、`variable_eval_stage_a_planner.json` 和阶段 A 三种子结果
比较。若完成率/合法率下降，或相对 S 形 transfer 超过 110%，停止阶段 C，先修复遗忘。
结果：完成/合法率 1.000，相对 S 形 transfer 100.41%，通过。见第 6.3 节。

### 步骤 2：补齐评价契约和并行评估（✅ 已完成 2026-07-31）

在继续消耗大规模算力前：

1. 在 episode 记录中增加累计血管应变、最差 10% 步应变、风险阈值以上步数比例、最大风险峰值、
   前沿张力和器官能量的对应统计；
2. 为 S 形/规划器增加 `--workers` 场景并行；
3. 保证并行输出按原 `scenario_id` 顺序恢复；
4. 在阶段 A/B 现有冻结集上验证新旧 transfer 和平均应变数值完全一致；
5. 新指标只用于外部评价，不要未经实验直接写入 reward。

已完成：风险指标与并行基线已实现，A/B 冻结集三方评估（PPO/S 形/规划器）新旧
`transfer_overhead`、`mean_vessel_strain` 全部逐字节一致；并行与串行 summary 及
`scenario_id` 顺序一致；新指标仅出现在 info/评估记录，reward 未改动。

### 步骤 3：阶段 C 纯尺寸 pilot（✅ 已完成 2026-07-31）

为缓存 CLI 增加了 `--current-stage-only`（纯当前阶段场景生成），并编写了
`pilot_stage_c.py`（逐场景计时/教师选择/障碍组件数）。pilot 记录要点见第 7.1 节：
单场景双回放均值 104.7 s、最大 180.9 s，全部完成，规划器在 C 尺度退化但能完成，
筛选正确回退 S 形。据此将缓存超时设为 300 s。

### 步骤 4：阶段 C 正式训练（✅ 三 seed 全部完成）

已完成（见第 6.4、6.5 节）：

- ✅ 192 混合场景教师缓存（A/B/C 各 64），可恢复分片；
- ✅ 三个独立 75k PPO 种子（2026073301/02/03，从阶段 B 最终模型初始化），实际各 75,776 步；
- ✅ A/B/C 三冻结集评估：三种子完成/合法率均 100%，相对 S 形 transfer 99.7%–105.5%
  （<110% 门槛），风险指标零越线，无小尺寸遗忘，跨种子方差极小；
- ⏳ 使用独立 Validation 选 checkpoint、冻结 Test 不参与选模——仍待实现（第 8.1-4 项）。

参考命令（缓存脚本具备断点续跑后再执行）：

```bash
CUDA_VISIBLE_DEVICES=6 \
python -u cache_variable_teachers.py \
  --output results/teacher_cache_stage_c_seed2026073301_192_isolated.npz \
  --stage c --scenarios 192 --seed 2026073301 --isolated
```

```bash
CUDA_VISIBLE_DEVICES=6 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl_codex PYTHONUNBUFFERED=1 \
python -u train_variable_masked_ppo.py \
  --output-dir results/variable_spatial_stage_c_from_b_seed2026073301_75k \
  --teacher-cache results/teacher_cache_stage_c_seed2026073301_192_isolated.npz \
  --stage c --timesteps 75000 --train-scenarios 192 \
  --n-envs 8 --n-steps 256 --batch-size 256 --bc-epochs 10 \
  --seed 2026073301 --device cuda \
  --init-model results/variable_spatial_stage_b_from_a3103_seed2026073201_50k/final_model.zip
```

阶段 C 候选必须分别评估 A、B、C 冻结集，不能只看 C。

### 步骤 5：阶段 D

只有阶段 C 的三个种子、跨 A/B/C 回归和完整风险门通过后才开始 D：

- D 训练池建议至少 256 个场景，使 A/B/C/D 各 64 个；
- 从选中的 C 模型初始化；
- 先 100k PPO，再根据 Validation 决定是否延长；
- 建立独立 D Test 和 Stress，覆盖 30×40、长宽比极端、凹形域、狭窄通道和多组件血管；
- 最终候选必须在 A–D 全部冻结集上回归，证明模型“能大能小”。

## 10. 每阶段晋升门

一个候选进入下一阶段前至少满足：

1. 当前阶段及所有以前阶段的完成率为 1.000；
2. 合法动作率为 1.000；
3. release 规则正确率为 1.000；
4. 相对 S 形和规则规划器的 transfer overhead 不超过各自的 110%；
5. 逐场景配对 bootstrap 区间已报告；
6. 血管风险主终点使用预先冻结的 CVaR 与累计应变，而不只看平均值；
7. 至少三个训练 seed 的方向一致；
8. Test 未参与 reward、超参数、早停或 checkpoint 选择。

若只满足前四项，可称为“工程课程候选”，不可称为最终科研评价通过。

## 11. 当前交接点

接手 Agent 当前执行步骤 5：阶段 D 训练准备。

- 步骤 1（遗忘检查）、步骤 2（评价契约补强）、步骤 3（纯 C pilot）均已完成，见第 6.3、7.1 节。
- ✅ C 教师缓存 192/192 完成：`results/teacher_cache_stage_c_seed2026073301_192_isolated.npz`
  （`--workers 12 --timeout 300 --shard-dir results/shards_stage_c_seed2026073301_192`）。
- ✅ 阶段 C 三个独立种子 2026073301/02/03（各 75,776 步）全部训练完成，A/B/C 三冻结集工程
  门槛全过（第 6.5 节），风险面对比见第 6.6 节
  （`variable_eval_stage_c_{model,seed...}_on_stage_{a,b,c}.json`）。
- 🔄 阶段 D 已启动：纯尺寸 pilot 完成（第 7.2 节），256 混合场景 D 教师缓存生成中
  （`teacher_cache_stage_d_seed2026074401_256_isolated.npz`，workers 12 / timeout 900，
  约 1 小时）。缓存完成后从 C 模型初始化训练 100k PPO（种子建议 2026074402/03/04）。
- ⏳ 进入 D 训练前仍待补（第 8.1 节）：独立 Validation/Stress 集分离、逐场景配对 bootstrap
  置信区间、阶段 B 第二个独立 seed。

