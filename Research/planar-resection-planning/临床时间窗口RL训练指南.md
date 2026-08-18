# 临床时间窗口 RL 训练与冻结评估指南

> 版本：v2（2026-08-04，替代 v1）
> 目标读者：接手执行训练、验证和结果整理的代理 Agent  
> 适用目录：`Research/planar-resection-planning`  
> 结论边界：本任务输出的是“面积模型估算的期望模拟失血”，不是经临床验证的术中失血预测。

> **2026-08-05 v8 提示：**本文中 `clinical-window-v2`、21 通道和旧 reward 参数只作历史
> 参考。新训练必须使用 `临床时间窗口奖励函数v8设计与验收.md` 的
> `clinical-window-v3`、25 通道和 v8 CLI。

> **2026-08-04 执行顺序更新：**当前应先执行 `固定15分钟到自主松夹两阶段训练指南.md`。
> 本文继续作为底层环境、reward、命令和冻结评估参考；不得跳过两阶段指南的代码
> 改造、Gate 0 时序机会审计和导航准入门槛，直接运行本文中的正式三 seed 命令。

## 1. 本轮任务的最终目标

训练一个五动作 MaskablePPO 策略，使其在完成二维切除任务的同时，学会联合安排切除路径和
夹闭窗口：

1. 每轮夹闭最长 15 min，策略可提前结束夹闭；
2. 结束夹闭后必须开放 5 min，之后由环境自动重新夹闭；
3. 夹闭时、血管尚未暴露时或血管已经切断封闭时，模拟失血为 0；
4. 开放期内，已围绕解离但尚未封闭的血管按暴露面积累计期望模拟失血；
5. 总耗时是主要效率指标，期望模拟失血是新增安全指标；
6. 最终与机械 S 形 baseline 在完全相同的冻结场景上做逐场景配对比较。

本任务必须重新训练。旧 v3 模型使用约 1,200 维动态前沿格动作；新环境动作空间只有
`上、下、左、右、提前结束夹闭` 五个动作，旧 checkpoint 不能直接加载。

## 2. 不允许擅自修改的冻结约定

代理 Agent 在首轮正式训练中不得自行改变以下语义：

| 项目 | 冻结值 |
| --- | --- |
| 单格尺寸 | $4\,\mathrm{mm}\times4\,\mathrm{mm}=16\,\mathrm{mm^2}$ |
| 暂定切除速度 | $2.3\,\mathrm{cm^2/min}$ |
| 普通方向动作耗时 | $0.16/2.3\approx0.069565$ min，即约 4.17 s |
| 大血管判据 | 完整局部横截面不少于 2 格 |
| 大血管处理耗时 | 普通动作的 3 倍 |
| 夹闭规则 | 每轮最多 15 min，可提前结束 |
| 开放规则 | 固定 5 min，期间不能提前重新夹闭 |
| 血管处理 | 切断即封闭，封闭后不再出血 |
| 训练出血概率 | 所有血管共用 $p_{\mathrm{bleed}}=1$ |
| baseline | 机械 S 形，不感知血管风险，不提前结束，机械执行 15/5 |
| transfer | 继续记录，但不单独进入 reward；其耗时已计入时间成本 |
| Test 使用 | 完成模型和超参数选择前禁止查看 Test/Stress 结果 |

如需改变上述内容，应停止训练并把改动理由、预期影响和兼容性风险交给负责人确认。

## 3. 已准备的代码

| 文件 | 作用 |
| --- | --- |
| `clinical_window_environment.py` | 五动作环境、15/5 状态机、血管状态、时间/失血积分、分项 reward |
| `clinical_window_scenarios.py` | 生成并冻结 Train/Validation/Test/Stress 场景 |
| `calibrate_clinical_window.py` | 仅用 Train 场景冻结全局时间和失血尺度 |
| `clinical_window_policy.py` | 21 通道网格观测的 CNN 特征提取器 |
| `train_clinical_window_ppo.py` | MaskablePPO 多环境训练、checkpoint 和完整元数据 |
| `clinical_window_evaluation.py` | 机械 S 形或 PPO 的冻结评估 |
| `audit_clinical_window_checkpoints.py` | 在小型冻结 Validation 子集上审计全部 checkpoints |
| `compare_clinical_window_results.py` | RL 与 S 形的逐场景配对差值和 bootstrap 95% CI |
| `tests/test_clinical_window.py` | 环境、状态机、面积、失血、动作 mask 和基线单元测试 |
| `临床时间窗口与模拟出血奖励模型设计.md` | 数学模型、临床假设和评价原则 |

所有新入口默认拒绝覆盖已有训练目录、冻结尺度或评估结果。需要重跑时应换一个带版本号的输出
目录，不要删除旧结果。

## 4. 环境与 reward 摘要

### 4.1 血管状态

一个互相隔离的血管连通分量代表一个局部完整横截面：

```text
hidden --周围四邻域组织环全部切除--> exposed --进入血管格处理--> sealed
```

进入 exposed 横截面的任意一格，会一次性切除并封闭整个横截面。横截面 1 格时耗时
$\Delta t_0$；不少于 2 格时耗时 $3\Delta t_0$。

### 4.2 出血模型

默认体重 $W=70$ kg，参考肝血流为：

$$
Q_{\mathrm{ref}}=17W=1190\,\mathrm{mL/min}.
$$

参考面积为 $80\,\mathrm{mm^2}$。当开放期存在总暴露面积 $A_{\mathrm{exposed}}$ 时：

$$
Q_t=\min\left(
Q_{\mathrm{ref}},
p_{\mathrm{bleed}}\frac{Q_{\mathrm{ref}}}{80}A_{\mathrm{exposed}}
\right),
\qquad
\Delta B_t=Q_t\Delta t_t.
$$

训练固定 $p_{\mathrm{bleed}}=1$。它与失血权重只形成乘积，缺少标定数据时不可分别辨识；正式
比较可以在冻结策略上用其他统一概率做敏感性评估，但不得据此重新选择模型。

### 4.3 reward

每步成本为：

$$
r_t=-\left(
\lambda_T\frac{\Delta t_t}{T_{\mathrm{scale}}}
+\lambda_B\frac{\Delta B_t}{B_{\mathrm{scale}}}
+\lambda_Fc_t^F+\lambda_Ec_t^E+\lambda_Vc_t^V
\right)
+\lambda_P\frac{\Delta N_{\mathrm{cut},t}}{N_{\mathrm{domain}}-1}.
$$

进度项只奖励本步新切除的格子，回穿和提前结束均为 0；任意完整轨迹累计固定获得 `+5`，因此
不恢复独立 transfer 奖励，也不改变已完成路径之间的原始时间—失血评价。完成奖励仍为 `+5`，
失败或超时惩罚为 `-5`。v2 首轮权重使用：

```text
time=1.0, blood=1.0, progress=5.0, front_tension=0.10,
organ_energy=0.10, vessel_strain=1.0, completion=5.0, failure=5.0
```

首轮训练默认 `--mechanics-update-interval 0`，因此三个力学观测及成本均为 0，只训练时间和
失血主目标。原因是当前力学求解器逐动作计算非常慢。时间—失血模型通过全部验收后，才可用
`--mechanics-update-interval 10` 做小规模联合微调；此时必须单独编号实验，不能与首轮结果
混在一起。

训练器必须保持 `norm_reward=False`。时间和失血已经由冻结尺度完成量纲归一化，再使用
`VecNormalize(norm_reward=True)` 会让固定权重随运行时回报方差漂移。每个 rollout 结束后，
训练器会把最近 100 个终止 episode 的完成率、覆盖率、回穿率、提前结束次数、耗时和失血写入
`training_health/` 与 `training_health_latest.json`。

## 5. 运行前准备

### 5.1 工作区保护

先在仓库根目录查看状态：

```bash
git status --short
```

现有不相关未跟踪文件和用户数据不得删除、覆盖或提交。训练产物很大，只保存在单独的结果
目录；除非负责人明确要求，不执行 `git add`、`git commit` 或任何远程操作。

### 5.2 Python 环境

当前已完成冒烟验证的版本为：

```text
Python 3.10.20
PyTorch 2.5.1+cu121
Gymnasium 0.29.1
stable-baselines3 2.5.0
sb3-contrib 2.5.0
NumPy 2.2.6
```

确认导入与设备：

```bash
python -c "import torch,gymnasium,stable_baselines3,sb3_contrib,numpy; print(torch.__version__, gymnasium.__version__, stable_baselines3.__version__, sb3_contrib.__version__, numpy.__version__, torch.cuda.is_available())"
```

当前 Codex 沙箱检查结果是 `torch.cuda.is_available() == False`，但服务器的代理 Agent 训练
环境已确认 CUDA 可用；这两者不能混为一谈。正式 Pilot 和多 seed 应优先在 CUDA 环境运行，
并记录实际机器、GPU、驱动、CUDA、PyTorch 和内存信息。CPU 仅用于本地冒烟或沙箱诊断。
不要为了获得 GPU 而擅自更换整个 Python 环境。若 Matplotlib 报缓存目录不可写，可设置：

```bash
MPLCONFIGDIR=/tmp/clinical-window-mpl
```

训练器默认 `--torch-threads 1`，适合 CUDA 或多个训练任务并发。若仅做 CPU 诊断，可在单个
Pilot 中设置 `--torch-threads 8`；线程数属于运行性能配置，必须写入元数据，不改变 reward 或
场景。PPO 评估和 checkpoint 审计也支持同名参数。不要同时并发多个各自使用 8 线程的正式
run 或评估。

### 5.3 固定工作目录

后续命令均从以下目录执行：

```bash
cd Research/planar-resection-planning
```

建议把本轮所有输出放在一个新目录，例如：

```text
results/clinical_window_v5/
├── frozen/
├── smoke/
├── runs/
└── evaluation/
```

## 6. 训练前的强制检查

### 6.1 编译和单元测试

```bash
python -m py_compile clinical_window_environment.py clinical_window_scenarios.py clinical_window_evaluation.py clinical_window_policy.py calibrate_clinical_window.py train_clinical_window_ppo.py audit_clinical_window_checkpoints.py compare_clinical_window_results.py
```

```bash
PYTHONPATH=. python -m unittest tests/test_clinical_window.py -v
```

预期：10 个测试全部 `OK`。任何失败都应先修复，不得带错开始正式训练。

### 6.2 生成一次性冻结数据

```bash
python clinical_window_scenarios.py --output results/clinical_window_v5/frozen/splits_v5.json --stage d --train-count 256 --validation-count 96 --test-count 120 --stress-count 80
```

该命令生成固定 seed 的四个不重叠集合。生成后检查：

```bash
sha256sum results/clinical_window_v5/frozen/splits_v5.json
```

把 SHA-256 写入实验记录。正式训练开始后，不得重新生成同名数据，也不得移动 Test 场景到
Train。

### 6.3 仅用 Train 冻结归一化尺度

```bash
python calibrate_clinical_window.py --splits results/clinical_window_v5/frozen/splits_v5.json --output results/clinical_window_v5/frozen/scales_v5.json --limit 32 --weight-kg 70 --bleeding-probability 1
```

该步骤用 Train 中 32 条机械 S 形轨迹计算一组全局 $T_{\mathrm{scale}}$ 和
$B_{\mathrm{scale}}$。这里的 S 形只作为统一的 Train-only 校准控制器，不针对每个场景调整
reward，也不读取 Validation/Test。检查输出：

- `source_split` 必须为 `train`；
- `completed_episode_count` 必须等于 `episode_count`；
- `time_scale_minutes > 0`；
- `blood_scale_ml >= 100`；
- 记录冻结尺度文件的 SHA-256。

如果没有非零失血，说明 Train 场景或暴露语义无法提供学习信号，应停止并排查；不得从 Test
取尺度。

### 6.4 冻结 baseline 的 Validation 结果

```bash
python clinical_window_evaluation.py --splits results/clinical_window_v5/frozen/splits_v5.json --split validation --algorithm serpentine --scales results/clinical_window_v5/frozen/scales_v5.json --output results/clinical_window_v5/evaluation/validation_serpentine_v5.json
```

baseline 必须 100% 完成、100% 合法，不应有提前结束夹闭。若失败，先修环境或控制器。

## 7. 冒烟训练：先证明整条链路能跑

先运行很小的训练，不判断模型优劣：

```bash
python train_clinical_window_ppo.py --splits results/clinical_window_v5/frozen/splits_v5.json --scales results/clinical_window_v5/frozen/scales_v5.json --output-dir results/clinical_window_v5/smoke/seed_2026080501 --timesteps 4096 --n-envs 1 --n-steps 256 --batch-size 128 --n-epochs 2 --progress-bonus 5 --seed 2026080501 --device cpu --train-limit 8
```

成功标准：

- 生成 `run_metadata.json`、`final_model.zip`、`vecnormalize.pkl` 和
  `training_complete.json`；
- 元数据中的 split/scales SHA-256 与冻结文件一致；
- 历史 v2 元数据必须显示 `environment_version=clinical-window-v2`、`reward_normalization=false` 和
  `progress_bonus=5.0`；
- 没有 NaN、维度错误、非法动作崩溃或覆盖旧目录；
- 4096 步模型可以不完成任务，这不是学习验收。

然后只评估 4 个 Validation 场景：

```bash
python clinical_window_evaluation.py --splits results/clinical_window_v5/frozen/splits_v5.json --split validation --algorithm ppo --model results/clinical_window_v5/smoke/seed_2026080501/final_model.zip --scales results/clinical_window_v5/frozen/scales_v5.json --output results/clinical_window_v5/smoke/validation_4.json --limit 4 --max-steps-multiplier 12 --progress-bonus 5
```

只检查评估文件能生成且字段完整。冒烟模型不应进入正式比较。

## 8. Pilot 与正式训练

### 8.1 Pilot

先用一个 seed 训练 100k 步：

```bash
python train_clinical_window_ppo.py --splits results/clinical_window_v5/frozen/splits_v5.json --scales results/clinical_window_v5/frozen/scales_v5.json --output-dir results/clinical_window_v5/runs/pilot_seed_2026080501_cuda --timesteps 100000 --n-envs 8 --n-steps 1024 --batch-size 512 --n-epochs 5 --learning-rate 0.0003 --gamma 0.999 --gae-lambda 0.98 --ent-coef 0.01 --target-kl 0.03 --progress-bonus 5 --seed 2026080501 --device cuda --torch-threads 1 --mechanics-update-interval 0
```

先检查 `training_health_latest.json`，再在 Validation 上评估 Pilot：

- 完成率低于 10% 且滚动覆盖率没有持续上升：停止，不得直接启动三 seed 2M；
- 完成率为 10%–90% 或覆盖率明显上升：可将同一 Pilot 延长到 500k 后复评；
- 完成率至少 90%：才允许进入正式三 seed；正式选模仍要求 100%。

未通过时优先依次检查：

1. 动作 mask 与终止原因；
2. 是否频繁提前结束夹闭；
3. 是否大量回穿导致超过步数；
4. `progress_bonus` 是否在完整 episode 中累计为固定 `+5`；
5. 必要时将单 seed 延长到 500k；
6. 必要时提高 `--max-steps-multiplier` 仅用于诊断，不能用它掩盖路径失效。

审计一个 run 的所有中间 checkpoint，不要只比较最终模型：

```bash
python audit_clinical_window_checkpoints.py --run-dir results/clinical_window_v5/runs/pilot_seed_2026080501_cuda --splits results/clinical_window_v5/frozen/splits_v5.json --scales results/clinical_window_v5/frozen/scales_v5.json --output-dir results/clinical_window_v5/evaluation/pilot_checkpoint_audit --limit 16 --progress-bonus 5 --torch-threads 1
```

### 8.2 正式三 seed 训练

Pilot 通过上述门槛后，才用完全相同超参数运行至少三个 seed。建议每个 2,000,000 步：

> 2026-08-04 实测状态：无教师 v5 Pilot 在 106,496 步后，训练期随机轨迹覆盖率约 69.9%，但
> 冻结 Validation 前 16 场景的确定性完成率为 0%、覆盖率仅 3.10%。随后进行 16 个 Train
> 场景 × 3 epoch 的 S 形方向行为克隆，教师 48/48 完成，但冻结 Validation 完成率仍为 0%、
> 覆盖率仅 2.37%。因此当前全局压缩 CNN 尚未通过导航可学习性门槛；下列三 seed 命令仅作为
> 门槛通过后的模板，当前禁止执行。下一步应先验证保留当前位置局部邻域特征的 policy 架构或
> A→D 课程学习，并重新做单 seed Pilot。

```bash
python train_clinical_window_ppo.py --splits results/clinical_window_v5/frozen/splits_v5.json --scales results/clinical_window_v5/frozen/scales_v5.json --output-dir results/clinical_window_v5/runs/formal_seed_2026080501 --timesteps 2000000 --n-envs 8 --n-steps 1024 --batch-size 512 --n-epochs 5 --learning-rate 0.0003 --gamma 0.999 --gae-lambda 0.98 --ent-coef 0.01 --target-kl 0.03 --progress-bonus 5 --seed 2026080501 --device cuda --mechanics-update-interval 0
```

```bash
python train_clinical_window_ppo.py --splits results/clinical_window_v5/frozen/splits_v5.json --scales results/clinical_window_v5/frozen/scales_v5.json --output-dir results/clinical_window_v5/runs/formal_seed_2026080502 --timesteps 2000000 --n-envs 8 --n-steps 1024 --batch-size 512 --n-epochs 5 --learning-rate 0.0003 --gamma 0.999 --gae-lambda 0.98 --ent-coef 0.01 --target-kl 0.03 --progress-bonus 5 --seed 2026080502 --device cuda --mechanics-update-interval 0
```

```bash
python train_clinical_window_ppo.py --splits results/clinical_window_v5/frozen/splits_v5.json --scales results/clinical_window_v5/frozen/scales_v5.json --output-dir results/clinical_window_v5/runs/formal_seed_2026080503 --timesteps 2000000 --n-envs 8 --n-steps 1024 --batch-size 512 --n-epochs 5 --learning-rate 0.0003 --gamma 0.999 --gae-lambda 0.98 --ent-coef 0.01 --target-kl 0.03 --progress-bonus 5 --seed 2026080503 --device cuda --mechanics-update-interval 0
```

若机器资源不足，可把 `--n-envs` 降到 4 或 1；同时保证 `batch-size` 能整除
`n_envs * n_steps`。不要因改变 `n-envs` 而更换 seed 或数据划分。若中途故障，可用最近
checkpoint 的 `.zip` 作为 `--init-model` 开新目录续训；新目录元数据必须说明来源。

## 9. Validation 选模

为每个正式 seed 分别运行完整 Validation：

```bash
python clinical_window_evaluation.py --splits results/clinical_window_v5/frozen/splits_v5.json --split validation --algorithm ppo --model results/clinical_window_v5/runs/formal_seed_2026080501/final_model.zip --scales results/clinical_window_v5/frozen/scales_v5.json --output results/clinical_window_v5/evaluation/validation_ppo_seed_2026080501.json --progress-bonus 5
```

另外两个 seed 只替换模型路径、seed 和输出文件名。随后分别与同一 baseline 比较：

```bash
python compare_clinical_window_results.py --ppo results/clinical_window_v5/evaluation/validation_ppo_seed_2026080501.json --baseline results/clinical_window_v5/evaluation/validation_serpentine_v5.json --output results/clinical_window_v5/evaluation/validation_compare_seed_2026080501.json
```

选模规则必须按以下优先级执行，不能直接选累计 reward 最大者：

1. `completion_rate == 1.0`；
2. `legal_action_rate == 1.0`；
3. `clamp_rule_violations == 0` 且 `unclamp_rule_violations == 0`；
4. RL 平均总耗时不超过 S 形的 105%；
5. 满足前四项的候选中，平均期望模拟失血最低者胜出；
6. 若接近，再比较峰值出血率、开放期暴露时间和最差 10% 场景。

如果没有模型通过完成率和时间门槛，应继续使用 Train/Validation 调参或延长训练。Test 必须保持
封存。

### 9.1 必查的奖励投机行为

逐场景检查以下指标，不能只看平均失血：

- `early_end_count`：是否几乎每轮一开始就提前开放；
- `total_clamped_minutes` 与 `total_unclamped_minutes`：是否用大量 5 min 开放拖延；
- `transfer_overhead`：是否反复回穿；
- `coverage` 和 `failure_reason`：是否以不完成换取低失血；
- replay 中是否在开放前故意暴露血管后长时间不封闭；
- 时间成本与血液成本的累计量级是否相差几个数量级。

`end_clamp_early` 本身耗时为 0，但只能在夹闭期执行一次，随后必须经过完整 5 min 开放；因此
连续零耗时刷动作已被 mask 阻止。策略仍可能形成“每轮极早结束”的坏习惯，必须用上述指标
和时间门槛识别，不能额外增加未经确认的提前结束惩罚。

## 10. 冻结 Test 与 Stress 评估

只有 Validation 选出唯一模型并冻结所有参数后，才运行一次 Test 和一次 Stress。先运行对应
S 形，再运行选定 PPO，最后配对比较。例如：

```bash
python clinical_window_evaluation.py --splits results/clinical_window_v5/frozen/splits_v5.json --split test --algorithm serpentine --scales results/clinical_window_v5/frozen/scales_v5.json --output results/clinical_window_v5/evaluation/test_serpentine_v5.json --progress-bonus 5
```

```bash
python clinical_window_evaluation.py --splits results/clinical_window_v5/frozen/splits_v5.json --split test --algorithm ppo --model MODEL_SELECTED_ON_VALIDATION.zip --scales results/clinical_window_v5/frozen/scales_v5.json --output results/clinical_window_v5/evaluation/test_ppo_selected_v5.json --progress-bonus 5
```

```bash
python compare_clinical_window_results.py --ppo results/clinical_window_v5/evaluation/test_ppo_selected_v5.json --baseline results/clinical_window_v5/evaluation/test_serpentine_v5.json --output results/clinical_window_v5/evaluation/test_comparison_v5.json --bootstrap-draws 10000
```

Stress 使用相同命令，只把 `--split test` 和文件名改为 `stress`。如果 Test 不理想，仍需如实
报告；不得再根据 Test 调权重、seed、网络、步数、概率或场景。

## 11. 概率敏感性与消融

主结果使用 $p_{\mathrm{bleed}}=1$。在策略完全冻结后，可把同一 PPO 和 S 形分别以
`--bleeding-probability 0.25`、`0.5`、`0.75` 重评。因为统一概率主要缩放失血，敏感性分析
用于展示绝对 mL 的不确定范围，不得重新选模。

最低消融建议：

| 消融 | 参数变化 | 回答的问题 |
| --- | --- | --- |
| 时间窗口但无失血成本 | `--blood-cost 0` | 策略优势是否真正来自出血 reward |
| 完整模型 | `--blood-cost 1` | 时间与失血联合优化效果 |
| 不允许提前结束 | 需要新增环境开关后单独编号 | 提前开放动作的独立贡献 |
| 力学联合微调 | `--mechanics-update-interval 10` | 加入原有代理风险后是否保持优势 |

当前代码尚未提供“禁止提前结束”CLI 开关。不要用手改动作 mask 的临时代码混入正式结果；如
要做该消融，应先实现版本化开关和单元测试。

## 12. 结果中必须提供的数据

代理 Agent 最终交付以下文件或汇总：

1. 冻结 `splits_v5.json`、`scales_v5.json` 及两者 SHA-256；
2. 每个 seed 的完整训练命令、机器配置、开始/结束时间；
3. 每个 run 的 `run_metadata.json`、`training_complete.json`、最终模型和 checkpoints；
4. 三个 seed 的完整 Validation JSON；
5. 选模表：完成率、合法率、总耗时、相对 S 时间比、失血、峰值率、暴露时间、transfer、
   提前结束次数；
6. 唯一选定模型的 Test/Stress 原始 JSON 和配对比较 JSON；
7. 每个主要指标的均值、中位数、配对差值 bootstrap 95% CI、RL 胜/平/负场景数；
8. 最差 10% 失血场景和代表性 replay；
9. 所有失败、续训、参数偏离和代码变更记录；
10. 清楚写明模拟失血模型的假设和非临床验证性质。

后续若要提升临床可信度，优先补充的数据是：真实器械和操作条件下的单位面积离断速度、大血管
额外封闭时间、局部血管类型与管径/横截面、血管树拓扑和分支流量、暴露后实际出血发生率、
夹闭/开放时序，以及术中逐时间段失血记录。在这些数据缺失时，不要把 $p_{\mathrm{bleed}}$ 或
模拟 mL 解释为患者级临床预测。

## 13. 常见问题与处置

### 训练很慢

- 确认 `--mechanics-update-interval 0`；
- 优先减少 `n-envs`，不要减少冻结场景；
- 确认每个子进程没有重复创建 Matplotlib 缓存；
- 当前 Codex 沙箱只做 CPU 诊断；代理 Agent 应在已确认可用的 CUDA 环境完成正式 Pilot；
- 不要在同一机器并发启动多个正式 seed，除非确认 CPU/内存足够。

### PPO 一直不完成

- 查看 `failure_reason` 是步数、总时间还是非法动作；
- 用机械 S 形确认场景本身可解；
- 检查 `transfer_overhead` 和 `early_end_count`；
- 先延长到 500k，再判断是否需要课程学习；
- 若引入 A→D 课程，必须另建冻结 Train 版本，Validation/Test 仍保持 stage D，且记录变化。

### 失血一直为 0

- 检查场景是否存在血管面积不少于 1 格；
- 检查周围组织环是否有机会在开放期完成；
- 检查 `exposed_component_ids` 和 replay 中的 `expose_vessel`；
- 检查开放期是否真的持续了方向动作；
- 不要靠调大血液权重制造不存在的信号。

### OOM 或多进程失败

- 改为 `--n-envs 1 --device cpu` 确认单进程可运行；
- 再逐步升到 4 或 8；
- 保持 `n_envs * n_steps` 能被 `batch_size` 整除；
- 新开输出目录，不要覆盖半成品 run。

## 14. 代理 Agent 的停止条件

遇到以下任一情况应停止并向负责人报告，不得自行扩大任务范围：

- 单元测试失败且原因不明确；
- 冻结场景或尺度文件被意外修改；
- baseline 无法 100% 完成；
- 所有正式 seed 均无法通过完成率或 105% 时间门槛；
- 需要修改临床语义、出血公式、大血管阈值或 15/5 规则；
- Test 已被提前查看或用于调参；
- 发现模拟器把血管长度误当作横截面积；
- 训练出现 NaN、不可复现或明显奖励投机而现有指标无法解释。

满足全部验收门槛后，代理 Agent 应交付原始文件、配对统计和一页结论摘要；结论只能表述为
“在该二维模拟假设下，RL 相对机械 S 形的时间与期望模拟失血表现”，不能推广为真实临床
有效性或安全性。
