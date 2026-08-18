# PlanarResectionEnv 环境契约

版本：1。`PlanarResectionEnv` 是二维/2.5D 试验中唯一的状态转换实现；规划器仍是独立的规则基线。本文区分其稳定的 50×50 画布接口与 v3 PPO 使用的局部 7×7 训练适配器，二者不可混用。

## 规范 50×50 画布接口

`reset(seed=None, options=None)` 返回 `(observation, info)`。场景必须含有 `rows`、`cols`、`domain_cells`、`obstacle_cells` 和 `start_cell`；Pilot 场景可用 `starts` 加 `options={"start_index": n}`。起点在 reset 时已经执行第一刀，和 `plan_resection()` 一致。

`step(action)` 返回 Gymnasium 五元组 `(observation, reward, terminated, truncated, info)`。规范动作空间始终是 2500 个画布格：`action` 为 $0\ldots2499$，`row = action // 50`、`col = action % 50`，即 `action = row * 50 + col`。只允许未切、非活跃血管、属于动态前沿的格子；环境会再次验证动作，非法动作会以失败结束 episode，不依赖调用方正确使用 mask。

`action_masks()` 返回长度 2500 的布尔数组，其第 `row * 50 + col` 项标识该画布格能否作为下一刀。动作只选择下一切割格；已有 cut 区内的移动由环境自动记录为 `transfer` 事件，不是独立动作。

### 规范观测

规范观测是 `float32` 的 CHW 张量 `(15, 50, 50)`，固定通道顺序为：

`domain, cut, vessel, released_vessel, frontier, current_position, start, thickness, normal_tension, shear_tension, front_tension, organ_energy, vessel_strain, tip, valid_cell_mask`。

连续力学量使用固定尺度归一化，而不是按单 episode 最大值归一化：`thickness` 除以 `DEFAULT_MECHANICS["thickness_max"]`，`normal_tension`、`shear_tension`、`front_tension` 和 `organ_energy` 除以 1，`vessel_strain` 除以 `DEFAULT_MECHANICS["tear_vessel_strain"]`；所有结果上限裁剪到 10。规范动作与观测画布仍固定为 50×50，即使实际场景更小。

## v3 局部训练适配器（7×7）

`LocalGridScenarioPoolEnv` 不改变底层 `PlanarResectionEnv` 语义。它仅接受每个 `rows`、`cols` 都等于 `grid_size` 的方形场景；v3 的已训练及评估范围为 `grid_size=7`，不是对任意尺寸、临床影像或 50×50 全画布策略的宣称。

适配器先取规范观测的左上 `grid_size × grid_size` 区域，再在末尾追加两个 `float32` 坐标通道：`row_coordinate` 与 `column_coordinate`。因此 v3 的 PPO 输入是 `(17, 7, 7)`，通道顺序为上述 15 个规范通道，再加 `row_coordinate, column_coordinate`。两坐标通道分别在行和列方向以 `linspace(0.0, 1.0, 7)` 广播；它们是显式位置特征，而非新的力学状态。

局部动作空间为 49。对局部动作 `local_action`，令 `row = local_action // 7`、`col = local_action % 7`，底层画布动作是 `row * 50 + col`；反向地，训练脚本从画布动作的 `(row, col)` 写为 `row * 7 + col`。局部 `action_masks()` 是规范 50×50 mask 的左上 7×7 裁剪并展平，长度为 49。该映射只在场景确为 7×7 时有效。

## 事件、终止和复现

每次 `step` 在 `info["events"]` 中返回本步自动 `transfer`、`cut` 与 `release` 事件序列。`episode_replay()` 和 `write_replay(path)` 输出场景、力学和奖励配置、事件及终止原因，供逐步复现。

成功条件为切除全部 domain；无前沿、无法在 cut 区中找到 transfer 路径或非法动作时 `terminated=True`；超过 `max_steps` 时 `truncated=True`。默认最大步数等于 domain 格数。

## 奖励范围与外部评估

环境默认 `DEFAULT_REWARD` 系数为：`transfer_cost=1.0`、`lookahead_transfer_cost=0.0`、`tension_cost=0.10`、`organ_energy_cost=0.10`、`vessel_strain_cost=1.0`、`completion_bonus=25.0`、`failure_penalty=25.0`、`invalid_action_penalty=25.0`。每步从奖励中扣除 transfer 数、最大前沿张力、峰值器官能量、峰值血管应变及可选的下一前沿 transfer 前瞻项的加权成本；成功加 `completion_bonus`，失败或非法动作扣相应罚项。

当前 v3 训练器在 `LocalGridScenarioPoolEnv` 上将 `transfer_cost` 覆盖为 `2.0`，并显式使用 `lookahead_transfer_cost=0.0`；其余六项沿用上述默认值。因此当前 v3 的完整有效权重为：`transfer_cost=2.0`、`lookahead_transfer_cost=0.0`、`tension_cost=0.10`、`organ_energy_cost=0.10`、`vessel_strain_cost=1.0`、`completion_bonus=25.0`、`failure_penalty=25.0`、`invalid_action_penalty=25.0`。PPO 训练还启用 reward normalization；训练 reward 只是一组可审计、未校准的研究用力学代理，不能作为外部性能指标或临床结论。

候选策略应以冻结的泛化评估衡量：120 个 Test 与 80 个 Stress 场景，检查完成率、合法动作率、release 规则及 transfer overhead（而非只看累计 reward）。这些实验和 v3 策略的可解释范围仅为 7×7 人工构造平面网格；它们不验证真实组织力学、手术安全性或临床疗效。
