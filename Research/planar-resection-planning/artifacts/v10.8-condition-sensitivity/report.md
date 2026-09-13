# v10.8 同条件预算敏感性纠正

原 v10.8 入口把 S0 的蛇形出血量用于全部条件，导致 S1/S2 的部分场景预算过紧、S3/S4 的部分场景预算过宽。此次按各条件重算 C0，再加同一个固定裕量；不修改模型、候选、安全判据或旧实验记录。

已核对 1920 个分片，每个条件、控制器 128 个相同场景。固定裕量 16.070543478261 mL。

完成与实际超预算分别统计；失败或不可行的场景也会检查已经发生的血量是否超预算。配对时间和血量只比较双方均完成的场景。以下是描述统计，不作显著性或临床有效性判断。

| 条件 | 控制器 | 完成 | 失败 | 不可行 | 实际超预算 | 安全不变量事件 |
|---|---|---:|---:|---:|---:|---:|
| S0 | C0 | 128 | 0 | 0 | 0 | 0 |
| S0 | C4L | 128 | 0 | 0 | 0 | 0 |
| S0 | C5 | 128 | 0 | 0 | 14 | 0 |
| S1 | C0 | 128 | 0 | 0 | 0 | 0 |
| S1 | C4L | 128 | 0 | 0 | 0 | 0 |
| S1 | C5 | 128 | 0 | 0 | 10 | 0 |
| S2 | C0 | 128 | 0 | 0 | 0 | 0 |
| S2 | C4L | 128 | 0 | 0 | 0 | 0 |
| S2 | C5 | 128 | 0 | 0 | 12 | 0 |
| S3 | C0 | 128 | 0 | 0 | 0 | 0 |
| S3 | C4L | 128 | 0 | 0 | 0 | 0 |
| S3 | C5 | 128 | 0 | 0 | 11 | 0 |
| S4 | C0 | 128 | 0 | 0 | 0 | 0 |
| S4 | C4L | 128 | 0 | 0 | 0 | 0 |
| S4 | C5 | 128 | 0 | 0 | 8 | 0 |

| 条件 | 配对差值方向 | 配对完成数 | 排除失败数 | 平均 ΔT (min) | 平均 ΔB (mL) |
|---|---|---:|---:|---:|---:|
| S0 | C4L_minus_C0 | 128 | 0 | -0.891848 | -106.275408 |
| S0 | C5_minus_C0 | 128 | 0 | -0.982065 | -94.343071 |
| S0 | C4L_minus_C5 | 128 | 0 | 0.090217 | -11.932337 |
| S1 | C4L_minus_C0 | 128 | 0 | -0.938043 | -152.808288 |
| S1 | C5_minus_C0 | 128 | 0 | -0.989130 | -137.157201 |
| S1 | C4L_minus_C5 | 128 | 0 | 0.051087 | -15.651087 |
| S2 | C4L_minus_C0 | 128 | 0 | -0.940217 | -192.582745 |
| S2 | C5_minus_C0 | 128 | 0 | -1.008696 | -181.119293 |
| S2 | C4L_minus_C5 | 128 | 0 | 0.068478 | -11.463451 |
| S3 | C4L_minus_C0 | 128 | 0 | -0.892935 | -50.081861 |
| S3 | C5_minus_C0 | 128 | 0 | -0.977717 | -44.762432 |
| S3 | C4L_minus_C5 | 128 | 0 | 0.084783 | -5.319429 |
| S4 | C4L_minus_C0 | 128 | 0 | -0.914674 | -24.846909 |
| S4 | C5_minus_C0 | 128 | 0 | -0.978804 | -22.639912 |
| S4 | C4L_minus_C5 | 128 | 0 | 0.064130 | -2.206997 |

来源与复现：

- 源目录：`D:\26SummerCamp\VoxelSage\Research\planar-resection-planning\results\clinical_window_v10_8_lazy_shield\sensitivity_condition_budget_20260913`
- Manifest SHA256：`b35842638025cc1320e10372a65a9901688bf4106f24a2052cd93341a489595f`
- Runner SHA256：`325aa6d2cdcdd1cd810558ca9766deb1485ef279aa87f31a7ff3461d672685ab`
- Checkpoint SHA256：`c07904502d6b71a74484adb1c27971c77cdf6a61bb20b04f1f39d786d61a70be`
- 汇总脚本 SHA256：`8164d49ce01f3bec1bfd9ed6885da5238778132bbea52e08d27ba7f4a8115061`
- 每个条件的基线哈希、各控制器分片集合哈希、场景清单、配对分布与裕量/平均基线血量比见 `summary.json`。

评估命令须使用已安装 PyTorch 的研究环境；汇总脚本只需 Python 标准库。

```text
C:\Users\Bingh\miniconda3\envs\v108\python.exe D:\26SummerCamp\VoxelSage\Research\planar-resection-planning\evaluate_v108_sensitivity.py --controllers C0,C4L,C5 --conditions S0,S1,S2,S3,S4 --limit 128 --offset 0 --scene-workers 20 --margin 16.07054347826075 --split-file D:\26SummerCamp\VoxelSage\Research\planar-resection-planning\results\clinical_window_v10_8_lazy_shield\frozen\split_lazy_replication.json --baseline-file D:\26SummerCamp\VoxelSage\Research\planar-resection-planning\results\clinical_window_v10_8_lazy_shield\frozen\baseline_lazy_replication.json --checkpoint D:\26SummerCamp\VoxelSage\Research\planar-resection-planning\results\clinical_window_v10_6_shielded_learning\runs\bc\config_05_seed_2026081603\epoch_05.pt --output-root D:\26SummerCamp\VoxelSage\Research\planar-resection-planning\results\clinical_window_v10_8_lazy_shield\sensitivity_condition_budget_20260913
D:\26SummerCamp\VoxelSage\.venv\Scripts\python.exe D:\26SummerCamp\VoxelSage\Research\planar-resection-planning\report_v108_condition_sensitivity.py --source D:\26SummerCamp\VoxelSage\Research\planar-resection-planning\results\clinical_window_v10_8_lazy_shield\sensitivity_condition_budget_20260913 --output D:\26SummerCamp\VoxelSage\Research\planar-resection-planning\artifacts\v10.8-condition-sensitivity --split-file D:\26SummerCamp\VoxelSage\Research\planar-resection-planning\results\clinical_window_v10_8_lazy_shield\frozen\split_lazy_replication.json --reference-baseline-file D:\26SummerCamp\VoxelSage\Research\planar-resection-planning\results\clinical_window_v10_8_lazy_shield\frozen\baseline_lazy_replication.json --checkpoint D:\26SummerCamp\VoxelSage\Research\planar-resection-planning\results\clinical_window_v10_6_shielded_learning\runs\bc\config_05_seed_2026081603\epoch_05.pt --expected-count 128 --evaluation-python C:\Users\Bingh\miniconda3\envs\v108\python.exe
```
