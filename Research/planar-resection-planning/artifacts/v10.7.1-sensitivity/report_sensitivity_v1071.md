# v10.7.1 Condition-specific-baseline Sensitivity Correction

## Audit conclusion

- Robustness classification: **robustness GO**.
- Perturbation conditions passed: **4/4**.
- Every condition used its own frozen C0 baseline and the unchanged 16.0705 mL margin.
- C0 paired deltas are exactly zero by construction; C2/C4 used the frozen v10.6 checkpoint.
- No model training, hyperparameter tuning, margin change, or v10.7 main-result recomputation occurred.
- Tail risk uses the upper (worst) 10%, correcting the old lower-tail implementation.
- Figure font requested: Times New Roman; renderer used **Liberation Serif**.

## Condition results

| Condition | C4-C0 time, min (95% CI) | C4-C2 time, min (95% CI) | mean ΔB, mL | max ΔB, mL | Gate |
|---|---:|---:|---:|---:|:---:|
| S0 | -0.964 [-1.063, -0.868] | -3.755 [-4.251, -3.264] | -120.633 | 0.000 | PASS |
| S1 | -0.963 [-1.078, -0.859] | -4.342 [-4.854, -3.849] | -166.843 | 0.000 | PASS |
| S2 | -0.927 [-1.027, -0.833] | -4.274 [-4.846, -3.732] | -176.010 | 8.278 | PASS |
| S3 | -0.959 [-1.056, -0.863] | -3.892 [-4.390, -3.401] | -60.300 | 0.000 | PASS |
| S4 | -0.963 [-1.057, -0.870] | -4.145 [-4.640, -3.662] | -29.406 | 4.139 | PASS |

## Interpretation

This supplement replaces the invalid v10.7 sensitivity paragraph only. The independent Replication-256 result remains unchanged. Robustness claims must follow the classification above and must not reuse the original S1-S4 decisions.
