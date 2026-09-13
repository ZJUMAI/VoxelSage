"""Compatibility entry point for condition-specific v10.8 sensitivity budgets.

Use the shared runner so this entry point cannot silently reuse the S0
baseline for S1--S4. Outputs default to a fresh corrected shard root.
"""
from evaluate_v108_sensitivity import SENSITIVITY_CONDITIONS, main


if __name__ == "__main__":
    raise SystemExit(main())
