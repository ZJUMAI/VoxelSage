# Manuscript Revision Notes

## Updated sources

- Summer-research thesis: `暑研论文/thesis.tex`
- arXiv technical report: `arXiv_tech_report/main.tex`

## Claims now supported

1. The final model is a behavior-cloned macro-target ranker, not a PPO policy.
2. Clamp release is automatic and fixed within each condition; the final model
   does not choose END or early unclamping.
3. Replication-256 supports C4 versus C0 time improvement and zero simulated-
   blood budget overruns under the frozen main condition.
4. C4 outperforms the prespecified myopic C2, but the corrected teacher C3 is
   still faster; the paper must not claim superiority over every deterministic
   method.
5. The exact policy-external shield is necessary: C5 had 18/256 overruns.
6. v10.7.1 replaces the invalid v10.7 S1--S4 decisions. All four perturbations
   pass with condition-specific baselines, yielding simulator-level
   `robustness GO`.

## Figures added to both manuscripts

- `learned_ordering_pipeline.pdf`
- `replication_controller_effects.pdf`
- `replication_paired_results.pdf`
- `shield_ablation.pdf`
- `corrected_sensitivity.pdf`

All newly generated figures use English labels. Times New Roman was requested,
but no Times New Roman font file is installed in the execution environment;
the generated files therefore use Liberation Serif, a metrically compatible
Times-style substitute. Replace or regenerate after supplying a licensed Times
New Roman font if exact font identity is mandatory.

## Remaining publication limits

- Simulated blood loss, the 16.0705 mL margin, and tension/mechanics proxies are
  not clinically calibrated.
- The exact shield guarantees only the frozen planar simulator budget and is
  computationally expensive.
- Online CT engineering validation and synthetic sequencing experiments are
  separate evidence streams.
- The local environment has no TeX engine. Static label, reference, brace,
  environment, image-existence, and whitespace checks pass, but final PDF
  layout must be compiled in Overleaf or another complete TeX environment.
