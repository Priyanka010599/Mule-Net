# Red-team experiments

`metrics.json` -- `ml/redteam.py`'s own output: Gemini designs evasion
patterns against the velocity rule's five published clauses, and the real
pipeline (`ml.features`, `ml.rule`, `ml.stage2`) scores each one. Generate
with `python -m ml.redteam`.

`model_comparison.json` -- the same 3 cached strategies replayed against
Model A and Model B (`ml.model_runtime`) as well, not only the velocity
rule. Generate with `python -m ml.redteam_vs_models`.

## Caveats

- **Isolated graph.** Each strategy's feature table holds only that
  strategy's own synthetic accounts and transfers -- no background noise,
  no other rings. A detector tuned against a busy real graph may behave
  differently in production than it does here.
- **The attacker knows the thresholds.** Gemini is told the velocity
  rule's five clauses verbatim and asked to beat them. Model A/B were
  never targeted directly, but they are trained on the same `cb_`
  features those clauses read, so a pattern built to clear one clause
  tends to clear the model's learned boundary too -- this is not
  independent evidence that the models resist evasion.
- **Replayed, not regenerated.** `model_comparison.json` scores the same
  3 patterns `ml/redteam.py` already produced; it is not a fresh
  adversarial search aimed at Model A/B specifically. A strategy that
  targets Model A/B's actual decision boundary (rather than the rule's
  clauses) could look very different.
