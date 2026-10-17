# `bank_alert_queues`: the single-bank alert queue table

`ml/out/bank_alert_queues.parquet`, registered in `ml/out/manifest.json`
under `tables.bank_alert_queues`. Why it exists: the bank tiles currently
show cross-bank output (see `docs/single_bank_audit.md`). This table is what
a bank would see running **single-bank (`sb_`) features alone**.

Additive. No existing table, metric, threshold or `docs/claims.md` entry is
touched; the five contract tables are byte-identical before and after.

## Provenance

- Model: XGBoost, `ml.evaluate.XGB_PARAMS` unchanged, grouped 5-fold CV
  (`ml.evaluate.fold_assignment`, whole rings per fold), seed 42, target
  `is_passthrough_mule`. The same recipe as the cross-bank `scores.parquet`
  except the column set (`sb_*` only). Scores are out-of-fold.
- Full-window single-bank OOF predictions did not exist (`ml.early` writes
  them only at early cutoffs), so `ml/bank_queues.py` fits them.
- `model_version = mulenet-xgboost-sb-oof-v1`. Hashed account ids only.
- Regenerate: `python -m ml.publish_all` (new last step), or
  `python -m ml.publish_all --only bank_alert_queues` to write just this
  table and add its manifest entry. Deterministic: two runs give identical bytes.

## Columns (one row per account, 30,000 rows, 10,000 per bank)

| column | type | meaning |
|---|---|---|
| `hashed_account_id` | str | unique key |
| `bank_id` | str | `BANK_A` / `BANK_B` / `BANK_C` |
| `score` | float64 in [0,1] | single-bank out-of-fold P(mule) |
| `model_version` | str | `mulenet-xgboost-sb-oof-v1` |
| `scored_at` | timestamp (UTC) | observation instant (full window), as in `scores` |
| `top_factors` | list of struct | 4 entries, largest absolute TreeSHAP first; each `{description: str, contribution: float (signed; + raises risk), feature_name: str}`. Same shape as `account_explanations.top_factors`. `feature_name` always starts `sb_`; render `description`, not `feature_name` |
| `rank_in_bank` | int | 1 = highest score in that bank (stable sort, ties by input order) |
| `queue_recall_target` | float | 0.8 |
| `queue_depth` | int | accounts to review, from rank 1, to reach 80% recall in this bank. Same value on every row of the bank |
| `queue_true_positives` | int | mules caught at that depth. **Label-derived** |
| `queue_false_positives` | int | `queue_depth - queue_true_positives`. **Label-derived** |
| `false_alerts_per_catch` | float | `fp / tp`, rounded to 2 dp. **Label-derived** |
| `mules_in_bank` | int | positives in the bank. **Label-derived** |
| `reached_recall` | bool | whether 80% was reached. **Label-derived** |

Queue rule is the tile's (`ui/build_snapshot._queue_at_recall`): needed =
`max(1, round(0.8 * positives))`, positives = `ground_truth.is_mule` (mule +
collector + source, 899). The model target is the narrower pass-through label
(712), as for the cross-bank model.

The label-derived columns come from simulator ground truth. The UI withholds
those behind its reveal toggle, so a consumer must do the same
(`labels_revealed`); `score`, `rank_in_bank`, `top_factors` are label-free.

## Numbers (queue to 80% recall, seed 42)

| bank | mules | tile today (cross-bank `cb_` model) | this table (single-bank `sb_` model) |
|---|---|---|---|
| BANK_A | 307 | depth 263 (246 mules, 17 not) | depth 397 (246 mules, 151 not) |
| BANK_B | 323 | depth 258 (258 mules, 0 not) | depth 343 (258 mules, 85 not) |
| BANK_C | 269 | depth 236 (215 mules, 21 not) | depth 334 (215 mules, 119 not) |

Single-bank queues are 1.3x to 1.5x deeper with 5x or more false alerts. The
tiles currently understate the single-bank burden. Full-window numbers are
ceiling-bound (see the `queue_caveat`); the early-horizon comparison on tab 3
is the more honest one.

## Follow-up (not wired here; `ui/` and `agents/` are not mine)

To make the tiles truthful the UI should read this table instead of
`scores`: in `ui/build_snapshot.py` rank by `score`/`rank_in_bank`, take
alert rows from `hashed_account_id`, `bank_id`, `score`, `top_factors[].description`
(first as the "top factor"), `queue_*`, `false_alerts_per_catch`,
`mules_in_bank`, `reached_recall` for the queue block, and
`model_version` for the label (replacing `score_model_version`); drop the
`queue_caveat` wording that says no single-bank column exists; and fix the
`ui/templates/index.html:192-193` copy. `AlertRow.top_factors` is `list[str]`,
so map `top_factors[*].description`. The flagged-by-velocity-rule line that
tiles show today is not in this table.
