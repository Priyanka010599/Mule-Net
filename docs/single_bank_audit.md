# Audit: do the single-bank bank tiles show cross-bank output?

**Yes.** Every bank tile's scores, alert order, queue depth and top factors
come from the published **cross-bank** model, restricted to that bank's
accounts. The tile copy says otherwise ("runs only its own velocity rule").

| what the tile shows | where it is chosen | scope |
|---|---|---|
| alert rows, `score` | `ui/build_snapshot.py:1581-1584` ranks `tables["scores"]` (`ml/out/scores.parquet`) per bank, descending | cross-bank |
| `top_factors` | `ui/build_snapshot.py:1592` copies `scores.top_factors`, built at `ml/publish.py:241-258` (`build_scores`): velocity-rule line + `evidence_flags` strings such as "KYC ID number reused at another bank" | cross-bank evidence |
| `queue` (depth to 80% recall) | `ui/build_snapshot.py:1617`, `_queue_at_recall` (`:267`) over the same ranking | cross-bank |
| model label | `ui/build_snapshot.py:1618` `scores["model_version"]` = `mulenet-xgboost-cb-oof-v1`; rendered in `ui/static/app.js` `bankTile` (`bank.score_model_version`) | cross-bank |
| why `scores.parquet` is cross-bank | `ml/publish.py:616-620`: `columns = feature_columns("cb_")`, `oof_scores_and_contributions(...)`, seed 42; manifest `settings.feature_scope = "cb_ (cross-bank)"` | cross-bank |
| copy "runs only its own velocity rule" | `ui/templates/index.html:192-193` | contradicted by the above |

`ui/datasource.py:banks()` (`:408`) only passes the snapshot through into
`ui/models.py:BankTile/AlertRow`; it does not rescore. The snapshot builder
even admits it: `queue_caveat` (`ui/build_snapshot.py:1619-1625`) says
"ml/ has not published a single-bank-only score column, so this queue is not
what this bank would see alone". That caveat is shown in `#bank-caveat`, but
the lede above the tiles and the "Sees its own N customers. Nothing else."
head say the opposite.

Where `cb_` reaches a tile: the **score** (the model was fitted on `cb_`
columns, including `cb_distinct_counterparty_banks`, `cb_shared_device_count`
and cross-bank document reuse), the **ordering and queue depth**, and the
**top-factor strings** (cross-bank identity evidence). Nothing in the tile
uses an `sb_` feature.

Existing single-bank OOF predictions: `ml/early.py` `write_oof` writes
`single_bank` OOF only at its early cutoffs
(`ml/experiments/early_detection/oof/`), not at full window, and
`ml.evaluate.run_scope` discards them (`strip_scores`). So a full-window
single-bank score had to be produced; see `docs/single_bank_view.md` and
`ml/bank_queues.py`.
