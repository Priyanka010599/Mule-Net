# Lane E: GNN embeddings + XGBoost, inductive

**Offline only.** No deploy, no Vertex AI registration, no Gemini calls,
no change to any shipped model (`ml/out/models/`), claim, or UI text.
Synthetic data from our own simulator -- the graph structure the GNN
learns from is the structure *we planted*, not an independently-observed
real-world transfer graph. See "What this does NOT show" below.

## Why this lane exists

`ml/gnn.py`'s existing GraphSAGE experiment is **transductive**: it
builds one graph over all 30,000 accounts -- train and test rows
together -- and masks only the *loss* to train rows. Every training
epoch still runs message passing that aggregates held-out nodes'
features and structure into their train-fold neighbours' embeddings,
and a held-out node's own embedding is built from its real neighbours
using weights simultaneously being fit against those neighbours'
labels. `ml/gnn.py`'s own docstring calls this a declared, uncorrected
asymmetry in the GNN's favour relative to `ml.evaluate`'s XGBoost, which
never instantiates a test row until after `model.fit` on train rows
only.

This lane asks two questions the shipped `ml/gnn.py` experiment cannot
answer on its own: **(1) how much of its reported performance is that
leak**, and **(2) does graph structure add anything once trained
honestly inductively, on top of or in place of the shipped XGBoost
detector.**

## Method

**Inductive GraphSAGE, per fold:** build `edge_index_train` from ONLY
the current fold's train accounts (both endpoints must be in-fold) --
test accounts and their edges are completely absent from the graph the
model trains on. Train for the same 150 epochs, same hyperparameters,
as `ml.gnn.GraphSAGE` (2-layer SAGEConv, hidden=64, dropout=0.3,
lr=0.01, weight_decay=5e-4) -- the architecture and training loop are
reused unmodified; only *which graph* each phase sees is new. After
training, freeze the weights, build a SEPARATE full graph (all 30,000
accounts, same cutoff-admission rule as `ml.gnn.build_edge_index`), and
run one forward pass in eval mode to get a score and a 64-dim embedding
for every account, test accounts included. `build_edge_index` itself
needed no change: its `accounts` argument already drives which node
positions exist, so passing a train-only index already gives a
train-only subgraph for free.

Same grouped 5-fold CV over `ring_id` as `ml.evaluate.fold_assignment`
(ring groups never straddle folds), same 3 seeds (42, 43, 44), same two
horizons as the ablation lane (T=5m primary / T=180m reference), same
`ml.evaluate.XGB_PARAMS` for every XGBoost model. The 1%-FPR threshold
is fixed on train-fold scores only, per fold, exactly as
`ml.evaluate.oof_scores` already does.

**A mid-run bug, caught and fixed before any number below was trusted:**
the first implementation trained 5 independent GraphSAGE models (one per
fold) and stitched their embeddings into one shared 64-column feature
matrix the way `ml.evaluate.oof_scores` correctly stitches *scalar*
out-of-fold scores. That is wrong for raw embedding vectors: each
fold's GraphSAGE starts from a fresh random initialisation, so
"dimension 23" in fold 0's model encodes something unrelated to
"dimension 23" in fold 1's model -- there is no shared basis across
independently-trained models. Stitching them made the embedding columns
look like structured noise to the downstream XGBoost, and PR-AUC for
"XGBoost + GNN" collapsed to ~0.03 against XGBoost-alone's ~0.62 on the
very first seed -- a 20x drop, not a "no gain" result. Caught
immediately (that size of drop does not look like a real finding), and
fixed: every GNN+XGBoost combination below pairs fold k's XGBoost
fit/predict with fold k's *own* embeddings only, never mixing embedding
spaces from different models within one XGBoost training set. Scalar
GNN scores (used alone, in the leak check, and in score averaging) were
never affected -- those were correctly OOF-stitched from the start,
since a probability is comparable across models in a way a raw latent
vector is not.

**Variants, all cross-bank scope (`cb_`), 3 seeds x 2 horizons:**

| name | what it is |
|---|---|
| `xgboost_model_a` | Shipped Model A recipe, all 16 features, reproduced here as the gate |
| `gnn_inductive_16feat` | Inductive GraphSAGE alone, same 16 features |
| `xgboost_model_a_plus_gnn` | XGBoost on the 16 features + that GNN's 64-dim embeddings (fold-matched) |
| `xgboost_transfer_only` | Model C's 11-feature transfer-only set alone (no identity, no account age) |
| `gnn_inductive_11feat_transfer_only` | Inductive GraphSAGE trained on ONLY those 11 features (no identity/age ever reaches the GNN) |
| `xgboost_transfer_only_plus_gnn` | XGBoost on the 11 features + that GNN's embeddings (fold-matched) |
| `score_average_xgboost_and_gnn` | Simple 50/50 average of `xgboost_model_a`'s and `gnn_inductive_16feat`'s own scores |
| `gnn_transductive_16feat_leak_check` | `ml.gnn.run_scope`, UNMODIFIED, same frame/fold/seed -- the leak check's other side |

**Gate:** `xgboost_model_a` at T=5m must reproduce
`ml/out/models/model_a_metadata.json`'s shipped ~0.622 PR-AUC / ~1.97
false alerts per catch. Result: **PR-AUC 0.6212 (shipped 0.6220, diff
-0.0008), false alerts/catch 2.020 (shipped 1.968) -- GATE PASSED**,
each individual seed also passing on its own (0.6183/0.6267/0.6188
PR-AUC; see `run_42.log`/`run_43.log`/`run_44.log`, not committed).
The -0.0008 PR-AUC difference from the shipped/ablation number is not a
methodology difference -- `run_scope`, `XGB_PARAMS`, `fold_assignment`,
and the seeds are all reused unmodified, identical to how
`ml/experiments/ablation/run_ablation.py` calls the same functions. The
likely cause is ordinary cross-machine XGBoost non-determinism: the
`hist` tree method's histogram construction is parallelised across
`n_jobs=4` threads, and floating-point summation is not associative
under reordering, so the same `random_state` on different hardware (this
run: a temporary GCE VM) can land a few trees' splits fractionally
differently. A diff this small (0.13% relative) is consistent with that,
not with a bug -- the gate's own 0.02 tolerance exists for exactly this.

## Results (3 seeds, mean +/- std)

### T=5m (primary, early-detection regime)

| variant | PR-AUC | recall@1% FPR | false alerts/catch @80% recall |
|---|---|---|---|
| `xgboost_model_a` | 0.6212 +/- 0.0038 | 0.6147 +/- 0.0043 | 2.020 +/- 0.056 |
| `gnn_inductive_16feat` | 0.6442 +/- 0.0038 | 0.6498 +/- 0.0140 | 1.832 +/- 0.081 |
| `xgboost_model_a_plus_gnn` | 0.6552 +/- 0.0074 | 0.6512 +/- 0.0078 | 2.039 +/- 0.126 |
| `score_average_xgboost_and_gnn` | **0.6654 +/- 0.0041** | 0.6428 +/- 0.0118 | **1.325 +/- 0.040** |
| `xgboost_transfer_only` (11 feat, no identity/age) | 0.0714 +/- 0.0010 | 0.0370 +/- 0.0048 | 15.87 +/- 0.40 |
| `gnn_inductive_11feat_transfer_only` | 0.1398 +/- 0.0037 | 0.1325 +/- 0.0093 | 13.47 +/- 0.33 |
| `xgboost_transfer_only_plus_gnn` | 0.1185 +/- 0.0006 | 0.1072 +/- 0.0046 | 13.86 +/- 0.38 |
| `gnn_transductive_16feat_leak_check` | 0.6401 +/- 0.0026 | 0.6456 +/- 0.0065 | 1.883 +/- 0.171 |

### T=180m (reference, full-window regime)

| variant | PR-AUC | recall@1% FPR | false alerts/catch @80% recall |
|---|---|---|---|
| `xgboost_model_a` | 0.9876 +/- 0.0012 | 0.9850 +/- 0.0018 | 0.00058 +/- 0.00083 |
| `gnn_inductive_16feat` | 0.9933 +/- 0.0006 | 0.9920 +/- 0.0007 | 0.00058 +/- 0.00083 |
| `xgboost_model_a_plus_gnn` | 0.9907 +/- 0.0001 | 0.9888 +/- 0.0020 | 0.00058 +/- 0.00083 |
| `score_average_xgboost_and_gnn` | 0.9941 +/- 0.0004 | 0.9925 +/- 0.0018 | 0.0 +/- 0.0 |
| `xgboost_transfer_only` (11 feat, no identity/age) | 0.9705 +/- 0.0011 | 0.9658 +/- 0.0018 | 0.00234 +/- 0.00083 |
| `gnn_inductive_11feat_transfer_only` | 0.9891 +/- 0.0005 | 0.9864 +/- 0.0007 | 0.0 +/- 0.0 |
| `xgboost_transfer_only_plus_gnn` | 0.9871 +/- 0.0003 | 0.9860 +/- 0.0011 | 0.00058 +/- 0.00083 |
| `gnn_transductive_16feat_leak_check` | 0.9934 +/- 0.0008 | 0.9930 +/- 0.0011 | 0.00058 +/- 0.00083 |

## What this shows

**The shipped detector is unchanged by this lane.** Nothing here is
registered, deployed, or wired into `agents/subset_upload.py` or any
other live path -- see "What this does NOT show" below.

- **The leak check found the inductive and transductive GNN within
  0.004 PR-AUC of each other at both horizons (3 seeds, no significance
  test): T=5m 0.6442 (inductive) vs 0.6401 (transductive), T=180m 0.9933
  vs 0.9934.** The inductive GNN -- the one with test accounts and their
  edges fully absent during training -- scored slightly *higher* than
  the transductive one at T=5m, not lower, so the earlier `ml/gnn.py`
  transductive numbers do not appear to have been inflated by the leak
  that module's own docstring declares. That is a narrower claim than
  "the leak doesn't matter in general": it means the specific gap this
  experiment can measure is small on this synthetic dataset, these 3
  seeds, this architecture -- not that no leak exists under any
  conditions.
- **Averaging XGBoost's and the GNN's scores lifts PR-AUC at T=5m from
  XGBoost's 0.6212 to 0.6654 (+0.044 absolute, ~7% relative), and lowers
  the false-alert rate too: 1.325 false alerts per catch at 80% recall,
  against 2.020 for this run's own XGBoost-alone reproduction (shipped:
  1.968).** Every GNN-involving variant beats `xgboost_model_a`'s 0.6212
  at T=5m -- the GNN by itself reaches 0.6442, concatenating its
  embeddings into XGBoost reaches 0.6552 -- but plain 50/50 score
  averaging, no feature engineering at all, is both the best PR-AUC and
  the best false-alert rate in the table.
- **With zero identity/account-age signal, an inductive GNN trained on
  only the 11 transfer-only features reaches 0.1398 PR-AUC at T=5m,
  compared with 0.0714 for XGBoost on the identical 11 features --
  roughly double, though the absolute PR-AUC remains low (0.14).** The
  GNN is extracting real signal from the graph's shape that the
  tree-based feature set does not capture on its own, even with no
  device/KYC/account-age information anywhere in the model; "roughly
  double a low number is still a low number" is the honest reading, not
  a claim that this makes the 11-feature detector strong in absolute
  terms.
- **Concatenating embeddings into XGBoost is not always the best way to
  use them -- score averaging did better in both feature sets tested.**
  For the 11-feature transfer-only case, `xgboost_transfer_only_plus_gnn`
  (0.1185) lands BELOW the GNN by itself (0.1398) -- both still above
  XGBoost by itself (0.0714), but concatenation gives up some of what
  the GNN found on its own. The same ordering holds for the 16-feature
  case (averaging 0.6654 beat concatenation 0.6552).
- **`xgboost_transfer_only`'s 0.0714 PR-AUC at T=5m closely reproduces
  `ml/experiments/ablation/results_ablation.json`'s
  `drop_all_identity_and_age` entry (~0.071)** despite being computed
  completely independently in this script -- a useful cross-check that
  both harnesses agree.

## What this does NOT show

- Not a claim about MuleNet's production detector or any number in
  `docs/claims.md` -- this is a separate, offline experiment. No model
  here is registered, deployed, or wired into `agents/subset_upload.py`
  or any other live path.
- Synthetic data from our own simulator: the ring structure the GNN
  learns to recognise is the structure our own generator planted, at
  30,000-account / 3-bank scale. Whether a GraphSAGE-style model adds
  the same lift on a real, independently-observed transfer graph is not
  tested here.
- Only 3 seeds. The GNN numbers' spread (std) is visibly larger than
  XGBoost's in several rows (e.g. `gnn_inductive_11feat_transfer_only`'s
  recall@1%FPR std of 0.0093 against `xgboost_transfer_only`'s 0.0048),
  consistent with a less stable training process on this scale of data;
  more seeds would narrow the confidence in "which combination is best"
  without changing the qualitative finding that graph structure helps.
- The GNN's own hyperparameters (150 epochs, hidden=64, lr=0.01, etc.)
  were inherited unmodified from `ml/gnn.py` and never tuned for the
  inductive setting specifically -- a tuned inductive GNN might do
  better or worse than what is reported here.
- `gnn_transductive_16feat_leak_check`'s numbers are NOT the same as
  `ml/experiments/gnn/metrics.json` (the shipped transductive
  experiment's own output): `ml.gnn.run_scope` is called fresh here on
  this script's own frame/fold construction for a fair side-by-side
  comparison, but it is the same function, same hyperparameters, same
  seeds -- any difference from the committed `metrics.json` would come
  from incidental differences in which horizons/strata were computed in
  that run, not from a different model.

## Files

- `run_gnn_xgb.py` -- the script, run with `python ml/experiments/gnn_xgb/run_gnn_xgb.py`
  (optionally `--seed {42,43,44}` to run one seed's work as its own
  process -- the three seeds' work is fully independent, so this lets
  the sweep run in parallel across processes; the combined
  `results_gnn_xgb.json` is written once every seed's checkpoints exist,
  by whichever process finishes last).
- `results_seed{42,43,44}_T{5,180}.json` -- one checkpoint per
  (seed, horizon), written as soon as that pair finishes.
- `results_gnn_xgb.json` -- the aggregated summary (the tables above).
