# Evidence ledger

A verifiable record of every model, experiment, and branch in this
repository. Every fact below was read directly from `git` or from the
files it names, on `main` at commit `875724b924bf72740eb4332e08e63e8889440624` unless a specific branch
is named. No hash, size, or date is quoted from memory — anyone can
re-run the command shown next to a fact and get the same value. Where
a fact could not be established this way, this file says **not found**
rather than guessing.

## 1. Models

### Model A — 16 features, identity + transfer

| field | value |
|---|---|
| file | `ml/out/models/model_a.ubj` |
| size | 625,794 bytes |
| sha256 | `0816c4b8b70452b37b03aaf3f4e2dd152f4e49762e6dca753479039ccf718478` |
| metadata file | `ml/out/models/model_a_metadata.json` (2,205 bytes, sha256 `b7bbbced47cdd63bff4b8d2489e80cd527f89e85af80f74adfad950423751c38`) |
| features (16, from metadata `feature_columns`) | `cb_in_degree, cb_out_degree, cb_unique_in_counterparties, cb_unique_out_counterparties, cb_unique_counterparties, cb_distinct_counterparty_banks, cb_amount_in_total, cb_amount_out_total, cb_passthrough_ratio, cb_passthrough_speed_seconds, cb_round_amount_ratio, cb_account_age_days, cb_device_count, cb_shared_device_count, cb_document_reuse_count, cb_document_reuse_flag` |
| trained on | 100% of 30,000 rows / 712 positives, `trained_at_cutoff` T=5m, seed 42 (per metadata, not re-derived) |
| served by (metadata's `served_by` field) | `"agents/subset.py, ml/coalition.py"` |
| served by (verified against code) | `agents/subset.py` confirmed (`from ml.model_runtime import MODEL_A`, line 88). `ml/coalition.py` does **not** import `ml.model_runtime` at all — it scores via `ml.rule.velocity_decision`, not this model. The metadata's `coalition.py` claim is **stale/unconfirmed by current code**. |
| Vertex Model Registry (from `ml/vertex/register_ab.py`'s `MODELS["a"]`) | `display_name="mulenet-model-a"`, `alias="model-a-16feat-v1"`, region `asia-south1` (`ml/vertex/config.py DEFAULT_LOCATION`) |
| live registration confirmed? | **not found** — no resource name, registry entry, or registration log is committed anywhere in this repository; only the script that would perform it |
| commit that added the file | `74730c70e604980b498704b6d6f4202765cceeb6`, 2026-10-02, "Serve trained XGBoost models live, with the rule as automatic fallback" (on `main`) |

### Model B — 12 features, no identity features (includes account age)

| field | value |
|---|---|
| file | `ml/out/models/model_b.ubj` |
| size | 610,358 bytes |
| sha256 | `9f586204c49104c3e3bc466d127c918c286ec60a77da24a9513a34f2bf2feadc` |
| metadata file | `ml/out/models/model_b_metadata.json` (2,096 bytes, sha256 `3c6472995ae20808b1d1faa604fe61e9a8ac6a904df68e8d739a1ca436e9b59e`) |
| features (12) | `cb_in_degree, cb_out_degree, cb_unique_in_counterparties, cb_unique_out_counterparties, cb_unique_counterparties, cb_distinct_counterparty_banks, cb_amount_in_total, cb_amount_out_total, cb_passthrough_ratio, cb_passthrough_speed_seconds, cb_round_amount_ratio, cb_account_age_days` |
| trained on | 100% of 30,000 rows / 712 positives, T=5m cutoff, seed 42 |
| served by (metadata's `served_by` field) | `"agents/subset_upload.py"` |
| served by (verified against code) | **This is stale.** `agents/subset_upload.py` currently imports `MODEL_C` (`from ml.model_runtime import MODEL_C`, line 51), not `MODEL_B` — Model B was replaced on this path (see Model C row and the "Model C upload fix" experiment below). Model B is actually imported by `ml/redteam_vs_models.py` (`from ml.model_runtime import MODEL_A, MODEL_B`, line 31), matching `ml/model_runtime.py`'s own docstring: "Still serves `ml/redteam_vs_models.py`'s own comparison; no longer serves live uploads." |
| Vertex Model Registry | `display_name="mulenet-model-b"`, `alias="model-b-12feat-v1"`, region `asia-south1` |
| live registration confirmed? | not found (same basis as Model A) |
| commit that added the file | `74730c70e604980b498704b6d6f4202765cceeb6`, 2026-10-02 (same commit as Model A) |

### Model C — 11 features, transfer-only minus account age; serves Judge Upload

| field | value |
|---|---|
| file | `ml/out/models/model_c.ubj` |
| size | 571,538 bytes |
| sha256 | `851cf8bfcd97be199db4c880171c344ffd0797ec1b80a95adc72b6941ac8c759` |
| metadata file | `ml/out/models/model_c_metadata.json` (2,598 bytes, sha256 `e0011df8e62b7db4460dfbffbcfd16025e5b7cf87fd653cb29df283264820971`) |
| features (11) | `cb_in_degree, cb_out_degree, cb_unique_in_counterparties, cb_unique_out_counterparties, cb_unique_counterparties, cb_distinct_counterparty_banks, cb_amount_in_total, cb_amount_out_total, cb_passthrough_ratio, cb_passthrough_speed_seconds, cb_round_amount_ratio` — Model B's 12 minus `cb_account_age_days` |
| trained on | 100% of 30,000 rows / 712 positives, T=5m cutoff, seed 42, same `ml.evaluate.XGB_PARAMS` recipe as A/B (reverse-engineered from their metadata, per `ml/train_model_c.py`'s own docstring) |
| served by (metadata) | `"agents/subset_upload.py (replacing model_b for this path)"` |
| served by (verified against code) | **Confirmed accurate.** `agents/subset_upload.py` line 51: `from ml.model_runtime import MODEL_C, load_model, score as model_score`. |
| validated_performance | T=5m PR-AUC 0.0708, T=180m PR-AUC 0.9708 — **not independently re-derived**; copied from `ml/experiments/ablation/results_ablation.json`'s `drop_all_identity_and_age` entry (per `ml/train_model_c.py`'s own docstring and `model_c_metadata.json`'s `trained_on` field) |
| Vertex Model Registry | `display_name="mulenet-model-c"`, `alias="model-c-11feat-v1"`, region `asia-south1` |
| live registration confirmed? | not found |
| commit that added the file | `88377d291846efb20774c98e347cbc7944474407`, 2026-10-05, "D1 follow-up: train and wire Model C, fix wording, extend claims" |
| note on `model_c.bst` | A file `ml/out/models/model_c.bst` exists on disk in this worktree but is **gitignored** (`.gitignore:33`, `ml/out/models/*.bst`) and untracked (`git ls-tree HEAD` returns nothing for it) — a transient Vertex-registration conversion artifact, not a repository file. |

## 2. Experiments

### Ablation (identity-feature ablation on Model A's recipe)

The three primary rows, quoted exactly from `results_ablation.json`'s `summary` (`pr_auc_mean`, mean over 3 seeds):

| config | T=5m | T=180m |
|---|---|---|
| `all16` (shipped Model A) | 0.6220 | 0.9876 |
| `drop_all_identity` (= Model B's 12 features) | 0.2448 | 0.9751 |
| `drop_all_identity_and_age` (= Model C's 11 features) | 0.0708 | 0.9708 |

| field | value |
|---|---|
| branch | `exp/ablation`, tip `6d1f7175c5141accef265cd4bc712c58eba3ee40`, 2026-10-05 |
| merged into main? | **Yes** — `c0c2a131f17610334ea380bf76b16f9535c42bfd`, "Merge exp/ablation: identity-feature ablation on Model A's recipe (D1)", 2026-10-06 |
| pushed? | No remote ref `origin/exp/ablation` exists (`git ls-remote origin` lists only `main` and `exp/adversarial`); moot since its content reached `origin/main` through the merge above |
| directory | `ml/experiments/ablation/` (`README.md`, `run_ablation.py`, `results_ablation.json`, `results_seed{42,43,44}.json`) |
| reproduce | `python ml/experiments/ablation/run_ablation.py` |
| headline | See the cascade table above — matches `docs/claims.md`'s `ablation_t5_drop_all_identity` and `ablation_t5_drop_all_identity_and_age` claims exactly |
| outcome | Confirmatory/diagnostic, not a negative result: dropping identity drops T=5m PR-AUC from 0.6220 to 0.2448, and dropping account age too drops it further to 0.0708 — most of the shipped recipe's early-detection power comes from identity and account-age signals, not the transfer-graph features alone. The gap mostly closes by T=180m (0.9876/0.9751/0.9708, all close together). A fourth entry, `identity_only` (the 4 identity features alone, no transfer-graph features at all) scores 0.4708 at T=5m — matches `docs/claims.md`'s `ablation_t5_identity_only` claim exactly, so it is quoted here rather than dropped; it is higher than the 12-feature transfer-only figure, meaning identity reuse alone outscores the whole transfer-graph feature set at this horizon. |

### IBM public benchmark (IBM AML HI-Small, D2)

| field | value |
|---|---|
| branch (clean) | `exp/ibm-aml-d2-clean`, tip `97bb7e6b5b995de23f7755e112730c82f5105f46`, 2026-10-06 |
| branch (superseded, local-only) | `exp/ibm-aml-d2`, tip `27b8f6a24d24d1f0ff6fa6481d8c36b02b13d367`, 2026-10-05 — **superseded by the `-clean` branch and never merged**; its own history contains `ml/experiments/ibm_aml/split.json` (commit `92985eb`), which the clean branch's history never does (`git log exp/ibm-aml-d2-clean -- ml/experiments/ibm_aml/split.json` returns nothing) |
| merged into main? | `-clean`: **Yes** — `348a89919338b3d1d443bb17ae3c193e74de94a5`, "Merge exp/ibm-aml-d2-clean: IBM AML HI-Small benchmark (D2)", 2026-10-06, plus a follow-up `8e0554ed22950bc8cb65896c40beb14ddf820d62` adding the two IBM claims. `exp/ibm-aml-d2` (old): **No**. |
| pushed? | Neither has an `origin/exp/ibm-aml-d2*` ref. **Publishing it would trigger the licence's share-alike terms, so it stays local.** Confirmed directly: `split.json` (read via `git show exp/ibm-aml-d2:ml/experiments/ibm_aml/split.json`) holds only a method description, date window, account/component counts, and SHA-256 hashes of the raw input CSVs — no account-level IDs — but is kept local anyway, out of caution, per the project's own CDLA-Sharing-1.0 reading. |
| directory | `ml/experiments/ibm_aml/` (`README.md`, `make_split.py`, `run_grouped_cv.py`, `results_grouped.json`, `results_seed{42,43,44}.json`) |
| reproduce | `python ml/experiments/ibm_aml/make_split.py` then `python ml/experiments/ibm_aml/run_grouped_cv.py` — requires the raw IBM dataset locally at the gitignored `ml/experiments/public_benchmark/`, not included in this repository |
| headline | `xgboost_full` PR-AUC 0.0813 mean over 3 seeds — matches `docs/claims.md`'s `ibm_aml_xgboost_pr_auc` claim |
| outcome | Confirmatory sanity check, not a claim of transfer to production: "produces *some* signal (5-9x base rate) on an independent dataset," explicitly "NOT comparable to this dataset's published GNN benchmarks." A second number, `ibm_aml_giant_component_realized_fpr` (0.7189), is a negative finding about one baseline (component-size alone is nearly uninformative at this scale) — not hidden, reported directly in the README and in `docs/claims.md`. |

### N-banks (does the cross-bank advantage hold past 3 banks?)

| field | value |
|---|---|
| branch | **None** — built directly on `main`, commit `715b0a3660d7b06426ed901cde3a7ee415d4e22b`, 2026-10-05, confirmed an ancestor of `main`'s current tip |
| merged into main? | N/A (never left main) |
| pushed? | Yes, as part of `origin/main` |
| directory | `ml/experiments/n_banks/` (`README.md`, `generate_and_sweep.py`, `results.json`, `results_hn.json`, `recall_vs_banks.png`) |
| reproduce | `python -m ml.experiments.n_banks.generate_and_sweep` |
| headline | Ring recall by coalition size at N=6: 1 bank 0.0%, 2 banks 13.5%, 3 banks 33.6%, 4 banks 58.9%, 5 banks 81.7%, 6 banks (all) 100% — **no `docs/claims.md` entry by design** (the README states explicitly: "Never touches... `docs/claims.md`") |
| outcome | Confirmatory: "recall rises monotonically with every bank that joins, reaching 100% once every bank has joined, for N=5 and N=6 exactly as for N=3 — the join effect is not a 3-bank artifact." A secondary, non-replicating observation is reported plainly too: recall at the same *number* of banks joined is lower for larger N (expected, not a contradiction, per the README's own explanation). |

### Red-team retraining (closed as inconclusive)

| field | value |
|---|---|
| branch | `exp/adversarial`, tip `ec8bb31c4ee0902addb0017e708caa336bcabaf6`, 2026-10-05 |
| merged into main? | **No** |
| pushed? | **Yes** — `origin/exp/adversarial` exists and matches the local tip exactly (confirmed via `git ls-remote origin`) |
| directory | `ml/experiments/adversarial/` on that branch (`README.md`, `build_data.py`, `build_controls.py`, `build_stress.py`, `train_and_evaluate.py`, `run_controls.py`, `run_stress.py`, `results.json`, `results_control_{a,b}.json`, `results_stress.json`, `split.json`, `stress_strategies.json`) — **not present on `main`**, read here via `git show exp/adversarial:ml/experiments/adversarial/README.md` |
| reproduce | Not independently confirmed from this ledger's read-only pass — see that branch's own README for its run scripts; this file does not execute them |
| headline | Retrained model's held-out-evasion catch rate: 0% → 100% (3 strategies, both horizons, both confound controls). **But** the same retrained model also flags 94.8% of 9,111 benign synthetic accounts at T=5m (64.8% at T=180m) that were never part of any ring — vs. 0.003%/0.008% before retraining |
| matching `docs/claims.md` entry | **None.** This result is deliberately not in `docs/claims.md`; it is documented in `README.md`'s "Limitations" section (line 328) and in the branch's own `ml/experiments/adversarial/README.md`, and the earlier, narrower evasion-detection baseline (before any retraining) lives in `ml/experiments/redteam/model_comparison.json` on `main`. |
| outcome | **Inconclusive, stated plainly in the source README's own words**: "the retrained model also flagged most benign synthetic accounts, so the result is inconclusive — dominated by a synthetic-data artifact, not evidence of robustness" (quoted from `README.md:328-337`). The branch's own README calls it "close to a smoking gun against the 'effect is real' reading." Real-account PR-AUC also measurably *dropped* after retraining (T=5m 0.6220 → 0.5784, T=180m 0.9876 → 0.9699) — retraining was not free even before the benign-false-alert finding. |

### Model C upload fix (judge-upload `account_age_days` placeholder)

| field | value |
|---|---|
| branch | `exp/model-c-upload`, tip `ba8be696cd44abb1bd62774e2e27a80385b28610`, 2026-10-06 |
| merged into main? | **Yes, by fast-forward** — its commits (`88377d2`, `40ab75b`, `ba8be69`, `80c745d`, `d90a619`, `f9365d0`) appear directly, linearly, in `main`'s own `git log main --oneline`; there is no separate merge commit because none was needed |
| pushed? | No `origin/exp/model-c-upload` ref exists; moot — its content is part of `origin/main` |
| directory | No dedicated `ml/experiments/` subdirectory. Code: `ml/train_model_c.py`, `ml/model_runtime.py` (added `MODEL_C`/`ALL_MODELS`), `agents/subset_upload.py`. Tests: `ml/test_model_c.py`, `agents/test_subset_upload.py`, `infra/test_agents_fetch.py`. Infra: `infra/agents_fetch.py`, `infra/deploy.sh` (the `AGENTS_FETCH` list). |
| reproduce | `python -m ml.train_model_c` (writes `ml/out/models/model_c.ubj`/`model_c_metadata.json`); `python -m pytest ml/test_model_c.py agents/test_subset_upload.py infra/test_agents_fetch.py` for the tests |
| headline | Same number as the ablation experiment's `drop_all_identity_and_age` (0.0708 / 0.9708) — **by design, not independently computed**; `ml/train_model_c.py`'s own docstring: "`validated_performance` is copied from `ml/experiments/ablation/`'s `drop_all_identity_and_age` entry... not re-derived here." |
| matching `docs/claims.md` entry | Reuses `ablation_t5_drop_all_identity_and_age` and `ablation_t180_drop_all_identity_and_age` directly — no separate Model-C-specific claim exists |
| outcome | Fix, not an experiment finding: the judge-upload path previously fed Model B a constant `account_age_days` placeholder (no real account-open date is available from an upload); Model C drops that feature so the placeholder is harmless rather than consequential. `d90a619` also closed a real deploy gap: `infra/deploy.sh`'s `AGENTS_FETCH` list never included Model C's files, caught and fixed with a regression test (`infra/test_agents_fetch.py`) that fails loudly if any future model named in `ml/model_runtime.ALL_MODELS` is missing from the fetch list. |

### GNN + XGBoost, inductive (`exp/gnn-xgb`, Lane E)

| field | value |
|---|---|
| branch | `exp/gnn-xgb`, tip `973031a0bd787db16207f76d39b46e137e27875c`, 2026-10-06 |
| merged into main? | **Yes** — `c17e4ea40e2e07ceadc05287363184ac92b0af8d`, "Merge exp/gnn-xgb: inductive GNN embeddings + XGBoost, leak check (Lane E)", 2026-10-06, merge commit (not fast-forward) |
| pushed? | No `origin/exp/gnn-xgb` ref exists; moot now — its content is part of `main`, which was pushed to `origin/main` at `5aad52a` (after an intervening roadmap-card commit; see below) |
| directory | `ml/experiments/gnn_xgb/` (`README.md`, `run_gnn_xgb.py`, `results_gnn_xgb.json`, `results_seed{42,43,44}_T{5,180}.json`) |
| reproduce | `python ml/experiments/gnn_xgb/run_gnn_xgb.py` (optionally `--seed {42,43,44}` to parallelise the 3 seeds' independent work across processes) |
| headline | T=5m score-averaging of XGBoost and an inductively-trained GNN: PR-AUC 0.6654 mean over 3 seeds, vs. 0.6212 for XGBoost alone (+0.044 absolute, ~7% relative) — matches the `gnn_xgb_t5_score_averaging` claim in `main`'s own `docs/claims.md` (confirmed present post-merge: `ml/claims.py`'s `SOURCES` now includes `gnn_xgb`) |
| outcome | Modest, honestly-measured gain; the shipped detector is unchanged. A companion leak check found the inductive GNN (test accounts and edges fully absent during training) scored within 0.004 PR-AUC of the existing transductive `ml/gnn.py` setup at both horizons — the inductive one slightly *higher*, not lower, at T=5m — so the transductive design's declared leak does not appear to have inflated its numbers in this setup. A mid-run bug (embeddings from 5 independently-trained, unaligned-latent-space GraphSAGE models incorrectly stitched into one feature matrix) was caught and fixed before any number was trusted; see `ml/experiments/gnn_xgb/README.md` for the full account. |

### UI refresh (`exp/ui-refresh`)

| field | value |
|---|---|
| branch | `exp/ui-refresh`, tip `18e1795e55be4911908b9b0405c434148f9a9c56`, built from `5aad52a` in 3 commits: `ac44724` (Phase 1 WIP), `ec93378` (hero polish, README screenshots), `18e1795` (Phase 2: tooltip/panel, tour, count-ups, pointers) |
| merged into main? | **Yes** — `875724b924bf72740eb4332e08e63e8889440624`, "Merge exp/ui-refresh: tab 1 hero (fragments/join animation), shared card styling", 2026-10-06, merge commit (not fast-forward) |
| pushed? | No `origin/exp/ui-refresh` ref existed at the time of this ledger entry's writing locally, but the branch was fetched from `origin/exp/ui-refresh` for this merge (it was pushed there by whoever built it) |
| scope | UI-only, confirmed: diffing the branch against its actual base (`5aad52a`, not `main`'s later tip) touches exactly 41 files, all under `ui/` — `ui/README.md`, `ui/static/{app.js,hero.js,hero.css}`, `ui/templates/index.html`, and 36 files under `ui/screenshots/` (1.80 MB total, under the ~5 MB budget — no trimming needed). No test file, backend, `agents/`, `ml/`, or `infra/` file touched. No new external URL (the one CDN reference in the diff, `cdnjs.cloudflare.com`'s Cytoscape, is pre-existing unchanged context, not new). No secret-looking string found. |
| numbers/claims | Unchanged — confirmed no edit to `productionRoadmapSection()`'s card content; the KPI strip's values are the same `d.accounts`/`d.transfers`/etc. fields, now animated via `countUp()` instead of a static assignment, not altered. One new line was added, explicitly authorized by the UI-refresh spec itself: a footnote stating scores tie at 1.0000 because the velocity rule is binary. |
| labels | Confirmed gated: `hero.js`'s own header states the canvas never reads any ground-truth field at all; the side panel's simulator-truth-label line is read only when `ctxApi.isRevealed()` (the existing Reveal checkbox) is true. |
| what it adds | Tab 1's hero: an animated three-bank network canvas (fragments before "Join the banks", merged after), ring picker, replay, auto-play tour, hover tooltip, click side panel, a KPI count-up on page load, and call-to-action pointers to tabs 3/4/5. The original three bank tiles move into a collapsed "Details" section, unchanged in content. |
| tests | No existing UI test was touched, weakened, or deleted (confirmed: empty diff for any `ui/test*`/`ui/*test*` path between `5aad52a` and `18e1795`). Post-merge: `python -m pytest ui/` → 3 failed / 140 passed / 60 skipped, all 3 failures pre-existing in this worktree's baseline (missing local `ui/out/demo_snapshot.json` / `agents/out/cases.parquet` data, confirmed against the saved baseline failing-test-ID set) — not a regression from this merge. `ml/test_claims.py` + `ml/vertex/test_claims_gate.py` → 112 passed. Full suite (`python -m pytest`) post-merge: 80 failed / 1080 passed / 165 skipped / 10 errors — the exact same counts AND the exact same failing/erroring test-ID set as the `5aad52a` baseline (diffed both directions, empty both ways). Zero regressions from this merge. |

## 3. Worktrees and branches

`git worktree list`:

```
D:/LLM/MuleNet                                   3650222 (detached HEAD)
D:/LLM/MuleNet/.claude/worktrees/exp+adversarial c17e4ea [main]
D:/LLM/MuleNet/.claude/worktrees/lane-e          973031a [exp/gnn-xgb]
```

`git branch -a -vv`:

```
  exp/ablation                   6d1f717 D1: identity-feature ablation on Model A's recipe
  exp/adversarial                ec8bb31 [origin/exp/adversarial] Lane C: benign false-alert rate overturns the held-out-catch verdict
+ exp/gnn-xgb                    973031a (.../lane-e) Add two Lane E claims to docs/claims.md; clarify leak-check/README wording
  exp/ibm-aml-d2                 27b8f6a D2: grouped-CV results on the IBM AML HI-Small public benchmark
  exp/ibm-aml-d2-clean           97bb7e6 D2: grouped-CV results on the IBM AML HI-Small public benchmark
  exp/model-c-upload             ba8be69 D1: add the two IBM AML claims now that the licence is confirmed
* main                           c17e4ea [origin/main: ahead 4] Merge exp/gnn-xgb: inductive GNN embeddings + XGBoost, leak check (Lane E)
  remotes/origin/HEAD            -> origin/main
  remotes/origin/exp/adversarial ec8bb31 Lane C: benign false-alert rate overturns the held-out-catch verdict
  remotes/origin/main            3650222 D2 README: replace the paper-citation TODO with neutral attribution
```

At the time the table above was captured, `main` was 4 commits ahead of `origin/main` (not yet pushed): the EVIDENCE.md commit (`0009718`), the two `exp/gnn-xgb` commits it brought in (`e95b729`, `973031a`), and the merge commit itself (`c17e4ea`). **Since then, `main` (through a 5th commit, `5aad52a`, updating the roadmap cards) has been pushed: `origin/main` now matches local `main` at `5aad52a97b918d31c1537dc7ec3c78b46d9efb12`** (confirmed via `git ls-remote origin main`).

| branch | purpose | status |
|---|---|---|
| `main` | shipped repository state | pushed; matches `origin/main` at `5aad52a` |
| `exp/adversarial` | Lane C, red-team retraining experiment | pushed, unmerged (deliberately — the inconclusive result lives on its own branch, referenced from `README.md`, not merged in) |
| `exp/ablation` | Lane D1, identity-feature ablation | merged (into `main` via `c0c2a13`); not independently pushed (moot — content is in `origin/main`) |
| `exp/ibm-aml-d2` | Lane D2, IBM AML benchmark, first attempt | **kept local; publishing it would trigger the licence's share-alike terms** (superseded by `exp/ibm-aml-d2-clean`) |
| `exp/ibm-aml-d2-clean` | Lane D2, IBM AML benchmark, rebuilt history with no `split.json` in any commit | merged (into `main` via `348a899`/`8e0554e`); not independently pushed (moot — content is in `origin/main`) |
| `exp/model-c-upload` | Lane D1 follow-up, Model C + deploy-gap fix | merged (fast-forward into `main`, no separate merge commit); not independently pushed (moot) |
| `exp/gnn-xgb` | Lane E, inductive GNN + XGBoost | merged into `main` via merge commit `c17e4ea`; not independently pushed (moot — content is in `main`, which is pushed to `origin/main`) |

No IBM data, account IDs, or account-level labels appear anywhere in this section or in any branch's history reachable from `origin/main` or `origin/exp/adversarial` (the only two refs actually on the remote).

## 4. What was NOT shipped

- **GNN + XGBoost combination** (`exp/gnn-xgb`, merged into `main` via `c17e4ea`). Modest, honestly-measured gain (+0.044 PR-AUC absolute at T=5m via score averaging). Merged into `main`'s history and git-blame, but not registered in Vertex, not wired into `agents/subset.py`/`agents/subset_upload.py`, and not deployed. **The shipped detector (Model A/B/C) is unchanged** — merging the experiment's code and results is not the same as shipping a new model.
- **Adversarial retraining as a defence** (`exp/adversarial`). Closed as inconclusive: the apparent 100% held-out-evasion catch was a synthetic-data artifact (the retrained model flagged 94.8% of benign synthetic accounts too), and real-account PR-AUC measurably dropped after retraining. Not merged, not deployed. The shipped models were never retrained on this data.
- **Unsupervised Isolation Forest, slow-drip stress test** (`ml/experiments/slow_drip/`). Every shipped detector is supervised against this dataset's labelled pass-through pattern; this experiment asks whether an unlabelled anomaly detector does better against a designed slow pattern (smaller amounts, hops paced under the velocity rule's speed ceiling, split 2-3 banks) and against the canonical data. At a 1%-per-bank alert budget: on slow-drip cross-bank it catches 60.0% of rings against the velocity rule's 67.5% and XGBoost's 89.2%; single-bank it beats the rule (41.7% vs 10.8%) but still trails XGBoost (85.8%) heavily. **Not evidence the system can catch a genuinely novel fraud pattern** -- this measures one designed pattern, not fraud nobody anticipated. Not merged into the shipped detector, not deployed. Numbers in `docs/claims.md`'s "Slow-drip stress test" section, every one machine-checked by the same claims gate as the rest.
- **Model B's own training/registration code**: `ml/vertex/register_ab.py`'s `register_one()` function can register Models A, B, or C in Vertex Model Registry, but this ledger found **no committed evidence that a live registration call was ever executed and recorded** (no resource name, no registry log, anywhere in any branch's tracked files) — only the script that would do it.
- **`ml/coalition.py` serving Model A**: Model A's own metadata claims `ml/coalition.py` as a server, but the current code shows `ml/coalition.py` uses `ml.rule.velocity_decision` (the rule), not Model A, with no `ml.model_runtime` import at all. Flagged here rather than silently repeated as fact.

## 5. How to verify

```bash
# 1. The whole test suite (offline, free)
python -m pytest

# 2. The claims gate specifically: every number in docs/claims.md still
#    resolves to the metrics file it claims, and the gate catches a
#    sabotaged number
python -m pytest ml/test_claims.py ml/vertex/test_claims_gate.py

# 3. Regenerate docs/claims.md itself and confirm it doesn't change
#    (every number is derived, never hand-written)
python -m ml.claims

# 4. Regenerate one experiment's result from scratch (ablation, ~minutes)
python ml/experiments/ablation/run_ablation.py

# 5. Spot-check one model file's hash against this ledger
sha256sum ml/out/models/model_c.ubj
# expect: 851cf8bfcd97be199db4c880171c344ffd0797ec1b80a95adc72b6941ac8c759
```

**Results when commands 1, 2, 3 and 5 were actually run** (this worktree, 2026-10-06): (1) the full suite's failed/errored tests are a pre-existing set, independent of any commit made while building this ledger — confirmed by running the identical suite at both `3650222` and `5aad52a` and diffing the exact FAILED/ERROR test IDs in both directions: empty diff, same 80 failed / 10 errors / 165 skipped at each (passed differs by exactly 4, fully explained by the 2 new GNN+XGBoost claims only existing at `5aad52a`). (2) `python -m pytest ml/test_claims.py ml/vertex/test_claims_gate.py` → 112 passed. (3) `python -m ml.claims` **left `docs/claims.md` byte-identical** (diffed before/after, no difference) — every number in it is already correctly derived, not hand-written. (5) `sha256sum ml/out/models/model_c.ubj` matched the expected hash above exactly.

## 6. Live proof (Confidential Space) — visual evidence

The frozen record at `ui/data/enclave_evidence.json` (described in its own `note` field: "Captured from one real, credentialed Confidential Space run against Google Cloud on 2026-09-29") is the structured evidence. The two screenshots below are visual corroboration, captured independently during this session against the live production app, re-triggering both the honest run and the tampered-image run fresh rather than reusing the frozen JSON's numbers.

| screenshot | sha256 | what it shows |
|---|---|---|
| `docs/evidence/live-proof-honest-run-completed.png` | `790520527b1b92d25461cf2402d5db99a09a6c749aefaf394d5076e7a5f0efc1` | A fresh honest run (`run_id` distinct from the frozen JSON's `stage3v2`), all 8 steps checked including both decrypt attempts, with its own real Cloud Logging `Confidential VM created` entry and timestamp. |
| `docs/evidence/live-proof-tamper-run-refused.png` | `1c2febed0b15eca2d6969df4de5e69710975ca97bfa2ac0adff9a73657fd9fea` | A fresh tampered-image run's completion card: the digest comparison (`approved` vs `tampered: one line changed`, different SHA-256 image digests) and the real `refusal_kind: kms_decrypt_refused` body, including a genuine `403 IAM_PERMISSION_DENIED` from `cloudkms.googleapis.com`. |

**How these were captured, and one bug found along the way:** both runs were triggered through the live `/api/live-proof/start` and `/api/live-proof/tampered-start` endpoints via a scripted browser session, not simulated. The first several capture attempts produced a false "done" reading on the tampered run — its own step list pre-renders a future, not-yet-reached step literally labelled "KMS refused the key -- exactly as designed" (see `TAMPER_STAGES` in `ui/static/app.js`), which is textually identical to the real completion state. Detecting completion by that text alone reports "done" at step 1, seconds in. The fix was to detect only the DOM element the app renders exclusively on genuine completion (`.lp-digests`, from `tamperResultCard()`), never by matching step-label text. The two screenshots above are from the run after that fix, confirmed against the real elapsed-time counter and a real Cloud Logging entry, not the false-positive runs.
