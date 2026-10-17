# MuleNet

Cross-bank money mule detection for the Google Cloud AI Builder Cup
(BFSI track). Detects mule rings from onboarding documents to live
transactions and drafts regulation-cited compliance reports.
Must use Gemini on Vertex AI and deploy on Cloud Run. Public GitHub
repo: never commit secrets, keys, or real data.

Gemini runs in **asia-south1** (Mumbai), not us-central1: that region
serves both `gemini-2.5-flash` and `gemini-3.5-flash`, while us-central1
serves no Gemini 3.x at all and 2.5 retires 16-20 Oct 2026. The data has
since moved to asia-south1 too, in a parallel stack; the original
us-central1 stack is kept running as a fallback during the soak period.
See `docs/residency.md` and `infra/asia_south1_migration.md`.

## Architecture
simulator -> Pub/Sub -> Cloud Run consumer -> AlloyDB + BigQuery
-> Vertex AI model (XGBoost baseline, GNN if better)
-> ring clustering + explanations -> BigQuery/AlloyDB
KYC docs -> Document AI -> reuse detection -> risk signals
AML regulations -> Vertex AI Search -> citations for reports
ADK agents (Gemini) -> Cloud Run UI
Stretch: joint scoring inside Confidential Space.

## Repo layout and ownership
shared/     contracts: pydantic models, SQL DDL, BQ schemas  (architect)
simulator/  synthetic 3-bank data + KYC docs                 (data-engineer)
ingest/     Pub/Sub publisher + Cloud Run consumer           (data-engineer)
ml/         features, models, rings, explanations            (ml-engineer)
docai/      KYC parsing and document reuse detection         (docai-engineer)
agents/     ADK agents, tools, prompts, eval                 (agents-engineer)
ui/         FastAPI + frontend on Cloud Run                  (frontend-engineer)
infra/      setup scripts, IAM, Cloud Build                  (platform-engineer)
enclave/    Confidential Space workload (stretch)            (platform-engineer)
docs/       README, architecture, demo script                (platform-engineer)

## Rules for every agent
- Only edit files in your own folders. If you need a change in
  shared/ or another folder, stop and report what you need.
- Code against shared/ contracts; never redefine schemas locally.
- Python 3.11, type hints, ruff, pytest. Tests alongside code.
- Config via env vars in one config.py per module. Secrets in
  Secret Manager only.
- Only hashed account IDs anywhere.
- Derive lists (steps, tables, fields) from their source of truth;
  never duplicate them by hand.
- Verify current Google Cloud and ADK APIs against docs; flag
  uncertainty instead of guessing.
- Finish with: tests run, summary of changes, open issues.
