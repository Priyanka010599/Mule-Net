"""Register Model A, Model B and Model C in Vertex AI Model Registry.

Registration only -- no endpoint is created or deployed. Converts each
local ``ml/out/models/model_{a,b,c}.ubj`` booster to the single-file
``model.bst`` the prebuilt XGBoost serving container requires (it
rejects the multi-file, per-fold layout the existing ``cb-oof-v1``
model needed), uploads each to its own GCS folder under
``gs://mulenet-mulenet-data-asia/ml/out/models/registry/model-{a,b,c}/``,
and calls ``aiplatform.Model.upload()`` once per model:

    display name   mulenet-model-a              mulenet-model-b             mulenet-model-c
    alias          model-a-16feat-v1            model-b-12feat-v1           model-c-11feat-v1
    features        16 (incl. device/KYC reuse)  12 (transfer-graph only)    11 (transfer-graph,
                                                                              no account_age_days)
    served by       agents/subset.py,            ml/experiments/redteam/     agents/subset_upload.py
                     ml/coalition.py              's own comparison only

Both use the prebuilt ``xgboost-cpu.2-1`` image and
``ml.vertex.config.VertexConfig``'s project/location
(``mulenet`` / ``asia-south1``), matching where Gemini inference
already runs (``docs/residency.md``).

Re-runnable: running it again creates a fresh Model Registry resource
under the same display name (not a new version of the existing one --
pass ``--parent-model`` yourself if you want version-on-top-of-version
behaviour instead).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from ml.vertex.config import VertexConfig

SERVING_IMAGE = "asia-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"


def _gcloud_executable() -> str:
    """Resolve the real gcloud executable name/path for this OS.

    On Windows, gcloud installs as ``gcloud.cmd``; passing the bare
    string "gcloud" to ``subprocess.run`` without ``shell=True`` does
    not resolve it at all (confirmed: that is exactly what broke the
    real registration run). ``shutil.which`` finds whatever this
    platform actually calls it (``gcloud.cmd``, ``gcloud.exe``, or
    plain ``gcloud`` on Linux/macOS); if it can't be found at all, this
    still returns the bare "gcloud" string so the caller's own
    subprocess call is the one that fails -- with a clear message added
    there, not a different, unrelated-looking resolution failure here.
    """
    return shutil.which("gcloud") or "gcloud"

MODELS = {
    "a": {
        "display_name": "mulenet-model-a",
        "alias": "model-a-16feat-v1",
        "booster": "ml/out/models/model_a.ubj",
        "metadata": "ml/out/models/model_a_metadata.json",
        "summary": "16-feature XGBoost (incl. device/KYC identity reuse)",
    },
    "b": {
        "display_name": "mulenet-model-b",
        "alias": "model-b-12feat-v1",
        "booster": "ml/out/models/model_b.ubj",
        "metadata": "ml/out/models/model_b_metadata.json",
        "summary": "12-feature XGBoost (transfer-graph only, no identity signal)",
    },
    "c": {
        "display_name": "mulenet-model-c",
        "alias": "model-c-11feat-v1",
        "booster": "ml/out/models/model_c.ubj",
        "metadata": "ml/out/models/model_c_metadata.json",
        "summary": (
            "11-feature XGBoost (transfer-graph only, no identity signal, "
            "no account_age_days -- serves judge uploads, which cannot supply "
            "a real account-open date)"
        ),
    },
}


def _build_description(meta: dict, summary: str, metadata_path: str) -> str:
    perf = meta["validated_performance"]
    t5, t180 = perf["T=5m"], perf["T=180m"]
    return (
        f"{summary}. Served by {meta['served_by']}. seed={meta['seed']}. "
        f"Validated (grouped 5-fold CV x 3 seeds, not re-scored on this "
        f"exact booster -- see {metadata_path}): "
        f"T=5m: PR-AUC {t5['pr_auc_mean']:.3f}, "
        f"recall@1%FPR {t5['recall_at_fpr_1pct_mean']:.3f}, "
        f"{t5['false_alerts_per_catch_at_80pct_recall_mean']:.2f} false "
        f"alerts/catch. "
        f"T=180m: PR-AUC {t180['pr_auc_mean']:.3f}, "
        f"recall@1%FPR {t180['recall_at_fpr_1pct_mean']:.3f}, "
        f"{t180['false_alerts_per_catch_at_80pct_recall_mean']:.3f} false "
        f"alerts/catch. Registration only -- no endpoint deployed. Trained "
        f"with xgboost 3.4.1 locally; the prebuilt serving container here "
        f"supports XGBoost 2.1, unverified for actually serving this exact "
        f"booster."
    )


def register_one(key: str, cfg: VertexConfig, dry_run: bool) -> None:
    import xgboost as xgb  # noqa: PLC0415
    from google.cloud import aiplatform  # noqa: PLC0415

    spec = MODELS[key]
    booster_path = Path(spec["booster"])
    meta = json.loads(Path(spec["metadata"]).read_text())
    description = _build_description(meta, spec["summary"], spec["metadata"])

    gcs_folder = f"gs://{cfg.bucket}/ml/out/models/registry/model-{key}"
    bst_path = booster_path.with_name(f"model_{key}.bst")

    booster = xgb.Booster()
    booster.load_model(str(booster_path))
    booster.save_model(str(bst_path))

    if dry_run:
        print(f"[dry-run] would upload {bst_path} -> {gcs_folder}/model.bst")
        print(f"[dry-run] would register display_name={spec['display_name']} "
              f"alias={spec['alias']}")
        return

    try:
        subprocess.run(
            [_gcloud_executable(), "storage", "cp", str(bst_path), f"{gcs_folder}/model.bst",
             "--project", cfg.project],
            check=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Could not run gcloud (tried shutil.which(\"gcloud\"), then the bare "
            "name \"gcloud\" as a fallback -- neither resolved to a runnable "
            "executable). Install the Google Cloud CLI and ensure it is on "
            "PATH, or run this script from a shell where `gcloud --version` "
            "works."
        ) from exc

    aiplatform.init(project=cfg.project, location=cfg.location)
    model = aiplatform.Model.upload(
        display_name=spec["display_name"],
        artifact_uri=gcs_folder,
        serving_container_image_uri=SERVING_IMAGE,
        project=cfg.project,
        location=cfg.location,
        version_aliases=[spec["alias"]],
        version_description=description,
        labels={"component": "ml", "scope": "cross-bank", "seed": "42"},
        sync=True,
    )
    print(f"{spec['display_name']}: resource_name={model.resource_name} "
          f"version_id={model.version_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["a", "b", "c", "both", "all"], default="both")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print what would happen; no upload, no GCP call.")
    args = parser.parse_args(argv)

    cfg = VertexConfig.from_env()
    if args.model == "both":
        keys = ["a", "b"]
    elif args.model == "all":
        keys = ["a", "b", "c"]
    else:
        keys = [args.model]
    for key in keys:
        register_one(key, cfg, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
