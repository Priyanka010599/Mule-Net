"""Build, submit, register and log the Vertex AI training job.

Subcommands, each independently runnable and each printing what it
*verified* rather than what it attempted:

``build``     assemble a clean build context and build the training image
``run``       submit the custom job and wait for it
``download``  fetch a run's artefacts from GCS for the gate
``register``  upload the exported boosters to Vertex AI Model Registry
``log``       log params and metrics to Vertex AI Experiments

Every Google Cloud call is pinned to an explicit project and location
from :class:`ml.vertex.config.VertexConfig`. Nothing here reads gcloud's
configured default project, and nothing here runs ``gcloud config set``:
the default on a developer machine may be an unrelated project, and a
command that omits ``--project`` would act on it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ml.vertex.config import (
    BUILD_SERVICE_ACCOUNT,
    EXPERIMENT_NAME,
    MODEL_DISPLAY_NAME,
    VertexConfig,
)

_VERTEX_DIR = Path(__file__).resolve().parent
_ML_DIR = _VERTEX_DIR.parent

#: ``Model.upload`` requires a serving container URI, and the registry
#: entry here is a lineage record rather than a deployment, so the honest
#: choice is not obvious. The prebuilt
#: ``vertex-ai/prediction/xgboost-cpu.2-1`` was tried first and **rejected
#: the artefact**:
#:
#:   Model directory gs://.../model/local-canonical is expected to contain
#:   exactly one of: [model.pkl, model.joblib, model.bst]
#:
#: which is the registry telling the truth: five per-fold boosters are not
#: one servable model. Renaming ``fold0.ubj`` to ``model.bst`` would have
#: satisfied the check by claiming one model where there are five, so the
#: training image is named instead -- the only image that can actually
#: read these files. It carries no HTTP server, so the entry is still not
#: deployable, and ``metadata.json`` says ``servable: false``.
SERVING_IMAGE_IS_TRAINING_IMAGE = True


def _run(cmd: list[str]) -> str:
    """Run a command, echo it, and raise on a non-zero exit.

    ``cmd[0]`` is resolved through :func:`shutil.which` because on Windows
    ``gcloud`` is ``gcloud.cmd`` and ``subprocess`` does not search
    ``PATHEXT`` -- without this the call fails with ``WinError 2`` and
    looks like a missing SDK rather than a lookup problem.
    """
    resolved = shutil.which(cmd[0]) or cmd[0]
    cmd = [resolved, *cmd[1:]]
    print("RUN: " + " ".join(cmd), flush=True)
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    if done.returncode != 0:
        raise SystemExit(f"command failed with exit {done.returncode}: {cmd[0]}")
    return done.stdout


def new_run_id() -> str:
    """UTC-stamped run id used for the image tag and the GCS output folder."""
    return "run-" + datetime.now(UTC).strftime("%Y%m%dt%H%M%S")


def assemble_context(dest: Path) -> Path:
    """Copy only what the image needs into a fresh build context.

    ``ml/experiments/`` (~109 MB of out-of-fold scores), ``ml/out/`` and
    ``ml/features.parquet`` are excluded: the job regenerates all of them
    and uploading them would make every build minutes slower for nothing.
    """
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "ml").mkdir(parents=True)
    for path in sorted(_ML_DIR.rglob("*.py")):
        rel = path.relative_to(_ML_DIR)
        if rel.parts[0] in {"experiments", "out"} or "__pycache__" in rel.parts:
            continue
        target = dest / "ml" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(_VERTEX_DIR / "Dockerfile", dest / "Dockerfile")
    shutil.copy2(_VERTEX_DIR / "requirements-train.txt", dest / "requirements-train.txt")
    files = [p for p in dest.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    print(f"build context: {len(files)} files, {size} bytes at {dest}")
    return dest


def cmd_build(args: argparse.Namespace) -> int:
    """Assemble the context and build the training image with Cloud Build."""
    cfg = VertexConfig.from_env()
    image = f"{cfg.image_uri_base}:{args.run_id}"
    context = assemble_context(Path(args.context_dir))
    _run(
        [
            "gcloud",
            "builds",
            "submit",
            str(context),
            "--project",
            cfg.project,
            "--region",
            cfg.location,
            "--config",
            str(_VERTEX_DIR / "cloudbuild.yaml"),
            "--substitutions",
            f"_IMAGE={image}",
            "--service-account",
            BUILD_SERVICE_ACCOUNT,
        ]
    )
    # Verified against the registry, not against the build's exit code.
    digest = _run(
        [
            "gcloud",
            "artifacts",
            "docker",
            "images",
            "describe",
            image,
            "--project",
            cfg.project,
            "--format",
            "value(image_summary.digest)",
        ]
    ).strip()
    if not digest.startswith("sha256:"):
        raise SystemExit(f"image {image} is not in Artifact Registry after the build")
    print(f"VERIFIED image {image} digest {digest}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Submit the custom job and wait for it, then print its terminal state."""
    from google.cloud import aiplatform  # noqa: PLC0415 - heavy, CLI-only import

    cfg = VertexConfig.from_env()
    image = f"{cfg.image_uri_base}:{args.run_id}"
    output_uri = cfg.run_uri(args.run_id)
    aiplatform.init(project=cfg.project, location=cfg.location, staging_bucket=f"gs://{cfg.bucket}")
    job = aiplatform.CustomJob(
        display_name=f"mulenet-early-detection-{args.run_id}",
        project=cfg.project,
        location=cfg.location,
        base_output_dir=f"{output_uri}/vertex",
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": cfg.machine_type},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": image,
                    "args": [
                        "--input-uri",
                        cfg.input_uri,
                        "--output-uri",
                        output_uri,
                        "--seed",
                        str(args.seed),
                    ],
                },
            }
        ],
        labels={"component": "ml", "purpose": "canonical-training-reproduction"},
    )
    job.run(service_account=args.service_account, sync=True)
    print(f"resource_name: {job.resource_name}")
    print(f"state: {job.state.name}")
    print(f"start: {job.gca_resource.start_time}  end: {job.gca_resource.end_time}")
    print(f"output_uri: {output_uri}")
    return 0 if job.state.name == "JOB_STATE_SUCCEEDED" else 1


def cmd_download(args: argparse.Namespace) -> int:
    """Copy one run's artefacts out of GCS so the gate can read them."""
    cfg = VertexConfig.from_env()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "gcloud",
            "storage",
            "rsync",
            "--recursive",
            "--project",
            cfg.project,
            f"{cfg.run_uri(args.run_id)}/",
            str(dest),
        ]
    )
    files = [p for p in dest.rglob("*") if p.is_file()]
    print(f"downloaded {len(files)} files to {dest}")
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    """Register exported boosters in Vertex AI Model Registry.

    ``--artifact-uri`` overrides the default "this run's ``model/``
    folder". It exists because **local is the source of truth**: the
    boosters that provably reproduce the published scores are the ones
    exported from the local canonical run, and if a cloud run fails the
    gate its boosters must not be the thing carrying the published
    model's name. The alias and description then have to say which run
    the artefact came from, which is why both are arguments.
    """
    from google.cloud import aiplatform  # noqa: PLC0415

    cfg = VertexConfig.from_env()
    aiplatform.init(project=cfg.project, location=cfg.location)
    artifact_uri = args.artifact_uri or f"{cfg.run_uri(args.run_id)}/model"
    serving_image = f"{cfg.image_uri_base}:{args.serving_tag}"
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_image,
        project=cfg.project,
        location=cfg.location,
        version_aliases=[args.alias or args.run_id.replace("_", "-")],
        version_description=args.description,
        labels={"component": "ml", "scope": "cross-bank", "seed": "42"},
        sync=True,
    )
    print(f"resource_name: {model.resource_name}")
    print(f"version_id: {model.version_id}")
    print(f"artifact_uri: {artifact_uri}")
    print(f"serving_container: {serving_image}")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    """Log the run's params and headline metrics to Vertex AI Experiments."""
    from google.cloud import aiplatform  # noqa: PLC0415

    from ml.evaluate import SEEDS, XGB_PARAMS  # noqa: PLC0415
    from ml.metrics import METRIC_KEYS  # noqa: PLC0415

    cfg = VertexConfig.from_env()
    root = Path(args.artifacts)
    early = json.loads((root / "experiments/early_detection/metrics.json").read_text("utf-8"))
    lead = json.loads((root / "experiments/lead_time/metrics.json").read_text("utf-8"))

    params: dict[str, float | int | str] = {
        "seed": args.seed,
        "seeds": ",".join(str(s) for s in SEEDS),
        "machine_type": cfg.machine_type,
        "region": cfg.location,
        "input_uri": cfg.input_uri,
        "evaluation": early["evaluation"],
        # Logged so the Experiments entry cannot be read as authoritative on
        # its own. Local is the source of truth; if this run did not
        # reproduce it, the run's own record has to say so.
        "gate_verdict": args.gate_verdict,
        **{f"xgb_{k}": str(v) for k, v in XGB_PARAMS.items()},
    }
    # Metric keys are derived from METRIC_KEYS so the logged set cannot
    # drift from what ml.metrics actually computes.
    metrics: dict[str, float | int | str] = {}
    for scope, summary in early["summary"].items():
        for key in METRIC_KEYS:
            metrics[f"full_{scope}_{key}"] = float(summary[f"{key}_mean"])
    for horizon in early["horizons"]:
        if horizon["cutoff_minutes"] == 5:
            for scope, summary in horizon["summary"].items():
                for key in METRIC_KEYS:
                    metrics[f"T5_{scope}_{key}"] = float(summary[f"{key}_mean"])
    for scope, entry in lead["seed_mean"]["by_scope"].items():
        metrics[f"leadtime_{scope}_share_with_positive_warning"] = float(
            entry["share_with_positive_warning"]
        )

    aiplatform.init(
        project=cfg.project,
        location=cfg.location,
        experiment=EXPERIMENT_NAME,
        experiment_description="Cross-bank vs single-bank early detection, canonical seed 42.",
    )
    aiplatform.start_run(args.run_id)
    aiplatform.log_params(params)
    aiplatform.log_metrics(metrics)
    aiplatform.end_run()
    print(f"logged {len(params)} params, {len(metrics)} metrics to {EXPERIMENT_NAME}/{args.run_id}")
    for key in sorted(metrics):
        print(f"  {key} = {metrics[key]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build the training image")
    build.add_argument("--run-id", required=True)
    build.add_argument("--context-dir", required=True)
    build.set_defaults(func=cmd_build)

    run = sub.add_parser("run", help="submit the custom job")
    run.add_argument("--run-id", required=True)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--service-account", default=None)
    run.set_defaults(func=cmd_run)

    down = sub.add_parser("download", help="fetch a run's artefacts")
    down.add_argument("--run-id", required=True)
    down.add_argument("--dest", required=True)
    down.set_defaults(func=cmd_download)

    reg = sub.add_parser("register", help="register the model")
    reg.add_argument("--run-id", required=True)
    reg.add_argument(
        "--artifact-uri",
        default=None,
        help="gs:// folder holding the boosters; defaults to this run's model/",
    )
    reg.add_argument("--alias", default=None, help="version alias; defaults to the run id")
    reg.add_argument(
        "--serving-tag",
        required=True,
        help="training-image tag named as the serving container; see the module constant",
    )
    reg.add_argument(
        "--description",
        required=True,
        help="what this version is and which run produced it; it is a provenance record",
    )
    reg.set_defaults(func=cmd_register)

    log = sub.add_parser("log", help="log params and metrics to Experiments")
    log.add_argument("--run-id", required=True)
    log.add_argument("--artifacts", required=True, help="local copy of the run's artefacts")
    log.add_argument("--seed", type=int, default=42)
    log.add_argument(
        "--gate-verdict",
        required=True,
        help="result of ml.vertex.gate for this run, e.g. PASSED or FAILED plus the reason",
    )
    log.set_defaults(func=cmd_log)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
