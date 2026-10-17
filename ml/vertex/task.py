"""Container entrypoint for the Vertex AI custom training job.

Runs the canonical training -- ``ml.early`` then ``ml.leadtime``, in that
order because ``ml.leadtime`` reads the out-of-fold store ``ml.early``
writes -- against inputs staged in GCS, then exports the boosters behind
the published scores and copies every artefact back to GCS.

This module deliberately contains **no modelling code**. It stages files,
sets the two environment variables ``ml.config`` already reads, and calls
the same ``main()`` functions a developer calls locally. If it did any of
the work itself the cloud run would no longer be the same run, and the
gate would be comparing two different programs.

The six input tables are enumerated from :data:`ml.config.TABLE_FILES`,
the same mapping ``ml.features.load_tables`` reads them through, so the
staged set cannot drift from the set the training needs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from google.cloud import storage

from ml.config import TABLE_FILES


def _split_uri(uri: str) -> tuple[str, str]:
    """``gs://bucket/a/b`` -> ``("bucket", "a/b")``."""
    if not uri.startswith("gs://"):
        raise ValueError(f"expected a gs:// URI, got {uri!r}")
    bucket, _, prefix = uri[len("gs://") :].partition("/")
    return bucket, prefix.strip("/")


def download_inputs(client: storage.Client, input_uri: str, data_dir: Path) -> list[Path]:
    """Stage the six simulator tables locally, failing loudly on any absence."""
    bucket_name, prefix = _split_uri(input_uri)
    bucket = client.bucket(bucket_name)
    data_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for filename in sorted(TABLE_FILES.values()):
        blob = bucket.blob(f"{prefix}/{filename}")
        if not blob.exists():
            raise FileNotFoundError(
                f"missing training input gs://{bucket_name}/{prefix}/{filename}"
            )
        target = data_dir / filename
        blob.download_to_filename(str(target))
        staged.append(target)
        print(f"staged {filename} ({target.stat().st_size} bytes)", flush=True)
    return staged


def upload_outputs(client: storage.Client, output_uri: str, root: Path) -> int:
    """Copy every file under ``root`` to ``output_uri``, keeping the layout."""
    bucket_name, prefix = _split_uri(output_uri)
    bucket = client.bucket(bucket_name)
    count = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        bucket.blob(f"{prefix}/{rel}").upload_from_filename(str(path))
        count += 1
    print(f"uploaded {count} files to {output_uri}", flush=True)
    return count


def print_runtime() -> None:
    """Print the interpreter and library versions this run actually used.

    The gate compares numbers produced here to numbers produced locally,
    and every one of them is a property of these builds. Printing them
    into the job log means a gate failure can be diagnosed from the run's
    own record instead of from an assumption about the image.
    """
    import numpy
    import pandas
    import pyarrow
    import scipy
    import sklearn
    import xgboost

    print(f"python      {sys.version.split()[0]}", flush=True)
    for module in (xgboost, numpy, pandas, sklearn, scipy, pyarrow):
        print(f"{module.__name__:<11} {module.__version__}", flush=True)
    print(f"cpu_count   {os.cpu_count()}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Stage inputs, run the canonical training, export and upload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-uri", required=True, help="gs:// folder holding the six tables")
    parser.add_argument("--output-uri", required=True, help="gs:// folder to write artefacts to")
    parser.add_argument("--work-dir", default="/tmp/mulenet", help="local scratch root")
    parser.add_argument("--seed", type=int, default=42, help="canonical seed; 42")
    args = parser.parse_args(argv)

    work = Path(args.work_dir)
    data_dir = work / "data"
    ml_dir = work / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    print_runtime()
    client = storage.Client()
    download_inputs(client, args.input_uri, data_dir)

    # Set before any ``MlConfig.from_env()`` call, which happens inside the
    # ``main()`` functions below rather than at import time.
    os.environ["MULENET_DATA_DIR"] = str(data_dir)
    os.environ["MULENET_ML_DIR"] = str(ml_dir)
    os.environ["MULENET_SEED"] = str(args.seed)

    from ml import early, leadtime  # noqa: PLC0415 - must follow the env setup above
    from ml.config import MlConfig  # noqa: PLC0415
    from ml.vertex.export_model import export  # noqa: PLC0415

    print("=== ml.early ===", flush=True)
    early.main()
    print("=== ml.leadtime ===", flush=True)
    leadtime.main()

    cfg = MlConfig.from_env()
    print("=== export_model ===", flush=True)
    metadata = export(cfg, ml_dir / "model")
    print(f"exported {len(metadata['booster_files'])} boosters", flush=True)

    upload_outputs(client, args.output_uri, ml_dir)
    print("TASK COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
