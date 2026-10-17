"""Pull a service's data files out of Cloud Storage at container start.

``ui/`` and ``agents/`` read their data from local paths, and in the cloud
those paths are filled from one bucket so both services see the identical
bytes (verified by comparing the object's md5 with the downloaded file).
Nothing in ``ui/`` or ``agents/`` changes: the paths they already resolve
from their own config are simply populated before uvicorn starts.

Configured with two environment variables:

``MULENET_DATA_BUCKET``
    Bucket holding the published snapshot. When unset this script is a
    no-op, so the image still runs against files baked in at build time.

``MULENET_FETCH``
    Comma-separated object names, e.g. ``ui/out/demo_snapshot.json``. Each
    is written to the same relative path under ``MULENET_FETCH_DEST``
    (default ``/app``), which is where the Dockerfiles put the code.

``MULENET_FETCH_DEST``
    Destination root. Defaults to ``/app``.

Object names are deliberately relative and the destination is a separate
variable: an earlier version took ``object=/abs/path`` pairs, and Git Bash on
Windows rewrote the absolute half of that value into
``C:/Program Files/Git/app/...`` before gcloud ever saw it, so the UI
downloaded its snapshot to a path it never reads. Nothing here now looks like
an absolute path to MSYS.

Usage:
  python -m infra.fetch_data
"""

from __future__ import annotations

import base64
import hashlib
import os
import sys
from pathlib import Path


def parse_objects(raw: str, dest_root: Path) -> list[tuple[str, Path]]:
    """Pair each object name with its destination under ``dest_root``."""
    pairs: list[tuple[str, Path]] = []
    for item in raw.split(","):
        obj = item.strip()
        if not obj:
            continue
        if obj.startswith("/"):
            raise SystemExit(
                f"MULENET_FETCH entry {obj!r} must be a relative object name"
            )
        pairs.append((obj, dest_root / obj))
    return pairs


def local_md5_b64(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - matching GCS's own md5, not for security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def main() -> None:
    bucket_name = os.environ.get("MULENET_DATA_BUCKET")
    raw = os.environ.get("MULENET_FETCH", "")
    if not bucket_name or not raw:
        print("fetch_data: no bucket or nothing to fetch, skipping")
        return

    from google.cloud import storage

    dest_root = Path(os.environ.get("MULENET_FETCH_DEST", "/app"))
    bucket = storage.Client().bucket(bucket_name)
    for obj, local in parse_objects(raw, dest_root):
        blob = bucket.blob(obj)
        blob.reload()
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local))
        size = local.stat().st_size
        # A truncated download would leave the app serving a half-parsed
        # snapshot, so confirm the bytes match the object before starting.
        if blob.md5_hash and local_md5_b64(local) != blob.md5_hash:
            print(
                f"fetch_data: md5 mismatch for gs://{bucket_name}/{obj}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"fetch_data: gs://{bucket_name}/{obj} -> {local} ({size} bytes, md5 ok)")


if __name__ == "__main__":
    main()
