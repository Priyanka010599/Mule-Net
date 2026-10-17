"""Create the MuleNet BigQuery dataset and load the published tables.

``shared/bq_schemas.py`` is the contract. Getting BigQuery to actually honour
it on a Parquet load takes more than passing it along, because a Parquet load
derives its own schema from the file:

* With no explicit schema, "the table schema is automatically retrieved from
  the self-describing source data", and ``WRITE_TRUNCATE`` "also deletes the
  table schema" -- so a truncate-load silently replaces TABLE_SCHEMAS with
  whatever pandas happened to write, modes and descriptions included.
* With an explicit schema, BigQuery still compares it against the file and
  rejects a disagreement. Pandas writes every column as optional, so the
  first attempt here failed with::

      Field transfer_id has changed mode from REQUIRED to NULLABLE

  and ``agents/out/cases.parquet`` carries ``closed_at`` as Parquet type
  ``null`` (no case is closed yet), which cannot be read as a TIMESTAMP.

So the published Parquet is restated against the contract before it is
loaded: :func:`arrow_schema_for` turns a list of ``SchemaField`` into the
equivalent Arrow schema -- REQUIRED becomes a non-nullable field, REPEATED a
list, RECORD a struct -- and every table is cast to it. That makes the
contract authoritative over the bytes as well as over the table, and it
turns a contract violation into a loud failure at cast time: a null in a
REQUIRED column now raises here instead of quietly loading.

Nested columns survive because the Arrow schema is built recursively from the
nested ``SchemaField`` lists: ``rings.members`` and
``account_explanations.top_factors`` stay REPEATED RECORDs rather than being
flattened to strings.

Usage:
  PROJECT_ID=... BUCKET=... python -m infra.load_bigquery
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery, storage
from google.cloud.bigquery import SchemaField

from shared.bq_schemas import TABLE_SCHEMAS

#: Table -> local Parquet file published by the owning module. Tables in
#: TABLE_SCHEMAS but absent here are created empty: they are part of the
#: contract but nothing has published rows for them yet.
SOURCES: dict[str, Path] = {
    "accounts": Path("simulator/out/accounts.parquet"),
    "devices": Path("simulator/out/devices.parquet"),
    "kyc_documents": Path("simulator/out/kyc_documents.parquet"),
    "kyc_submissions": Path("simulator/out/kyc_submissions.parquet"),
    "ground_truth": Path("simulator/out/ground_truth.parquet"),
    "transfers": Path("simulator/out/transfers.parquet"),
    "scores": Path("ml/out/scores.parquet"),
    "rings": Path("ml/out/rings.parquet"),
    "ring_explanations": Path("ml/out/ring_explanations.parquet"),
    "account_explanations": Path("ml/out/account_explanations.parquet"),
    "account_evidence": Path("ml/out/account_evidence.parquet"),
    "cases": Path("agents/out/cases.parquet"),
    "str_reports": Path("agents/out/str_reports.parquet"),
}

#: Where the restated copies land in the bucket. Kept separate from the
#: published paths so the originals the owning modules wrote are never
#: overwritten by this script.
STAGING_PREFIX = "bq_staging"

#: BigQuery scalar type -> Arrow type. NUMERIC is BigQuery's decimal(38, 9).
_ARROW_SCALARS: dict[str, pa.DataType] = {
    "STRING": pa.string(),
    "BYTES": pa.binary(),
    "INT64": pa.int64(),
    "INTEGER": pa.int64(),
    "FLOAT64": pa.float64(),
    "FLOAT": pa.float64(),
    "NUMERIC": pa.decimal128(38, 9),
    "BOOL": pa.bool_(),
    "BOOLEAN": pa.bool_(),
    "DATE": pa.date32(),
    "TIMESTAMP": pa.timestamp("us", tz="UTC"),
}


def arrow_field_for(field: SchemaField) -> pa.Field:
    """Translate one contract field into its Arrow equivalent."""
    if field.field_type in ("RECORD", "STRUCT"):
        inner: pa.DataType = pa.struct(
            [arrow_field_for(sub) for sub in field.fields]
        )
    else:
        try:
            inner = _ARROW_SCALARS[field.field_type]
        except KeyError as exc:  # pragma: no cover - new type in the contract
            raise SystemExit(
                f"no Arrow mapping for BigQuery type {field.field_type!r} "
                f"on field {field.name!r}; add it to _ARROW_SCALARS"
            ) from exc

    if field.mode == "REPEATED":
        # A repeated column is a list, and its child field MUST be named
        # "element". pyarrow's pa.list_(inner) names the child "item", which
        # BigQuery does not recognise as a Parquet LIST -- it reads the
        # column as a RECORD instead and the load fails with "field
        # banks_involved has changed type from STRING to RECORD", or for a
        # repeated record with "has array levels of 1, while the
        # corresponding field path to Parquet column has 0 repeated fields".
        # The files each module published already use "element"; naming it
        # explicitly keeps the restated copy identical in that respect.
        #
        # The list itself is nullable: BigQuery represents an empty repeated
        # field as an empty list, so a null list loads as [].
        return pa.field(
            field.name, pa.list_(pa.field("element", inner)), nullable=True
        )
    return pa.field(field.name, inner, nullable=field.mode != "REQUIRED")


def arrow_schema_for(fields: list[SchemaField]) -> pa.Schema:
    """Translate a whole contract table schema into an Arrow schema."""
    return pa.schema([arrow_field_for(f) for f in fields])


#: ``scores`` needs two columns added before it can be cast. SCORES_SCHEMA is
#: the union of RiskScore and GnnScore, so it carries a discriminator and an
#: embedding reference that the RiskScore-shaped ml/out/scores.parquet does
#: not have:
#:   model_type    REQUIRED, "xgboost or gnn"
#:   embedding_ref NULLABLE, "null for the XGBoost baseline"
#: ml/out/manifest.json records validated_against="shared.contracts.RiskScore"
#: and score_model_version="mulenet-xgboost-...", so for this file the
#: discriminator is "xgboost" and embedding_ref is null. Both values are the
#: contract's own documented semantics; neither is invented data.
SCORES_MANIFEST = Path("ml/out/manifest.json")


def scores_model_type() -> str:
    """Read the score model family out of ml/out/manifest.json."""
    version = json.loads(SCORES_MANIFEST.read_text(encoding="utf-8")).get(
        "score_model_version", ""
    )
    if "xgboost" in version:
        return "xgboost"
    if "gnn" in version:
        return "gnn"
    raise SystemExit(
        f"cannot derive scores.model_type from score_model_version={version!r}; "
        "SCORES_SCHEMA requires it to be 'xgboost' or 'gnn'"
    )


def restate(name: str, source: Path, out_dir: Path) -> Path:
    """Cast one published Parquet file to the contract's Arrow schema."""
    table = pq.read_table(source)

    if name == "scores":
        rows = table.num_rows
        if "model_type" not in table.column_names:
            table = table.append_column(
                "model_type", pa.array([scores_model_type()] * rows, pa.string())
            )
        if "embedding_ref" not in table.column_names:
            table = table.append_column("embedding_ref", pa.nulls(rows, pa.string()))

    schema = arrow_schema_for(TABLE_SCHEMAS[name])
    # select() reorders to the contract's column order and drops anything the
    # contract does not declare; cast() then fixes types and nullability.
    table = table.select(schema.names).cast(schema)

    out = out_dir / f"{name}.parquet"
    pq.write_table(table, out)
    print(f"    {name}: {table.num_rows} rows restated -> {out.name}")
    return out


def md5_b64(path: Path) -> str:
    """Base64 md5 of a local file, in the form GCS reports in ``md5_hash``."""
    digest = hashlib.md5()  # noqa: S324 - matching GCS's checksum, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def ensure_dataset(client: bigquery.Client, dataset_id: str, location: str) -> None:
    dataset = bigquery.Dataset(f"{client.project}.{dataset_id}")
    dataset.location = location
    client.create_dataset(dataset, exists_ok=True)
    print(f"==> dataset {client.project}.{dataset_id} ready ({location})")


def ensure_table(client: bigquery.Client, dataset_id: str, name: str) -> str:
    """Create ``name`` from the contract schema if absent. Returns table id."""
    table_id = f"{client.project}.{dataset_id}.{name}"
    client.create_table(
        bigquery.Table(table_id, schema=TABLE_SCHEMAS[name]), exists_ok=True
    )
    return table_id


def load_table(client: bigquery.Client, table_id: str, uri: str, name: str) -> None:
    """Empty the table, keeping its schema, then append the restated file."""
    # TRUNCATE TABLE is DDL that drops the rows and keeps the schema, unlike
    # WRITE_TRUNCATE, which would delete the schema along with the data.
    client.query(f"TRUNCATE TABLE `{table_id}`").result()

    # enable_list_inference is what makes the REPEATED columns survive. A
    # Parquet LIST is a group annotated LIST containing a repeated "list"
    # group containing "element"; without this flag BigQuery reads that
    # group as a STRUCT instead of a repeated field, which fails as
    #   rings.banks_involved: "changed type from STRING to RECORD"
    #   rings.members: "has array levels of 1, while the corresponding
    #                   field path to Parquet column has 0 repeated fields"
    # This is exactly the silent-corruption case to guard: without it a
    # repeated record does not load as a repeated record.
    parquet_options = bigquery.ParquetOptions()
    parquet_options.enable_list_inference = True

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        autodetect=False,
        schema=TABLE_SCHEMAS[name],
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    job_config.parquet_options = parquet_options

    job = client.load_table_from_uri(uri, table_id, job_config=job_config)
    job.result()

    table = client.get_table(table_id)
    verify_schema(name, table.schema)
    print(f"    {name}: {table.num_rows} rows in BigQuery")


#: The API reports the legacy name for some types, so INT64 comes back as
#: INTEGER and FLOAT64 as FLOAT. Same types, different spelling -- normalise
#: before comparing or the check fires on every table that has a float.
_TYPE_ALIASES = {
    "INTEGER": "INT64",
    "FLOAT": "FLOAT64",
    "BOOLEAN": "BOOL",
    "STRUCT": "RECORD",
}


def _shape(fields: list[SchemaField]) -> list[tuple[str, str, str, tuple]]:
    """(name, type, mode, nested shape) for each field, for comparison."""
    return [
        (
            f.name,
            _TYPE_ALIASES.get(f.field_type, f.field_type),
            f.mode,
            tuple(_shape(list(f.fields))),
        )
        for f in fields
    ]


def verify_schema(name: str, loaded: list[SchemaField]) -> None:
    """Fail if the loaded table drifted from the contract.

    The failure this exists for is a repeated RECORD arriving as a STRING or
    as a plain RECORD: the rows load, the counts look right, and the nesting
    is gone. Comparing the shape after the load makes that loud.
    """
    want = _shape(TABLE_SCHEMAS[name])
    got = _shape(list(loaded))
    if want != got:
        raise SystemExit(
            f"{name}: loaded schema does not match shared/bq_schemas.py\n"
            f"  contract: {want}\n"
            f"  loaded:   {got}"
        )


def main() -> None:
    project = os.environ["PROJECT_ID"]
    bucket_name = os.environ["BUCKET"]
    dataset_id = os.environ.get("BQ_DATASET", "mulenet")
    location = os.environ.get("REGION", "us-central1")

    bucket = storage.Client(project=project).bucket(bucket_name)
    client = bigquery.Client(project=project)

    ensure_dataset(client, dataset_id, location)

    print(f"==> ensuring {len(TABLE_SCHEMAS)} tables from shared/bq_schemas.py")
    table_ids = {name: ensure_table(client, dataset_id, name) for name in TABLE_SCHEMAS}
    for name in sorted(set(TABLE_SCHEMAS) - set(SOURCES)):
        print(f"    {name}: created empty (no published Parquet source)")

    with tempfile.TemporaryDirectory(prefix="mulenet-bq-") as tmp:
        out_dir = Path(tmp)
        print("==> restating published Parquet against the contract")
        staged = {name: restate(name, src, out_dir) for name, src in SOURCES.items()}

        print(f"==> uploading restated files to gs://{bucket_name}/{STAGING_PREFIX}")
        uris: dict[str, str] = {}
        for name, path in staged.items():
            obj = f"{STAGING_PREFIX}/{name}.parquet"
            uris[name] = f"gs://{bucket_name}/{obj}"
            blob = bucket.blob(obj)
            # Skip an object that already holds these exact bytes. The
            # restated transfers file is ~100 MB and re-sending it on every
            # run is both slow and the most likely thing to fail on a home
            # uplink, which is how this script first died mid-upload.
            if blob.exists():
                blob.reload()
                if blob.md5_hash == md5_b64(path):
                    print(f"    {obj}: unchanged, skipping upload")
                    continue
            # The default 120s timeout is not enough for the large file.
            blob.upload_from_filename(str(path), timeout=1800)
            print(f"    {obj} ({path.stat().st_size} bytes)")

    print("==> loading BigQuery")
    for name, uri in uris.items():
        load_table(client, table_ids[name], uri, name)

    print("==> done.")


if __name__ == "__main__":
    main()
