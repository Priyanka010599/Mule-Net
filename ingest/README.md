# ingest

Pub/Sub publisher and the Cloud Run consumer that lands transaction events into
AlloyDB and BigQuery. Owned by the **data-engineer** agent. Message and row
shapes come from `shared/` — do not redefine them here.
