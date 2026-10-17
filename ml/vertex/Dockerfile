# Training image for the MuleNet Vertex AI custom job.
#
# Python 3.14 to match the local canonical run, NOT the 3.11 that
# pyproject.toml names as the supported *service* runtime. The gate
# compares this job's metrics.json to a file produced on 3.14.5, and a
# different minor version is a different set of compiled wheels.
#
# Build context is assembled by ml/vertex/submit.py: ml/*.py only, with
# experiments/, out/ and features.parquet excluded, so the context stays
# kilobytes rather than the ~420 MB those directories hold.
FROM python:3.14-slim

WORKDIR /app

COPY requirements-train.txt /app/requirements-train.txt
RUN pip install --no-cache-dir -r /app/requirements-train.txt

COPY ml /app/ml

# No PYTHONHASHSEED is set: nothing in ml/ iterates a set or dict in a
# way that reaches a number -- ring and fold ordering both go through an
# explicit sort or a seeded Generator. Verified by the local run being
# byte-reproducible across two separate invocations.
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "ml.vertex.task"]
