"""Vertex AI packaging for the canonical ``ml/`` training run.

The **local** run of ``python -m ml.early`` followed by
``python -m ml.leadtime`` is the source of truth for every published
number. This subpackage runs the *same* code as a Vertex AI custom job
and then proves the cloud run reproduced the local file, leaf for leaf.
It exists for reviewability and lineage, not to replace the local run.
See ``ml/README.md`` section "Two ways to run the training".
"""
