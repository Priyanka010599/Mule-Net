"""Test-session defaults for ``agents/``."""

import os

# The service warms its subset feature builds in a background thread at
# startup (agents.subset.warm). Tests that start the app against real local
# data would otherwise burn CPU on six builds nobody asked for.
os.environ.setdefault("MULENET_AGENTS_WARM_SUBSETS", "0")

# Likewise the Ask tools' warm-up (agents.warm): it builds the real transfer
# index, which a test run has no use for.
os.environ.setdefault("MULENET_AGENTS_WARM_TOOLS", "0")
