"""Where every served number came from, and whether it is older than its data.

The Metrics tab quotes a comparison generated on one day, the footer a snapshot
built on another, and the red-team / lead-time numbers come from bundled copies
on a third. ``GET /api/meta`` lists each artifact the browser can read with a
content hash, seed, generation time and live-vs-bundled source; this module
builds that list, decides ``stale`` and derives the ``ETag`` the JSON endpoints
send. Everything is read from the artifacts themselves -- no version, seed or
count is hardcoded here -- and nothing here ever reads a raw account id.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ui.config import UiConfig

log = logging.getLogger("mulenet.ui.provenance")

#: Browsers may reuse a metric artifact for this long without asking. Short on
#: purpose: the ETag, not the lifetime, is what keeps a stale copy from living.
CACHE_MAX_AGE_S = 300

#: How each artifact is regenerated (for ``stale_reason``). The bundled copies are
#: refreshed from ``ml/experiments/`` by ``python -m ui.build_snapshot``.
_REGENERATE: dict[str, str] = {
    "metrics": "python -m ml.early && python -m ui.build_snapshot",
    "lead_time": "python -m ml.leadtime && python -m ui.build_snapshot",
    "coalition": "python -m ml.coalition && python -m ui.build_snapshot",
    "redteam": (
        "MULENET_AGENT_MODE=vertex python -m ml.redteam && python -m ui.build_snapshot "
        "(calls live Gemini with no rate limit)"
    ),
    "snapshot": "python -m ui.build_snapshot",
}

#: Rule parameters the red-team file records under ``detector``, and where the
#: manifest records the same ones under ``settings``.
_RULE_PARAMS = (
    "velocity_max_seconds",
    "velocity_ratio_range",
    "velocity_min_in_degree",
    "velocity_min_out_degree",
    "min_distinct_counterparty_banks",
)

_SEED_IN_TEXT = re.compile(r"\bseed (\d+)\b")


def _parse(ts: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts))
    except ValueError:
        return None


class Provenance:
    """Resolves, hashes and describes the artifacts one ``UiConfig`` points at."""

    def __init__(self, cfg: UiConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        #: path -> ((mtime_ns, size), sha256 prefix, parsed json | None)
        self._cache: dict[Path, tuple[tuple[int, int], str, Any]] = {}

    # -- files ---------------------------------------------------------

    def _candidates(self) -> dict[str, tuple[Path, Path | None]]:
        c = self.cfg
        return {
            "snapshot": (c.snapshot_path, None),
            "ml_manifest": (c.ml_out_dir / "manifest.json", None),
            "metrics": (c.metrics_path, c.metrics_fallback_path),
            "lead_time": (c.lead_metrics_path, c.lead_metrics_fallback_path),
            "coalition": (c.coalition_path, c.coalition_fallback_path),
            "redteam": (c.redteam_path, c.redteam_fallback_path),
            "enclave": (c.enclave_evidence_path, None),
        }

    def resolve(self, name: str) -> tuple[Path, str] | None:
        """The file a request would actually open, and its source label."""
        live, fallback = self._candidates()[name]
        if live.exists():
            return live, "bundled" if name == "enclave" else "live"
        if fallback is not None and fallback.exists():
            return fallback, "bundled fallback"
        return None

    def _load(self, path: Path) -> tuple[str, Any]:
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            hit = self._cache.get(path)
            if hit is not None and hit[0] == key:
                return hit[1], hit[2]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()[:12]
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        with self._lock:
            self._cache[path] = (key, digest, parsed)
        return digest, parsed

    def version(self, *names: str) -> str:
        """Data version of the artifacts an endpoint reads (``-`` joined hashes)."""
        parts = []
        for name in names:
            found = self.resolve(name)
            parts.append(self._load(found[0])[0] if found else "none")
        return "-".join(parts)

    # -- per-artifact facts ---------------------------------------------

    @staticmethod
    def _seeds(name: str, doc: Any) -> list[str] | None:
        if not isinstance(doc, dict):
            return None
        found: set[str] = set()
        if name == "metrics":
            for horizon in doc.get("horizons", []):
                found.update(str(r["seed"]) for r in horizon.get("runs", []) if "seed" in r)
        elif name == "lead_time":
            found.update(str(k) for k in doc.get("per_seed", {}))
        elif name == "ml_manifest":
            seed = doc.get("settings", {}).get("score_seed")
            if seed is not None:
                found.add(str(seed))
        elif name == "snapshot":
            found.update(_SEED_IN_TEXT.findall(str(doc.get("source", ""))))
        return sorted(found, key=int) if found else None

    @staticmethod
    def _versions(name: str, doc: Any) -> dict[str, str]:
        if not isinstance(doc, dict):
            return {}
        if name == "snapshot":
            block = doc.get("detector", {})
            keys = ("detector_version", "score_model_version")
        elif name == "ml_manifest":
            block, keys = doc, ("detector_version", "score_model_version", "explainer_version")
        else:
            return {}
        return {k: str(block[k]) for k in keys if k in block}

    def _reference(self) -> tuple[datetime | None, str | None]:
        """When the ml/out run the artifacts describe was generated."""
        found = self.resolve("ml_manifest")
        if found:
            doc = self._load(found[0])[1]
            at = _parse(doc.get("generated_at")) if isinstance(doc, dict) else None
            if at:
                return at, "ml/out/manifest.json"
        found = self.resolve("snapshot")
        if found:
            doc = self._load(found[0])[1]
            basis = doc.get("data_basis", {}) if isinstance(doc, dict) else {}
            at = _parse(basis.get("ml_manifest_generated_at"))
            if at:
                return at, "snapshot data_basis"
        return None, None

    def _rule_params_note(self, doc: Any) -> str:
        found = self.resolve("ml_manifest")
        if not found or not isinstance(doc, dict):
            return ""
        settings = (self._load(found[0])[1] or {}).get("settings", {})
        mine = doc.get("detector", {})
        if not all(k in mine and k in settings for k in _RULE_PARAMS):
            return ""
        same = all(mine[k] == settings[k] for k in _RULE_PARAMS)
        return (
            " The rule thresholds it records match the current manifest."
            if same
            else " The rule thresholds it records DIFFER from the current manifest."
        )

    def artifact(self, name: str) -> dict[str, Any]:
        found = self.resolve(name)
        entry: dict[str, Any] = {
            "name": name,
            "available": found is not None,
            "data_version": None,
            "seed": None,
            "generated_at": None,
            "source": None,
            "stale": False,
            "stale_reason": None,
        }
        if found is None:
            return entry
        path, source = found
        digest, doc = self._load(path)
        generated = doc.get("generated_at") if isinstance(doc, dict) else None
        entry.update(
            data_version=digest,
            seed=self._seeds(name, doc),
            generated_at=str(generated) if generated else None,
            source=source,
            file=path.name,
            **self._versions(name, doc),
        )
        reference, reference_source = self._reference()
        made = _parse(generated)
        if name not in ("ml_manifest", "enclave") and made and reference and made < reference:
            how = _REGENERATE.get(name, "")
            entry["stale"] = True
            entry["stale_reason"] = (
                f"generated {str(generated)[:16]}Z, before the ml/out run it describes "
                f"({reference.isoformat()[:16]}Z, {reference_source})."
                + (self._rule_params_note(doc) if name == "redteam" else "")
                + (f" Regenerate: {how}" if how else "")
            )
        return entry

    # -- /api/meta -------------------------------------------------------

    def meta(self) -> dict[str, Any]:
        artifacts = [self.artifact(n) for n in self._candidates()]
        versions: dict[str, str] = {}
        for entry in artifacts:  # manifest first would win; snapshot is the served one
            for key in ("detector_version", "score_model_version", "explainer_version"):
                if key in entry:
                    versions.setdefault(key, entry[key])
        reference, reference_source = self._reference()
        return {
            "artifacts": artifacts,
            "versions": versions,
            "data_reference": {
                "ml_out_generated_at": reference.isoformat() if reference else None,
                "from": reference_source,
            },
            "consistency_warnings": self.consistency_warnings(),
        }

    # -- startup consistency ----------------------------------------------

    def _doc(self, name: str) -> dict[str, Any]:
        found = self.resolve(name)
        doc = self._load(found[0])[1] if found else None
        return doc if isinstance(doc, dict) else {}

    def consistency_warnings(self) -> list[str]:
        """Disagreements between artifacts. Counts and versions only, no ids."""
        warnings: list[str] = []
        snap, manifest = self._doc("snapshot"), self._doc("ml_manifest")

        seeds = {n: self._seeds(n, self._doc(n)) for n in ("snapshot", "ml_manifest")}
        pipeline = {s for v in seeds.values() if v for s in v}
        if len(pipeline) > 1:
            warnings.append(f"seed disagrees between snapshot and ml manifest: {seeds}")
        for n in ("metrics", "lead_time"):
            have = self._seeds(n, self._doc(n))
            if pipeline and have and not pipeline <= set(have):
                warnings.append(f"{n} was not run on pipeline seed {sorted(pipeline)}: {have}")

        rules = {
            v
            for v in (
                snap.get("detector", {}).get("detector_version"),
                manifest.get("detector_version"),
            )
            if v
        }
        if len(rules) > 1:
            warnings.append(f"rule version disagrees (snapshot vs manifest): {sorted(rules)}")

        detected = {
            "snapshot.dataset": snap.get("dataset", {}).get("detected_rings_total"),
            "snapshot.detector": snap.get("detector", {}).get("rings_found"),
            "manifest": manifest.get("tables", {}).get("rings", {}).get("rows"),
        }
        if len({v for v in detected.values() if v is not None}) > 1:
            warnings.append(f"detected ring count disagrees: {detected}")

        accounts = {
            "snapshot": snap.get("dataset", {}).get("accounts"),
            "manifest.scores": manifest.get("tables", {}).get("scores", {}).get("rows"),
        }
        if len({v for v in accounts.values() if v is not None}) > 1:
            warnings.append(f"account count disagrees: {accounts}")

        true_rings: dict[str, Any] = {
            "lead_time": self._doc("lead_time").get("exit_distribution", {}).get("rings"),
            "coalition": self._doc("coalition").get("true_rings_total"),
        }
        first = (self._doc("metrics").get("horizons") or [{}])[0]
        per_seed = first.get("ring_catch_per_seed", {}).get("cross_bank", {})
        if per_seed:
            true_rings["metrics"] = max(int(r.get("rings", 0)) for r in per_seed.values())
        if len({v for v in true_rings.values() if v is not None}) > 1:
            warnings.append(f"true ring count disagrees: {true_rings}")
        return warnings

    def log_startup_check(self) -> None:
        """WARNING per disagreement; one INFO line if clean. Never raises."""
        try:
            problems = self.consistency_warnings()
            stale = [a["name"] for a in map(self.artifact, self._candidates()) if a["stale"]]
        except Exception:  # noqa: BLE001 - a diagnostic must not stop the app starting
            log.warning("artifact consistency check could not run", exc_info=True)
            return
        for problem in problems:
            log.warning("artifact consistency: %s", problem)
        if stale:
            log.warning("artifact stale against ml/out run: %s", ", ".join(stale))
        if not problems and not stale:
            log.info("artifact consistency: ok")


def etag_matches(header: str | None, etag: str) -> bool:
    """RFC 9110 ``If-None-Match`` weak comparison against one ETag."""
    if not header:
        return False
    if header.strip() == "*":
        return True
    wanted = etag.removeprefix("W/")
    return any(t.strip().removeprefix("W/") == wanted for t in header.split(","))
