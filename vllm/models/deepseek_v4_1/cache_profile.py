# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional checkpoint-specific warm starts for the hybrid expert cache."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)
MAX_PROFILE_BYTES = 1024 * 1024


def model_fingerprint(model_config):
    model = Path(model_config.model)
    identity = {
        "model": str(model.resolve()) if model.is_dir() else model_config.model,
        "revision": model_config.revision,
        "commit": getattr(model_config.hf_config, "_commit_hash", None),
        "config": model_config.hf_config.to_dict(),
    }
    if model.is_dir():
        identity["weights"] = [
            (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(model.glob("*.safetensors"))
        ]
        index = model / "model.safetensors.index.json"
        if index.is_file():
            identity["index"] = hashlib.sha256(index.read_bytes()).hexdigest()
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()


class ExpertCacheProfile:
    def __init__(self, path, fingerprint, num_experts, layers, interval=16):
        self.path = Path(path).expanduser()
        self.fingerprint = fingerprint
        self.num_experts = num_experts
        self.layers = set(map(str, layers))
        self.interval = interval
        self.boundaries = 0
        self.last_saved = -interval
        self._entries = {}

    def _validate(self, profile):
        if profile["version"] != 1 or profile["fingerprint"] != self.fingerprint:
            raise ValueError("checkpoint fingerprint or profile version differs")
        entries = profile["layers"]
        if not isinstance(entries, dict) or not set(entries) <= self.layers:
            raise ValueError("invalid cache profile layers")
        for entry in entries.values():
            selected = entry["selected"]
            history = entry["history"]
            requests, mass = entry["requests"], entry["mass"]
            if (
                not isinstance(selected, list)
                or not selected
                or any(
                    type(e) is not int or not 0 <= e < self.num_experts
                    for e in selected
                )
                or len(set(selected)) != len(selected)
                or type(requests) is not int
                or not 0 <= requests < 2**63
                or type(mass) not in (int, float)
                or not 0 <= mass <= 1e6
                or not math.isfinite(mass)
            ):
                raise ValueError("invalid cache profile selection or learning state")
            if history is not None and (
                not isinstance(history, list)
                or len(history) != self.num_experts
                or any(
                    type(n) not in (int, float)
                    or not math.isfinite(n)
                    or not 0 <= n <= 1
                    for n in history
                )
            ):
                raise ValueError("invalid cache profile demand history")
        return entries

    def load(self):
        self._entries = {}
        try:
            with self.path.open("rb") as file:
                data = file.read(MAX_PROFILE_BYTES + 1)
            if len(data) > MAX_PROFILE_BYTES:
                raise ValueError("cache profile exceeds its byte limit")
            self._entries = self._validate(json.loads(data))
            return self._entries
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            logger.warning("Ignoring expert cache profile %s: %s", self.path, error)
            return {}

    def selection(self, entry, capacity, *, rank_by_history=False):
        selected = entry["selected"]
        history = entry["history"]
        if history is not None:
            ranking = sorted(range(self.num_experts), key=lambda e: (-history[e], e))
            if rank_by_history:
                return ranking[:capacity]
            if len(selected) > capacity:
                selected = sorted(selected, key=lambda e: (-history[e], e))
        else:
            ranking = list(range(self.num_experts))
        return (selected + [e for e in ranking if e not in selected])[:capacity]

    def save(self, caches):
        """Keep inactive layers' last saved entries as historical warm starts."""
        self.boundaries += 1
        if self.boundaries - self.last_saved < self.interval:
            return
        entries = dict(self._entries)
        for layer, cache in caches:
            state = cache.learning_state()
            entries[str(layer)] = {
                "selected": list(cache.selected),
                **{key: state[key] for key in ("history", "requests", "mass")},
            }
        if not entries or not any(e["history"] is not None for e in entries.values()):
            return
        self.last_saved = self.boundaries
        profile = {"version": 1, "fingerprint": self.fingerprint, "layers": entries}
        temporary = None
        try:
            self._validate(profile)
            data = json.dumps(profile, separators=(",", ":")).encode()
            if len(data) > MAX_PROFILE_BYTES:
                raise ValueError("cache profile exceeds its byte limit")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=self.path.parent, delete=False
            ) as file:
                temporary = Path(file.name)
                file.write(data)
            os.replace(temporary, self.path)
            self._entries = entries
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            logger.warning("Cannot save expert cache profile %s: %s", self.path, error)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    logger.warning(
                        "Cannot remove expert cache profile temporary %s: %s",
                        temporary,
                        error,
                    )
