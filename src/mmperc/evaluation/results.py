"""Reader for the output of scripts/evaluate_all.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mmperc.evaluation.suffstats import GROUPS

SEP = "__"


class Results:
    """res = Results(dir_of_one_split); res.ep(scenario, model, group) -> list over seeds of (n_ep, K) sums."""

    def __init__(self, split_dir):
        self.dir = Path(split_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self._cache: dict[int, dict] = {}

    def _scen(self, si: int):
        if si not in self._cache:
            f = self.dir / f"s{si:02d}.npz"
            if not f.exists():
                raise FileNotFoundError(f"{f} (scenario {self.meta['scenarios'][si]!r} not evaluated)")
            self._cache[si] = dict(np.load(f))
        return self._cache[si]

    def scenarios(self):
        return self.meta["scenarios"]

    def models(self):
        return list(self.meta["models"]) + list(self.meta["classical"])

    def seeds(self, model: str):
        return self.meta["models"].get(model, [0])

    def get(self, scenario: str, model: str, what: str, seed: int):
        return self._scen(self.scenarios().index(scenario))[f"{model}{SEP}{seed}{SEP}{what}"]

    def ep(self, scenario: str, model: str, group: str = "coupled") -> list[np.ndarray]:
        j = GROUPS.index(group)
        return [self.get(scenario, model, "ep", s)[:, j, :] for s in self.seeds(model)]
