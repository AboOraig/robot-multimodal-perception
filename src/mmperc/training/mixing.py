"""Mixed-degradation training protocol.

* Every `refresh_every` epochs the whole TRAIN set is re-degraded with a fresh random scenario per episode
  (stateless RNG keyed by (seed, epoch, episode id) -> reproducible, resumable).
* Scenario grammar: each relevant family independently active with prob `p_family` at a random level 1..max_level.
* Held-out combinations: scenarios containing BOTH families of a forbidden pair are never sampled in training, so the
  matching rows of the evaluation table (e.g. occlusion + proprioception noise) are genuinely UNSEEN combinations.
* Selection uses a FIXED degraded validation set drawn from the same grammar (clean val is logged alongside).
"""
from __future__ import annotations

import numpy as np

from mmperc.common.seeding import make_rng
from mmperc.sensors.degradation import FAMILIES, MODALITY_OF, Degrader, Scenario, sample_scenario

DEFAULT_FORBIDDEN = (("vision_occlusion", "proprio_noise"), ("vision_occlusion", "proprio_bias"))
KIND_MODALITIES = {"vision": {"vision"}, "proprio": {"proprio"}, "fusion": {"vision", "proprio", "contact"}}


def relevant_families(kind: str) -> tuple[str, ...]:
    mods = KIND_MODALITIES[kind]
    return tuple(f for f in FAMILIES if MODALITY_OF[f] in mods)


def is_forbidden(sc: Scenario, forbidden) -> bool:
    active = {f for f, _ in sc.levels}
    return any(a in active and b in active for a, b in forbidden)


def sample_train_scenario(rng: np.random.Generator, families, forbidden=DEFAULT_FORBIDDEN, p_clean: float = 0.15,
                          p_family: float = 0.3, max_level: int = 4) -> Scenario:
    for _ in range(100):                                   # rejection sampling; a pair-free scenario is always reachable
        sc = sample_scenario(rng, families, p_clean, p_family, max_level)
        if not is_forbidden(sc, forbidden):
            return sc
    return Scenario()


class Mixer:
    """Callable epoch -> degraded model inputs for the train split; `val(...)` gives the fixed mixed validation inputs."""

    def __init__(self, kind: str, cfg: dict, degrader: Degrader, make_inputs, seed: int):
        m = cfg["mixing"]
        self.kind, self.dg, self.make_inputs, self.seed, self.cfg = kind, degrader, make_inputs, seed, cfg
        self.families = relevant_families(kind)
        self.forbidden = tuple(tuple(p) for p in m.get("forbidden_pairs", DEFAULT_FORBIDDEN))
        self.kw = {"p_clean": m["p_clean"], "p_family": m["p_family"], "max_level": m["max_level"]}
        self.every = int(m["refresh_every"])
        self.val_seed = int(m["val_seed"])
        self._cache: tuple[int, dict] | None = None

    def _scenario_fn(self, stream: int):
        return lambda ep_id: sample_train_scenario(make_rng(self.seed, stream, int(ep_id)), self.families, self.forbidden, **self.kw)

    def train_inputs(self, data: dict, epoch: int) -> dict:
        block = epoch // self.every                        # same degraded copy for `refresh_every` consecutive epochs
        if self._cache is None or self._cache[0] != block:
            self._cache = None                             # free the previous copy before building the next
            d = self.dg.apply(data, self._scenario_fn(1 + block), seed=self.seed * 1000 + block)
            self._cache = (block, self.make_inputs(self.kind, d, self.cfg))
        return self._cache[1]

    def val_inputs(self, data: dict) -> dict:
        d = self.dg.apply(data, self._scenario_fn(10 ** 6), seed=self.val_seed)
        return self.make_inputs(self.kind, d, self.cfg)
