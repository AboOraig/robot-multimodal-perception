"""Degradation tests: identity at level 0, determinism, order independence, statistics."""
import numpy as np
import pytest

from mmperc.common.config import load_config
from mmperc.sensors.degradation import (FAMILIES, Degrader, Scenario, markov_mask, sample_scenario)


@pytest.fixture(scope="module")
def dg():
    return Degrader(load_config("configs/degradation.yaml"), load_config("configs/simulation.yaml"))


def synth_episode(dg, T=40, seed=0, with_image=True):
    rng = np.random.default_rng(seed)
    ep = {"q": rng.normal(size=(T, 7)), "qd": rng.normal(size=(T, 7)), "finger_q": rng.uniform(0, 0.04, (T, 2)),
          "wrench": rng.normal(size=(T, 6)), "cube_pos": np.tile([0.5, 0.0, 0.02], (T, 1))}
    if with_image:
        img = np.full((T, 96, 96, 3), 100, np.uint8)
        u, v, _ = dg.proj.project(ep["cube_pos"][:1])
        x0, y0 = int(round(u[0])) - 3, int(round(v[0])) - 3
        img[:, y0:y0 + 7, x0:x0 + 7] = (230, 25, 25)                     # 7x7 "cube"
        ep["image"] = img
    return ep


def red(img):
    i = img.astype(int)
    return ((i[..., 0] > 150) & (i[..., 1] < 90) & (i[..., 2] < 90)).sum(axis=(1, 2))


def test_level_zero_is_exact_identity(dg):
    ep = synth_episode(dg)
    sc = Scenario.make(**{f: 0 for f in FAMILIES})
    assert sc.is_clean and sc.name == "clean"
    o = dg.apply_episode(ep, sc, 0, 1)
    for k in ("q", "qd", "finger_q", "wrench", "image"):
        assert np.array_equal(o[k], ep[k]), k
    assert o["proprio_mask"].all() and o["contact_mask"].all()


def test_deterministic_seeded_and_input_not_mutated(dg):
    ep = synth_episode(dg)
    before = {k: v.copy() for k, v in ep.items()}
    sc = Scenario.make(vision_noise=3, vision_occlusion=2, proprio_noise=2, proprio_dropout=2,
                       contact_bias=3, contact_dropout=1, vision_illumination=2)
    a = dg.apply_episode(ep, sc, 7, 5)
    b = dg.apply_episode(ep, sc, 7, 5)
    c = dg.apply_episode(ep, sc, 8, 5)
    d = dg.apply_episode(ep, sc, 7, 6)
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    assert not np.array_equal(a["image"], c["image"]) and not np.array_equal(a["q"], c["q"])
    assert not np.array_equal(a["q"], d["q"])                    # different episode id -> different stream
    for k, v in before.items():
        assert np.array_equal(ep[k], v), f"input {k} was mutated"


def test_apply_is_order_independent_and_keeps_ground_truth(dg):
    e0, e1 = synth_episode(dg, seed=1), synth_episode(dg, seed=2)
    cat = lambda k: np.concatenate([e0[k], e1[k]])
    d = {k: cat(k) for k in e0}
    d["ep_id"] = np.repeat([10, 3], 40)
    d["target"] = np.zeros((80, 4), np.float32)
    sc = Scenario.make(vision_blur=2, proprio_noise=3, contact_noise=2)
    full = dg.apply(d, sc, 11)
    # episode with id 3 alone gives exactly the same values as inside the batch
    alone = dg.apply_episode(e1, sc, 11, 3)
    for k in ("q", "wrench", "image"):
        assert np.array_equal(full[k][40:], alone[k]), k
    assert full["target"] is d["target"] and full["cube_pos"] is d["cube_pos"]     # untouched
    per = dg.apply(d, lambda eid: sc if eid == 10 else Scenario(), 11)
    assert np.array_equal(per["q"][40:], e1["q"]) and not np.array_equal(per["q"][:40], e0["q"])


@pytest.mark.parametrize("level,sigma", [(1, 0.002), (3, 0.01), (4, 0.02)])
def test_proprio_noise_std(dg, level, sigma):
    ep = synth_episode(dg, T=20000, with_image=False)
    o = dg.apply_episode(ep, Scenario.make(proprio_noise=level), 0, 0)
    assert np.std(o["q"] - ep["q"]) == pytest.approx(sigma, rel=0.03)
    assert np.std(o["qd"] - ep["qd"]) == pytest.approx(5.0 * sigma, rel=0.03)
    assert np.std(o["finger_q"] - ep["finger_q"]) == pytest.approx(0.05 * sigma, rel=0.03)


def test_contact_noise_torque_scales_with_moment_arm(dg):
    ep = synth_episode(dg, T=20000, with_image=False)
    o = dg.apply_episode(ep, Scenario.make(contact_noise=4), 0, 0)
    d = o["wrench"] - ep["wrench"]
    sf = dg.levels["contact_noise"][4]                              # level value comes from the (calibrated) config
    assert np.std(d[:, :3]) == pytest.approx(sf, rel=0.03)
    assert np.std(d[:, 3:]) == pytest.approx(sf * dg.cfg["contact_torque_arm_m"], rel=0.03)


def test_bias_offsets_distribution_and_slow_drift(dg):
    assert dg.cfg["bias_drift_frac"] == 0.1
    sigma = 0.02
    first, spread = [], []
    for e in range(400):
        ep = synth_episode(dg, T=60, seed=0, with_image=False)
        o = dg.apply_episode(ep, Scenario.make(proprio_bias=3), 0, e)
        off = o["q"] - ep["q"]
        first.append(off[0])
        spread.append(np.abs(off[-1] - off[0]).mean())
    assert np.std(first) == pytest.approx(sigma, rel=0.1)
    # drift magnitude follows the random-walk law: E|N(0, frac*sigma*sqrt(t))| over (T-1)*dt seconds
    expected = np.sqrt(2 / np.pi) * 0.1 * sigma * np.sqrt(59 * dg.dt)
    assert np.mean(spread) == pytest.approx(expected, rel=0.1)
    assert np.mean(spread) < 0.3 * sigma                           # slow compared with the constant offset


def test_markov_dropout_rate_and_burst_length():
    rng = np.random.default_rng(0)
    for p in (0.05, 0.2, 0.4):
        m = markov_mask(rng, 300000, p, 5.0)
        assert 1 - m.mean() == pytest.approx(p, abs=0.01)
        d = np.diff(np.concatenate([[0], (~m).astype(int), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        assert np.mean(ends - starts) == pytest.approx(5.0, rel=0.08)
    assert markov_mask(rng, 10, 0.0, 5.0).all()


def test_dropout_zero_fills_and_flags(dg):
    ep = synth_episode(dg, T=400, with_image=False)
    o = dg.apply_episode(ep, Scenario.make(proprio_dropout=4, proprio_bias=2, contact_dropout=4), 0, 0)
    pm, cm = ~o["proprio_mask"], ~o["contact_mask"]
    assert pm.any() and cm.any()
    assert (o["q"][pm] == 0).all() and (o["qd"][pm] == 0).all() and (o["finger_q"][pm] == 0).all()
    assert (o["wrench"][cm] == 0).all()
    assert not np.array_equal(o["q"][~pm], ep["q"][~pm])          # bias still applied to available frames


def test_targeted_occlusion_ladder_hides_the_cube(dg):
    ep = synth_episode(dg, T=80)
    vis = [red(dg.apply_episode(ep, Scenario.make(vision_occlusion=k), 0, 0)["image"]).mean() / 49 for k in range(1, 5)]
    assert all(vis[i] >= vis[i + 1] for i in range(3)), vis
    assert vis[0] < 0.95 and vis[3] == 0.0, vis


def test_random_occlusion_area_fraction(dg):
    ep = synth_episode(dg, T=60)
    ep["image"][:] = 0
    for k, a in ((1, 0.10), (3, 0.50)):
        o = dg.apply_episode(ep, Scenario.make(vision_occlusion_random=k), 0, 0)["image"]
        assert (o[..., 0] == 127).mean() == pytest.approx(a, abs=0.04)


def test_image_ops_shape_dtype_and_effect(dg):
    ep = synth_episode(dg)
    base = ep["image"].astype(float)
    for fam in ("vision_noise", "vision_blur", "vision_illumination"):
        o = dg.apply_episode(ep, Scenario.make(**{fam: 4}), 0, 0)["image"]
        assert o.dtype == np.uint8 and o.shape == ep["image"].shape
        assert not np.array_equal(o, ep["image"])
    blur = dg.apply_episode(ep, Scenario.make(vision_blur=4), 0, 0)["image"].astype(float)
    energy = lambda x: (np.diff(x, axis=2) ** 2).mean()           # (total variation of a step edge is blur-invariant)
    assert energy(blur) < 0.4 * energy(base)
    ill = [dg.apply_episode(ep, Scenario.make(vision_illumination=4), 0, e)["image"].mean() for e in range(20)]
    assert np.std(ill) > 5                                         # lighting varies between episodes


def test_scenario_validation_and_sampling():
    with pytest.raises(KeyError):
        Scenario.make(nonsense=1)
    with pytest.raises(ValueError):
        Scenario.make(vision_noise=5)
    assert Scenario.make(vision_noise=2, proprio_noise=1).name == "proprio_noise:1+vision_noise:2"
    rng = np.random.default_rng(0)
    fam = ("vision_noise", "proprio_noise")
    sc = [sample_scenario(rng, fam) for _ in range(2000)]
    assert all(set(dict(s.levels)) <= set(fam) for s in sc)
    assert 0.10 < np.mean([s.is_clean for s in sc]) < 0.20
    assert all(0 < v <= 4 for s in sc for _, v in s.levels)


def _grey_centroid_error_px(dg_, ep, level, n_ep=60):
    """Median over episodes (ep_id = decoy/jitter draw) of |centroid of occluder-grey pixels - projected cube| in pixels."""
    ys, xs = np.mgrid[:96, :96]
    u, v, _ = dg_.proj.project(ep["cube_pos"])
    errs, areas = [], []
    for e in range(n_ep):
        o = dg_.apply_episode(ep, Scenario.make(vision_occlusion=level), 0, e)["image"]
        m = (np.abs(o.astype(int) - 127).sum(-1) < 6)
        n = np.maximum(m.sum((1, 2)), 1)
        errs.append(np.hypot((m * xs).sum((1, 2)) / n - u, (m * ys).sum((1, 2)) / n - v).mean())
        areas.append(m.sum((1, 2)).mean())
    return float(np.median(errs)), float(np.mean(areas))


def test_decoys_make_the_occluder_uninformative_about_the_cube_position():
    """Regression (found with the cube-invisible control): a single occluder centred on the cube leaks its position."""
    import copy
    sim, cfg = load_config("configs/simulation.yaml"), load_config("configs/degradation.yaml")
    plain_cfg = copy.deepcopy(cfg); plain_cfg["occlusion_decoys"] = 0
    plain, decoy = Degrader(plain_cfg, sim), Degrader(cfg, sim)
    ep = synth_episode(plain, T=5)
    e0, a0 = _grey_centroid_error_px(plain, ep, 4)
    e2, a2 = _grey_centroid_error_px(decoy, ep, 4)
    assert e0 < 2.0                                    # single square: its centre IS the cube (up to the jitter)
    assert e2 > 3.0 * e0 and e2 > 5.0 and a2 > 1.5 * a0   # decoys: grey blobs no longer mark the cube (~9 px vs 1.5 px)
    cover = decoy.apply_episode(ep, Scenario.make(vision_occlusion=4), 0, 0)["image"]
    assert red(cover).max() == 0                       # the cube is still fully hidden
    again = decoy.apply_episode(ep, Scenario.make(vision_occlusion=4), 0, 0)["image"]
    assert np.array_equal(cover, again)                # deterministic


def test_vision_dropout_blanks_frames_flags_them_and_leaves_other_streams_untouched(dg):
    from mmperc.sensors.degradation import _FAMILY_ID
    assert _FAMILY_ID["vision_occlusion"] == 100 and _FAMILY_ID["contact_dropout"] == 110   # old RNG streams unchanged
    ep = synth_episode(dg, T=400)
    o = dg.apply_episode(ep, Scenario.make(vision_dropout=4), 0, 5)
    m = o["vision_mask"]
    assert 1 - m.mean() == pytest.approx(0.40, abs=0.10) and (~m).any() and m.any()
    assert (o["image"][~m] == 0).all() and np.array_equal(o["image"][m], ep["image"][m])
    assert o["proprio_mask"].all() and o["contact_mask"].all() and np.array_equal(o["q"], ep["q"])
    assert dg.apply_episode(ep, Scenario(), 0, 5)["vision_mask"].all()
    assert np.array_equal(o["image"], dg.apply_episode(ep, Scenario.make(vision_dropout=4), 0, 5)["image"])   # deterministic
