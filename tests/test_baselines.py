"""Baseline and metric tests (synthetic where possible; one tiny generated dataset for integration)."""
import copy

import numpy as np
import pytest
from scipy.stats import chi2

from mmperc.baselines.common import Estimate, Prior, wrap
from mmperc.baselines.contact import skew, static_mask
from mmperc.baselines.fusion import solve_position
from mmperc.baselines.kinematics import HandKinematics, hand_yaw, to_hand, to_world
from mmperc.baselines.proprio import FKProprio
from mmperc.baselines.vision import centroid, plane_jacobian, ray_plane, yaw_harmonic
from mmperc.common.config import load_config
from mmperc.data.dataset import load_split
from mmperc.data.generate import generate_dataset
from mmperc.evaluation.metrics import cluster_bootstrap_ci, per_frame, summarize, tangent_error
from mmperc.sensors.degradation import CubeProjector, Degrader, Scenario


@pytest.fixture(scope="module")
def sim():
    return load_config("configs/simulation.yaml")


@pytest.fixture(scope="module")
def proj(sim):
    return CubeProjector(sim)


@pytest.fixture(scope="module")
def tiny(sim, tmp_path_factory):
    ds = copy.deepcopy(load_config("configs/dataset.yaml"))
    ds["splits"].update(n_train=10, n_val=4, n_test=3, n_test_ood=2)
    ds["store_images"] = False
    out = tmp_path_factory.mktemp("tiny")
    generate_dataset(sim, ds, out, workers=1)
    return out


# ------------------------------------------------------------------ metrics
def test_tangent_error_wraps_phi():
    e = tangent_error(np.array([[0, 0, 0, 3.0]]), np.array([[0, 0, 0, -3.0]]))
    assert e[0, 3] == pytest.approx(2 * np.pi - 6.0)


def test_nll_matches_closed_form_and_calibration_of_a_true_gaussian():
    rng = np.random.default_rng(0)
    n = 40000
    A = rng.normal(size=(4, 4))
    S = 0.01 * (A @ A.T + 0.5 * np.eye(4))          # small angular spread: phi errors are wrapped at pi
    err = rng.multivariate_normal(np.zeros(4), S, n)
    est = Estimate(np.zeros((n, 4)), np.tile(S, (n, 1, 1)), np.ones(n, bool))
    f = per_frame(est, err)
    assert f["d2"].mean() == pytest.approx(4.0, rel=0.03)
    nll_closed = 0.5 * (4 * np.log(2 * np.pi) + np.linalg.slogdet(S)[1] + 4.0)
    assert f["nll"].mean() == pytest.approx(nll_closed, rel=0.02)
    r = summarize(est, err)
    assert r["cov95"] == pytest.approx(0.95, abs=0.01) and r["cov68"] == pytest.approx(0.68, abs=0.01)
    assert chi2.ppf(0.95, 4) > 9


def test_cluster_bootstrap_ci_contains_truth():
    rng = np.random.default_rng(1)
    ep = np.repeat(np.arange(60), 30)
    v = (rng.normal(size=60)[ep] * 0.5 + rng.normal(size=len(ep))) ** 2
    est, lo, hi = cluster_bootstrap_ci(v, ep, stat="rmse", n_boot=400)
    assert lo < est < hi and hi - lo > 0


# ------------------------------------------------------------------ geometry / vision
def test_skew_is_cross_product():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    assert np.allclose(np.einsum("nij,nj->ni", skew(a), b), np.cross(a, b))


def test_centroid_subpixel_and_ray_plane_roundtrip(proj):
    H = 96
    yy, xx = np.mgrid[0:H, 0:H] + 0.5
    for cu, cv in [(30.3, 40.7), (70.6, 20.2)]:
        w = np.exp(-((xx - cu) ** 2 + (yy - cv) ** 2) / (2 * 2.2 ** 2))
        img = np.full((H, H, 3), 100, np.uint8)
        img[..., 0] = (100 + 130 * w).astype(np.uint8)
        img[..., 1:] = (100 - 75 * w)[..., None].astype(np.uint8)
        u, v, m = centroid(img[None], 60.0)
        assert abs(u[0] - cu) < 0.05 and abs(v[0] - cv) < 0.05 and m[0] > 0
    p = np.array([[0.5, 0.1, 0.02], [0.45, -0.15, 0.1]])
    u, v, _ = proj.project(p)
    assert np.allclose(ray_plane(proj, u, v, p[:, 2]), p, atol=1e-9)


def test_yaw_harmonic_recovers_four_fold_angle():
    H = 96
    yy, xx = np.mgrid[0:H, 0:H] + 0.5
    for deg in (-40, -10, 0, 25, 40):
        psi = np.radians(deg)
        c, s = np.cos(psi), np.sin(psi)
        x, y = xx - 48, yy - 48
        w = (1 / (1 + np.exp((np.abs(c * x + s * y) - 3.5) / 0.6)) / (1 + np.exp((np.abs(-s * x + c * y) - 3.5) / 0.6)))
        img = np.full((H, H, 3), 100, np.uint8)
        img[..., 0] = (100 + 130 * w).astype(np.uint8)
        img[..., 1:] = (100 - 75 * w)[..., None].astype(np.uint8)
        S4 = yaw_harmonic(img[None], np.eye(2)[None], 60.0)
        assert abs(np.degrees(wrap(np.angle(S4[0]) - 4 * psi)) / 4) < 3.0


def test_solve_position_recovers_table_object_and_trusts_precise_measurement(proj, sim):
    prior = Prior(np.array([0.5, 0.0, 0.05]), np.diag([0.06, 0.1, 0.04]) ** 2)
    table = prior.table(0.02)
    rng = np.random.default_rng(0)
    p = np.column_stack([rng.uniform(0.4, 0.6, 50), rng.uniform(-0.2, 0.2, 50), np.full(50, 0.02)])
    u, v, _ = proj.project(p)
    est, cov = solve_position(proj, table, np.stack([u, v], 1), np.ones(50, bool), np.eye(2) * 0.2 ** 2)
    assert np.abs(est - p).max() < 1e-3
    # a lifted cube on the table prior gives a (confident) error: the documented monocular weakness
    pl = p.copy(); pl[:, 2] = 0.14
    ul, vl, _ = proj.project(pl)
    estl, _ = solve_position(proj, table, np.stack([ul, vl], 1), np.ones(50, bool), np.eye(2) * 0.2 ** 2)
    assert np.linalg.norm((estl - pl)[:, :2], axis=1).mean() > 0.03
    # no vision, precise 3-D measurement -> posterior equals the measurement
    est2, cov2 = solve_position(proj, prior, np.zeros((50, 2)), np.zeros(50, bool), np.eye(2),
                                p_meas=pl, cov_meas=np.tile(np.eye(3) * 1e-6, (50, 1, 1)), meas_use=np.ones(50, bool))
    assert np.abs(est2 - pl).max() < 1e-3 and np.sqrt(cov2[:, 0, 0]).max() < 2e-3


def test_static_mask_rejects_motion_start_and_first_frame():
    qd = np.zeros((6, 7)); qd[3] = 0.0; qd[4] = 0.2; qd[5] = 0.2
    ep = np.zeros(6, int)
    m = static_mask(qd, ep, 0.05)
    assert not m[0] and m[1] and m[2] and m[3] and not m[4] and not m[5]
    qd2 = np.zeros((4, 7)); qd2[2] = 0.005; qd2[3] = 0.005       # a velocity *jump* of 0.013 rad/s in one frame
    m2 = static_mask(qd2, np.zeros(4, int), 0.05)               # = 0.26 rad/s^2 > a0: acceleration test rejects it
    assert m2[1] and not m2[2]


def test_kinematics_helpers_roundtrip(sim):
    hk = HandKinematics(sim)
    q = np.array(sim["robot"]["home_qpos"])[None] + np.random.default_rng(0).normal(0, 0.05, (3, 7))
    P, R = hk.fk(q)
    v = np.random.default_rng(1).normal(size=(3, 3))
    assert np.allclose(to_hand(R, to_world(R, v)), v)
    Pj, Rj, Jp, Jz = hk.fk(q, jac=True)
    assert np.allclose(P, Pj) and Jp.shape == (3, 3, 7)
    # Jacobian vs finite differences
    dq = np.zeros((1, 7)); dq[0, 2] = 1e-6
    P2, _ = hk.fk(q[:1] + dq)
    assert np.allclose((P2 - P[:1])[0] / 1e-6, Jp[0][:, 2], atol=1e-4)


# ------------------------------------------------------------------ integration on a tiny dataset
def test_kinematic_baseline_gating_and_accuracy(tiny, sim):
    hk = HandKinematics(sim)
    tr, va = load_split(tiny, "train", load_images=False), load_split(tiny, "val", load_images=False)
    b3 = FKProprio(hk).fit(tr, sim["robot"]["tcp_offset_z"], sim["cube"]["mass"])
    est = b3.predict(va)
    t = va["target"].astype(float)
    coupled = va["stage"] == 3
    assert est.informative[coupled].mean() > 0.95
    assert not est.informative[va["stage"] == 0].any()                 # gated: free-motion frames carry no estimate
    r = summarize(est, t, coupled)
    assert r["rmse_xy_mm"] < 9 and r["mae_z_mm"] < 6 and r["yaw_mae_deg"] < 5
    # leaky (ungated) variant is informative everywhere -- and overconfident in free motion
    leaky = FKProprio(hk, gated=False).fit(tr, sim["robot"]["tcp_offset_z"], sim["cube"]["mass"]).predict(va)
    assert leaky.informative.all() and summarize(leaky, t, va["stage"] == 0)["mean_d2"] > 100


def test_gate_uses_degraded_finger_readings_and_dropout_masks(tiny, sim):
    hk = HandKinematics(sim)
    deg = load_config("configs/degradation.yaml")
    dg = Degrader(deg, sim)
    tr, va = load_split(tiny, "train", load_images=False), load_split(tiny, "val", load_images=False)
    b3 = FKProprio(hk).fit(tr, sim["robot"]["tcp_offset_z"], sim["cube"]["mass"])
    d = dg.apply(va, Scenario.make(proprio_dropout=4), 3)
    est = b3.predict(d)
    assert not est.informative[~d["proprio_mask"]].any()
    assert np.array_equal(d["holding_gate"], d["holding_gate"] & d["proprio_mask"])
    # a large finger bias must be able to break the gate (it is recomputed from corrupted readings)
    d2 = dict(va); d2["finger_q"] = va["finger_q"] + 0.02
    from mmperc.data.dataset import holding_gate
    assert not holding_gate(d2["finger_q"]).any()


def test_contact_offset_is_accurate_when_valid_and_only_when_lifted(tiny, sim):
    hk = HandKinematics(sim)
    tr, va = load_split(tiny, "train", load_images=False), load_split(tiny, "val", load_images=False)
    b = FKProprio(hk, use_contact=True).fit(tr, sim["robot"]["tcp_offset_z"], sim["cube"]["mass"])
    c = b.contact
    assert -c.a == pytest.approx(0.757, rel=0.05)                      # fitted hand mass (Menagerie hand + fingers)
    est = b.predict(va)
    v = b.last_contact_valid
    assert v.sum() > 20 and (va["stage"][v] == 3).all()
    assert (est.mean[v, 2] > 0.05).all()
    P, R = hk.fk(va["q"])
    d_true = to_hand(R, va["cube_pos"] - P)
    d_hat = to_hand(R, est.mean[:, :3] - P)
    assert np.abs(d_hat - d_true)[v, 0].max() < 3e-3                   # x offset within 3 mm where valid
    base = FKProprio(hk).fit(tr, sim["robot"]["tcp_offset_z"], sim["cube"]["mass"]).predict(va)
    t = va["target"].astype(float)
    assert summarize(est, t, v)["rmse_xy_mm"] < summarize(base, t, v)["rmse_xy_mm"]


def test_wls_fusion_does_not_assume_table_height_when_proprioception_is_missing(proj):
    """Regression (M4 follow-up): proprio dropout means 'holding state unknown', so B6 must use the broad
    height prior; the table prior would be confidently wrong (tiny covariance at z = cube_half) for a lifted cube."""
    from mmperc.baselines.fusion import WLSFusion
    from mmperc.baselines.vision import ColourVision

    class NoKin:                                           # kinematics unavailable (dropout)
        def predict(self, d, **kw):
            n = len(d["image"])
            return Prior(np.zeros(3), np.eye(3)).estimate(n)

    pos = np.array([[0.5, 0.05, 0.15]])                    # cube lifted 15 cm above the table
    u, v, _ = proj.project(pos)
    img = np.zeros((1, 96, 96, 3), np.uint8)
    r, c = int(round(float(v[0]))), int(round(float(u[0])))
    img[0, r - 2:r + 3, c - 2:c + 3] = (220, 20, 20)
    vis = ColourVision.__new__(ColourVision)
    vis.proj, vis.thr, vis.min_mass = proj, 60.0, 1.0
    vis.bias, vis.cov_pix, vis.phi_off, vis.var_phi = np.zeros(2), np.eye(2), 0.0, 0.1
    prior = Prior(np.array([0.5, 0.0, 0.1]), np.diag([0.06, 0.1, 0.06]) ** 2)
    fusion = WLSFusion(vision=vis, kin=NoKin(), prior=prior, cube_half=0.02)
    d = {"image": img, "proprio_mask": np.array([False])}
    broad = fusion.predict(d).cov[0, 2, 2]
    d_ok = {"image": img, "proprio_mask": np.array([True])}
    table = fusion.predict(d_ok).cov[0, 2, 2]
    assert broad > 50 * table                              # unknown holding state -> z far less certain than table prior
    assert np.sqrt(broad) > 0.02                           # ...at least ~2 cm of height uncertainty
