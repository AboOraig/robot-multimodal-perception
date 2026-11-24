"""Construction of the classical (non-learned) estimators, shared by run_baselines.py and evaluate_all.py.
All calibration (priors, in-hand offset statistics, tare, pixel noise) comes from the TRAIN split only."""
from __future__ import annotations

from mmperc.baselines.common import Prior
from mmperc.baselines.fusion import VisionOnly, WLSFusion
from mmperc.baselines.kinematics import HandKinematics
from mmperc.baselines.proprio import FKProprio
from mmperc.baselines.vision import ColourVision
from mmperc.sensors.degradation import CubeProjector


def build_classical(sim: dict, train: dict):
    """Returns (estimators: name -> fn(degraded_data, kin_kwargs) -> Estimate, calibration objects)."""
    half = 0.5 * sim["cube"]["size"]
    proj, hk = CubeProjector(sim), HandKinematics(sim)
    vis = ColourVision(proj).fit(train)
    prior = Prior.fit(train)
    tcp_z, mass = sim["robot"]["tcp_offset_z"], sim["cube"]["mass"]
    b3 = FKProprio(hk).fit(train, tcp_z, mass)
    b3c = FKProprio(hk, use_contact=True).fit(train, tcp_z, mass)
    est = {
        "B1a vision(table)": lambda d, kw: VisionOnly(vis, prior, half).predict(d),
        "B1b vision(broad)": lambda d, kw: VisionOnly(vis, prior).predict(d),
        "B3 kin(gated)": lambda d, kw: b3.predict(d, **kw),
        "B3c kin+contact": lambda d, kw: b3c.predict(d, **kw),
        "B6 WLS": lambda d, kw: WLSFusion(vis, b3, prior, half).predict(d, kin_kwargs=kw),
        "B6c WLS+contact": lambda d, kw: WLSFusion(vis, b3c, prior, half).predict(d, kin_kwargs=kw),
    }
    return est, {"vis": vis, "b3": b3, "b3c": b3c}


def oracle_kwargs(dg, sc, deg: dict) -> dict:
    """Noise standard deviations the classical estimators are told (upper bound for the family): the TRUE degradation levels."""
    sq = dg._val("proprio_noise", sc.level("proprio_noise")) if sc.level("proprio_noise") else 1e-3
    cn = dg._val("contact_noise", sc.level("contact_noise"))
    return {"sigma_q": sq, "sigma_qd": 5.0 * sq if sc.level("proprio_noise") else 0.0,
            "sigma_wrench_torque": deg["contact_torque_arm_m"] * cn, "sigma_wrench_force": cn}
