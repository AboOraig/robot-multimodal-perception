"""Smoke test: the figure script runs end-to-end on synthetic result files and writes all figures."""
import glob
import subprocess
import sys

import numpy as np

from mmperc.common.config import repo_root
from mmperc.evaluation import hypotheses as H
from test_hypotheses import _good_world, _world


def test_make_figures_runs_and_writes_every_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "N_BOOT", 150)
    _world(tmp_path, _good_world)
    for f in glob.glob(str(tmp_path / "test" / "s*.npz")):          # add the fusion-weight arrays the real evaluation stores
        z = dict(np.load(f))
        for k in [k for k in z if k.startswith("fusion_ci_mixed__") and k.endswith("__counts")]:
            z[k[:-8] + "__w_sum"] = np.outer(z[k], [.5, .3, .2])
        np.savez(f, **z)
    root = repo_root()
    r = subprocess.run([sys.executable, str(root / "scripts/analyze.py"), "--results", str(tmp_path), "--split", "test", "--out", str(tmp_path / "an")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    r = subprocess.run([sys.executable, str(root / "scripts/make_figures.py"), "--results", str(tmp_path), "--split", "test",
                        "--analysis", str(tmp_path / "an"), "--out", str(tmp_path / "fig")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    assert len(list((tmp_path / "fig").glob("fig*.png"))) == 7
