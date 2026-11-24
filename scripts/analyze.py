"""Apply the pre-registered decision rules and write the result tables.

    python scripts/analyze.py --results F:/results/final --split val  --out F:/results/analysis_val     # dry run of the pipeline
    python scripts/analyze.py --results F:/results/final --split test --out F:/results/analysis_test    # the final numbers
Outputs: decisions.json (every test with CI, p, Holm p), tables.md (coupled-stage metric tables, mean +- std over seeds),
and a printed verdict per hypothesis.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from mmperc.data.dataset import resolve
from mmperc.evaluation import hypotheses as H
from mmperc.evaluation.results import Results
from mmperc.evaluation.stats import metric


def table(res: Results, name: str, group: str, fmt: str) -> str:
    models = res.models()
    lines = [f"### {name} — group `{group}`, mean ± std over seeds", "", "| scenario | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    for s in res.scenarios():
        cells = []
        for m in models:
            try:
                v = [metric(e, name) for e in res.ep(s, m, group)]
                cells.append(fmt.format(np.mean(v)) + (f" ± {fmt.format(np.std(v))}" if len(v) > 1 else ""))
            except FileNotFoundError:
                cells.append("n/a")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = Results(resolve(a.results) / a.split)
    out = resolve(a.out); out.mkdir(parents=True, exist_ok=True)
    dec = {"split": a.split, "h1": H.h1(res), "h2": H.h2(res), "h3": H.h3(res), "contact_ablation": H.contact_ablation(res)}
    (out / "decisions.json").write_text(json.dumps(dec, indent=1))
    md = [f"# Tables ({a.split}, simulation only)\n"]
    for g in ("coupled", "all"):
        for m, f in (("rmse_xy_mm", "{:.1f}"), ("nll", "{:.2f}"), ("mean_d2", "{:.1f}"), ("cov95", "{:.2f}"), ("yaw_mae_deg", "{:.1f}")):
            md.append(table(res, m, g, f))
    (out / "tables.md").write_text("\n".join(md))
    print(f"=== {a.split}: verdicts (pre-registered rules) ===")
    print(f"H1  {dec['h1']['verdict']}")
    for t in dec["h1"]["tests"]:
        print(f"    {t['scenario']:<18} ci {t['a']:6.1f} vs vision {t['b']:6.1f} mm  diff {t['diff']:+6.1f} [{t['lo']:+.1f},{t['hi']:+.1f}]  p_holm {t['p_holm']:.3f}  seeds {t['seeds_a_lower']}/{t['n_seeds']}  {'SIG' if t['significant'] else ''}")
    c = dec["h1"]["contrast"]
    print(f"    relative gain occl L4 {c['rho1']:+.2f} vs clean {c['rho0']:+.2f}, contrast CI [{c['lo']:+.2f},{c['hi']:+.2f}]")
    for t in dec["h1"]["reported_not_decisive"]:
        print(f"    (not decisive) {t['scenario']:<26} ci {t['a']:6.1f} vs vision {t['b']:6.1f} mm")
    print(f"H2  {dec['h2']['verdict']}   per comparator: {json.dumps(dec['h2']['per_comparator'])}")
    print(f"H3  {dec['h3']['verdict']}   (H3a {dec['h3']['H3a']}, H3b {dec['h3']['H3b_noninferior']})")
    for t in dec["h3"]["tests"]:
        print(f"    {t['scenario']:<28} NLL mixed {t['a']:6.2f} vs clean {t['b']:6.2f}  diff {t['diff']:+.2f} [{t['lo']:+.2f},{t['hi']:+.2f}]  p_holm {t['p_holm']:.3f}")
    for t in dec["h3"]["control"]:
        print(f"    control {t['scenario']:<26} dNLL hi {t['nll']['hi']:+.2f} (<=0.5)  RMSE ratio hi {t['rmse_ratio']['hi']:.2f} (<=1.25)  noninferior {t['noninferior']}")
    print("saved", out)


if __name__ == "__main__":
    main()
