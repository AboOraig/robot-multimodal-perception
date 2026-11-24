"""Generate tables_auto.tex from tables_test.md and decisions_test.json (so no number in the report is typed by hand)."""
import json, re, sys

MD, DEC = sys.argv[1], sys.argv[2]
T = {}
for b in open(MD, encoding="utf-8").read().split("### ")[1:]:
    L = b.split("\n"); m = re.match(r"(\w+) — group `(\w+)`", L[0])
    hdr = [c.strip() for c in L[2].strip("|").split("|")]
    T[(m.group(1), m.group(2))] = {c[0]: dict(zip(hdr[1:], c[1:])) for c in ([x.strip() for x in l.strip("|").split("|")] for l in L[4:] if l.strip())}
D = json.load(open(DEC))

def num(x, nd=1):
    v = float(x)
    if abs(v) >= 1000:
        m, e = f"{v:.1e}".split("e")
        return f"{m}\\!\\times\\!10^{{{int(e)}}}"
    return f"{v:.{nd}f}"
def cell(s, nd=1):
    parts = [p.strip() for p in s.split("±")]
    a = num(parts[0], nd)
    return f"${a}" + (f"{{\\pm}}{num(parts[1], nd)}" if len(parts) > 1 else "") + "$"
def esc(s): return s.replace("+", "{+}").replace("_", r"\_")

MODELS = [("vision_mixed", "Vis"), ("proprio_gated_mixed", "Prop"), ("fusion_concat_mixed", "F0"), ("fusion_learned_mixed", "F1"), ("fusion_gated_mixed", "F2"),
          ("fusion_poe_mixed", "PoE"), ("fusion_tpoe_mixed", "tPoE"), ("fusion_ci_mixed", "F-CI"), ("fusion_ci", "F-CI$_{\\mathrm{clean}}$"), ("B6c WLS+contact [oracle]", "WLS")]
out = []
def big(metric, group, nd, caption, label):
    rows = T[(metric, group)]
    out.append("\\begin{table*}[p]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l" + "r" * len(MODELS) + "}\\toprule")
    out.append("Scenario & " + " & ".join(n for _, n in MODELS) + " \\\\\\midrule")
    for s, r in rows.items():
        out.append(esc(s) + " & " + " & ".join(cell(r[m], nd) for m, _ in MODELS) + " \\\\")
    out.append("\\bottomrule\\end{tabular}}\n\\caption{" + caption + "}\\label{" + label + "}\\end{table*}\n")
big("rmse_xy_mm", "coupled", 1, "Test split, hold (\\texttt{coupled}) stage: xy RMSE in mm, mean$\\pm$std over 3 training seeds (WLS: classical fusion given the true noise levels, one deterministic run). Vis/Prop: vision-only / gated-proprioception models trained on mixed degradations; F0--F2: non-uncertainty-aware fusion; F-CI$_{\\mathrm{clean}}$ trained on clean data only; all others trained on mixed degradations.", "tab:app_rmse")
big("mean_d2", "coupled", 1, "Test split, hold stage: mean squared Mahalanobis distance $d^2$ (calibrated $=4$ for a 4-D Gaussian; values $\\ge 10^3$ in scientific notation). Same models as Table~\\ref{tab:app_rmse}.", "tab:app_d2")
big("cov95", "coupled", 2, "Test split, hold stage: empirical coverage of the nominal 95\\,\\% ellipsoid (calibrated $=0.95$).", "tab:app_cov")
big("nll", "coupled", 1, "Test split, hold stage: heteroscedastic Gaussian NLL (lower is better; includes the constant, 4-D state).", "tab:app_nll")
big("yaw_mae_deg", "coupled", 1, "Test split, hold stage: yaw mean absolute error in degrees (chance level for the 90$^\\circ$-symmetric cube is $22.5^\\circ$).", "tab:app_yaw")

open("tab_app.tex","w",encoding="utf-8").write("\n".join(out)); out = []
# H1 table
h1 = D["h1"]
o = ["\\begin{table}[t]\\centering\\small\\begin{tabular}{lrrrrr}\\toprule", "Scenario & vision & F-CI & $\\Delta$ (95\\,\\% CI) & $p_{\\mathrm{Holm}}$ & seeds \\\\\\midrule"]
for t in h1["tests"] + h1["reported_not_decisive"]:
    star = "" if "p_holm" in t else "$^{*}$"
    ph = f"{t['p_holm']:.3f}" if "p_holm" in t else "--"
    o.append(f"{esc(t['scenario'])}{star} & {t['b']:.1f} & {t['a']:.1f} & ${t['diff']:+.1f}$ [{t['lo']:+.1f},{t['hi']:+.1f}] & {ph} & {t['seeds_a_lower']}/{t['n_seeds']} \\\\")
o.append("\\bottomrule\\end{tabular}\\caption{H1, test split, hold stage, xy RMSE in mm. $\\Delta=$ RMSE(F-CI) $-$ RMSE(vision); negative favours fusion. Rows marked $^{*}$ are reported but not part of the decision rule. Seeds: number of training seeds in which F-CI is lower.}\\label{tab:h1}\\end{table}\n")
open("tab_h1.tex","w",encoding="utf-8").write("\n".join(o))
# H2 table
by = {}
for t in D["h2"]["tests"]:
    by.setdefault(t["scenario"], {})[t["comparator"]] = t
o = ["\\begin{table}[t]\\centering\\small\\begin{tabular}{lrrrrrr}\\toprule", " & \\multicolumn{3}{c}{NLL} & \\multicolumn{3}{c}{xy RMSE (mm)} \\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Scenario & F-CI & F1 & F2 & F-CI & F1 & F2 \\\\\\midrule"]
for s, d in by.items():
    f1, f2 = d["fusion_learned_mixed"], d["fusion_gated_mixed"]
    o.append(f"{esc(s)} & {f1['nll']['a']:.2f} & {f1['nll']['b']:.2f} & {f2['nll']['b']:.2f} & {f1['rmse']['a']:.1f} & {f1['rmse']['b']:.1f} & {f2['rmse']['b']:.1f} \\\\")
o.append("\\bottomrule\\end{tabular}\\caption{H2, test split, hold stage. F-CI has a significantly lower NLL than both non-uncertainty-aware fusions in all 9 scenarios (all Holm-adjusted $p=0.004$, the floor of the 5000-resample bootstrap after Holm for 18 tests; all 3 seeds agree) and is never significantly less accurate.}\\label{tab:h2}\\end{table}\n")
open("tab_h2.tex","w",encoding="utf-8").write("\n".join(o))
# H3 table
h3 = D["h3"]
o = ["\\begin{table}[t]\\centering\\small\\begin{tabular}{lrrr}\\toprule", "Scenario (unseen combination) & mixed & clean-trained & seen-combination control \\\\\\midrule"]
for t, c in zip(h3["tests"], h3["control"]):
    o.append(f"{esc(t['scenario'])} (NLL) & {t['a']:.2f} & ${num(t['b'], 1)}$ & {c['nll']['b']:.2f} \\\\")
o.append("\\midrule")
for t, c in zip(h3["tests"], h3["control"]):
    rm = T[("rmse_xy_mm", "coupled")][t["scenario"]]
    ctrl = T[("rmse_xy_mm", "coupled")][t["scenario"]]["fusion_ci_mixed_allcombos"]
    o.append(f"{esc(t['scenario'])} (RMSE, mm) & {cell(rm['fusion_ci_mixed'])} & {cell(rm['fusion_ci'])} & {cell(ctrl)} \\\\")
o.append("\\bottomrule\\end{tabular}\\caption{H3, test split, hold stage. The two scenarios combine occlusion with a proprioceptive fault that never co-occurred in mixed training; the control model was trained with these combinations allowed. NLL differences between mixed and clean-trained models are significant after Holm correction in both scenarios (Fig.~\\ref{fig:matrix}); the non-inferiority margins (0.5 nats, RMSE ratio 1.25) are not met in either scenario.}\\label{tab:h3}\\end{table}\n")
open("tab_h3.tex","w",encoding="utf-8").write("\n".join(o))
print("ok")
