"""Non-destructive repository check before publishing.

    python scripts/final_check.py            # static checks only
    python scripts/final_check.py --tests    # also run pytest

Exit code 1 if any FAIL. It never modifies files.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
res: list[tuple[str, str, str]] = []


def rec(level, name, msg=""):
    res.append((level, name, msg))


def gitignore_lines():
    p = ROOT / ".gitignore"
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--max-mb", type=float, default=5.0, help="flag files larger than this")
    args = ap.parse_args()

    for f in ["README.md", "docs/design.md", "docs/preregistration.md", "docs/loss_math.md", "docs/FREEZE.txt"]:
        rec("OK" if (ROOT / f).exists() else "FAIL", f"file {f}")
    rec("OK" if (ROOT / "pyproject.toml").exists() or (ROOT / "requirements.txt").exists() else "FAIL",
        "dependency spec (pyproject.toml or requirements.txt)")
    rec("OK" if any((ROOT / n).exists() for n in ("LICENSE", "LICENSE.txt", "LICENSE.md")) else "WARN",
        "LICENSE file", "add one before making the repo public")

    gi = gitignore_lines()
    if not gi:
        rec("FAIL", ".gitignore", "missing")
    else:
        for need in ["data", "experiments", ".pt", ".npz", ".venv", "robots", "__pycache__"]:
            ok = any(need in l for l in gi)
            rec("OK" if ok else "FAIL", f".gitignore covers '{need}'")

    # large files and heavy extensions inside the repo (outside .venv / .git)
    skip = {".venv", ".git", "__pycache__", ".pytest_cache", "robots"}
    big = []
    for p in ROOT.rglob("*"):
        if any(part in skip for part in p.parts) or not p.is_file():
            continue
        if p.stat().st_size > args.max_mb * 1e6 or p.suffix in {".pt", ".npz", ".h5", ".hdf5", ".mp4"}:
            big.append(f"{p.relative_to(ROOT)} ({p.stat().st_size/1e6:.1f} MB)")
    rec("OK" if not big else "WARN", "large/heavy files in the repo tree",
        "; ".join(big[:10]) + (" ..." if len(big) > 10 else "") + (" (fine if git-ignored)" if big else ""))

    # git: is anything heavy tracked?
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
        heavy = [t for t in tracked if t.endswith((".pt", ".npz", ".h5", ".mp4")) or t.startswith(("data/", "experiments/"))]
        rec("OK" if not heavy else "FAIL", "tracked heavy files", ", ".join(heavy[:10]))
        rec("OK", f"git tracked files: {len(tracked)}")
    except Exception as e:  # not a git repo or git missing
        rec("WARN", "git ls-files", f"unavailable ({type(e).__name__}); run `git init` / check git is installed")

    # hard-coded absolute paths in executable code (docstring/comment mentions are tolerated)
    pat = re.compile(r"""["'](?:[A-Za-z]:[\\/]|/home/|/Users/|/mnt/)""")
    bad = []
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py")):
        if p.name == "final_check.py":
            continue
        in_doc = False
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            s = line.strip()
            if s.count('"""') % 2 == 1:
                in_doc = not in_doc
                continue
            if in_doc or s.startswith("#"):
                continue
            if "help=" in s or "default=" in s and "F:" in s and False:
                continue
            if pat.search(line):
                bad.append(f"{p.relative_to(ROOT)}:{i}")
    rec("OK" if not bad else "WARN", "absolute paths in code", ", ".join(bad[:10]) + (" (inspect: may be help strings)" if bad else ""))

    # report placeholders
    for tex in ROOT.rglob("main.tex"):
        if ".venv" in tex.parts:
            continue
        txt = tex.read_text(encoding="utf-8", errors="ignore")
        for ph in ["Project report", "TODO", "FIXME", "XXX", "\\today"]:
            if ph in txt:
                rec("WARN", f"{tex.relative_to(ROOT)} contains '{ph}'", "replace author/date placeholders")

    if args.tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        rec("OK" if r.returncode == 0 else "FAIL", "pytest", tail)

    w = max(len(n) for _, n, _ in res)
    for lvl, n, m in res:
        print(f"[{lvl:4}] {n:<{w}}  {m}")
    nf = sum(l == "FAIL" for l, _, _ in res)
    print(f"\n{nf} FAIL, {sum(l == 'WARN' for l, _, _ in res)} WARN")
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
