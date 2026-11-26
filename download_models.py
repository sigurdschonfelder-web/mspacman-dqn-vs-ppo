#!/usr/bin/env python3
"""Download the trained models from the GitHub release into their run folders.

The repo holds the logs and traces; the models (1.9 GB) are release assets, one zip per run:
<run>.zip becomes runs/<group>/<run>/model.zip. Only needed for make_full_games.py.

    python download_models.py                     # all 85 models
    python download_models.py A_dqn_seed4_base_1m # just these runs
    python download_models.py --snapshots         # also the policy snapshots (0.6 GB, seed-0 runs)
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "sigurdschonfelder-web/mspacman-dqn-vs-ppo"
TAG = "models-v1"
URL = f"https://github.com/{REPO}/releases/download/{TAG}/"
ROOT = Path(__file__).parent


def fetch(name: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(URL + name) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run names (default: all)")
    p.add_argument("--snapshots", action="store_true", help="also download and unpack snapshots.zip")
    args = p.parse_args()

    run_dirs = sorted(c.parent for c in (ROOT / "runs").rglob("config.json"))
    if args.runs:
        run_dirs = [d for d in run_dirs if d.name in args.runs]
    for i, d in enumerate(run_dirs, 1):
        dest = d / "model.zip"
        if dest.exists():
            continue
        print(f"[{i}/{len(run_dirs)}] {d.name}", flush=True)
        try:
            fetch(f"{d.name}.zip", dest)
        except Exception as e:
            print(f"  failed: {e}", file=sys.stderr)

    if args.snapshots:
        archive = ROOT / "snapshots.zip"
        if not archive.exists():
            print("snapshots.zip", flush=True)
            fetch("snapshots.zip", archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(ROOT)  # paths inside are runs/<group>/<run>/snapshots/policy_<step>.pt


if __name__ == "__main__":
    main()
