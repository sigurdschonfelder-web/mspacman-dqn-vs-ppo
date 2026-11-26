#!/usr/bin/env python3
"""Collect all runs into docs/data.json for the results site (GitHub Pages serves docs/).

Reads every run folder under runs/ (config.json, eval_<setting>_<policy>.csv,
curve.csv, logs/progress.csv), the greedy-vs-epsilon diagnostic and the random baselines,
computes group stats and tests, and copies the full-game videos from make_full_games.py.

    python build_site.py      (needs numpy and scipy)
"""

import csv
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent
EXTRACTED = ROOT / "runs"
GIF_DIR = ROOT / "gifs"
FULL_GAMES = ROOT / "full_games"
SITE = ROOT / "docs"
OUT = SITE / "data.json"
MAIN_TAG = "base_1m"
RNG = np.random.default_rng(0)


def read_rows(path):
    with path.open(newline="") as f:
        return [{"seed": int(r["eval_seed"]), "ep": int(r["episode"]), "outcome": r["outcome"],
                 "score": float(r["score"]), "length": int(r["length"])} for r in csv.DictReader(f)]


def summarize(rows):
    scores = [r["score"] for r in rows]
    n = len(rows)
    return {
        "mean": round(statistics.mean(scores), 1),
        "median": statistics.median(scores),
        "std": round(statistics.pstdev(scores), 1),
        "min": min(scores), "max": max(scores),
        "success": round(sum(r["outcome"] == "success" for r in rows) / n, 3),
        "timeout": round(sum(r["outcome"] == "timeout" for r in rows) / n, 3),
        "mean_length": round(statistics.mean(r["length"] for r in rows)),
        # a "game" is the exact (score, length) pair; identical pairs are almost surely replays
        "distinct": len({(r["score"], r["length"]) for r in rows}),
    }


def boot_ci(values, n=5000):
    v = np.asarray(values, dtype=float)
    means = RNG.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


def read_curve(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {"steps": [int(r["interactions"]) for r in rows],
            "score": [round(float(r["mean_return"]), 1) for r in rows],
            "success": [float(r["success_rate"]) for r in rows]}


def read_entropy(path, points=40):
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("train/entropy_loss")]
    if not rows:  # DQN logs have no entropy
        return None
    idx = np.linspace(0, len(rows) - 1, points).round().astype(int)
    return {"steps": [int(rows[i]["time/total_timesteps"]) for i in idx],
            "entropy": [round(-float(rows[i]["train/entropy_loss"]), 4) for i in idx]}


def main():
    runs, duplicates = [], []
    for cfg_path in sorted(EXTRACTED.rglob("config.json")):
        d = cfg_path.parent
        if not any(d.glob("eval_*_*.csv")):  # models live in the GitHub release, not needed here
            continue
        cfg = json.loads(cfg_path.read_text())
        evals = {}
        for p in sorted(d.glob("eval_*_*.csv")):
            setting, policy = p.stem.removeprefix("eval_").split("_", 1)
            rows = read_rows(p)
            evals[setting] = {"policy": policy, **summarize(rows),
                              "scores": [int(r["score"]) for r in rows],
                              "outcomes": "".join(r["outcome"][0] for r in rows)}
        run = {
            "name": d.name, "agent": cfg["algo"].upper(), "setting": cfg["setting"], "seed": cfg["seed"],
            "tag": cfg["tag"], "group": f'{cfg["algo"].upper()}_{cfg["setting"]}_{cfg["tag"]}',
            "params": {k: v for k, v in cfg["hparams"].items()
                       if k not in ("optimize_memory_usage", "replay_buffer_kwargs")},
            "overrides": cfg.get("overrides", {}),
            "runtime_s": round(cfg["runtime_train_s"]), "interactions": cfg["interactions"],
            "n_envs": cfg["n_envs"], "curve": read_curve(d / "curve.csv"), "evals": evals,
        }
        if (d / "logs" / "progress.csv").exists() and (ent := read_entropy(d / "logs" / "progress.csv")):
            run["entropy"] = ent
        if cfg["tag"] == "base_logged":
            duplicates.append(run)  # same config and seed as base_1m, rerun with SB3 logging
        else:
            runs.append(run)

    # base_logged reruns: did they reproduce base_1m exactly?
    by_name = {r["name"]: r for r in runs}
    reproduced = []
    for dup in duplicates:
        orig = by_name.get(dup["name"].replace("base_logged", MAIN_TAG))
        same = orig is not None and all(
            orig["evals"][s]["scores"] == e["scores"] for s, e in dup["evals"].items())
        reproduced.append({"run": dup["name"], "identical": same})
        if same and "entropy" in dup:
            orig["entropy"] = dup["entropy"]  # the rerun is identical, so its SB3 log describes base_1m too

    # ---- groups over seeds ----
    groups = defaultdict(list)
    for r in runs:
        for s in r["evals"]:
            groups[(r["group"], s)].append(r)
    group_stats = []
    for (g, s), rs in sorted(groups.items()):
        e = [r["evals"][s] for r in rs]
        pooled_scores = [x for ev in e for x in ev["scores"]]
        pooled_out = "".join(ev["outcomes"] for ev in e)
        group_stats.append({
            "group": g, "agent": rs[0]["agent"], "train": rs[0]["setting"], "tag": rs[0]["tag"],
            "eval": s, "policy": e[0]["policy"], "n_seeds": len(rs),
            "seed_scores": [ev["mean"] for ev in e], "seed_success": [ev["success"] for ev in e],
            "score": round(statistics.mean(ev["mean"] for ev in e), 1),
            "success": round(statistics.mean(ev["success"] for ev in e), 3),
            "score_ci": boot_ci([ev["mean"] for ev in e]), "success_ci": boot_ci([ev["success"] for ev in e]),
            "seed_sd_success": round(statistics.stdev(ev["success"] for ev in e), 3),
            "pooled_median": statistics.median(pooled_scores),
            "pooled_max": max(pooled_scores),
            "failure": round(pooled_out.count("f") / len(pooled_out), 3),
            "timeout": round(pooled_out.count("t") / len(pooled_out), 3),
            "hist": np.histogram(pooled_scores, bins=np.arange(0, 1701, 100))[0].tolist(),
            "hist_success": np.histogram([x for x, o in zip(pooled_scores, pooled_out) if o == "s"],
                                         bins=np.arange(0, 1701, 100))[0].tolist(),
            "distinct": round(statistics.mean(ev["distinct"] for ev in e), 1),
            "mean_length": round(statistics.mean(ev["mean_length"] for ev in e)),
            "runtime_min": round(statistics.mean(r["runtime_s"] for r in rs) / 60, 1),
        })

    def G(group, ev):
        return next(x for x in group_stats if x["group"] == group and x["eval"] == ev)

    def pooled(group, ev):
        return [x for r in runs if r["group"] == group for x in r["evals"][ev]["scores"]]

    tests = []
    for s, train in (("A", "A"), ("B", "B"), ("C", "C"), ("D", "B")):
        d, p = G(f"DQN_{train}_{MAIN_TAG}", s), G(f"PPO_{train}_{MAIN_TAG}", s)
        ts = stats.ttest_ind(d["seed_scores"], p["seed_scores"], equal_var=False)
        tsu = stats.ttest_ind(d["seed_success"], p["seed_success"], equal_var=False)
        pd_, pp_ = pooled(d["group"], s), pooled(p["group"], s)
        u = stats.mannwhitneyu(pd_, pp_)
        tests.append({"name": f"agent_{s}", "kind": "agent", "setting": s,
                      "diff_score": round(d["score"] - p["score"], 1), "p_score": float(ts.pvalue),
                      "diff_success": round(d["success"] - p["success"], 3), "p_success": float(tsu.pvalue),
                      "cles": round(float(u.statistic) / (len(pd_) * len(pp_)), 3)})
    for a in ("DQN", "PPO"):
        rb = [r for r in runs if r["group"] == f"{a}_B_{MAIN_TAG}"]
        b = [r["evals"]["B"]["mean"] for r in rb]
        dd = [r["evals"]["D"]["mean"] for r in rb]
        t = stats.ttest_rel(dd, b)
        tests.append({"name": f"shift_{a}", "kind": "shift", "agent": a,
                      "diff_score": round(statistics.mean(dd) - statistics.mean(b), 1),
                      "rel": round(statistics.mean(dd) / statistics.mean(b) - 1, 3), "p_score": float(t.pvalue)})
    # does the flip hurt DQN more than PPO? compare the per-seed relative drops
    drop = {a: [r["evals"]["D"]["mean"] / r["evals"]["B"]["mean"] - 1
                for r in runs if r["group"] == f"{a}_B_{MAIN_TAG}"] for a in ("DQN", "PPO")}
    t = stats.ttest_ind(drop["DQN"], drop["PPO"], equal_var=False)
    tests.append({"name": "shift_dqn_vs_ppo", "kind": "shift_cmp", "p": float(t.pvalue)})

    # hyperparameter sweep on A: each tag vs the base_1m group of the same agent
    sweep = [g for g in group_stats if g["train"] == "A" and g["tag"] != MAIN_TAG]
    for g in sweep:
        base = G(f'{g["agent"]}_A_{MAIN_TAG}', "A")
        ts = stats.ttest_ind(g["seed_success"], base["seed_success"], equal_var=False)
        tsc = stats.ttest_ind(g["seed_scores"], base["seed_scores"], equal_var=False)
        tests.append({"name": g["group"], "kind": "hparam", "agent": g["agent"],
                      "diff_success": round(g["success"] - base["success"], 3), "p_success": float(ts.pvalue),
                      "diff_score": round(g["score"] - base["score"], 1), "p_score": float(tsc.pvalue)})
    hp = sorted((t for t in tests if t["kind"] == "hparam"), key=lambda t: t["p_success"])
    running = 0.0
    for i, t in enumerate(hp):
        running = max(running, min(1.0, (len(hp) - i) * t["p_success"]))
        t["holm_success"] = running

    # ---- greedy vs epsilon diagnostic (DQN on A) ----
    diag_dir = next(EXTRACTED.rglob("diagnostic"))
    diagnostic = []
    for r in (x for x in runs if x["group"] == f"DQN_A_{MAIN_TAG}"):
        row = {"run": r["name"], "greedy": {k: r["evals"]["A"][k] for k in ("success", "mean", "distinct")}}
        for eps in ("0.01", "0.05"):
            p = diag_dir / f'{r["name"]}_eps{eps}.csv'
            if p.exists():
                s = summarize(read_rows(p))
                row[f"eps{eps}"] = {k: s[k] for k in ("success", "mean", "distinct")}
        diagnostic.append(row)

    random_baseline = {}
    for p in EXTRACTED.rglob("random_*.json"):
        j = json.loads(p.read_text())
        random_baseline[j["eval_setting"]] = {"score": j["mean_return"], "success": j["success_rate"]}

    # ---- full-game videos from make_full_games.py ----
    manifest = json.loads((FULL_GAMES / "manifest.json").read_text())
    video_out = SITE / "videos"
    if video_out.exists():
        shutil.rmtree(video_out)
    video_out.mkdir()
    videos = []
    for v in manifest["videos"]:
        shutil.copy(FULL_GAMES / v["file"], video_out / v["file"])
        videos.append({**v, "file": f"videos/{v['file']}"})
    video_agents = manifest["agents"]

    # one-life GIF replays: how many matched the log (shown as a note)
    episodes = list(csv.DictReader((GIF_DIR / "episodes.csv").open())) if (GIF_DIR / "episodes.csv").exists() else []
    match = defaultdict(lambda: [0, 0])
    for e in episodes:
        match[e["eval_setting"]][0] += e["matches_csv"] == "True"
        match[e["eval_setting"]][1] += 1

    for r in runs:  # page payload: scores and outcomes are all the page needs per episode
        for ev in r["evals"].values():
            ev.pop("scores")

    data = {"runs": runs, "groups": group_stats, "tests": tests, "diagnostic": diagnostic,
            "random": random_baseline, "reproduced": reproduced, "videos": videos,
            "video_agents": video_agents, "replay_match": {k: v for k, v in sorted(match.items())},
            "n_episodes_total": sum(len(ev["outcomes"]) for r in runs for ev in r["evals"].values())}
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"{len(runs)} runs (+{len(duplicates)} reruns), {len(group_stats)} groups, {len(videos)} videos "
          f"({sum(v['kb'] for v in videos) // 1024} MB), {OUT.stat().st_size // 1024} KB data")


if __name__ == "__main__":
    main()
