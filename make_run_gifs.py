#!/usr/bin/env python3
"""Replay trained MsPacman runs (DQN/PPO) and save GIFs of selected evaluation episodes.

For every run folder (config.json + eval CSVs, plus model.zip for model-format runs) the script picks episodes
from the eval CSV (worst/median/best by default), replays them and writes GIFs.

Two run formats are understood:

  model format   config.json has "name"/"agent"; eval_<setting>.csv has seed,score,length,success.
                 Episodes are replayed by running the saved model on the same env seed.
  trace format   config.json has "algo"; eval_<setting>_<policy>.csv has eval_seed,episode,
                 outcome,score,length, and traces_<setting>_<policy>/ holds the actions of every
                 eval episode. Episodes are replayed from the recorded actions (no model needed),
                 and every step's reward is checked against the trace, so a match is exact.

Output:

    <out>/<eval setting>/<agent>/<run>/<label>_seed<seed>[_ep<n>].gif
    <out>/runs.csv       one row per run and eval setting, sorted by mean score
    <out>/episodes.csv   one row per GIF, incl. whether the replay matched the eval CSV
    <out>/index.html     overview page: runs ranked by score, GIFs inline

Inputs can be run zips, folders of run zips, or already extracted run folders.

Examples:
    python make_run_gifs.py runs/ -o gifs --workers 8
    python make_run_gifs.py runs/ --seeds 1006 1042      # specific eval seeds
    python make_run_gifs.py runs/ --dry-run              # only stats + plan, no replays
"""

import argparse
import csv
import html
import json
import statistics
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ENV_ID = "ALE/MsPacman-v5"
PICKS = ("worst", "median", "best")
HEADER_H = 32
HEADER_BG = (20, 20, 20)
LIFE_LOST_BG = (170, 30, 30)
FLASH_STEPS = 10
END_HOLD_MS = 1500


# Trace format: eval env settings (the run's config only stores its own training setting).
# D is the B agent on a vertically flipped maze, with UP/DOWN actions swapped to match.
TRACE_SETTINGS = {
    "A": {"sticky": 0.0, "flip": False},
    "B": {"sticky": 0.25, "flip": False},
    "C": {"sticky": 0.25, "flip": False},
    "D": {"sticky": 0.25, "flip": True},
}
FLIP_SWAP = [0, 4, 2, 3, 1, 7, 8, 5, 6]


@dataclass
class Eval:
    setting: str
    setting_def: dict
    max_steps: int
    rows: list[dict]
    policy: str = "greedy"
    trace_dir: Path | None = None  # set for trace-format runs


@dataclass
class Run:
    path: Path
    name: str
    agent: str
    train_setting: str
    seed: int
    evals: list[Eval]
    fmt: str = "model"  # "model" or "trace"


# ---------- finding and reading runs ----------

def extract_zips(zips: list[Path], workdir: Path) -> None:
    """Extract run zips (and zips inside them), but never a run's model.zip."""
    queue = list(zips)
    while queue:
        zip_path = queue.pop()
        dest = workdir / zip_path.stem
        if dest.exists():
            continue
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        queue += [p for p in dest.rglob("*.zip") if p.name != "model.zip"]


def find_runs(inputs: list[Path], workdir: Path) -> list[Run]:
    zips = [p for p in inputs if p.suffix == ".zip"]
    zips += [z for d in inputs if d.is_dir() for z in d.rglob("*.zip") if z.name != "model.zip"]
    extract_zips(zips, workdir)

    roots = [d for d in inputs if d.is_dir()] + [workdir]
    runs = {}
    for root in roots:
        for cfg_path in sorted(root.rglob("config.json")):
            d = cfg_path.parent
            # model runs need their model; trace runs replay from traces_*/ and need none
            if (d / "model.zip").exists() or any(d.glob("traces_*")):
                run = read_run(cfg_path.parent)
                runs.setdefault(run.name, run)  # same run as zip and folder: keep one
    return sorted(runs.values(), key=lambda r: (r.agent, r.train_setting, r.seed))


def read_run(path: Path) -> Run:
    cfg = json.loads((path / "config.json").read_text())
    if "algo" in cfg:
        return read_trace_run(path, cfg)
    evals = []
    for csv_path in sorted(path.glob("eval_*.csv")):
        setting = csv_path.stem.removeprefix("eval_")
        meta_path = path / f"eval_{setting}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        evals.append(Eval(
            setting=setting,
            setting_def=meta.get("setting_definition", cfg["setting_definition"]),
            max_steps=meta.get("max_steps", 27_000),
            rows=read_eval_csv(csv_path),
        ))
    return Run(path, cfg["name"], cfg["agent"], cfg["setting"], cfg["seed"], evals)


def read_eval_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [
            {"seed": int(r["seed"]), "score": float(r["score"]), "length": int(r["length"]),
             "success": r["success"] == "True"}
            for r in csv.DictReader(f)
        ]


def read_trace_run(path: Path, cfg: dict) -> Run:
    evals = []
    for csv_path in sorted(path.glob("eval_*_*.csv")):
        setting, policy = csv_path.stem.removeprefix("eval_").split("_", 1)
        trace_dir = path / f"traces_{setting}_{policy}"
        if not trace_dir.is_dir():
            continue
        with csv_path.open(newline="") as f:
            rows = [
                {"seed": int(r["eval_seed"]), "episode": int(r["episode"]), "score": float(r["score"]),
                 "length": int(r["length"]), "success": r["outcome"] == "success", "outcome": r["outcome"]}
                for r in csv.DictReader(f)
            ]
        setting_def = {**TRACE_SETTINGS[setting], "goal": cfg.get("success_score", 1000)}
        evals.append(Eval(setting, setting_def, cfg.get("max_eval_steps", 5000), rows, policy, trace_dir))
    return Run(path, path.name, cfg["algo"].upper(), cfg["setting"], cfg["seed"], evals, fmt="trace")


def run_stats(run: Run) -> list[dict]:
    stats = []
    for ev in run.evals:
        scores = [r["score"] for r in ev.rows]
        stats.append({
            "eval_setting": ev.setting, "policy": ev.policy, "agent": run.agent, "run": run.name,
            "train_setting": run.train_setting, "seed": run.seed, "episodes": len(scores),
            "mean_score": round(statistics.mean(scores), 1),
            "std_score": round(statistics.pstdev(scores), 1),
            "median_score": statistics.median(scores),
            "min_score": min(scores), "max_score": max(scores),
            "success_rate": round(sum(r["success"] for r in ev.rows) / len(scores), 3),
            "mean_length": round(statistics.mean(r["length"] for r in ev.rows)),
        })
    return stats


def pick_episodes(rows: list[dict], picks: list[str], seeds: list[int] | None) -> list[tuple]:
    """Return (label, env_seed, csv_row or None) for each episode to record."""
    if seeds:
        by_seed = {}
        for r in rows:
            by_seed.setdefault(r["seed"], r)  # trace format: first episode of that eval seed
        return [("seed", s, by_seed.get(s)) for s in seeds]
    if not rows:
        return []
    ranked = sorted(rows, key=lambda r: (r["score"], r["length"]))
    index = {"worst": 0, "median": len(ranked) // 2, "best": -1}
    return [(p, ranked[index[p]]["seed"], ranked[index[p]]) for p in picks]


# ---------- env, model and replay ----------

def make_env(setting: dict, noop_max: int):
    """Rebuild the eval env. If replays don't match eval_*.csv, make this match your own env code."""
    import ale_py
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    gym.register_envs(ale_py)
    env = gym.make(
        ENV_ID,
        render_mode="rgb_array",
        frameskip=1,  # AtariPreprocessing does the frame skipping
        repeat_action_probability=setting.get("sticky", 0.0),
        mode=setting.get("mode", 0),
    )
    env = AtariPreprocessing(env, noop_max=noop_max, frame_skip=4, screen_size=84, grayscale_obs=True)
    return FrameStackObservation(env, stack_size=4)


def make_trace_env(setting: dict):
    """The trace-format eval env: full game, raw rewards, SB3 AtariWrapper on top of frameskip=1.
    Episode ends (first life lost, goal score, step cap) are decided by the trace lengths."""
    import ale_py
    import gymnasium as gym
    from stable_baselines3.common.atari_wrappers import AtariWrapper

    gym.register_envs(ale_py)
    env = gym.make(ENV_ID, render_mode="rgb_array", frameskip=1,
                   repeat_action_probability=setting["sticky"], full_action_space=False)
    return AtariWrapper(env, terminal_on_life_loss=False, clip_reward=False)


def load_trace(trace_dir: Path, seed: int, episode: int):
    import numpy as np

    with np.load(trace_dir / f"seed{seed}_ep{episode:03d}.npz") as d:
        return d["actions"].astype(int), d["rewards"]


def play_trace(env, ev: Eval, row: dict, check: dict):
    """Yield (rgb_frame, score, lives, action_name, done) while replaying one recorded episode.

    The eval ran all episodes of an eval seed back to back in one env, so the episodes before
    this one are replayed first (without rendering) to reach the same RNG and game state.
    check["ok"] ends up False if any reward differs from the trace."""
    seed, target = row["seed"], row["episode"]
    flip = ev.setting_def["flip"]
    to_env = (lambda a: FLIP_SWAP[a]) if flip else (lambda a: a)
    env.reset(seed=seed)
    for ep in range(target):
        actions, _ = load_trace(ev.trace_dir, seed, ep)
        for a in actions:
            env.step(to_env(a))
        env.reset()

    actions, rewards = load_trace(ev.trace_dir, seed, target)
    action_names = env.unwrapped.get_action_meanings()
    ale = env.unwrapped.ale
    score = 0.0
    for i, a in enumerate(actions):
        _, reward, *_ = env.step(to_env(a))
        if reward != rewards[i]:
            check["ok"] = False
        score += float(reward)
        rgb = env.render()
        yield (rgb[::-1] if flip else rgb), score, ale.lives(), action_names[a], i == len(actions) - 1


def load_model(agent: str, path: Path):
    import stable_baselines3

    return getattr(stable_baselines3, agent.upper()).load(path, device="cpu")


def play_episode(model, env, seed: int, max_steps: int):
    """Yield (rgb_frame, score, lives, action_name, done) after every agent step."""
    obs, _ = env.reset(seed=seed)
    ale = env.unwrapped.ale
    action_names = env.unwrapped.get_action_meanings()
    score = 0.0
    for step in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)  # greedy, like the eval
        obs, reward, terminated, truncated, _ = env.step(int(action))
        score += float(reward)
        done = terminated or truncated or step == max_steps
        yield env.render(), score, ale.lives(), action_names[int(action)], done
        if done:
            break


def draw_frame(rgb, title: str, info: str, life_lost: bool, scale: int) -> Image.Image:
    game = Image.fromarray(rgb)
    game = game.resize((game.width * scale, game.height * scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (game.width, game.height + HEADER_H), LIFE_LOST_BG if life_lost else HEADER_BG)
    canvas.paste(game, (0, HEADER_H))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=11)
    draw.text((4, 3), title, fill="white", font=font)
    draw.text((4, 17), info, fill="white", font=font)
    return canvas.convert("P", palette=Image.Palette.ADAPTIVE)  # palette mode keeps memory low


def record_episode(steps, title: str, args) -> tuple:
    """Turn a play_* step stream into GIF frames.

    Returns (frames, score, length). Halves the frame rate if the episode gets very long."""
    frames, stride, flash, prev_lives = [], 1, 0, None
    step, score = 0, 0.0
    for step, (rgb, score, lives, action, done) in enumerate(steps, start=1):
        flash = FLASH_STEPS if prev_lives is not None and lives < prev_lives else max(flash - 1, 0)
        prev_lives = lives
        if step % stride == 0 or done:  # always keep the final frame
            info = f"step {step}   score {score:.0f}   lives {lives}   {action}"
            frames.append(draw_frame(rgb, title, info, flash > 0, args.scale))
        if len(frames) > args.max_frames:
            frames, stride = frames[::2], stride * 2
    return frames, score, step


def save_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = [round(1000 / fps)] * len(frames)
    durations[-1] = END_HOLD_MS  # pause on the final frame before looping
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


def process_run(run: Run, args) -> list[dict]:
    """Worker: load one model, replay its picked episodes and save the GIFs."""
    model = None
    if run.fmt == "model":
        import torch

        torch.set_num_threads(1)  # parallelism comes from the worker processes
        model = load_model(run.agent, run.path / "model.zip")
    results = []
    for ev in run.evals:
        if run.fmt == "trace":
            env = make_trace_env(ev.setting_def)
        else:
            env = make_env(ev.setting_def, args.noop_max)
            if env.observation_space.shape != model.observation_space.shape:
                raise ValueError(
                    f"env gives observations {env.observation_space.shape}, model expects "
                    f"{model.observation_space.shape}; adjust make_env() to your training wrappers"
                )
        for label, env_seed, row in pick_episodes(ev.rows, args.pick, args.seeds):
            if run.fmt == "trace" and row is None:
                print(f"{run.name} [{ev.setting}]: no recorded episode for seed {env_seed}, skipped")
                continue
            episode = row.get("episode") if row else None
            ep_tag = "" if episode is None else f"_ep{episode:03d}"
            title = f"{run.name}  eval {ev.setting}  {label}  seed {env_seed}{ep_tag.replace('_', ' ')}"
            check = {"ok": True}
            steps = (play_trace(env, ev, row, check) if run.fmt == "trace"
                     else play_episode(model, env, env_seed, ev.max_steps))
            frames, score, length = record_episode(steps, title, args)
            gif = args.out / ev.setting / run.agent / run.name / f"{label}_seed{env_seed}{ep_tag}.gif"
            save_gif(frames, gif, args.fps)
            matches = row is None or (check["ok"] and score == row["score"] and length == row["length"])
            results.append({
                "eval_setting": ev.setting, "policy": ev.policy, "agent": run.agent, "run": run.name,
                "label": label, "env_seed": env_seed, "episode": episode, "score": score, "length": length,
                "csv_score": row and row["score"], "csv_length": row and row["length"],
                "matches_csv": matches, "gif": gif.relative_to(args.out).as_posix(),
            })
            print(f"{run.name} [{ev.setting}] {label} seed {env_seed}: score {score:.0f}, "
                  f"{length} steps{'' if matches else '  <-- differs from eval CSV'}", flush=True)
        env.close()
    return results


# ---------- output ----------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


INDEX_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MsPacman runs</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }}
  table {{ border-collapse: collapse; margin-bottom: 40px; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #333; text-align: left; vertical-align: top; }}
  th {{ cursor: pointer; position: sticky; top: 0; background: #111; }}
  figure {{ display: inline-block; margin: 0 8px 0 0; }}
  figcaption {{ font-size: 12px; color: #aaa; }}
  .warn {{ color: #f88; }}
</style></head><body>
<h1>MsPacman runs</h1>
<p>Ranked by mean eval score. Click a column header to sort. Red caption = replay differs from eval CSV.</p>
{sections}
<script>
document.querySelectorAll("th").forEach((th, col) => th.addEventListener("click", () => {{
  const body = th.closest("table").tBodies[0];
  const key = tr => {{ const t = tr.cells[th.cellIndex].innerText; return isNaN(t) ? t : -Number(t); }};
  const rows = [...body.rows].sort((a, b) => key(a) > key(b) ? 1 : key(a) < key(b) ? -1 : 0);
  if (th.dataset.asc) rows.reverse();
  th.dataset.asc = th.dataset.asc ? "" : "1";
  rows.forEach(r => body.appendChild(r));
}}));
</script>
</body></html>
"""

COLUMNS = ["agent", "run", "mean_score", "std_score", "median_score", "max_score", "success_rate", "mean_length"]


def write_index(path: Path, stats: list[dict], episodes: list[dict]) -> None:
    gifs = {}
    for e in episodes:
        gifs.setdefault((e["run"], e["eval_setting"]), []).append(e)

    sections = []
    for setting in sorted({s["eval_setting"] for s in stats}):
        rows = []
        for rank, s in enumerate((s for s in stats if s["eval_setting"] == setting), start=1):
            figures = "".join(
                f'<figure><img src="{html.escape(e["gif"])}" loading="lazy">'
                f'<figcaption class="{"" if e["matches_csv"] else "warn"}">'
                f'{e["label"]}, seed {e["env_seed"]}, score {e["score"]:.0f}</figcaption></figure>'
                for e in gifs.get((s["run"], setting), [])
            )
            cells = "".join(f"<td>{html.escape(str(s[c]))}</td>" for c in COLUMNS)
            rows.append(f"<tr><td>{rank}</td>{cells}<td>{figures}</td></tr>")
        header = "".join(f"<th>{c}</th>" for c in ["#", *COLUMNS, "episodes"])
        sections.append(f"<h2>Eval setting {html.escape(setting)}</h2>"
                        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>")
    path.write_text(INDEX_TEMPLATE.format(sections="\n".join(sections)), encoding="utf-8")


# ---------- main ----------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="run zips and/or folders containing runs")
    p.add_argument("-o", "--out", type=Path, default=Path("gifs"), help="output folder (default: gifs)")
    p.add_argument("--pick", nargs="+", choices=PICKS, default=list(PICKS), help="which eval episodes per run")
    p.add_argument("--seeds", nargs="+", type=int, help="record these eval seeds instead of --pick")
    p.add_argument("--fps", type=int, default=30, help="GIF speed; 15 = real time (default: 30)")
    p.add_argument("--scale", type=int, default=2, help="pixel upscaling of the 210x160 screen (default: 2)")
    p.add_argument("--max-frames", type=int, default=2000, help="frame cap per GIF; longer episodes skip frames")
    p.add_argument("--noop-max", type=int, default=30, help="random no-ops at reset, must match your eval env")
    p.add_argument("--workers", type=int, default=4, help="runs replayed in parallel (default: 4)")
    p.add_argument("--dry-run", action="store_true", help="write runs.csv and show the plan, no replays")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(args.inputs, args.out / "_extracted")
    if not runs:
        raise SystemExit("No runs found (a run is a folder with config.json and model.zip).")

    stats = sorted((s for run in runs for s in run_stats(run)), key=lambda s: (s["eval_setting"], -s["mean_score"]))
    write_csv(args.out / "runs.csv", stats)
    print(f"Found {len(runs)} runs. Stats written to {args.out / 'runs.csv'}")

    if args.dry_run:
        for run in runs:
            for ev in run.evals:
                plan = ", ".join(f"{label} (seed {seed})" for label, seed, _ in pick_episodes(ev.rows, args.pick, args.seeds))
                print(f"  {run.name} [{ev.setting}]: {plan}")
        return

    episodes = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_run, run, args): run for run in runs}
        for future in as_completed(futures):
            try:
                episodes += future.result()
            except Exception as e:  # one broken run should not stop the rest
                print(f"[skipped] {futures[future].name}: {e}")

    episodes.sort(key=lambda e: (e["eval_setting"], e["agent"], e["run"], e["label"]))
    write_csv(args.out / "episodes.csv", episodes)
    write_index(args.out / "index.html", stats, episodes)

    mismatches = sum(not e["matches_csv"] for e in episodes)
    print(f"\n{len(episodes)} GIFs written. Open {args.out / 'index.html'}")
    if mismatches:
        by_setting = {}
        for e in episodes:
            if not e["matches_csv"]:
                by_setting[e["eval_setting"]] = by_setting.get(e["eval_setting"], 0) + 1
        print(f"Warning: {mismatches} replays differ from the eval CSV "
              f"({', '.join(f'{k}: {v}' for k, v in sorted(by_setting.items()))}). The replay env "
              f"(make_env for model runs, make_trace_env/TRACE_SETTINGS for trace runs) differs from "
              f"the one the eval used for those settings.")


if __name__ == "__main__":
    main()