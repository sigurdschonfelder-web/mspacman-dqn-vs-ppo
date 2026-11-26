#!/usr/bin/env python3
"""Record full Ms. Pac-Man games (all lives) for trace-format runs, as MP4 videos.

The KV evaluation stops each episode at the first lost life (or at 1,000 points), so the logs
only cover one life. For each candidate episode this script

  1. rebuilds the eval env and replays the logged actions of that episode (life 1), checking
     every reward against the trace and the model's Q-values / action probabilities against the
     logged ones, so we know the env and the observation pipeline match the original eval;
  2. lets the saved model keep playing the same game until game over (DQN greedy, PPO sampled,
     as in the eval);
  3. ranks the candidates by full-game score and renders the best, median and worst as MP4.

Output: <out>/<setting>_<AGENT>_<label>.mp4 and <out>/manifest.json

    python download_models.py            # fetch the models from the GitHub release first
    python make_full_games.py runs -o full_games --workers 6
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from make_run_gifs import TRACE_SETTINGS, draw_frame, load_trace, read_run

MAX_STEPS = 10_000  # agent steps per full game (40,000 frames), a safety cap
PICKS = ("best", "median", "worst")
SETTINGS = ("A", "B", "C")  # D cannot be reproduced from the logs, see make_run_gifs.py


def make_env(setting: dict):
    """The KV eval env without the one-life episode cut: noop reset, 4-frame max-skip,
    RGB 84x84 (cv2 INTER_AREA). Verified against logged Q-values to ~1e-5."""
    import ale_py
    import cv2
    import gymnasium as gym
    from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv

    class RGBWarp(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)

        def observation(self, frame):
            return cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

    gym.register_envs(ale_py)
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array", frameskip=1,
                   repeat_action_probability=setting["sticky"], full_action_space=False)
    return RGBWarp(MaxAndSkipEnv(NoopResetEnv(env, noop_max=30), skip=4))


class Stack:
    """SB3 VecFrameStack + VecTransposeImage for one env: 4 RGB frames -> (12, 84, 84)."""

    def __init__(self, obs):
        self.buf = np.zeros((84, 84, 12), np.uint8)
        self.buf[..., -3:] = obs

    def push(self, obs):
        self.buf = np.roll(self.buf, -3, axis=2)
        self.buf[..., -3:] = obs

    def model_input(self):
        return self.buf.transpose(2, 0, 1)[None]


def action_scores(model, algo, stack):
    import torch

    with torch.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(stack.model_input())
        if algo == "DQN":
            return model.q_net(obs_t)[0].numpy()
        return model.policy.get_distribution(obs_t).distribution.probs[0].numpy()


def reach_episode(env, trace_dir, seed, episode):
    """Reset with the eval seed and replay the earlier episodes of that seed (the eval ran them
    back to back in one env), so the env RNG and game state match the start of `episode`."""
    obs, _ = env.reset(seed=seed)
    for ep in range(episode):
        actions, _ = load_trace(trace_dir, seed, ep)
        for a in actions:
            env.step(int(a))
        obs, _ = env.reset()
    return obs


def play_full_game(model, algo, env, trace_dir, seed, episode):
    """Return a dict with the full action list and stats. Life 1 = the logged episode."""
    obs = reach_episode(env, trace_dir, seed, episode)
    stack = Stack(obs)
    logged_actions, logged_rewards = load_trace(trace_dir, seed, episode)
    with np.load(trace_dir / f"seed{seed}_ep{episode:03d}.npz") as d:
        logged_scores = d["scores"]
    ale = env.unwrapped.ale
    actions, score, life_scores, lives = [], 0.0, [], ale.lives()
    reward_ok, max_score_diff = True, 0.0

    for i, a in enumerate(logged_actions):  # life 1, replayed from the log
        sc = action_scores(model, algo, stack)
        max_score_diff = max(max_score_diff, float(np.abs(sc - logged_scores[i]).max()))
        obs, r, term, trunc, _ = env.step(int(a))
        reward_ok &= r == logged_rewards[i]
        actions.append(int(a)); score += r; stack.push(obs)
        if ale.lives() < lives:
            life_scores.append(score); lives = ale.lives()
    logged_len, logged_score = len(actions), score

    rng = np.random.default_rng(seed * 1000 + episode)
    done = term or trunc
    while not done and len(actions) < MAX_STEPS:  # the model plays on
        sc = action_scores(model, algo, stack)
        a = int(np.argmax(sc)) if algo == "DQN" else int(rng.choice(len(sc), p=sc / sc.sum()))
        obs, r, term, trunc, _ = env.step(a)
        actions.append(a); score += r; stack.push(obs)
        if ale.lives() < lives:
            life_scores.append(score); lives = ale.lives()
        done = term or trunc
    return {"seed": seed, "episode": episode, "actions": actions, "score": score, "length": len(actions),
            "life_scores": life_scores, "logged_len": logged_len, "logged_score": logged_score,
            "game_over": bool(done), "reward_ok": bool(reward_ok), "max_score_diff": max_score_diff}


def render(env, trace_dir, game, title, path: Path, fps: int, scale: int):
    import imageio.v2 as imageio

    reach_episode(env, trace_dir, game["seed"], game["episode"])
    ale = env.unwrapped.ale
    names = env.unwrapped.get_action_meanings()
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1,
                                pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"])
    score, lives, flash, life = 0.0, ale.lives(), 0, 1
    for step, a in enumerate(game["actions"], start=1):
        _, r, *_ = env.step(a)
        score += r
        if ale.lives() < lives:
            lives, flash, life = ale.lives(), 10, life + 1
        flash = max(flash - 1, 0)
        part = f"life {min(life, 3)}: " + ("logged" if step <= game["logged_len"] else "model plays on")
        info = f"step {step}  score {score:.0f}  lives {lives}  {names[a]}"
        frame = draw_frame(env.render(), f"{title}  {part}", info, flash > 0, scale)
        writer.append_data(np.asarray(frame.convert("RGB")))
    for _ in range(fps):  # hold the final frame for a second
        writer.append_data(np.asarray(frame.convert("RGB")))
    writer.close()
    assert score == game["score"], f"render replay diverged: {score} vs {game['score']}"


def process(run, setting, n_candidates, out: Path, fps: int, scale: int):
    import torch
    from stable_baselines3 import DQN, PPO

    torch.set_num_threads(1)
    ev = next(e for e in run.evals if e.setting == setting)
    model = (DQN if run.agent == "DQN" else PPO).load(run.path / "model.zip", device="cpu")
    env = make_env(TRACE_SETTINGS[setting])
    # candidates: the first episodes of every eval seed, spread evenly over the seeds
    seeds = sorted({r["seed"] for r in ev.rows})
    per_seed = -(-n_candidates // len(seeds))
    games = [play_full_game(model, run.agent, env, ev.trace_dir, s, k) for s in seeds for k in range(per_seed)]
    ranked = sorted(games, key=lambda g: (g["score"], -g["length"]))
    picks = {"worst": ranked[0], "median": ranked[len(ranked) // 2], "best": ranked[-1]}
    results = []
    for label, g in picks.items():
        path = out / f"{setting}_{run.agent}_{label}.mp4"
        render(env, ev.trace_dir, g, f"{run.name.replace('_base_1m', '')} {label}", path, fps, scale)
        results.append({"setting": setting, "agent": run.agent, "run": run.name, "label": label,
                        "file": path.name, "kb": path.stat().st_size // 1024,
                        **{k: v for k, v in g.items() if k != "actions"}})
    env.close()
    summary = {"setting": setting, "agent": run.agent, "run": run.name, "candidates": len(games),
               "scores": [g["score"] for g in games], "game_over": sum(g["game_over"] for g in games),
               "life1_rewards_match": sum(g["reward_ok"] for g in games),
               "life1_max_score_diff": max(g["max_score_diff"] for g in games)}
    return results, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="folders containing trace-format runs")
    p.add_argument("-o", "--out", type=Path, default=Path("full_games"))
    p.add_argument("--tag", default="base_1m", help="run tag to use (default: base_1m)")
    p.add_argument("--candidates", type=int, default=24, help="full games played per agent before picking")
    p.add_argument("--fps", type=int, default=30, help="video speed; 15 = real time (default: 30)")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = [read_run(c.parent) for d in args.inputs for c in sorted(d.rglob("config.json"))
            if (c.parent / "model.zip").exists() and c.parent.name.endswith(args.tag)]
    runs = [r for r in runs if r.fmt == "trace"]
    jobs = []
    for setting in SETTINGS:
        for agent in ("DQN", "PPO"):
            cands = [r for r in runs if r.agent == agent and r.train_setting == setting
                     and any(e.setting == setting for e in r.evals)]
            if not cands:
                continue
            ev = lambda r: next(e for e in r.evals if e.setting == setting)
            # the same agent the results page shows: highest success rate, then mean score
            top = max(cands, key=lambda r: (sum(x["success"] for x in ev(r).rows), sum(x["score"] for x in ev(r).rows)))
            jobs.append((top, setting))

    videos, summaries = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, run, s, args.candidates, args.out, args.fps, args.scale): (run.name, s)
                   for run, s in jobs}
        for f in as_completed(futures):
            name, s = futures[f]
            try:
                res, summ = f.result()
            except Exception as e:
                print(f"[failed] {name} on {s}: {e}", flush=True)
                continue
            videos += res; summaries.append(summ)
            print(f"{name} on {s}: {summ['candidates']} full games, life 1 rewards match "
                  f"{summ['life1_rewards_match']}/{summ['candidates']}, max Q/prob diff "
                  f"{summ['life1_max_score_diff']:.1e}; picked " +
                  ", ".join(f"{v['label']} {v['score']:.0f}" for v in res), flush=True)
    (args.out / "manifest.json").write_text(json.dumps({"videos": videos, "agents": summaries}, indent=1))
    print(f"{len(videos)} videos written to {args.out}")


if __name__ == "__main__":
    main()
