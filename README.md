# DQN vs PPO on Ms. Pac-Man

Results, logs and analysis code for our comparison of DQN and PPO on `ALE/MsPacman-v5`.

**Results page:** https://sigurdschonfelder-web.github.io/mspacman-dqn-vs-ppo/

The page covers the headline comparison with statistical tests, score distributions, learning
curves, the mirrored-maze robustness test, the hyperparameter sweep, full-game replays and a
table of every run. Everything on it is computed from the files in this repository.

## Experiment in short

| | |
|---|---|
| Agents | DQN and PPO (Stable-Baselines3 2.9.2, `CnnPolicy`) |
| Training | 1,000,000 environment steps per agent, 5 seeds per configuration. PPO uses 8 parallel envs, DQN one |
| Observation | RGB frames resized to 84×84 (`cv2.INTER_AREA`), 4 frames stacked, so `(12, 84, 84)` |
| Evaluation episode | One life: ends at the first lost life, at 1,000 points (success) or after 5,000 steps (timeout) |
| Evaluation set | 200 episodes per agent: 50 back to back on each of env seeds 1000–1003 |
| Test-time policy | DQN greedy (highest Q-value); PPO samples from its action distribution |

| Setting | Sticky actions | Step penalty (training) | Maze |
|---|---|---|---|
| A | 0.00 | 0 | normal |
| B | 0.25 | 0 | normal |
| C | 0.25 | 0.01 (rewards clipped to ±1 in training) | normal |
| D | 0.25 | 0 | vertically mirrored, UP/DOWN swapped (evaluation only, B agents) |

The hyperparameter sweep runs on setting A, changing one hyperparameter at a time.

## What is where

```
docs/                 the results page (GitHub Pages serves this folder)
  index.html          the page
  data.json           everything the page shows, built by build_site.py
  videos/             full-game videos shown on the page
runs/
  main/               base runs: A, B, C for DQN and PPO (+ B agents evaluated on D)
    summary.csv       per-configuration summary
    random_B.json     random-agent baseline on B (and random_D.json on D)
  sweep/              hyperparameter sweep on A (+ base_logged: PPO baseline rerun with SB3 logging)
    part3_summary.csv per-configuration summary of the sweep
    diagnostic/       DQN on A evaluated with epsilon 0.01 and 0.05 instead of greedy
gifs/                 one-life replays (best/median/worst per run and eval setting), see gifs/index.html
full_games/           full three-life games (MP4) and manifest.json, made by make_full_games.py
make_run_gifs.py      replays logged episodes from their traces and writes GIFs
make_full_games.py    plays full games: logged life 1, then the saved model plays on
build_site.py         computes all statistics and writes docs/data.json
download_models.py    downloads the trained models from the release
```

### One run folder

`runs/<group>/<setting>_<algo>_seed<n>_<tag>/`, for example `runs/main/B_ppo_seed0_base_1m/`:

| File | Content |
|---|---|
| `config.json` | algorithm, setting, seed, hyperparameters, budget, runtime, library versions |
| `eval_<S>_<policy>.csv` | one row per evaluation episode: `eval_seed, episode, outcome, score, length` (`outcome` is success, failure or timeout) |
| `eval_<S>_<policy>.json` | summary of that evaluation (success/failure/timeout rates, mean score, lengths) |
| `traces_<S>_<policy>/seed<s>_ep<k>.npz` | per episode: `actions`, `rewards` and `scores` (DQN Q-values or PPO action probabilities) for every step |
| `curve.csv` | learning curve: every 50,000 steps, 10 evaluation episodes on seed 500 |
| `monitor/` | SB3 training monitor files |
| `logs/progress.csv` | SB3 training log (sweep runs only) |

`S` is the evaluation setting (A–D); `policy` is `greedy` for DQN and `sampled` for PPO.

## Models

The trained models (1.9 GB) are not in the git history. They are attached to the
[`models-v1` release](https://github.com/sigurdschonfelder-web/mspacman-dqn-vs-ppo/releases/tag/models-v1)
as one zip per run, plus `snapshots.zip` with policy snapshots at 0, 100k, 250k and 500k steps
for the seed-0 runs. `download_models.py` puts them in place:

```
python download_models.py                       # all models -> runs/*/*/model.zip
python download_models.py B_ppo_seed0_base_1m   # one run
python download_models.py --snapshots           # also the snapshots
```

## Reproducing the page

Python 3.12. The pinned OpenCV version has a prebuilt wheel; newer ones may try to compile.

```
# statistics and data for the page (no models needed)
uv run --no-project --with numpy --with scipy python build_site.py

# one-life replay GIFs, straight from the traces (no models needed)
uv run --python 3.12 --no-project --with gymnasium==1.4.0 --with ale-py==0.12.1 \
  --with stable-baselines3==2.9.2 --with pillow --with opencv-python-headless==4.10.0.84 \
  python make_run_gifs.py runs -o gifs --workers 8

# full-game videos (needs the models)
python download_models.py
uv run --python 3.12 --no-project --with gymnasium==1.4.0 --with ale-py==0.12.1 \
  --with stable-baselines3==2.9.2 --with pillow --with opencv-python-headless==4.10.0.84 \
  --with imageio --with imageio-ffmpeg python make_full_games.py runs -o full_games --workers 6
uv run --no-project --with numpy --with scipy python build_site.py
```

To preview the page locally: `python -m http.server -d docs` and open http://localhost:8000.

## Checks and known limits

- Replaying the logged actions reproduces every reward of every logged episode on A, B and C
  (255 of 255 replays). For full games, the model's Q-values or action probabilities on life 1
  match the logged ones to within 3e-5, so the observation pipeline is the one used in the evaluation.
- Setting D cannot be replayed exactly with the mirrored-maze wrapper we have, so there are no
  D videos and the D GIFs are flagged as not matching. The D results come straight from the logs.
- In full-game videos only life 1 is a logged evaluation episode. Lives 2 and 3 are the same model
  playing on; they illustrate behaviour and are not part of the statistics.
- The five `base_logged` PPO runs are a rerun of the A baseline with logging switched on. They
  reproduce the original evaluation scores exactly, so the page counts them once.
