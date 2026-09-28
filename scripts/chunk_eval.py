"""Evaluate how ACT's chunk-execution modes hold up under observation latency and cube displacement.

The rollout loop, seeding and success bookkeeping are LeRobot's own `eval_policy`, and the env,
policy and processors are built the way `lerobot-eval` builds them. This file adds two layers:

* `ChunkExecutor` decides which predicted actions reach the robot: the rest of each chunk, the
  next k actions, or LeRobot's `ACTTemporalEnsembler`. It feeds the policy observations that are
  `delay` control steps old and time-aligns every chunk it executes.
* `CubeDisplacement` moves the cube once per episode and records when each task stage is first
  reached.

Episodes are split into independent tasks (condition x seed block) and run in parallel worker
processes. Each worker keeps its envs in-process: LeRobot's AsyncVectorEnv workers are started
with the "forkserver" context and never import gym_aloha, so they cannot create the ALOHA env.

Example, from the repo root inside the LeRobot environment:
    python scripts/chunk_eval.py --checkpoint /tmp/policy --grid validation --seeds 1000:1050 --out out/val
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.policies.pretrained import PreTrainedPolicy

# Initial cube positions are sampled uniformly from this x/y region (gym_aloha.utils.sample_box_pose).
CUBE_REGION = ((0.0, 0.2), (0.4, 0.6))
# Control steps, approved 2026-09-28. The first plan was 70-100, but in 50 official-checkpoint episodes
# the earliest right-gripper contact came at step 84 (full chunk), so 50-75 keeps every move before contact.
DISPLACEMENT_WINDOW = (50, 75)
MODES = (("full", None), ("replan", 50), ("replan", 25), ("replan", 10), ("te", None))
DELAYS = (0, 2, 5, 10, 20)  # control steps at 50 Hz: 0, 40, 100, 200, 400 ms
DISPLACEMENTS_CM = (2, 4, 6)


@dataclass(frozen=True)
class Condition:
    mode: str  # "full", "replan" or "te"
    k: int | None = None  # actions executed per prediction in "replan" mode
    delay: int = 0  # observation latency in control steps
    displacement_cm: float = 0.0
    te_coeff: float = 0.01  # temporal-ensembling weight exp(-c * i), i = 0 for the oldest prediction (ACT's default)

    @property
    def name(self) -> str:
        mode = f"replan{self.k}" if self.mode == "replan" else self.mode
        if self.mode == "te" and self.te_coeff != 0.01:
            mode = f"te{self.te_coeff:g}"  # e.g. te0, te-0.05; the default keeps the plain "te" name
        return f"{mode}_d{self.delay:02d}_m{self.displacement_cm:g}"


def parse_modes(names: list[str] | None) -> list[tuple[str, int | None]]:
    """'full', 'te' or 'replan<k>' -> (mode, k); None -> the default MODES."""
    if not names:
        return list(MODES)
    out = []
    for n in names:
        if n in ("full", "te"):
            out.append((n, None))
        elif n.startswith("replan") and n[6:].isdigit():
            out.append(("replan", int(n[6:])))
        else:
            raise ValueError(f"unknown mode {n!r}")
    return out


def build_grid(name: str, modes: list[tuple[str, int | None]] | None = None) -> list[Condition]:
    modes = modes or list(MODES)
    if name == "validation":
        return [Condition(mode, k) for mode, k in modes]
    if name == "core":
        delay_sweep = [Condition(mode, k, delay=d) for mode, k in modes for d in DELAYS]
        displacement_sweep = [Condition(mode, k, displacement_cm=m) for mode, k in modes for m in DISPLACEMENTS_CM]
        return delay_sweep + displacement_sweep
    if name == "te_weights":
        # Does favoring recent predictions let temporal ensembling react to a moved cube, and at
        # what latency cost? The default (0.01) runs are in the core grid.
        return [Condition("te", delay=d, displacement_cm=m, te_coeff=c)
                for c in (0.0, -0.05) for d, m in ((0, 0), (0, 6), (20, 0))]
    raise ValueError(f"unknown grid {name!r}")


class ChunkExecutor(PreTrainedPolicy):
    """Runs an ACT policy's action chunks when the newest usable observation is `delay` steps old.

    At control step t the policy sees o_s with s = max(0, t - delay): there is no older frame during
    the first `delay` steps of an episode. A chunk predicted from o_s holds the actions for steps
    s, s+1, ..., s+H-1, so execution starts at index t - s.

    Modes:
        full    execute the rest of the chunk, then predict again (H - delay actions per prediction)
        replan  execute k actions, then predict again
        te      predict every step and combine all predictions for the current step with ACT's
                exponential weights (LeRobot's ACTTemporalEnsembler, horizon H - delay)

    With delay = 0 these reproduce LeRobot's ACT with n_action_steps = H, n_action_steps = k, and
    temporal_ensemble_coeff = te_coeff with n_action_steps = 1.

    It subclasses PreTrainedPolicy because LeRobot's eval_policy only accepts PreTrainedPolicy
    instances; training-side methods are passed through to the wrapped policy.
    """

    config_class = ACTConfig
    name = "act_chunk_executor"

    def __init__(self, policy: PreTrainedPolicy, mode: str, delay: int = 0, k: int | None = None, te_coeff: float = 0.01):
        super().__init__(policy.config)
        horizon = policy.config.chunk_size
        if mode not in ("full", "replan", "te"):
            raise ValueError(f"unknown mode {mode!r}")
        if not 0 <= delay < horizon:
            raise ValueError(f"delay must be in [0, {horizon}), got {delay}")
        if mode == "replan" and (k is None or k < 1 or delay + k > horizon):
            raise ValueError(f"replan needs 1 <= k <= {horizon - delay}, got k={k}")
        self.policy = policy
        self.mode, self.delay, self.k, self.te_coeff, self.horizon = mode, delay, k, te_coeff, horizon
        self.n_predictions = 0
        self.reset()

    def get_optim_params(self):
        return self.policy.get_optim_params()

    def forward(self, batch: dict):
        return self.policy.forward(batch)

    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        return self.policy.predict_action_chunk(batch)

    def reset(self) -> None:
        self.policy.reset()
        self.t = 0
        self.history: deque[tuple[int, dict]] = deque(maxlen=self.delay + 1)
        self.queue: deque[torch.Tensor] = deque()
        self.ensembler = (
            ACTTemporalEnsembler(self.te_coeff, self.horizon - self.delay) if self.mode == "te" else None
        )

    def _predict(self) -> tuple[torch.Tensor, int]:
        s, batch = self.history[0]
        self.n_predictions += 1
        return self.predict_action_chunk(batch), self.t - s

    @torch.no_grad()
    def select_action(self, batch: dict, **kwargs) -> torch.Tensor:
        self.history.append((self.t, batch))
        if self.mode == "te":
            chunk, offset = self._predict()
            action = self.ensembler.update(chunk[:, offset : offset + self.horizon - self.delay])
        else:
            if not self.queue:
                chunk, offset = self._predict()
                n = self.horizon - offset if self.mode == "full" else self.k
                self.queue.extend(chunk[:, offset : offset + n].transpose(0, 1))
            action = self.queue.popleft()
        self.t += 1
        return action


class CubeDisplacement(gym.Wrapper):
    """Moves the cube once per seeded episode and records when each reward stage is first reached.

    The move happens at a step drawn uniformly from `window` (inclusive). Its direction is drawn
    uniformly among the directions that keep the cube inside CUBE_REGION, so the new position is
    one the policy saw during training. All draws depend only on the episode seed, so every
    execution mode faces the same perturbation. Episodes begun by vector-env autoreset
    (seed=None) are not recorded.

    ALOHA transfer-cube rewards: 1 right gripper touches the cube, 2 cube lifted, 3 left gripper
    touches it, 4 transfer succeeded (the episode terminates).
    """

    def __init__(self, env: gym.Env, displacement_m: float, window: tuple[int, int] = DISPLACEMENT_WINDOW,
                 seed_offset: int = 10_000, log_steps: int = 0):
        super().__init__(env)
        self.displacement_m = displacement_m
        self.window = window
        self.log_steps = log_steps  # log right-arm joints and gripper-cube distance for the first N steps
        self.seed_offset = seed_offset
        self.records: list[dict] = []
        self._episode: dict | None = None
        self._t = 0

    def _physics(self):
        return self.env.unwrapped._env.physics

    def _log_row(self) -> list[float]:
        physics = self._physics()
        grip = physics.named.data.xpos["vx300s_right/gripper_link"]
        box = physics.named.data.xpos["box"]
        right_arm = physics.data.qpos[8:14]  # vx300s_right waist ... wrist_rotate
        return [self._t, *np.round(right_arm, 5).tolist(), round(float(np.linalg.norm(grip - box)), 5)]

    def _cube_xy(self) -> np.ndarray:
        return self._physics().data.qpos[-7:-5].copy()  # the cube's free joint is the last 7 qpos

    def _sample_delta(self, rng: np.random.Generator, xy: np.ndarray) -> np.ndarray:
        (x0, x1), (y0, y1) = CUBE_REGION
        for _ in range(10_000):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            delta = self.displacement_m * np.array([np.cos(theta), np.sin(theta)])
            x, y = xy + delta
            if x0 <= x <= x1 and y0 <= y <= y1:
                return delta
        raise RuntimeError(f"no direction keeps a {self.displacement_m} m move inside {CUBE_REGION}")

    def _move_cube(self, delta: np.ndarray) -> None:
        # Edit the state in place and recompute derived quantities. physics.reset_context() is
        # not an option here: it resets the whole simulation, arms included, before yielding.
        physics = self._physics()
        (x0, x1), (y0, y1) = CUBE_REGION
        qpos = physics.data.qpos
        qpos[-7] = np.clip(qpos[-7] + delta[0], x0, x1)
        qpos[-6] = np.clip(qpos[-6] + delta[1], y0, y1)
        physics.data.qvel[-6:] = 0.0
        physics.forward()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._t = 0
        self._episode = None
        if seed is None:
            return obs, info
        rng = np.random.default_rng(seed + self.seed_offset)
        t_move = int(rng.integers(self.window[0], self.window[1] + 1))
        start_xy = self._cube_xy()
        moving = self.displacement_m > 0
        delta = self._sample_delta(rng, start_xy) if moving else np.zeros(2)
        self._episode = {
            "seed": int(seed),
            "cube_start_xy": start_xy.round(4).tolist(),
            "t_move": t_move if moving else None,
            "delta_m": delta.round(4).tolist(),
            "reward_before_move": None,
            "first_step_at_reward": {},
        }
        if self.log_steps:
            self._episode["trajectory"] = [self._log_row()]  # rows: step, 6 right-arm joints, gripper-cube distance
        self.records.append(self._episode)
        return obs, info

    def step(self, action):
        ep = self._episode
        if ep is not None and ep["t_move"] is not None and self._t == ep["t_move"]:
            ep["reward_before_move"] = max(map(int, ep["first_step_at_reward"]), default=0)
            qpos = self._physics().data.qpos
            arms_before, xy_before = qpos[:-7].copy(), qpos[-7:-5].copy()
            self._move_cube(np.asarray(ep["delta_m"]))
            ep["moved_m"] = round(float(np.linalg.norm(qpos[-7:-5] - xy_before)), 4)
            ep["arms_unchanged_by_move"] = bool(np.array_equal(qpos[:-7], arms_before))
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        if ep is not None:
            for level in range(1, int(reward) + 1):
                ep["first_step_at_reward"].setdefault(str(level), self._t)
            if self.log_steps and self._t <= self.log_steps and not (terminated or truncated):
                ep["trajectory"].append(self._log_row())
        return obs, reward, terminated, truncated, info


_LOADED: dict[str, tuple] = {}


def _load(checkpoint: str, device: str) -> tuple:
    """Build env config, policy and processors once per worker, the way lerobot-eval does."""
    if checkpoint not in _LOADED:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.configs import AlohaEnv
        from lerobot.policies import make_policy, make_pre_post_processors

        env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
        policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
        policy_cfg.pretrained_path = Path(checkpoint)
        policy_cfg.device = device
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
        _LOADED[checkpoint] = (env_cfg, policy, preprocessor, postprocessor, env_pre, env_post)
    return _LOADED[checkpoint]


def run_task(task: dict) -> dict:
    """Evaluate one condition on one block of seeds and write the episodes to task['out_path']."""
    from lerobot.envs import make_env
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True  # as in lerobot-eval
    if task["deterministic"]:
        # Fixed kernels, so reruns of a condition give identical episodes. The cuBLAS setting
        # must be in place before this worker first touches CUDA, which happens in _load().
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True  # as in lerobot-eval
    set_seed(1000)

    cond = Condition(**task["condition"])
    env_cfg, policy, preprocessor, postprocessor, env_pre, env_post = _load(task["checkpoint"], task["device"])
    vec = make_env(env_cfg, n_envs=task["n_envs"], use_async_envs=False)[env_cfg.type][0]
    vec.envs = [CubeDisplacement(e, cond.displacement_cm / 100.0, window=tuple(task['window']),
                                 log_steps=task.get('log_steps', 0)) for e in vec.envs]
    executor = ChunkExecutor(policy, cond.mode, delay=cond.delay, k=cond.k, te_coeff=cond.te_coeff)
    videos_dir = Path(task["out_path"]).with_suffix("") if task["n_videos"] else None

    start = time.time()
    with torch.no_grad():
        info = eval_policy(
            env=vec,
            policy=executor,
            env_preprocessor=env_pre,
            env_postprocessor=env_post,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=task["n_seeds"],
            max_episodes_rendered=task["n_videos"],
            videos_dir=videos_dir,
            start_seed=task["seed_start"],
        )
    records = {r["seed"]: r for env in vec.envs for r in env.records}
    vec.close()

    episodes = []
    for ep in info["per_episode"]:
        extra = {k: v for k, v in records.get(ep["seed"], {}).items() if k != "seed"}
        episodes.append(
            {
                "condition": cond.name,
                **asdict(cond),
                "seed": ep["seed"],
                "success": bool(ep["success"]),
                "max_reward": float(ep["max_reward"]),
                "sum_reward": float(ep["sum_reward"]),
                **extra,
            }
        )
    result = {
        "condition": cond.name,
        "checkpoint": task["checkpoint_label"],
        "seed_start": task["seed_start"],
        "n_seeds": task["n_seeds"],
        "n_envs": task["n_envs"],
        "n_predictions": executor.n_predictions,
        "eval_s": round(time.time() - start, 1),
        "video_paths": info.get("video_paths", []),
        "episodes": episodes,
    }
    out = Path(task["out_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(out)
    return {k: result[k] for k in ("condition", "seed_start", "n_seeds", "eval_s")} | {
        "n_success": sum(e["success"] for e in episodes)
    }


def _sync_to_hub(out_dir: Path, repo_id: str, path_in_repo: str, message: str) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=message,
        allow_patterns=["*.json", "*.mp4"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="local pretrained_model directory")
    parser.add_argument("--checkpoint-label", default=None, help="name recorded in the results")
    parser.add_argument("--grid", choices=("validation", "core", "te_weights"), required=True)
    parser.add_argument("--only", nargs="*", default=None, help="run only these condition names from the grid")
    parser.add_argument("--modes", nargs="*", default=None, help="execution modes, e.g. full replan50 replan25 replan10 te")
    parser.add_argument("--window", default=f"{DISPLACEMENT_WINDOW[0]}:{DISPLACEMENT_WINDOW[1]}",
                        help="displacement step window start:end (inclusive)")
    parser.add_argument("--seeds", default="1000:1200", help="start:stop, stop exclusive")
    parser.add_argument("--block", type=int, default=25, help="seeds per task")
    parser.add_argument("--n-envs", type=int, default=5, help="envs per task (one batched forward pass)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--videos-per-condition", type=int, default=0)
    parser.add_argument("--log-trajectory", type=int, default=0, metavar="N",
                        help="log right-arm joints and gripper-cube distance for the first N steps of each episode")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="deterministic GPU kernels so reruns are identical (default on; --no-deterministic matches lerobot-eval)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hub-repo", default=None, help="dataset repo to sync results to, e.g. user/act-chunking-study")
    parser.add_argument("--sync-every", type=int, default=16, help="sync to the Hub after this many finished tasks")
    args = parser.parse_args()

    seed_start, seed_stop = map(int, args.seeds.split(":"))
    if args.block % args.n_envs:
        raise SystemExit("--block must be a multiple of --n-envs so every batch uses consecutive seeds")
    hub_path = f"results/{args.out.name}"  # results for --out X live at results/X in the dataset repo
    if args.hub_repo:
        import shutil
        import tempfile

        from huggingface_hub import snapshot_download

        with tempfile.TemporaryDirectory() as tmp:
            try:  # resume: fetch results a previous job already uploaded
                snapshot_download(args.hub_repo, repo_type="dataset", allow_patterns=[f"{hub_path}/**"], local_dir=tmp)
            except Exception as exc:  # noqa: BLE001 - a missing repo or folder just means a fresh start
                print(f"no previous results fetched ({type(exc).__name__})")
            if (Path(tmp) / hub_path).exists():
                shutil.copytree(Path(tmp) / hub_path, args.out, dirs_exist_ok=True)

    window = tuple(map(int, args.window.split(":")))
    grid = build_grid(args.grid, parse_modes(args.modes))
    if args.only:
        unknown = set(args.only) - {c.name for c in grid}
        if unknown:
            raise SystemExit(f"not in grid {args.grid}: {sorted(unknown)}")
        grid = [c for c in grid if c.name in args.only]
    tasks = []
    for cond in grid:
        for i, start in enumerate(range(seed_start, seed_stop, args.block)):
            n = min(args.block, seed_stop - start)
            out_path = args.out / cond.name / f"seeds_{start}-{start + n - 1}.json"
            if out_path.exists():
                continue
            tasks.append(
                {
                    "condition": asdict(cond),
                    "checkpoint": args.checkpoint,
                    "checkpoint_label": args.checkpoint_label or args.checkpoint,
                    "device": args.device,
                    "seed_start": start,
                    "n_seeds": n,
                    "n_envs": min(args.n_envs, n),
                    "n_videos": args.videos_per_condition if i == 0 else 0,
                    "deterministic": args.deterministic,
                    "window": list(window),
                    "log_steps": args.log_trajectory,
                    "out_path": str(out_path),
                }
            )
    total = len(grid) * len(range(seed_start, seed_stop, args.block))
    print(f"{len(tasks)} of {total} tasks to run with {args.workers} workers", flush=True)

    meta = {
        "grid": args.grid,
        "seeds": args.seeds,
        "deterministic": args.deterministic,
        "window": list(window),
        "modes": [c for c in dict.fromkeys(f"replan{k}" if m == "replan" else m for m, k in parse_modes(args.modes))],
        "checkpoint": args.checkpoint_label or args.checkpoint,
        "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device,
    }
    try:
        import lerobot

        meta["lerobot"] = lerobot.__version__
    except Exception:  # noqa: BLE001
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))

    done = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for fut in as_completed(futures):
            summary = fut.result()
            done += 1
            elapsed = time.time() - started
            print(f"[{done}/{len(tasks)} {elapsed / 60:.1f} min] {summary}", flush=True)
            if args.hub_repo and done % args.sync_every == 0:
                _sync_to_hub(args.out, args.hub_repo, hub_path, f"{hub_path}: {done}/{len(tasks)} tasks")
    if args.hub_repo:
        _sync_to_hub(args.out, args.hub_repo, hub_path, f"{hub_path}: finished")
    print(f"finished {len(tasks)} tasks in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
