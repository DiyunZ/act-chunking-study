"""Unit test for ChunkExecutor's time alignment. It needs no simulator or GPU.

A fake policy returns chunks whose entry i equals the absolute control step it is meant for
(observation step s + i). With correct alignment every executed action equals the current step
t in every mode and at every delay, including temporal ensembling, which averages equal values.
The test also checks which observation each prediction used and how often it predicted.

Run: python scripts/test_chunk_executor.py
"""

import sys
from pathlib import Path

import torch
from lerobot.policies.act.configuration_act import ACTConfig
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from chunk_eval import ChunkExecutor  # noqa: E402

H = 100


class FakeACT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = ACTConfig(chunk_size=H, n_action_steps=H)
        self.seen: list[int] = []

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        s = int(batch["t"])
        self.seen.append(s)
        return torch.arange(s, s + H, dtype=torch.float64).reshape(1, H, 1)


def run(mode: str, delay: int, k: int | None = None, steps: int = 400) -> tuple[list[float], list[int]]:
    policy = FakeACT()
    executor = ChunkExecutor(policy, mode, delay=delay, k=k)
    executor.reset()
    actions = [float(executor.select_action({"t": t})) for t in range(steps)]
    return actions, policy.seen


def main() -> None:
    for mode, k in (("full", None), ("replan", 25), ("replan", 10), ("te", None)):
        for delay in (0, 2, 5, 10, 20):
            actions, seen = run(mode, delay, k)
            # Tolerance: ACTTemporalEnsembler keeps its weights in float32 (errors ~1e-5).
            worst = max(abs(a - t) for t, a in enumerate(actions))
            assert worst < 1e-4, (mode, k, delay, f"misaligned by {worst}")
            # Each prediction must use the observation from max(0, t - delay) at its request step t.
            if mode == "te":
                expected = [max(0, t - delay) for t in range(len(actions))]
            else:
                expected, t = [], 0
                while t < len(actions):
                    s = max(0, t - delay)
                    expected.append(s)
                    t += (H - (t - s)) if mode == "full" else k
            assert seen == expected, (mode, k, delay, seen[:6], expected[:6])
            print(f"ok  {mode:<6} k={str(k):<4} delay={delay:<2} predictions={len(seen)}")
    print("all alignment checks passed")


if __name__ == "__main__":
    main()
