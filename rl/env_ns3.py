import os
import subprocess
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class RewardConfig:
    reward_mode: str = "fair"   # "global" or "fair"
    beta: float = 0.05          # penalty on Q (ms)
    gamma: float = 5.0          # fairness bonus

def jains_fairness(x: np.ndarray) -> float:
    x = np.maximum(np.array(x, dtype=float), 0.0)
    s1 = float(np.sum(x))
    s2 = float(np.sum(x * x))
    if s1 < 1e-12 or s2 < 1e-12:
        return 0.0
    n = len(x)
    return (s1 * s1) / (n * s2)

def compute_reward(T, Q, cfg: RewardConfig) -> float:
    T = np.array(T, dtype=float)
    Q = np.array(Q, dtype=float)

    if cfg.reward_mode == "global":
        return float(np.sum(T) - cfg.beta * np.sum(Q))

    if cfg.reward_mode == "fair":
        J = jains_fairness(T)
        return float(np.sum(T) + cfg.gamma * J - cfg.beta * np.sum(Q))

    raise ValueError(f"Unknown reward_mode: {cfg.reward_mode}")

class Ns3Env:
    """
    Runs ns-3 scratch/wifi_congestion_dataset and reads per_ap_step_metrics.csv

    Observation: dict with
      T: list[Mbps] per AP
      Q: list[ms] mean delay per AP
      loss: float
      nAps: int
    """
    def __init__(self, ns3_root: str, reward_cfg: RewardConfig, seed: int = 1):
        self.ns3_root = os.path.abspath(ns3_root)
        self.csv_path = os.path.join(self.ns3_root, "per_ap_step_metrics.csv")
        self.reward_cfg = reward_cfg
        self.seed = seed
        self.run_id = 1

    def _run_ns3(self, args: str):
        cmd = ["./ns3", "run", f"scratch/wifi_congestion_dataset -- {args}"]
        subprocess.run(cmd, cwd=self.ns3_root, check=True)

    def step(self, action_args: str) -> tuple[dict, float]:
        # action_args is the string of CLI args passed to ns-3
        self._run_ns3(action_args)

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        last = df.iloc[-1]

        nAps = int(last["nAps"])
        T = [float(last[f"T{i}_Mbps"]) for i in range(nAps)]

        # Q columns are "Q{i}_ms" in our version-safe C++ file
        Q = [float(last[f"Q{i}_ms"]) for i in range(nAps)]

        loss = float(last["lossRate"])

        r = compute_reward(T, Q, self.reward_cfg)
        obs = {"nAps": nAps, "T": T, "Q": Q, "loss": loss}

        self.run_id += 1
        return obs, r
