import os
import subprocess
import pandas as pd
import numpy as np
from dataclasses import dataclass

NS3_DIR = os.path.expanduser("~/ns-3-dev")
NS3_CSV = os.path.join(NS3_DIR, "per_ap_step_metrics.csv")

@dataclass
class RewardConfig:
    reward_mode: str = "global"   # "global" or "fair"
    beta: float = 0.2             # penalty weight for Q (packets by default)
    gamma: float = 5.0            # fairness weight

def jains_fairness(x: np.ndarray) -> float:
    x = np.maximum(x, 0.0)
    s1 = np.sum(x)
    s2 = np.sum(x * x)
    if s1 <= 1e-12 or s2 <= 1e-12:
        return 0.0
    n = len(x)
    return float((s1 * s1) / (n * s2))

def compute_reward(T, Q, cfg: RewardConfig) -> float:
    T = np.array(T, dtype=float)
    Q = np.array(Q, dtype=float)

    if cfg.reward_mode == "global":
        return float(np.sum(T) - cfg.beta * np.sum(Q))

    if cfg.reward_mode == "fair":
        J = jains_fairness(T)
        return float(np.sum(T) + cfg.gamma * J - cfg.beta * np.sum(Q))

    raise ValueError(f"Unknown reward_mode={cfg.reward_mode}")

class Ns3PerApEnv:
    def __init__(self, nAps=3, nStaPerAp=25, nBg=15, stepTime=10.0, seed=1, reward_cfg=None):
        self.nAps = nAps
        self.nStaPerAp = nStaPerAp
        self.nBg = nBg
        self.stepTime = stepTime
        self.seed = seed
        self.run_id = 1
        self.reward_cfg = reward_cfg or RewardConfig()

    def reset_csv(self):
        if os.path.exists(NS3_CSV):
            os.remove(NS3_CSV)

    def step(self, action: dict, useQueuePkts: bool = True):
        params = {
            "nAps": self.nAps,
            "nStaPerAp": self.nStaPerAp,
            "nBg": self.nBg,
            "stepTime": self.stepTime,
            "channels": ",".join(map(str, action["channels"])),
            "txPowers": ",".join(map(str, action["txPowers"])),
            "cwMins": ",".join(map(str, action["cwMins"])),
            "cwMaxs": ",".join(map(str, action["cwMaxs"])),
            "dataRates": ",".join(map(str, action["dataRates"])),
            "useQueuePkts": "1" if useQueuePkts else "0",
            "seed": self.seed,
            "run": self.run_id,
        }

        cmd = ["./ns3", "run", "scratch/per_ap_step_env", "--"] + [f"--{k}={v}" for k, v in params.items()]
        subprocess.run(cmd, cwd=NS3_DIR, check=True)

        df = pd.read_csv(NS3_CSV)
        last = df.iloc[-1]

        T = [float(last[f"T{i}_Mbps"]) for i in range(self.nAps)]

        # Q columns can be _pkts or _ms depending on fallback.
        # We detect the column suffix.
        q_col = None
        for suffix in ["_pkts", "_ms"]:
            if f"Q0{suffix}" in df.columns:
                q_col = suffix
                break
        if q_col is None:
            raise RuntimeError("Could not find Q columns in CSV")

        Q = [float(last[f"Q{i}{q_col}"]) for i in range(self.nAps)]
        loss = float(last["lossRate"])

        r = compute_reward(T, Q, self.reward_cfg)

        obs = {"T": T, "Q": Q, "loss": loss, "q_units": ("pkts" if q_col == "_pkts" else "ms")}

        self.run_id += 1
        return obs, r
