import os
import pandas as pd
import numpy as np

from env_ns3 import Ns3Env, RewardConfig
from actions import action_to_ns3_args
from agent_qlearn import QLearningAgent

# ---- CONFIG ----
NS3_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

N_APS = 3
STEP_TIME = 10
N_STA_PER_AP = 25
N_BG = 15

EPISODES = 2
STEPS_PER_EP = 1  # keep 1-step episodes initially

# Discrete CWmin choices (must be sorted)
CWMIN_VALUES = [7, 15, 31, 63, 127]

# Reward config
R_CFG = RewardConfig(reward_mode="fair", beta=0.05, gamma=5.0)

# State binning
T_BINS = [5, 15]     # Mbps
Q_BINS = [10, 50]    # ms (or whatever your env returns)

# Global delta action space (ONE value applied to all APs)
ACTIONS = [-1, 0, +1]  # decrease / keep / increase CWmin on the grid


def bin_value(x, bins):
    """returns 0..len(bins)"""
    for i, b in enumerate(bins):
        if x < b:
            return i
    return len(bins)


def nearest_cw_index(cw, cw_values):
    diffs = [abs(cw - v) for v in cw_values]
    return int(np.argmin(diffs))


def apply_global_delta(cwmins_current, delta, cw_values):
    """Apply one delta to all AP CWmins (move along discrete grid)."""
    d = int(delta)
    next_cwmins = []
    for cw in cwmins_current:
        idx = nearest_cw_index(cw, cw_values)
        idx2 = max(0, min(len(cw_values) - 1, idx + d))
        next_cwmins.append(cw_values[idx2])
    return next_cwmins


def obs_to_state_id(obs, cwmins_current):
    """
    State = bins(T per AP) + bins(Q per AP) + bins(CWmin per AP)
    Mixed radix encoding:
      T bins: 3 each
      Q bins: 3 each
      CW bins: len(CWMIN_VALUES) each
    """
    Tb = [bin_value(t, T_BINS) for t in obs["T"]]
    Qb = [bin_value(q, Q_BINS) for q in obs["Q"]]
    Cb = [nearest_cw_index(c, CWMIN_VALUES) for c in cwmins_current]

    digits = Tb + Qb + Cb
    radices = ([3] * len(Tb)) + ([3] * len(Qb)) + ([len(CWMIN_VALUES)] * len(Cb))

    sid = 0
    for d, base in zip(digits, radices):
        sid = sid * base + int(d)
    return sid


def main():
    n_actions = len(ACTIONS)  # 3 actions total

    # State space size:
    # (3^(2*N_APS)) * ((len(CWMIN_VALUES))^N_APS)
    n_states = int((3 ** (2 * N_APS)) * ((len(CWMIN_VALUES)) ** N_APS))

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        lr=0.15,
        gamma=0.95,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay=0.98
    )

    env = Ns3Env(ns3_root=NS3_ROOT, reward_cfg=R_CFG, seed=1)

    rows = []

    # Persist CWmins across episodes (adaptive)
    cwmins_current = [31] * N_APS  # start mid-grid
    s = 0  # initial dummy state; becomes meaningful after first obs

    for ep in range(1, EPISODES + 1):
        ep_reward = 0.0

        for step in range(STEPS_PER_EP):
            a_idx = agent.select_action(s)
            delta = ACTIONS[a_idx]  # global delta: -1/0/+1

            cwmins_next = apply_global_delta(cwmins_current, delta, CWMIN_VALUES)

            args = action_to_ns3_args(
                cwmins=cwmins_next,
                seed=1,
                run=ep,
                stepTime=STEP_TIME,
                nStaPerAp=N_STA_PER_AP,
                nBg=N_BG
            )

            obs, r = env.step(args)

            s2 = obs_to_state_id(obs, cwmins_next)
            done = (step == STEPS_PER_EP - 1)

            agent.update(s, a_idx, r, s2, done)

            cwmins_current = cwmins_next
            s = s2
            ep_reward += r

            rows.append({
                "episode": ep,
                "step": step,
                "reward": r,
                "ep_reward": ep_reward,
                "delta": int(delta),
                "cwmins": ",".join(map(str, cwmins_current)),
                "T": ",".join([f"{x:.3f}" for x in obs["T"]]),
                "Q": ",".join([f"{x:.3f}" for x in obs["Q"]]),
                "loss": obs["loss"],
            })

        agent.decay_eps()
        print(f"EP {ep:03d}  ep_reward={ep_reward:.3f}  eps={agent.eps:.3f}  delta={delta:+d}  cwmins={cwmins_current}")

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), "rl_history.csv")
    df.to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()
