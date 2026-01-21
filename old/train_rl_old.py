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
STEP_TIME = 10 # allowing queue dynamics to evolve
N_STA_PER_AP = 25
N_BG = 15

EPISODES = 10
STEPS_PER_EP = 5  # keep 1-step episodes initially

# Discrete CWmin choices (must be sorted)
CWMIN_VALUES = [7, 15, 31, 63, 127]

# Reward config
R_CFG = RewardConfig(reward_mode="fair", beta=0.05, gamma=5.0)

# State binning
AVG_T_BINS = [5, 15]    # Mbps
AVG_Q_BINS = [10, 50]   # ms (or whatever your env returns)

# Delay bins (ms)
DELAY_BINS = [500, 1500, 2500]      # low / med / high congestion
DELAY_TREND_BINS = [-200, 200]   # decreasing / stable / increasing

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


def obs_to_state_id(obs, cwmins_current, prev_avg_delay):
    avg_delay = sum(obs["Q"]) / len(obs["Q"])
    delay_trend = avg_delay - prev_avg_delay

    avg_d_bin = bin_value(avg_delay, DELAY_BINS)
    trend_bin = bin_value(delay_trend, DELAY_TREND_BINS)
    cw_idx = nearest_cw_index(cwmins_current[0], CWMIN_VALUES)

    sid = (
        avg_d_bin * (3 * len(CWMIN_VALUES)) +
        trend_bin * len(CWMIN_VALUES) +
        cw_idx
    )
    return sid, avg_delay


def main():
    n_actions = len(ACTIONS)  # 3 actions total

    # State space size:
    n_states = 4 * 3 * len(CWMIN_VALUES)

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        lr=0.05,
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
    prev_avg_delay = None

    for ep in range(1, EPISODES + 1):
        ep_reward = 0.0

        for step in range(STEPS_PER_EP):
            a_idx = agent.select_action(s)
            delta = ACTIONS[a_idx]

            cwmins_next = apply_global_delta(cwmins_current, delta, CWMIN_VALUES)

            args = action_to_ns3_args(
                cwmins=cwmins_next,
                seed=1,
                run=ep * 100 + step,   # IMPORTANT: unique runs
                stepTime=STEP_TIME,
                nStaPerAp=N_STA_PER_AP,
                nBg=N_BG
            )

            obs, _ = env.step(args)

            avg_delay = sum(obs["Q"]) / len(obs["Q"])

            # --- REWARD: delta-delay ---
            if prev_avg_delay is None:
                r = 0.0
            else:
                raw_delta = prev_avg_delay - avg_delay

                # Normalize by scale (~2000 ms observed)
                r = np.clip(raw_delta / 500.0, -1.0, 1.0)

            s2, _ = obs_to_state_id(obs, cwmins_next, prev_avg_delay or avg_delay)

            done = (step == STEPS_PER_EP - 1)
            agent.update(s, a_idx, r, s2, done=done)

            prev_avg_delay = avg_delay
            cwmins_current = cwmins_next
            s = s2
            ep_reward += r

            rows.append({
                "episode": ep,
                "step": step,
                "reward": r,
                "ep_reward": ep_reward,
                "delta": delta,
                "cwmins": ",".join(map(str, cwmins_current)),
                "avg_delay": avg_delay
            })

        agent.decay_eps()
        print(f"EP {ep:03d} reward={ep_reward:.3f} eps={agent.eps:.3f}")
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), "rl_history.csv")
    df.to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()
