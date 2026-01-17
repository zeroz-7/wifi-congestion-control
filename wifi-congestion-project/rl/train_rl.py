import os
import itertools
import pandas as pd
import numpy as np

from env_ns3 import Ns3Env, RewardConfig
from actions import ActionSpace, action_to_ns3_args
from agent_qlearn import QLearningAgent

# ---- CONFIG ----
NS3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ns3"))

N_APS = 3
STEP_TIME = 10
N_STA_PER_AP = 25
N_BG = 15

EPISODES = 80          # within your 150-200 run budget with repeats/ablation later
STEPS_PER_EP = 1       # keep 1-step episodes first (fast + stable); can increase later

# Discrete CWmin choices
CWMIN_VALUES = [7, 15, 31, 63, 127]

# Reward config (switch reward_mode here)
R_CFG = RewardConfig(reward_mode="fair", beta=0.05, gamma=5.0)

# State binning (simple)
# Throughput bins (Mbps): low/med/high
T_BINS = [5, 15]        # <5 low, 5-15 med, >15 high
# Delay bins (ms): low/med/high
Q_BINS = [10, 50]       # <10 low, 10-50 med, >50 high

def bin_value(x, bins):
    # returns 0..len(bins)
    for i, b in enumerate(bins):
        if x < b:
            return i
    return len(bins)

def obs_to_state_id(obs):
    # per-AP bins -> one state id
    Tb = [bin_value(t, T_BINS) for t in obs["T"]]
    Qb = [bin_value(q, Q_BINS) for q in obs["Q"]]

    # combine into integer (base = 3 bins each => 3^(2*N_APS) states)
    base = 3
    digits = Tb + Qb
    sid = 0
    for d in digits:
        sid = sid * base + d
    return sid

def build_action_list(nAps, cw_values):
    # all combinations (size len(cw_values)^nAps) -> for N_APS=3 and 5 values => 125 actions
    return [list(a) for a in itertools.product(cw_values, repeat=nAps)]

def main():
    # Build action list
    actions = build_action_list(N_APS, CWMIN_VALUES)
    n_actions = len(actions)

    # State size: 3 bins each for T and Q => 3^(2*N_APS)
    n_states = int(3 ** (2 * N_APS))

    agent = QLearningAgent(n_states=n_states, n_actions=n_actions,
                           lr=0.15, gamma=0.95, eps_start=1.0, eps_end=0.1, eps_decay=0.98)

    env = Ns3Env(ns3_root=NS3_ROOT, reward_cfg=R_CFG, seed=1)

    rows = []

    for ep in range(1, EPISODES + 1):
        ep_reward = 0.0

        # start with a dummy obs to get a state (or just state=0)
        s = 0

        for step in range(STEPS_PER_EP):
            a_idx = agent.select_action(s)
            cwmins = actions[a_idx]

            args = action_to_ns3_args(
                cwmins=cwmins,
                seed=1,
                run=ep,                 # different run per episode
                stepTime=STEP_TIME,
                nStaPerAp=N_STA_PER_AP,
                nBg=N_BG
            )

            obs, r = env.step(args)
            s2 = obs_to_state_id(obs)

            done = (step == STEPS_PER_EP - 1)
            agent.update(s, a_idx, r, s2, done)

            s = s2
            ep_reward += r

            rows.append({
                "episode": ep,
                "step": step,
                "reward": r,
                "ep_reward": ep_reward,
                "cwmins": ",".join(map(str, cwmins)),
                "T": ",".join([f"{x:.3f}" for x in obs["T"]]),
                "Q": ",".join([f"{x:.3f}" for x in obs["Q"]]),
                "loss": obs["loss"]
            })

        agent.decay_eps()
        print(f"EP {ep:03d}  ep_reward={ep_reward:.3f}  eps={agent.eps:.3f}")

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), "rl_history.csv")
    df.to_csv(out, index=False)
    print("Saved:", out)

if __name__ == "__main__":
    main()
