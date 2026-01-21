import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
HIST = os.path.join(HERE, "rl_history.csv")

def main():
    df = pd.read_csv(HIST)

    # --- Episode summary (last step of each episode) ---
    last = df.sort_values(["episode", "step"]).groupby("episode").tail(1)

    # --------------------------------------------------
    # 1. Episode reward
    # --------------------------------------------------
    plt.figure()
    plt.plot(last["episode"], last["ep_reward"], marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Δ-delay sum)")
    plt.title("RL Training: Episode Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "reward.png"), dpi=160)

    # --------------------------------------------------
    # 2. Average delay per episode
    # --------------------------------------------------
    plt.figure()
    plt.plot(last["episode"], last["avg_delay"], marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Average Delay (ms)")
    plt.title("Average Delay per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "avg_delay.png"), dpi=160)

    # --------------------------------------------------
    # 3. CWmin trajectory (AP0)
    # --------------------------------------------------
    def parse_vec(s):
        return [int(x) for x in str(s).split(",") if x.strip()]

    cw0 = [parse_vec(v)[0] for v in last["cwmins"]]

    plt.figure()
    plt.plot(last["episode"], cw0, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("CWmin (AP0)")
    plt.title("CWmin Trajectory (Learned Control)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "cwmin_traj.png"), dpi=160)

    # --------------------------------------------------
    # 4. Action (delta) trajectory
    # --------------------------------------------------
    plt.figure()
    plt.plot(last["episode"], last["delta"], marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Action ΔCWmin (-1 / 0 / +1)")
    plt.title("Chosen Actions per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "delta.png"), dpi=160)

    print("Saved plots:")
    print("  reward.png")
    print("  avg_delay.png")
    print("  cwmin_traj.png")
    print("  delta.png")

if __name__ == "__main__":
    main()
