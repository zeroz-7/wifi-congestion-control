import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
HIST = os.path.join(HERE, "rl_history.csv")

def parse_vec(s):
    return [float(x) for x in str(s).split(",") if x.strip()]

def main():
    df = pd.read_csv(HIST)

    # Episode summary (take last row per episode)
    last = df.sort_values(["episode", "step"]).groupby("episode").tail(1)

    # Episode reward
    plt.figure()
    plt.plot(last["episode"], last["ep_reward"])
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("RL training reward")
    plt.grid(True)
    plt.savefig(os.path.join(HERE, "reward.png"), dpi=160)

    # Average throughput and Q per episode
    avgT, avgQ = [], []
    for _, row in last.iterrows():
        T = parse_vec(row["T"])
        Q = parse_vec(row["Q"])
        avgT.append(sum(T) / len(T) if len(T) else 0.0)
        avgQ.append(sum(Q) / len(Q) if len(Q) else 0.0)

    plt.figure()
    plt.plot(last["episode"], avgT)
    plt.xlabel("Episode")
    plt.ylabel("Avg Throughput (Mbps)")
    plt.title("Average throughput per episode")
    plt.grid(True)
    plt.savefig(os.path.join(HERE, "avg_throughput.png"), dpi=160)

    plt.figure()
    plt.plot(last["episode"], avgQ)
    plt.xlabel("Episode")
    plt.ylabel("Avg Q (ms)")
    plt.title("Average Q per episode")
    plt.grid(True)
    plt.savefig(os.path.join(HERE, "avg_q.png"), dpi=160)

    # NEW (global delta): chosen delta per episode
    if "delta" in last.columns:
        plt.figure()
        plt.plot(last["episode"], last["delta"])
        plt.xlabel("Episode")
        plt.ylabel("Delta action (-1/0/+1)")
        plt.title("Chosen global delta per episode")
        plt.grid(True)
        plt.savefig(os.path.join(HERE, "delta.png"), dpi=160)

    # NEW: CWmin trajectory (plot AP0; with global delta all APs should match)
    if "cwmins" in last.columns:
        cw0 = []
        for s in last["cwmins"]:
            v = parse_vec(s)
            cw0.append(v[0] if len(v) else 0.0)

        plt.figure()
        plt.plot(last["episode"], cw0)
        plt.xlabel("Episode")
        plt.ylabel("CWmin (AP0)")
        plt.title("CWmin trajectory (global)")
        plt.grid(True)
        plt.savefig(os.path.join(HERE, "cwmin_traj.png"), dpi=160)

    print("Saved plots in:", HERE)

if __name__ == "__main__":
    main()
