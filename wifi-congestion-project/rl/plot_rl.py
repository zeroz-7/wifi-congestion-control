import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
HIST = os.path.join(HERE, "rl_history.csv")

def parse_vec(s):
    return [float(x) for x in s.split(",") if x.strip()]

def main():
    df = pd.read_csv(HIST)

    # Episode reward (take last row per episode)
    last = df.sort_values(["episode", "step"]).groupby("episode").tail(1)

    plt.figure()
    plt.plot(last["episode"], last["ep_reward"])
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("RL training reward")
    plt.grid(True)
    plt.savefig(os.path.join(HERE, "reward.png"), dpi=160)

    # Average throughput across APs per episode
    avgT = []
    avgQ = []
    for _, row in last.iterrows():
        T = parse_vec(row["T"])
        Q = parse_vec(row["Q"])
        avgT.append(sum(T)/len(T))
        avgQ.append(sum(Q)/len(Q))

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
    plt.ylabel("Avg Delay (ms)")
    plt.title("Average delay per episode")
    plt.grid(True)
    plt.savefig(os.path.join(HERE, "avg_delay.png"), dpi=160)

    print("Saved plots in:", HERE)

if __name__ == "__main__":
    main()
