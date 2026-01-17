import os
import subprocess
import itertools
import pandas as pd

NS3_DIR = os.path.expanduser("~/ns-3-dev")
OUT_CSV = os.path.join(os.getcwd(), "congestion_dataset.csv")

def run_ns3(nSta, nBg, offered, seed, run, simTime=20):
    cmd = [
        "./ns3", "run", "scratch/wifi_congestion_dataset",
        "--",
        f"--nSta={nSta}",
        f"--nBg={nBg}",
        f"--offeredMbps={offered}",
        f"--seed={seed}",
        f"--run={run}",
        f"--simTime={simTime}",
    ]
    subprocess.run(cmd, cwd=NS3_DIR, check=True)

def main():
    # Sweep parameters to generate dataset
    nSta_list = [5, 10, 20, 30]
    nBg_list  = [0, 5, 10, 20]
    offered_list = [5, 10, 20, 30]  # Mbps per flow

    # Clean old dataset generated in ns-3 folder if you want
    ns3_csv = os.path.join(NS3_DIR, "congestion_dataset.csv")
    if os.path.exists(ns3_csv):
        os.remove(ns3_csv)

    seed = 1
    run_id = 1
    for nSta, nBg, offered in itertools.product(nSta_list, nBg_list, offered_list):
        for r in range(1, 6):  # 5 repeats
            run_ns3(nSta, nBg, offered, seed, run_id, simTime=20)
            run_id += 1

    # Copy resulting CSV from ns-3 dir into ml dir
    df = pd.read_csv(ns3_csv)
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Dataset saved to: {OUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()
