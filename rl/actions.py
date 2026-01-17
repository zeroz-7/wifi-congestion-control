from dataclasses import dataclass
import numpy as np

@dataclass
class ActionSpace:
    nAps: int
    cwmin_values: list[int]

    def sample(self) -> list[int]:
        # random CWmin per AP
        idx = np.random.randint(0, len(self.cwmin_values), size=self.nAps)
        return [self.cwmin_values[i] for i in idx]

def action_to_ns3_args(cwmins: list[int],
                       channels: list[int] | None = None,
                       txpowers: list[int] | None = None,
                       cwmaxs: list[int] | None = None,
                       datarates: list[int] | None = None,
                       seed: int = 1,
                       run: int = 1,
                       stepTime: int = 10,
                       nStaPerAp: int = 25,
                       nBg: int = 15) -> str:
    """
    Converts action + config into ns-3 CLI args string.
    Our C++ supports: nAps,nStaPerAp,nBg,stepTime,channels,txPowers,cwMins,cwMaxs,dataRates,seed,run
    """
    nAps = len(cwmins)
    if channels is None: channels = [6] * nAps
    if txpowers is None: txpowers = [16] * nAps
    if cwmaxs is None:  cwmaxs  = [1023] * nAps
    if datarates is None: datarates = [24] * nAps

    args = []
    args.append(f"--nAps={nAps}")
    args.append(f"--nStaPerAp={nStaPerAp}")
    args.append(f"--nBg={nBg}")
    args.append(f"--stepTime={stepTime}")

    args.append(f"--channels={','.join(map(str, channels))}")
    args.append(f"--txPowers={','.join(map(str, txpowers))}")
    args.append(f"--cwMins={','.join(map(str, cwmins))}")
    args.append(f"--cwMaxs={','.join(map(str, cwmaxs))}")
    args.append(f"--dataRates={','.join(map(str, datarates))}")

    args.append(f"--seed={seed}")
    args.append(f"--run={run}")

    return " ".join(args)
