import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

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

# INCREASED for proper learning
EPISODES = 100  # Was 10
STEPS_PER_EP = 1  # Keep episodes short but many

CWMIN_VALUES = [7, 15, 31, 63, 127]

R_CFG = RewardConfig(reward_mode="fair", beta=0.05, gamma=5.0)

# REFINED state binning (more granular)
DELAY_BINS = [50, 150, 300, 600, 1000, 1500]  # 7 bins instead of 4
CWMIN_BINS = [0, 1, 2, 3, 4]  # Index directly into CWMIN_VALUES

ACTIONS = [-1, 0, +1]


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
    d = int(delta)
    next_cwmins = []
    for cw in cwmins_current:
        idx = nearest_cw_index(cw, cw_values)
        idx2 = max(0, min(len(cw_values) - 1, idx + d))
        next_cwmins.append(cw_values[idx2])
    return next_cwmins


def obs_to_state_id(avg_delay, cw_idx):
    """Simplified state: delay bin + CW index"""
    delay_bin = bin_value(avg_delay, DELAY_BINS)
    sid = delay_bin * len(CWMIN_VALUES) + cw_idx
    return sid


def compute_reward_fixed(obs, prev_obs):
    """
    FIXED REWARD: Negative of average delay (lower delay = higher reward)
    Plus improvement bonus
    """
    avg_delay = np.mean(obs["Q"])
    
    if prev_obs is None:
        # First step: just penalize current delay
        return -avg_delay / 100.0  # Normalize to reasonable scale
    
    prev_avg_delay = np.mean(prev_obs["Q"])
    
    # Improvement bonus: positive when delay decreases
    improvement = (prev_avg_delay - avg_delay) / 100.0
    
    # Base penalty for absolute delay
    delay_penalty = -avg_delay / 100.0
    
    # Combined reward (weighted toward improvement)
    reward = 0.7 * improvement + 0.3 * delay_penalty
    
    return reward


class MLPredictor:
    """
    ML predictor for delay given CWmin configuration
    """
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        
    def add_experience(self, cwmins, avg_delay, loss_rate):
        """Store experience for training"""
        self.training_data.append({
            'cwmins': cwmins.copy(),
            'avg_delay': avg_delay,
            'loss_rate': loss_rate
        })
    
    def train(self):
        """Train on collected experiences"""
        if len(self.training_data) < 20:
            return False
        
        X = []
        y = []
        
        for exp in self.training_data:
            # Features: CWmin values + loss rate
            features = exp['cwmins'] + [exp['loss_rate']]
            X.append(features)
            y.append(exp['avg_delay'])
        
        X = np.array(X)
        y = np.array(y)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return True
    
    def predict(self, cwmins, loss_rate=0.0):
        """Predict delay for given configuration"""
        if not self.is_trained:
            return None
        
        features = cwmins + [loss_rate]
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)[0]
    
    def get_best_action(self, current_cwmins, loss_rate=0.0):
        """
        Use ML to predict which action will minimize delay
        Returns: best action index
        """
        if not self.is_trained:
            return None
        
        predictions = []
        for delta in ACTIONS:
            next_cwmins = apply_global_delta(current_cwmins, delta, CWMIN_VALUES)
            pred_delay = self.predict(next_cwmins, loss_rate)
            predictions.append(pred_delay)
        
        return int(np.argmin(predictions))


def train_rl_only(episodes=50):
    """Pure RL approach"""
    n_actions = len(ACTIONS)
    n_states = len(DELAY_BINS) + 1  # +1 for overflow
    n_states *= len(CWMIN_VALUES)
    
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        lr=0.1,
        gamma=0.95,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995
    )
    
    env = Ns3Env(ns3_root=NS3_ROOT, reward_cfg=R_CFG, seed=1)
    
    rows = []
    cwmins_current = [31] * N_APS  # Start at middle value
    prev_obs = None
    
    for ep in range(1, episodes + 1):
        # Reset for new episode
        cw_idx = nearest_cw_index(cwmins_current[0], CWMIN_VALUES)
        avg_delay = prev_obs["Q"][0] if prev_obs else 500.0
        s = obs_to_state_id(avg_delay, cw_idx)
        
        ep_reward = 0.0
        
        for step in range(STEPS_PER_EP):
            # Select action
            a_idx = agent.select_action(s)
            delta = ACTIONS[a_idx]
            
            # Apply action
            cwmins_next = apply_global_delta(cwmins_current, delta, CWMIN_VALUES)
            
            args = action_to_ns3_args(
                cwmins=cwmins_next,
                seed=1,
                run=ep,  # Use episode as run number
                stepTime=STEP_TIME,
                nStaPerAp=N_STA_PER_AP,
                nBg=N_BG
            )
            
            # Execute in environment
            obs, _ = env.step(args)
            
            # Compute FIXED reward
            r = compute_reward_fixed(obs, prev_obs)
            
            # Next state
            avg_delay = np.mean(obs["Q"])
            cw_idx_next = nearest_cw_index(cwmins_next[0], CWMIN_VALUES)
            s2 = obs_to_state_id(avg_delay, cw_idx_next)
            
            # Update Q-table
            done = (step == STEPS_PER_EP - 1)
            agent.update(s, a_idx, r, s2, done=done)
            
            # Store for next iteration
            prev_obs = obs
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
                "avg_delay": avg_delay,
                "loss_rate": obs["loss"],
                "method": "RL_only"
            })
        
        agent.decay_eps()
        
        if ep % 10 == 0:
            print(f"RL-Only EP {ep:03d} | Reward={ep_reward:.3f} | "
                  f"Delay={avg_delay:.1f}ms | ε={agent.eps:.3f}")
    
    return pd.DataFrame(rows), agent


def train_rl_with_ml(episodes=50, ml_start_episode=30):
    """RL + ML predictor approach"""
    n_actions = len(ACTIONS)
    n_states = len(DELAY_BINS) + 1
    n_states *= len(CWMIN_VALUES)
    
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        lr=0.1,
        gamma=0.95,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995
    )
    
    ml_predictor = MLPredictor()
    env = Ns3Env(ns3_root=NS3_ROOT, reward_cfg=R_CFG, seed=1)
    
    rows = []
    cwmins_current = [31] * N_APS
    prev_obs = None
    ml_active = False
    
    for ep in range(1, episodes + 1):
        cw_idx = nearest_cw_index(cwmins_current[0], CWMIN_VALUES)
        avg_delay = prev_obs["Q"][0] if prev_obs else 500.0
        s = obs_to_state_id(avg_delay, cw_idx)
        
        ep_reward = 0.0
        
        for step in range(STEPS_PER_EP):
            # Decide action source
            use_ml = ml_active and np.random.rand() > agent.eps
            
            if use_ml:
                # ML-guided action
                ml_action = ml_predictor.get_best_action(
                    cwmins_current, 
                    prev_obs["loss"] if prev_obs else 0.0
                )
                a_idx = ml_action if ml_action is not None else agent.select_action(s)
                action_source = "ML"
            else:
                # Pure RL action
                a_idx = agent.select_action(s)
                action_source = "RL"
            
            delta = ACTIONS[a_idx]
            cwmins_next = apply_global_delta(cwmins_current, delta, CWMIN_VALUES)
            
            args = action_to_ns3_args(
                cwmins=cwmins_next,
                seed=1,
                run=ep + 1000,  # Different run numbers from RL-only
                stepTime=STEP_TIME,
                nStaPerAp=N_STA_PER_AP,
                nBg=N_BG
            )
            
            obs, _ = env.step(args)
            r = compute_reward_fixed(obs, prev_obs)
            
            avg_delay = np.mean(obs["Q"])
            
            # Store experience for ML
            ml_predictor.add_experience(
                cwmins_next, 
                avg_delay, 
                obs["loss"]
            )
            
            cw_idx_next = nearest_cw_index(cwmins_next[0], CWMIN_VALUES)
            s2 = obs_to_state_id(avg_delay, cw_idx_next)
            
            done = (step == STEPS_PER_EP - 1)
            agent.update(s, a_idx, r, s2, done=done)
            
            prev_obs = obs
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
                "avg_delay": avg_delay,
                "loss_rate": obs["loss"],
                "method": "RL+ML",
                "action_source": action_source,
                "ml_active": ml_active
            })
        
        # Train ML model periodically
        if ep == ml_start_episode:
            if ml_predictor.train():
                ml_active = True
                print(f"✓ ML Predictor activated at episode {ep}")
        elif ep > ml_start_episode and ep % 20 == 0:
            ml_predictor.train()
        
        agent.decay_eps()
        
        if ep % 10 == 0:
            status = "ML+RL" if ml_active else "RL"
            print(f"{status} EP {ep:03d} | Reward={ep_reward:.3f} | "
                  f"Delay={avg_delay:.1f}ms | ε={agent.eps:.3f}")
    
    return pd.DataFrame(rows), agent, ml_predictor


def main():
    print("="*60)
    print("PHASE 1: Training RL-Only (50 episodes)")
    print("="*60)
    df_rl, agent_rl = train_rl_only(episodes=50)
    df_rl.to_csv("rl_only_history.csv", index=False)
    print(f"✓ RL-Only complete. Final avg delay: {df_rl['avg_delay'].iloc[-1]:.1f}ms")
    
    print("\n" + "="*60)
    print("PHASE 2: Training RL+ML (50 episodes, ML starts at ep 30)")
    print("="*60)
    df_ml, agent_ml, predictor = train_rl_with_ml(episodes=50, ml_start_episode=30)
    df_ml.to_csv("rl_ml_history.csv", index=False)
    print(f"✓ RL+ML complete. Final avg delay: {df_ml['avg_delay'].iloc[-1]:.1f}ms")
    
    # Save ML model
    joblib.dump(predictor, "ml_predictor.pkl")
    print("✓ ML predictor saved")
    
    # Combined dataframe for comparison
    df_combined = pd.concat([df_rl, df_ml], ignore_index=True)
    df_combined.to_csv("rl_comparison.csv", index=False)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Final 10 episodes average
    rl_final = df_rl[df_rl['episode'] > 40]['avg_delay'].mean()
    ml_final = df_ml[df_ml['episode'] > 40]['avg_delay'].mean()
    
    print(f"RL-Only final 10-ep avg delay: {rl_final:.1f}ms")
    print(f"RL+ML final 10-ep avg delay:   {ml_final:.1f}ms")
    print(f"Improvement: {((rl_final - ml_final) / rl_final * 100):.1f}%")


if __name__ == "__main__":
    main()