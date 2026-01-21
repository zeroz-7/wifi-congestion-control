import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
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

EPISODES = 70
STEPS_PER_EP = 1

CWMIN_VALUES = [15, 31, 63, 127]

R_CFG = RewardConfig(reward_mode="fair", beta=0.05, gamma=5.0)

# State bins - include predicted delay
DELAY_BINS = [300, 500, 700, 900]  # 5 bins
DELAY_TREND_BINS = [-100, 100]     # decreasing/stable/increasing (3 bins)

ACTIONS = [-1, 0, +1]


def bin_value(x, bins):
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


def obs_to_state_id(current_delay, predicted_delay, cwmin_idx):
    """
    State includes BOTH current and predicted future delay
    This allows proactive decision making
    """
    delay_bin = bin_value(current_delay, DELAY_BINS)
    
    # Trend: are we heading toward higher or lower delay?
    trend = predicted_delay - current_delay
    trend_bin = bin_value(trend, DELAY_TREND_BINS)
    
    # State ID
    sid = (delay_bin * (len(DELAY_TREND_BINS) + 1) * len(CWMIN_VALUES) +
           trend_bin * len(CWMIN_VALUES) +
           cwmin_idx)
    return sid


def compute_reward_predictive(current_obs, predicted_delay_next):
    """
    PROACTIVE reward: Penalize both current AND predicted future delay
    This encourages the agent to prevent problems, not just react
    """
    current_delay = np.mean(current_obs["Q"])
    
    # Penalize current delay
    current_penalty = -current_delay / 200.0
    
    # CRITICAL: Also penalize predicted future delay
    # This makes the agent proactive
    if predicted_delay_next is not None:
        future_penalty = -predicted_delay_next / 200.0
        # Weight future more heavily to encourage prevention
        reward = 0.3 * current_penalty + 0.7 * future_penalty
    else:
        reward = current_penalty
    
    # Bonus for keeping delay low
    if current_delay < 500:
        reward += 0.5
    
    return np.clip(reward, -5.0, 5.0)


class TrafficPredictor:
    """
    ML model that predicts delay 1 step (10s) into the future
    Trained on: [current_cwmin, current_delay, current_throughput, current_loss]
    Predicts: delay at next timestep
    """
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        
    def add_experience(self, cwmin_idx, current_obs, next_delay):
        """
        Store transition: (state at t) -> (delay at t+1)
        """
        self.training_data.append({
            'cwmin_idx': cwmin_idx,
            'current_delay': np.mean(current_obs["Q"]),
            'current_throughput': np.mean(current_obs["T"]),
            'current_loss': current_obs["loss"],
            'next_delay': next_delay  # Ground truth future delay
        })
    
    def train(self):
        """Train predictor on collected transitions"""
        if len(self.training_data) < 25:
            return False
        
        X = []
        y = []
        
        for exp in self.training_data:
            # Features: current state
            features = [
                exp['cwmin_idx'],
                exp['current_delay'],
                exp['current_throughput'],
                exp['current_loss']
            ]
            X.append(features)
            y.append(exp['next_delay'])  # Target: future delay
        
        X = np.array(X)
        y = np.array(y)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate prediction accuracy
        y_pred = self.model.predict(X_scaled)
        mae = np.mean(np.abs(y - y_pred))
        
        return True, mae
    
    def predict_next_delay(self, cwmin_idx, current_obs):
        """
        Predict delay at next timestep given current state
        This is the KEY function for proactive control
        """
        if not self.is_trained:
            return None
        
        features = [
            cwmin_idx,
            np.mean(current_obs["Q"]),
            np.mean(current_obs["T"]),
            current_obs["loss"]
        ]
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        predicted_delay = self.model.predict(X_scaled)[0]
        return predicted_delay
    
    def predict_delay_for_action(self, cwmin_idx, delta, current_obs):
        """
        Predict what delay will be if we take action delta
        Used by RL+ML to evaluate actions before taking them
        """
        if not self.is_trained:
            return None
        
        new_cwmin_idx = max(0, min(len(CWMIN_VALUES) - 1, cwmin_idx + delta))
        return self.predict_next_delay(new_cwmin_idx, current_obs)


def train_rl_only(episodes=70):
    """
    BASELINE: Pure RL without predictions (reactive control)
    """
    n_actions = len(ACTIONS)
    n_states = (len(DELAY_BINS) + 1) * (len(DELAY_TREND_BINS) + 1) * len(CWMIN_VALUES)
    
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        lr=0.15,
        gamma=0.95,
        eps_start=1.0,
        eps_end=0.02,
        eps_decay=0.96
    )
    
    env = Ns3Env(ns3_root=NS3_ROOT, reward_cfg=R_CFG, seed=1)
    
    rows = []
    cwmins_current = [31] * N_APS
    prev_obs = None
    prev_delay = 500.0
    
    for ep in range(1, episodes + 1):
        cwmin_idx = nearest_cw_index(cwmins_current[0], CWMIN_VALUES)
        current_delay = prev_delay if prev_obs else 500.0
        
        # NO prediction - use current delay as "predicted" (reactive)
        s = obs_to_state_id(current_delay, current_delay, cwmin_idx)
        
        ep_reward = 0.0
        
        for step in range(STEPS_PER_EP):
            a_idx = agent.select_action(s)
            delta = ACTIONS[a_idx]
            
            cwmins_next = apply_global_delta(cwmins_current, delta, CWMIN_VALUES)
            
            args = action_to_ns3_args(
                cwmins=cwmins_next,
                seed=1,
                run=ep,
                stepTime=STEP_TIME,
                nStaPerAp=N_STA_PER_AP,
                nBg=N_BG
            )
            
            obs, _ = env.step(args)
            current_delay = np.mean(obs["Q"])
            
            # Reactive reward (only considers current delay)
            r = compute_reward_predictive(obs, current_delay)
            
            cwmin_idx_next = nearest_cw_index(cwmins_next[0], CWMIN_VALUES)
            s2 = obs_to_state_id(current_delay, current_delay, cwmin_idx_next)
            
            done = (step == STEPS_PER_EP - 1)
            agent.update(s, a_idx, r, s2, done=done)
            
            prev_obs = obs
            prev_delay = current_delay
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
                "avg_delay": current_delay,
                "predicted_delay": current_delay,  # No real prediction
                "avg_throughput": np.mean(obs["T"]),
                "loss_rate": obs["loss"],
                "method": "RL_only",
                "epsilon": agent.eps
            })
        
        agent.decay_eps()
        
        if ep % 10 == 0:
            print(f"RL-Only EP {ep:03d} | Reward={ep_reward:+.2f} | "
                  f"Delay={current_delay:.0f}ms | ε={agent.eps:.3f}")
    
    return pd.DataFrame(rows), agent


def train_rl_with_ml_predictor(episodes=70, ml_start_episode=20):
    """
    RL + ML PREDICTOR: Proactive control using traffic predictions
    """
    n_actions = len(ACTIONS)
    n_states = (len(DELAY_BINS) + 1) * (len(DELAY_TREND_BINS) + 1) * len(CWMIN_VALUES)
    
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        lr=0.15,
        gamma=0.95,
        eps_start=1.0,
        eps_end=0.02,
        eps_decay=0.96
    )
    
    predictor = TrafficPredictor()
    env = Ns3Env(ns3_root=NS3_ROOT, reward_cfg=R_CFG, seed=1)
    
    rows = []
    cwmins_current = [31] * N_APS
    prev_obs = None
    prev_delay = 500.0
    ml_active = False
    
    for ep in range(1, episodes + 1):
        cwmin_idx = nearest_cw_index(cwmins_current[0], CWMIN_VALUES)
        current_delay = prev_delay if prev_obs else 500.0
        
        # Predict future delay if ML is active
        if ml_active and prev_obs is not None:
            predicted_delay = predictor.predict_next_delay(cwmin_idx, prev_obs)
        else:
            predicted_delay = current_delay  # Fallback to reactive
        
        s = obs_to_state_id(current_delay, predicted_delay, cwmin_idx)
        
        ep_reward = 0.0
        
        for step in range(STEPS_PER_EP):
            a_idx = agent.select_action(s)
            delta = ACTIONS[a_idx]
            
            cwmins_next = apply_global_delta(cwmins_current, delta, CWMIN_VALUES)
            
            args = action_to_ns3_args(
                cwmins=cwmins_next,
                seed=1,
                run=ep + 1000,
                stepTime=STEP_TIME,
                nStaPerAp=N_STA_PER_AP,
                nBg=N_BG
            )
            
            obs, _ = env.step(args)
            current_delay = np.mean(obs["Q"])
            
            # Store transition for predictor training
            # (current state, action) -> (observed next delay)
            if prev_obs is not None:
                predictor.add_experience(
                    nearest_cw_index(cwmins_current[0], CWMIN_VALUES),
                    prev_obs,
                    current_delay  # This is the "future" delay we want to predict
                )
            
            # Proactive reward (considers predicted future)
            if ml_active:
                pred_future = predictor.predict_next_delay(
                    nearest_cw_index(cwmins_next[0], CWMIN_VALUES),
                    obs
                )
            else:
                pred_future = current_delay
            
            r = compute_reward_predictive(obs, pred_future)
            
            cwmin_idx_next = nearest_cw_index(cwmins_next[0], CWMIN_VALUES)
            
            # Next state includes prediction
            if ml_active:
                next_predicted = predictor.predict_next_delay(cwmin_idx_next, obs)
            else:
                next_predicted = current_delay
            
            s2 = obs_to_state_id(current_delay, next_predicted, cwmin_idx_next)
            
            done = (step == STEPS_PER_EP - 1)
            agent.update(s, a_idx, r, s2, done=done)
            
            prev_obs = obs
            prev_delay = current_delay
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
                "avg_delay": current_delay,
                "predicted_delay": next_predicted if ml_active else current_delay,
                "avg_throughput": np.mean(obs["T"]),
                "loss_rate": obs["loss"],
                "method": "RL+ML",
                "ml_active": ml_active,
                "epsilon": agent.eps
            })
        
        # Train predictor
        if ep == ml_start_episode:
            success, mae = predictor.train()
            if success:
                ml_active = True
                print(f"✓ Traffic Predictor ACTIVATED at ep {ep} (MAE={mae:.1f}ms)")
        elif ml_active and ep % 15 == 0:
            success, mae = predictor.train()
            print(f"  Predictor retrained at ep {ep} (MAE={mae:.1f}ms)")
        
        agent.decay_eps()
        
        if ep % 10 == 0:
            status = "PROACTIVE (ML)" if ml_active else "REACTIVE (RL)"
            pred_str = f"Pred={predicted_delay:.0f}ms" if ml_active else ""
            print(f"{status} EP {ep:03d} | Reward={ep_reward:+.2f} | "
                  f"Delay={current_delay:.0f}ms {pred_str} | ε={agent.eps:.3f}")
    
    return pd.DataFrame(rows), agent, predictor


def main():
    print("="*70)
    print("PHASE 1: Reactive Control (RL-Only, no predictions)")
    print("="*70)
    df_rl, agent_rl = train_rl_only(episodes=70)
    df_rl.to_csv("rl_only_history.csv", index=False)
    
    final_delay_rl = df_rl[df_rl['episode'] > 60]['avg_delay'].mean()
    print(f"✓ Reactive RL complete. Final avg delay: {final_delay_rl:.1f}ms")
    
    print("\n" + "="*70)
    print("PHASE 2: Proactive Control (RL + Traffic Predictor)")
    print("="*70)
    df_ml, agent_ml, predictor = train_rl_with_ml_predictor(episodes=70, ml_start_episode=20)
    df_ml.to_csv("rl_ml_history.csv", index=False)
    
    final_delay_ml = df_ml[df_ml['episode'] > 60]['avg_delay'].mean()
    print(f"✓ Proactive RL+ML complete. Final avg delay: {final_delay_ml:.1f}ms")
    
    # Save predictor
    joblib.dump(predictor, "traffic_predictor.pkl")
    print("✓ Traffic predictor saved")
    
    # Combined
    df_combined = pd.concat([df_rl, df_ml], ignore_index=True)
    df_combined.to_csv("rl_comparison.csv", index=False)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    improvement = ((final_delay_rl - final_delay_ml) / final_delay_rl * 100)
    print(f"Reactive (RL-Only):  {final_delay_rl:.1f}ms avg delay")
    print(f"Proactive (RL+ML):   {final_delay_ml:.1f}ms avg delay")
    print(f"Improvement: {improvement:+.1f}%")
    
    # Analyze prediction accuracy
    ml_data = df_ml[df_ml['ml_active'] == True]
    if len(ml_data) > 0:
        # Check how well predictions matched reality
        prediction_errors = []
        for i in range(len(ml_data) - 1):
            if ml_data.iloc[i]['episode'] == ml_data.iloc[i+1]['episode']:
                predicted = ml_data.iloc[i]['predicted_delay']
                actual = ml_data.iloc[i+1]['avg_delay']
                prediction_errors.append(abs(predicted - actual))
        
        if prediction_errors:
            mae = np.mean(prediction_errors)
            print(f"\nPredictor Performance:")
            print(f"  Mean Absolute Error: {mae:.1f}ms")
            print(f"  (Lower is better - perfect prediction = 0ms)")


if __name__ == "__main__":
    main()