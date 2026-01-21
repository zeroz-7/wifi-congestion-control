import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)

def moving_average(data, window=5):
    """Compute moving average for smoothing"""
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

def main():
    # Load both datasets
    df_rl = pd.read_csv(os.path.join(HERE, "rl_only_history.csv"))
    df_ml = pd.read_csv(os.path.join(HERE, "rl_ml_history.csv"))
    
    # Get last step of each episode
    rl_last = df_rl.groupby("episode").tail(1).reset_index(drop=True)
    ml_last = df_ml.groupby("episode").tail(1).reset_index(drop=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --------------------------------------------------
    # 1. Average Delay Comparison
    # --------------------------------------------------
    ax = axes[0, 0]
    
    # Plot raw data (lighter)
    ax.plot(rl_last["episode"], rl_last["avg_delay"], 
            'o-', alpha=0.3, color='blue', label='RL Only (raw)')
    ax.plot(ml_last["episode"], ml_last["avg_delay"], 
            'o-', alpha=0.3, color='red', label='RL+ML (raw)')
    
    # Plot smoothed (darker)
    rl_smooth = moving_average(rl_last["avg_delay"], window=5)
    ml_smooth = moving_average(ml_last["avg_delay"], window=5)
    
    ax.plot(rl_last["episode"], rl_smooth, 
            '-', linewidth=2, color='blue', label='RL Only (smooth)')
    ax.plot(ml_last["episode"], ml_smooth, 
            '-', linewidth=2, color='red', label='RL+ML (smooth)')
    
    # Mark ML activation
    if 'ml_active' in ml_last.columns:
        ml_start = ml_last[ml_last['ml_active'] == True]['episode'].min()
        if not pd.isna(ml_start):
            ax.axvline(ml_start, color='green', linestyle='--', 
                      label=f'ML Activated (ep {int(ml_start)})')
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average Delay (ms)", fontsize=12)
    ax.set_title("Delay Comparison: RL vs RL+ML", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --------------------------------------------------
    # 2. Episode Reward Comparison
    # --------------------------------------------------
    ax = axes[0, 1]
    
    rl_rew_smooth = moving_average(rl_last["ep_reward"], window=5)
    ml_rew_smooth = moving_average(ml_last["ep_reward"], window=5)
    
    ax.plot(rl_last["episode"], rl_rew_smooth, 
            '-', linewidth=2, color='blue', label='RL Only')
    ax.plot(ml_last["episode"], ml_rew_smooth, 
            '-', linewidth=2, color='red', label='RL+ML')
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Reward Progression", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --------------------------------------------------
    # 3. CWmin Trajectory (AP0)
    # --------------------------------------------------
    ax = axes[1, 0]
    
    def parse_cwmin(s):
        return int(str(s).split(",")[0])
    
    rl_cw = [parse_cwmin(v) for v in rl_last["cwmins"]]
    ml_cw = [parse_cwmin(v) for v in ml_last["cwmins"]]
    
    ax.plot(rl_last["episode"], rl_cw, 
            'o-', color='blue', label='RL Only', markersize=4)
    ax.plot(ml_last["episode"], ml_cw, 
            'o-', color='red', label='RL+ML', markersize=4)
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("CWmin (AP0)", fontsize=12)
    ax.set_title("Learned CWmin Control Strategy", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --------------------------------------------------
    # 4. Performance Summary (Bar Chart)
    # --------------------------------------------------
    ax = axes[1, 1]
    
    # Calculate metrics for last 10 episodes
    rl_final_delay = rl_last[rl_last['episode'] > 40]['avg_delay'].mean()
    ml_final_delay = ml_last[ml_last['episode'] > 40]['avg_delay'].mean()
    
    rl_final_reward = rl_last[rl_last['episode'] > 40]['ep_reward'].mean()
    ml_final_reward = ml_last[ml_last['episode'] > 40]['ep_reward'].mean()
    
    categories = ['Avg Delay\n(lower better)', 'Avg Reward\n(higher better)']
    rl_values = [rl_final_delay, rl_final_reward]
    ml_values = [ml_final_delay, ml_final_reward]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rl_values, width, label='RL Only', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, ml_values, width, label='RL+ML', color='red', alpha=0.7)
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Final Performance (Last 10 Episodes)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "comparison_analysis.png"), dpi=200)
    print(f"✓ Saved: comparison_analysis.png")
    
    # --------------------------------------------------
    # Additional: ML Action Source Analysis
    # --------------------------------------------------
    if 'action_source' in df_ml.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        ml_active_data = df_ml[df_ml['ml_active'] == True].copy()
        if len(ml_active_data) > 0:
            action_counts = ml_active_data.groupby(['episode', 'action_source']).size().unstack(fill_value=0)
            
            action_counts.plot(kind='bar', stacked=True, ax=ax2, 
                              color=['blue', 'orange'], alpha=0.7)
            ax2.set_xlabel("Episode", fontsize=12)
            ax2.set_ylabel("Number of Actions", fontsize=12)
            ax2.set_title("ML vs RL Action Selection (After ML Activation)", 
                         fontsize=14, fontweight='bold')
            ax2.legend(title="Action Source")
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(HERE, "ml_action_analysis.png"), dpi=200)
            print(f"✓ Saved: ml_action_analysis.png")
    
    # --------------------------------------------------
    # Print Summary Statistics
    # --------------------------------------------------
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"\nRL-Only (Last 10 episodes):")
    print(f"  Average Delay: {rl_final_delay:.2f} ms")
    print(f"  Average Reward: {rl_final_reward:.3f}")
    
    print(f"\nRL+ML (Last 10 episodes):")
    print(f"  Average Delay: {ml_final_delay:.2f} ms")
    print(f"  Average Reward: {ml_final_reward:.3f}")
    
    improvement = ((rl_final_delay - ml_final_delay) / rl_final_delay) * 100
    print(f"\nImprovement with ML: {improvement:.1f}%")
    
    if improvement > 0:
        print(f"✓ RL+ML performs BETTER (lower delay)")
    else:
        print(f"✗ RL+ML performs WORSE (needs tuning)")


if __name__ == "__main__":
    main()