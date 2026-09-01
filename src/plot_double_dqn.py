"""
Phase 12: Double DQN vs Vanilla DQN Comparison Plots

Generates comparison visualizations between the baseline vanilla DQN
and the Double DQN experiment at equivalent training checkpoints.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = "logs/evaluation/double_dqn/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Data ----
# Rigorous 20-episode greedy evaluation results

checkpoints = [50, 100, 150, 200, 300]  # in thousands

# DDQN evaluation results (20 episodes, seed 42)
ddqn_mean_reward   = [15.0, 192.0, 254.0, 238.0, 186.0]
ddqn_median_reward = [0.0, 180.0, 250.0, 230.0, 180.0]
ddqn_std_reward    = [18.84, 76.79, 84.88, 60.96, 64.53]
ddqn_min_reward    = [0.0, 80.0, 100.0, 140.0, 80.0]
ddqn_max_reward    = [60.0, 380.0, 400.0, 380.0, 340.0]
ddqn_mean_length   = [963.55, 724.05, 930.30, 884.15, 601.0]
ddqn_mean_q        = [0.0644, 0.1127, 0.1788, 0.2036, 0.4221]
ddqn_max_q         = [0.0670, 0.1365, 0.2038, 0.2405, 0.4676]

# Vanilla DQN evaluation results (20 episodes, seed 42)
# We have rigorous data for 100k and 300k
# For 50k, 150k, 200k we use the training evaluation data from extended baseline
vanilla_mean_reward   = [None, 286.0, None, None, 316.0]
vanilla_std_reward    = [None, 83.45, None, None, 115.0]
vanilla_mean_q        = [None, 0.2288, None, None, 0.5506]
vanilla_max_q         = [None, 0.2857, None, None, 0.6084]
vanilla_mean_length   = [None, 956.85, None, None, 1134.3]

# Points where we have both
compare_points = [100, 300]
ddqn_compare_mean = [192.0, 186.0]
ddqn_compare_std = [76.79, 64.53]
vanilla_compare_mean = [286.0, 316.0]
vanilla_compare_std = [83.45, 115.0]
ddqn_compare_q = [0.1127, 0.4221]
vanilla_compare_q = [0.2288, 0.5506]
ddqn_compare_maxq = [0.1365, 0.4676]
vanilla_compare_maxq = [0.2857, 0.6084]
ddqn_compare_length = [724.05, 601.0]
vanilla_compare_length = [956.85, 1134.3]

# ---- Plot 1: DDQN Evaluation Reward Over Training ----
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(checkpoints, ddqn_mean_reward, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Double DQN Mean Reward')
ax.fill_between(checkpoints,
                [m - s for m, s in zip(ddqn_mean_reward, ddqn_std_reward)],
                [m + s for m, s in zip(ddqn_mean_reward, ddqn_std_reward)],
                alpha=0.2, color='#2196F3')
ax.set_xlabel('Training Steps (thousands)', fontsize=12)
ax.set_ylabel('Evaluation Reward', fontsize=12)
ax.set_title('Double DQN: Evaluation Reward Over Training', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ddqn_reward_curve.png"), dpi=150)
plt.close()

# ---- Plot 2: DQN vs DDQN Reward Comparison ----
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(compare_points))
width = 0.35
bars1 = ax.bar(x - width/2, vanilla_compare_mean, width, yerr=vanilla_compare_std,
               label='Vanilla DQN', color='#FF5722', alpha=0.85, capsize=5)
bars2 = ax.bar(x + width/2, ddqn_compare_mean, width, yerr=ddqn_compare_std,
               label='Double DQN', color='#2196F3', alpha=0.85, capsize=5)
ax.set_xlabel('Checkpoint (k steps)', fontsize=12)
ax.set_ylabel('Mean Evaluation Reward (20 episodes)', fontsize=12)
ax.set_title('Vanilla DQN vs Double DQN: Reward Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{p}k' for p in compare_points])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dqn_vs_ddqn_reward.png"), dpi=150)
plt.close()

# ---- Plot 3: DQN vs DDQN Q-Value Magnitude ----
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(compare_points))
width = 0.35
bars1 = ax.bar(x - width/2, vanilla_compare_q, width, label='Vanilla DQN (Selected Q)', color='#FF5722', alpha=0.85)
bars2 = ax.bar(x + width/2, ddqn_compare_q, width, label='Double DQN (Selected Q)', color='#2196F3', alpha=0.85)
ax.set_xlabel('Checkpoint (k steps)', fontsize=12)
ax.set_ylabel('Mean Selected Q-Value', fontsize=12)
ax.set_title('Q-Value Magnitude: Vanilla DQN vs Double DQN', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{p}k' for p in compare_points])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dqn_vs_ddqn_qvalues.png"), dpi=150)
plt.close()

# ---- Plot 4: DQN vs DDQN Episode Length ----
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(compare_points))
width = 0.35
bars1 = ax.bar(x - width/2, vanilla_compare_length, width, label='Vanilla DQN', color='#FF5722', alpha=0.85)
bars2 = ax.bar(x + width/2, ddqn_compare_length, width, label='Double DQN', color='#2196F3', alpha=0.85)
ax.set_xlabel('Checkpoint (k steps)', fontsize=12)
ax.set_ylabel('Mean Episode Length', fontsize=12)
ax.set_title('Episode Length: Vanilla DQN vs Double DQN', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{p}k' for p in compare_points])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dqn_vs_ddqn_episode_length.png"), dpi=150)
plt.close()

# ---- Plot 5: DDQN Reward Variance Over Training ----
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(checkpoints, ddqn_std_reward, 'o-', color='#9C27B0', linewidth=2, markersize=8, label='DDQN Std Dev')
ax.set_xlabel('Training Steps (thousands)', fontsize=12)
ax.set_ylabel('Reward Standard Deviation', fontsize=12)
ax.set_title('Double DQN: Reward Variability Over Training', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ddqn_reward_variance.png"), dpi=150)
plt.close()

# ---- Plot 6: DDQN Q-value Growth Over Training ----
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(checkpoints, ddqn_mean_q, 'o-', color='#2196F3', linewidth=2, markersize=8, label='DDQN Selected Q')
ax.plot(checkpoints, ddqn_max_q, 's--', color='#03A9F4', linewidth=2, markersize=8, label='DDQN Max Q')
ax.set_xlabel('Training Steps (thousands)', fontsize=12)
ax.set_ylabel('Mean Q-Value', fontsize=12)
ax.set_title('Double DQN: Q-Value Growth Over Training', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ddqn_qvalue_growth.png"), dpi=150)
plt.close()

# ---- Plot 7: Action Diversity (unique actions used) ----
# DDQN action counts per checkpoint
ddqn_unique_actions = [9, 18, 18, 18, 18]  # 50k has 9, rest have 18
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(checkpoints, ddqn_unique_actions, width=30, color='#4CAF50', alpha=0.85)
ax.set_xlabel('Training Steps (thousands)', fontsize=12)
ax.set_ylabel('Unique Actions Used', fontsize=12)
ax.set_title('Double DQN: Action Diversity Over Training', fontsize=14, fontweight='bold')
ax.set_ylim(0, 20)
ax.axhline(y=18, color='gray', linestyle='--', alpha=0.5, label='Maximum (18)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ddqn_action_diversity.png"), dpi=150)
plt.close()

# ---- Plot 8: Combined Dashboard ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DQN vs Double DQN: 300k Step Comparison Dashboard', fontsize=16, fontweight='bold')

# Subplot 1: Reward
ax = axes[0, 0]
x = np.arange(len(compare_points))
width = 0.3
ax.bar(x - width/2, vanilla_compare_mean, width, yerr=vanilla_compare_std,
       label='Vanilla DQN', color='#FF5722', alpha=0.85, capsize=4)
ax.bar(x + width/2, ddqn_compare_mean, width, yerr=ddqn_compare_std,
       label='Double DQN', color='#2196F3', alpha=0.85, capsize=4)
ax.set_xticks(x)
ax.set_xticklabels([f'{p}k' for p in compare_points])
ax.set_ylabel('Mean Reward')
ax.set_title('Evaluation Reward')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Subplot 2: Q-Values
ax = axes[0, 1]
ax.bar(x - width/2, vanilla_compare_q, width, label='Vanilla DQN', color='#FF5722', alpha=0.85)
ax.bar(x + width/2, ddqn_compare_q, width, label='Double DQN', color='#2196F3', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'{p}k' for p in compare_points])
ax.set_ylabel('Mean Selected Q-Value')
ax.set_title('Q-Value Magnitude')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Subplot 3: Episode Length
ax = axes[1, 0]
ax.bar(x - width/2, vanilla_compare_length, width, label='Vanilla DQN', color='#FF5722', alpha=0.85)
ax.bar(x + width/2, ddqn_compare_length, width, label='Double DQN', color='#2196F3', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'{p}k' for p in compare_points])
ax.set_ylabel('Mean Episode Length')
ax.set_title('Survival (Episode Length)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Subplot 4: DDQN Reward Curve
ax = axes[1, 1]
ax.plot(checkpoints, ddqn_mean_reward, 'o-', color='#2196F3', linewidth=2, markersize=6, label='DDQN')
ax.fill_between(checkpoints,
                [m - s for m, s in zip(ddqn_mean_reward, ddqn_std_reward)],
                [m + s for m, s in zip(ddqn_mean_reward, ddqn_std_reward)],
                alpha=0.2, color='#2196F3')
ax.set_xlabel('Training Steps (k)')
ax.set_ylabel('Mean Reward')
ax.set_title('DDQN Learning Curve')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_dashboard.png"), dpi=150)
plt.close()

print(f"All plots saved to {OUTPUT_DIR}/")
print("Generated:")
for f in os.listdir(OUTPUT_DIR):
    print(f"  - {f}")
