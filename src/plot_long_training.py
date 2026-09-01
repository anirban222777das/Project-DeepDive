import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_oxygen_metrics():
    """Extracts oxygen metrics from evaluation logs."""
    logs = glob.glob("logs/evaluation/double_dqn_1m_eval_*.log")
    metrics = []
    
    for log_path in logs:
        # filename is like double_dqn_1m_eval_100000.log
        basename = os.path.basename(log_path)
        step_match = re.search(r"eval_(\d+).log", basename)
        if not step_match:
            continue
        step = int(step_match.group(1))
        
        with open(log_path, "r") as f:
            content = f.read()
            
        avg_oxy = re.search(r"Avg Oxygen at Death:\s+([0-9.]+)", content)
        low_oxy = re.search(r"Times Reached Low Oxygen \(<16\):\s+(\d+)", content)
        surf_low = re.search(r"Successful Resurfaces from Low:\s+(\d+)", content)
        recov = re.search(r"Total Oxygen Recovery Events:\s+(\d+)", content)
        
        if avg_oxy and low_oxy and surf_low and recov:
            metrics.append({
                "step": step,
                "avg_oxy_death": float(avg_oxy.group(1)),
                "low_oxy_count": int(low_oxy.group(1)),
                "surf_low_count": int(surf_low.group(1)),
                "total_recovery": int(recov.group(1))
            })
            
    df = pd.DataFrame(metrics)
    if not df.empty:
        df = df.sort_values("step")
    return df

def plot_training_metrics():
    os.makedirs("logs/evaluation/double_dqn_1m/plots", exist_ok=True)
    
    # 1. Training Reward, Length, Epsilon, Loss, Q-values
    train_df = pd.read_csv("logs/training/double_dqn_1m/training_metrics.csv", 
                           names=["step", "episode", "reward", "length", "epsilon", "loss", "mean_q", "max_q", "grad_norm", "learning_updates", "env_steps"])
                           
    for col in ["reward", "length", "epsilon", "loss", "mean_q", "max_q", "env_steps", "learning_updates"]:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
    
    # Plot Reward vs Env Steps
    plt.figure(figsize=(10, 6))
    plt.plot(train_df["env_steps"], train_df["reward"], alpha=0.3, color="blue", label="Episode Reward")
    plt.plot(train_df["env_steps"], train_df["reward"].rolling(50).mean(), color="red", label="Rolling Mean (50)")
    plt.title("Training Reward vs Environment Steps (1M DDQN)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/evaluation/double_dqn_1m/plots/reward_vs_env_steps.png")
    plt.close()
    
    # Plot Episode Length vs Env Steps
    plt.figure(figsize=(10, 6))
    plt.plot(train_df["env_steps"], train_df["length"], alpha=0.3, color="green", label="Episode Length")
    plt.plot(train_df["env_steps"], train_df["length"].rolling(50).mean(), color="orange", label="Rolling Mean (50)")
    plt.title("Episode Length vs Environment Steps (1M DDQN)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Length (frames)")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/evaluation/double_dqn_1m/plots/length_vs_env_steps.png")
    plt.close()
    
    # Plot Loss vs Learning Updates
    valid_loss = train_df.dropna(subset=["loss"])
    plt.figure(figsize=(10, 6))
    plt.plot(valid_loss["learning_updates"], valid_loss["loss"], alpha=0.3, color="purple")
    plt.plot(valid_loss["learning_updates"], valid_loss["loss"].rolling(50).mean(), color="red")
    plt.title("Training Loss vs Learning Updates (1M DDQN)")
    plt.xlabel("Learning Updates")
    plt.ylabel("Huber Loss")
    plt.grid(True)
    plt.savefig("logs/evaluation/double_dqn_1m/plots/loss_vs_updates.png")
    plt.close()
    
    # Plot Q-Values vs Env Steps
    valid_q = train_df.dropna(subset=["mean_q"])
    plt.figure(figsize=(10, 6))
    plt.plot(valid_q["env_steps"], valid_q["max_q"].rolling(50).mean(), color="red", label="Max Q (Rolling 50)")
    plt.plot(valid_q["env_steps"], valid_q["mean_q"].rolling(50).mean(), color="blue", label="Mean Q (Rolling 50)")
    plt.title("Q-Values vs Environment Steps (1M DDQN)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Q-Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/evaluation/double_dqn_1m/plots/q_values_vs_env_steps.png")
    plt.close()

    # Plot Epsilon vs Env Steps
    plt.figure(figsize=(10, 6))
    plt.plot(train_df["env_steps"], train_df["epsilon"], color="black")
    plt.title("Epsilon vs Environment Steps (1M DDQN)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.savefig("logs/evaluation/double_dqn_1m/plots/epsilon_vs_env_steps.png")
    plt.close()
    
def plot_evaluation_metrics():
    # 2. Evaluation metrics
    eval_df = pd.read_csv("logs/evaluation/evaluation_summary.csv")
    eval_df = eval_df[eval_df["checkpoint"].str.contains("dqn_step")]
    # Extract step from checkpoint filename
    eval_df["step"] = eval_df["checkpoint"].apply(lambda x: int(re.search(r"step_(\d+).pt", x).group(1)))
    eval_df = eval_df.sort_values("step")
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(eval_df["step"], eval_df["mean_reward"], yerr=eval_df["std_reward"], fmt='-o', color="blue", capsize=5)
    plt.title("Evaluation Mean Reward vs Training Steps (1M DDQN)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Mean Reward (20 episodes)")
    plt.grid(True)
    plt.savefig("logs/evaluation/double_dqn_1m/plots/eval_reward_vs_step.png")
    plt.close()
    
    # 3. Oxygen Metrics
    oxy_df = extract_oxygen_metrics()
    if not oxy_df.empty:
        # Plot Average Oxygen at Death
        plt.figure(figsize=(10, 6))
        plt.plot(oxy_df["step"], oxy_df["avg_oxy_death"], marker='o', color="red")
        plt.title("Average Oxygen at Death vs Training Steps")
        plt.xlabel("Environment Steps")
        plt.ylabel("Oxygen Level at Death (0-64)")
        plt.grid(True)
        plt.savefig("logs/evaluation/double_dqn_1m/plots/oxy_death_vs_step.png")
        plt.close()
        
        # Plot Oxygen Recovery Events
        plt.figure(figsize=(10, 6))
        plt.plot(oxy_df["step"], oxy_df["total_recovery"], marker='o', color="green", label="Total Oxygen Refills")
        plt.plot(oxy_df["step"], oxy_df["surf_low_count"], marker='s', color="blue", label="Refills Specifically from Low (<25%)")
        plt.title("Oxygen Recovery Events vs Training Steps (Total 20 Episodes)")
        plt.xlabel("Environment Steps")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.savefig("logs/evaluation/double_dqn_1m/plots/oxy_recovery_vs_step.png")
        plt.close()
        
        # Plot Low Oxygen Frequency
        plt.figure(figsize=(10, 6))
        plt.plot(oxy_df["step"], oxy_df["low_oxy_count"], marker='o', color="orange")
        plt.title("Frequency of Entering Low Oxygen State (<25%)")
        plt.xlabel("Environment Steps")
        plt.ylabel("Count")
        plt.grid(True)
        plt.savefig("logs/evaluation/double_dqn_1m/plots/low_oxy_freq_vs_step.png")
        plt.close()
        
if __name__ == "__main__":
    plot_training_metrics()
    plot_evaluation_metrics()
    print("Plots generated in logs/evaluation/double_dqn_1m/plots/")
