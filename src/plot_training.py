import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    training_file = "logs/training/training_metrics.csv"
    eval_file = "logs/training/evaluation_metrics.csv"
    out_dir = "logs/evaluation/plots"
    
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(training_file):
        print(f"Error: {training_file} not found.")
        return
        
    df_train = pd.read_csv(training_file)
    
    # 1. Training reward vs steps
    plt.figure(figsize=(10, 6))
    plt.plot(df_train['step'], df_train['episode_reward'], color='blue', alpha=0.3, label='Episode Reward')
    
    # 2. Rolling mean reward
    window = 100
    if 'mean_reward_100' in df_train.columns:
        plt.plot(df_train['step'], df_train['mean_reward_100'], color='darkblue', linewidth=2, label=f'Rolling Mean ({window})')
    else:
        df_train['rolling_reward'] = df_train['episode_reward'].rolling(window=window, min_periods=1).mean()
        plt.plot(df_train['step'], df_train['rolling_reward'], color='darkblue', linewidth=2, label=f'Rolling Mean ({window})')
        
    plt.title('Training Reward over Environment Steps')
    plt.xlabel('Environment Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_dir}/training_reward.png")
    plt.close()
    
    # 3. Loss vs learning updates
    if 'loss' in df_train.columns and not df_train['loss'].isna().all():
        plt.figure(figsize=(10, 6))
        # Filter out NaN rows where loss wasn't logged
        df_loss = df_train.dropna(subset=['loss', 'learning_updates'])
        plt.plot(df_loss['learning_updates'], df_loss['loss'], color='red', alpha=0.5)
        plt.title('Huber Loss over Learning Updates')
        plt.xlabel('Learning Updates')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/training_loss.png")
        plt.close()
        
    # 4. Epsilon vs environment steps
    if 'epsilon' in df_train.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df_train['step'], df_train['epsilon'], color='green', linewidth=2)
        plt.title('Epsilon Decay over Environment Steps')
        plt.xlabel('Environment Steps')
        plt.ylabel('Epsilon')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/training_epsilon.png")
        plt.close()
        
    # 5. Episode length vs steps
    if 'episode_length' in df_train.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df_train['step'], df_train['episode_length'], color='purple', alpha=0.3, s=5)
        if 'mean_length_100' in df_train.columns:
            plt.plot(df_train['step'], df_train['mean_length_100'], color='darkviolet', linewidth=2, label='Rolling Length (100)')
            plt.legend()
        plt.title('Episode Length over Environment Steps')
        plt.xlabel('Environment Steps')
        plt.ylabel('Episode Length (Frames)')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/training_length.png")
        plt.close()

    # 6. Evaluation reward vs step
    if os.path.exists(eval_file):
        df_eval = pd.read_csv(eval_file)
        plt.figure(figsize=(10, 6))
        
        plt.errorbar(df_eval['step'], df_eval['mean_reward'], yerr=df_eval['std_reward'], 
                     fmt='-o', color='orange', ecolor='lightsalmon', elinewidth=3, capsize=5)
        plt.title('Greedy Evaluation Reward over Training')
        plt.xlabel('Environment Steps')
        plt.ylabel('Mean Reward')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/evaluation_reward.png")
        plt.close()
        
    print(f"Plots saved to {out_dir}/")

if __name__ == "__main__":
    main()
