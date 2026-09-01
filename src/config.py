import argparse
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    seed: int = 42
    device: str = None
    
    # Run scale
    total_steps: int = 100000
    
    # Algorithm
    algorithm: str = "dqn"
    
    # Resuming and Logging
    resume: str = None
    log_dir: str = "logs/training"
    checkpoint_dir: str = "models/checkpoints"
    
    # DQN Hyperparameters
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 32
    replay_capacity: int = 100000
    
    # Epsilon (Exploration) schedule
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 1000000
    
    # Training cadence
    learning_starts: int = 20000
    train_frequency: int = 4
    target_update_frequency: int = 10000
    
    # Episode logic
    max_episode_steps: int = 27000  # Classic 27k frame limit (5 minutes of 60fps play)
    
    # Checkpointing and evaluation
    checkpoint_frequency: int = 50000
    evaluation_frequency: int = 50000
    evaluation_episodes: int = 5

def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Seaquest DQN Training")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--algorithm", type=str, default="dqn", choices=["dqn", "double-dqn"], help="RL algorithm to use")
    
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log-dir", type=str, default="logs/training", help="Directory for training logs")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints", help="Directory for saved checkpoints")
    
    parser.add_argument("--steps", type=int, default=100000, help="Total environment steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--replay-capacity", type=int, default=100000, help="Replay buffer capacity")
    
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.1, help="Ending epsilon")
    parser.add_argument("--epsilon-decay-steps", type=int, default=1000000, help="Epsilon decay steps")
    
    parser.add_argument("--learning-starts", type=int, default=20000, help="Steps before learning begins")
    parser.add_argument("--train-frequency", type=int, default=4, help="Steps between learning updates")
    parser.add_argument("--target-update-frequency", type=int, default=10000, help="Steps between target updates")
    
    parser.add_argument("--checkpoint-frequency", type=int, default=50000, help="Steps between checkpoints")
    parser.add_argument("--evaluation-frequency", type=int, default=50000, help="Steps between evaluations")
    parser.add_argument("--evaluation-episodes", type=int, default=5, help="Number of evaluation episodes")
    
    args, unknown = parser.parse_known_args()
    
    return TrainingConfig(
        seed=args.seed,
        device=args.device,
        algorithm=args.algorithm,
        resume=args.resume,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        total_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        replay_capacity=args.replay_capacity,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        learning_starts=args.learning_starts,
        train_frequency=args.train_frequency,
        target_update_frequency=args.target_update_frequency,
        checkpoint_frequency=args.checkpoint_frequency,
        evaluation_frequency=args.evaluation_frequency,
        evaluation_episodes=args.evaluation_episodes
    )
