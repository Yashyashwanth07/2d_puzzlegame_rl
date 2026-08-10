import argparse
import yaml
import os
import sys
import numpy as np
from datetime import datetime

# Add parent directory to path so we can import envs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from envs.dda_env import DDAEnv

class MetricsCallback(BaseCallback):
    """Logs per-episode metrics to CSV."""
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.csv_path = os.path.join(log_dir, 'training_metrics.csv')
        # Create CSV header
        os.makedirs(log_dir, exist_ok=True)
        with open(self.csv_path, 'w') as f:
            f.write('timestep,episode_reward,mean_reward\n')
    
    def _on_step(self):
        # Check if episode ended
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    if 'episode' in info:
                        ep_reward = info['episode']['r']
                        self.episode_rewards.append(ep_reward)
                        mean_reward = np.mean(self.episode_rewards[-100:])
                        with open(self.csv_path, 'a') as f:
                            f.write(f'{self.num_timesteps},{ep_reward:.3f},{mean_reward:.3f}\n')
        return True

def load_config(config_path):
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Train DQN DDA Director')
    parser.add_argument('--config', default='configs/dqn_config.yaml', help='Path to config file')
    parser.add_argument('--timesteps', type=int, default=None, help='Override total timesteps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    args = parser.parse_args()
    
    # Load and merge config
    config = load_config(args.config)
    if args.timesteps:
        config['total_timesteps'] = args.timesteps
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Setup directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config.get('log_dir', 'results/logs/dqn'), timestamp)
    model_dir = os.path.join(config.get('model_dir', 'results/models/dqn'), timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Create environment
    env_config = {
        'session_length': config.get('session_length', 10),
        't_min': config.get('t_min', 50),
        't_max': config.get('t_max', 200),
        'timeout_steps': config.get('timeout_steps', 500),
    }
    env = DDAEnv(config=env_config)
    
    # Create model
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=config.get('learning_rate', 1e-4),
        buffer_size=config.get('buffer_size', 1000000),
        learning_starts=config.get('learning_starts', 50000),
        batch_size=config.get('batch_size', 32),
        gamma=config.get('gamma', 0.99),
        train_freq=config.get('train_freq', 4),
        gradient_steps=config.get('gradient_steps', 1),
        target_update_interval=config.get('target_update_interval', 10000),
        exploration_fraction=config.get('exploration_fraction', 0.1),
        exploration_final_eps=config.get('exploration_final_eps', 0.05),
        max_grad_norm=config.get('max_grad_norm', 10),
        policy_kwargs={'net_arch': config.get('policy_kwargs', {}).get('net_arch', [64, 64])},
        seed=config.get('seed', 42),
        verbose=1,
    )
    
    # Callbacks
    metrics_cb = MetricsCallback(log_dir)
    checkpoint_cb = CheckpointCallback(
        save_freq=config.get('save_interval', 10000),
        save_path=model_dir,
        name_prefix='dqn_dda'
    )
    
    # Train
    total_timesteps = config.get('total_timesteps', 100000)
    print(f'Training DQN for {total_timesteps} timesteps...')
    print(f'Logs: {log_dir}')
    print(f'Models: {model_dir}')
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[metrics_cb, checkpoint_cb],
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(model_dir, 'dqn_dda_final')
    model.save(final_path)
    print(f'Final model saved to {final_path}')
    
    # Print summary
    if metrics_cb.episode_rewards:
        print(f'\n=== Training Summary ===')
        print(f'Episodes: {len(metrics_cb.episode_rewards)}')
        print(f'Mean reward (last 100): {np.mean(metrics_cb.episode_rewards[-100:]):.3f}')
        print(f'Max reward: {max(metrics_cb.episode_rewards):.3f}')
        print(f'Min reward: {min(metrics_cb.episode_rewards):.3f}')

if __name__ == '__main__':
    main()
