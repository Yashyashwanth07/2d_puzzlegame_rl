"""
Evaluation script for DDA Director Agent.

Evaluates multiple DDA modes (fixed, heuristic, PPO, DQN) across
simulated bot players and computes aggregate metrics.
"""
import sys
import os
import argparse
import yaml
import numpy as np
import csv
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.puzzle_game import PuzzleGame, LevelConfig
from envs.dda_env import DDAEnv, DDAAction
from players.bot_players import NoisyBFSBot


# ---------------------------------------------------------------------------
# DDA strategies (baselines + trained agents)
# ---------------------------------------------------------------------------

class FixedDDA:
    """Fixed difficulty — never changes parameters."""
    def __init__(self, config: dict):
        self.config = config

    def choose_action(self, obs) -> int:
        return DDAAction.KEEP_SAME


class HeuristicDDA:
    """
    Rule-based DDA:
      - If last level solved too fast (< t_min)  → increase complexity.
      - If last level timed out / failed          → decrease complexity.
      - Otherwise                                 → keep same.
    """
    def __init__(self, config: dict):
        self.t_min = config.get('t_min', 50)
        self.t_max = config.get('t_max', 200)
        self.last_solved = None
        self.last_solve_time = None

    def observe_result(self, solved: bool, solve_time: int):
        self.last_solved = solved
        self.last_solve_time = solve_time

    def choose_action(self, obs) -> int:
        if self.last_solved is None:
            return DDAAction.KEEP_SAME

        if not self.last_solved:
            # Too hard — decrease complexity
            return DDAAction.DECREASE_SIZE
        elif self.last_solve_time < self.t_min:
            # Too easy — increase complexity
            return DDAAction.INCREASE_SIZE
        elif self.last_solve_time > self.t_max:
            # Solved but slow — decrease a bit
            return DDAAction.DECREASE_ENEMIES
        else:
            return DDAAction.KEEP_SAME


class RLAgentDDA:
    """Wrapper around a trained SB3 model for evaluation."""
    def __init__(self, model_path: str):
        # Defer import so script works even without SB3 installed
        from stable_baselines3 import PPO, DQN
        # Try loading as PPO first, then DQN
        try:
            self.model = PPO.load(model_path)
        except Exception:
            self.model = DQN.load(model_path)

    def choose_action(self, obs) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_mode(
    mode: str,
    dda_agent,
    env_config: dict,
    num_episodes: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Run *num_episodes* sessions and collect per-level results.

    Returns a list of dicts, one per level played.
    """
    env = DDAEnv(config=env_config)
    records: List[Dict] = []
    rng = np.random.RandomState(seed)

    for ep in range(num_episodes):
        obs, info = env.reset(seed=int(rng.randint(0, 1_000_000)))
        terminated = False
        level_idx = 0

        # For heuristic: feed it the first "no result" observation
        if isinstance(dda_agent, HeuristicDDA):
            dda_agent.last_solved = None

        while not terminated:
            action = dda_agent.choose_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            records.append({
                'mode': mode,
                'episode': ep,
                'level': level_idx,
                'action': int(action),
                'reward': reward,
                'solved': info.get('solved', False),
                'solve_time': info.get('solve_time', 0),
                'deaths': info.get('deaths', 0),
                'grid_size': info.get('grid_size', 0),
                'num_keys': info.get('num_keys', 0),
                'num_enemies': info.get('num_enemies', 0),
                'num_hints': info.get('num_hints', 0),
                'bot_noise': info.get('bot_noise_prob', 0),
            })

            # Feed result to heuristic
            if isinstance(dda_agent, HeuristicDDA):
                dda_agent.observe_result(
                    info.get('solved', False),
                    info.get('solve_time', 0),
                )
            level_idx += 1

    return records


def compute_metrics(records: List[Dict], t_min: int, t_max: int) -> Dict:
    """Compute aggregate metrics from per-level records."""
    if not records:
        return {}

    solved = [r for r in records if r['solved']]
    solve_times = [r['solve_time'] for r in solved]
    in_window = [t for t in solve_times if t_min <= t <= t_max]

    total = len(records)
    return {
        'total_levels': total,
        'solved': len(solved),
        'solve_rate': len(solved) / total if total else 0,
        'in_window_rate': len(in_window) / total if total else 0,
        'failure_rate': 1 - len(solved) / total if total else 1,
        'mean_solve_time': np.mean(solve_times) if solve_times else 0,
        'std_solve_time': np.std(solve_times) if solve_times else 0,
        'mean_reward': np.mean([r['reward'] for r in records]),
        'total_deaths': sum(r['deaths'] for r in records),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate DDA strategies')
    parser.add_argument('--mode', default='all',
                        choices=['all', 'fixed', 'heuristic', 'ppo', 'dqn'],
                        help='Which DDA mode(s) to evaluate')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes per mode')
    parser.add_argument('--ppo-model', default=None,
                        help='Path to trained PPO model (.zip)')
    parser.add_argument('--dqn-model', default=None,
                        help='Path to trained DQN model (.zip)')
    parser.add_argument('--config', default='configs/ppo_config.yaml',
                        help='Environment config YAML')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: results/logs/eval_<timestamp>.csv)')
    args = parser.parse_args()

    # Load config
    env_config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        env_config = {
            'session_length': cfg.get('session_length', 10),
            't_min': cfg.get('t_min', 50),
            't_max': cfg.get('t_max', 200),
            'timeout_steps': cfg.get('timeout_steps', 500),
        }
    t_min = env_config.get('t_min', 50)
    t_max = env_config.get('t_max', 200)

    # Output path
    if args.output is None:
        os.makedirs('results/logs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'results/logs/eval_{timestamp}.csv'

    # Define modes to evaluate
    modes = {}
    if args.mode in ('all', 'fixed'):
        modes['fixed'] = FixedDDA(env_config)
    if args.mode in ('all', 'heuristic'):
        modes['heuristic'] = HeuristicDDA(env_config)
    if args.mode in ('all', 'ppo') and args.ppo_model:
        modes['ppo'] = RLAgentDDA(args.ppo_model)
    if args.mode in ('all', 'dqn') and args.dqn_model:
        modes['dqn'] = RLAgentDDA(args.dqn_model)

    if not modes:
        print("No modes to evaluate. If using RL modes, provide --ppo-model or --dqn-model.")
        if args.mode == 'all':
            # Still run fixed + heuristic
            modes['fixed'] = FixedDDA(env_config)
            modes['heuristic'] = HeuristicDDA(env_config)
        else:
            return

    # Run evaluations
    all_records = []
    print(f"\n{'='*60}")
    print(f"  DDA Evaluation — {args.episodes} episodes per mode")
    print(f"  Target window: [{t_min}, {t_max}] steps")
    print(f"{'='*60}\n")

    for mode_name, agent in modes.items():
        print(f"Evaluating [{mode_name}]...", end=' ', flush=True)
        records = evaluate_mode(
            mode=mode_name,
            dda_agent=agent,
            env_config=env_config,
            num_episodes=args.episodes,
            seed=args.seed,
        )
        all_records.extend(records)
        metrics = compute_metrics(records, t_min, t_max)
        print("Done!")
        print(f"  Solve rate:    {metrics['solve_rate']:.1%}")
        print(f"  In-window:     {metrics['in_window_rate']:.1%}")
        print(f"  Mean solve:    {metrics['mean_solve_time']:.1f} ± {metrics['std_solve_time']:.1f}")
        print(f"  Mean reward:   {metrics['mean_reward']:.3f}")
        print(f"  Deaths:        {metrics['total_deaths']}")
        print()

    # Save results CSV
    if all_records:
        keys = all_records[0].keys()
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"Results saved to {args.output}")

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"  Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Mode':<12} {'Solve%':>8} {'InWindow%':>10} {'MeanTime':>10} {'Reward':>8}")
    print(f"{'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for mode_name in modes:
        mode_records = [r for r in all_records if r['mode'] == mode_name]
        m = compute_metrics(mode_records, t_min, t_max)
        print(f"{mode_name:<12} {m['solve_rate']:>7.1%} {m['in_window_rate']:>9.1%} "
              f"{m['mean_solve_time']:>9.1f} {m['mean_reward']:>7.3f}")
    print()


if __name__ == '__main__':
    main()
