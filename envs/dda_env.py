import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
from enum import IntEnum

# Ensure the parent directory is in the path so we can import envs and players
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.puzzle_game import PuzzleGame, LevelConfig, Action as GameAction
from players.bot_players import NoisyBFSBot, RandomBot, BFSBot, OptimalBot

class DDAAction(IntEnum):
    """Actions the DDA agent can take to adjust difficulty."""
    KEEP_SAME = 0
    INCREASE_SIZE = 1
    DECREASE_SIZE = 2
    INCREASE_KEYS = 3
    DECREASE_KEYS = 4
    INCREASE_ENEMIES = 5
    DECREASE_ENEMIES = 6
    ADD_HINT = 7
    REMOVE_HINT = 8

class DDAEnv(gym.Env):
    """
    Gymnasium environment that acts as an RL director agent.
    The director adjusts the puzzle generation parameters to keep the bot in the "flow channel".
    Each step in this environment corresponds to one complete level played by the bot.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config=None):
        super().__init__()

        default_config = {
            'session_length': 10,       # levels per episode
            't_min': 50,                # target window lower bound (steps)
            't_max': 200,               # target window upper bound (steps)
            'timeout_steps': 500,       # max steps per level
            'grid_size_min': 8,
            'grid_size_max': 20,
            'max_keys': 3,
            'max_enemies': 5,
            'max_hints': 3,
            'initial_grid_size': 12,
            'initial_wall_density': 0.3,
            'initial_num_keys': 1,
            'initial_num_enemies': 0,
            'initial_num_hints': 0,
        }

        self.config = default_config.copy()
        if config is not None:
            self.config.update(config)

        # 16-dimensional observation vector, all normalized to [0, 1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)
        
        # 9 possible difficulty adjustment actions
        self.action_space = spaces.Discrete(9)

        # Extract config to instance variables for easy access
        self.t_min = self.config['t_min']
        self.t_max = self.config['t_max']
        self.timeout_steps = self.config['timeout_steps']
        self.session_length = self.config['session_length']
        self.grid_size_min = self.config['grid_size_min']
        self.grid_size_max = self.config['grid_size_max']
        self.max_keys = self.config['max_keys']
        self.max_enemies = self.config['max_enemies']
        self.max_hints = self.config['max_hints']

        # Session state
        self.current_grid_size = self.config['initial_grid_size']
        self.current_wall_density = self.config['initial_wall_density']
        self.current_num_keys = self.config['initial_num_keys']
        self.current_num_enemies = self.config['initial_num_enemies']
        self.current_num_hints = self.config['initial_num_hints']

        self.levels_completed = 0
        self.levels_attempted = 0
        self.levels_won = 0
        self.levels_failed = 0
        self.solve_times = []
        self.steps_since_last_win = 0

        self.bot = None
        self.bot_noise_prob = 0.0
        self.game = None

    def reset(self, seed=None, options=None):
        """Resets the session state, difficulty, and picks a new bot."""
        super().reset(seed=seed)

        # Reset session tracking
        self.levels_completed = 0
        self.levels_attempted = 0
        self.levels_won = 0
        self.levels_failed = 0
        self.solve_times = []
        self.steps_since_last_win = 0

        # Reset difficulty
        self.current_grid_size = self.config['initial_grid_size']
        self.current_wall_density = self.config['initial_wall_density']
        self.current_num_keys = self.config['initial_num_keys']
        self.current_num_enemies = self.config['initial_num_enemies']
        self.current_num_hints = self.config['initial_num_hints']

        # Pick a random bot
        noise_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.bot_noise_prob = float(self.np_random.choice(noise_probs))
        
        # Instantiate the bot (assuming NoisyBFSBot takes a noise_prob and seed)
        bot_seed = int(self.np_random.integers(0, 1000000))
        self.bot = NoisyBFSBot(noise_prob=self.bot_noise_prob, seed=bot_seed)

        # Generate the first level
        level_config = LevelConfig(
            grid_size=self.current_grid_size,
            wall_density=self.current_wall_density,
            num_keys=self.current_num_keys,
            num_enemies=self.current_num_enemies,
            num_hints=self.current_num_hints
        )
        game_seed = int(self.np_random.integers(0, 1000000))
        self.game = PuzzleGame(level_config, seed=game_seed)

        # Get initial state
        initial_state = self.game.get_state()
        obs = self._build_observation(initial_state)

        info = {
            'grid_size': self.current_grid_size,
            'num_keys': self.current_num_keys,
            'num_enemies': self.current_num_enemies,
            'num_hints': self.current_num_hints,
            'bot_noise_prob': self.bot_noise_prob,
        }

        return obs, info

    def step(self, action):
        """
        Takes a DDA action, modifies the puzzle parameters, generates a new level, 
        lets the bot play it, and returns the step information.
        """
        # 1. Apply DDA action
        if action == DDAAction.INCREASE_SIZE:
            self.current_grid_size = min(self.current_grid_size + 2, self.grid_size_max)
        elif action == DDAAction.DECREASE_SIZE:
            self.current_grid_size = max(self.current_grid_size - 2, self.grid_size_min)
        elif action == DDAAction.INCREASE_KEYS:
            self.current_num_keys = min(self.current_num_keys + 1, self.max_keys)
        elif action == DDAAction.DECREASE_KEYS:
            self.current_num_keys = max(self.current_num_keys - 1, 0)
        elif action == DDAAction.INCREASE_ENEMIES:
            self.current_num_enemies = min(self.current_num_enemies + 1, self.max_enemies)
        elif action == DDAAction.DECREASE_ENEMIES:
            self.current_num_enemies = max(self.current_num_enemies - 1, 0)
        elif action == DDAAction.ADD_HINT:
            self.current_num_hints = min(self.current_num_hints + 1, self.max_hints)
        elif action == DDAAction.REMOVE_HINT:
            self.current_num_hints = max(self.current_num_hints - 1, 0)

        # 2. Create a new LevelConfig and PuzzleGame
        level_config = LevelConfig(
            grid_size=self.current_grid_size,
            wall_density=self.current_wall_density,
            num_keys=self.current_num_keys,
            num_enemies=self.current_num_enemies,
            num_hints=self.current_num_hints
        )
        game_seed = int(self.np_random.integers(0, 1000000))
        self.game = PuzzleGame(level_config, seed=game_seed)

        # 3. Let the bot play the entire level
        result = self.bot.play_level(self.game)
        
        solved = result.get('solved', False)
        T = result.get('steps', self.timeout_steps)
        deaths = result.get('deaths', 0)
        keys_collected = result.get('keys_collected', 0)
        keys_total = result.get('keys_total', self.current_num_keys)
        optimal_steps = result.get('optimal_steps', 0)
        hints_collected = result.get('hints_collected', 0)

        # 4. Compute reward
        if solved:
            if self.t_min <= T <= self.t_max:
                reward = 1.0  # In the engagement window!
            elif T < self.t_min:
                reward = -0.5  # Too easy
            else:
                reward = -0.3  # Solved but took too long
        else:
            reward = -1.0  # Too hard (timeout/death)

        # Penalties
        reward -= 0.1 * min(deaths / 5.0, 1.0)
        if keys_total > 0:
            hints_penalty = 0.05 * (hints_collected / max(self.current_num_hints, 1))
            reward -= hints_penalty

        # 5. Update session state
        self.levels_attempted += 1
        if solved:
            self.levels_won += 1
            self.levels_completed += 1
            self.steps_since_last_win = 0
        else:
            self.levels_failed += 1
            self.steps_since_last_win += T
        self.solve_times.append(T)

        # 6. Check termination
        terminated = (self.levels_attempted >= self.session_length)
        truncated = False

        # 7. Build observation from the END of this level's play
        final_state = self.game.get_state()
        obs = self._build_observation(final_state)

        # 8. Build info dict
        info = {
            'solve_time': T,
            'solved': solved,
            'deaths': deaths,
            'reward': reward,
            'grid_size': self.current_grid_size,
            'num_keys': self.current_num_keys,
            'num_enemies': self.current_num_enemies,
            'num_hints': self.current_num_hints,
            'bot_noise_prob': self.bot_noise_prob
        }

        # 9. Return tuple
        return obs, reward, terminated, truncated, info

    def _build_observation(self, game_state):
        """Builds the 16-dimensional normalized observation vector."""
        step_count = game_state.get('step_count', 0)
        deaths_this_level = game_state.get('deaths', 0)
        hints_collected = game_state.get('hints_collected', 0)
        keys_total = game_state.get('keys_total', self.current_num_keys)
        keys_collected = game_state.get('keys_collected', 0)
        optimal_steps = game_state.get('optimal_steps', 0)

        # Manhattan distance to exit
        player_pos = game_state.get('player_pos', (0, 0))
        exit_pos = game_state.get('exit_pos', (0, 0))
        dist_to_exit = abs(player_pos[0] - exit_pos[0]) + abs(player_pos[1] - exit_pos[1])

        # 0. time_elapsed_norm
        time_elapsed_norm = step_count / max(self.timeout_steps, 1)
        # 1. time_since_success_norm
        time_since_success_norm = self.steps_since_last_win / max(self.timeout_steps * 3, 1)
        # 2. deaths_norm
        deaths_norm = deaths_this_level / 5.0
        # 3. retries_norm
        retries_norm = self.levels_failed / max(self.session_length, 1)
        # 4. hints_used_norm
        hints_used_norm = hints_collected / max(self.current_num_hints, 1)
        # 5. steps_vs_optimal
        steps_vs_opt = step_count / max(optimal_steps, 1)
        steps_vs_optimal = min(steps_vs_opt, 3.0) / 3.0
        # 6. keys_collected_frac
        keys_collected_frac = keys_collected / max(keys_total, 1)
        # 7. dist_to_exit_norm
        dist_to_exit_norm = dist_to_exit / max(self.current_grid_size * 2, 1)
        # 8. grid_size_norm
        grid_size_norm = (self.current_grid_size - self.grid_size_min) / max(self.grid_size_max - self.grid_size_min, 1)
        # 9. wall_density_norm
        wall_density_norm = self.current_wall_density
        # 10. num_keys_norm
        num_keys_norm = self.current_num_keys / max(self.max_keys, 1)
        # 11. num_enemies_norm
        num_enemies_norm = self.current_num_enemies / max(self.max_enemies, 1)
        # 12. num_hints_norm
        num_hints_norm = self.current_num_hints / max(self.max_hints, 1)
        # 13. levels_completed_frac
        levels_completed_frac = self.levels_completed / max(self.session_length, 1)
        
        # 14. avg_solve_time_norm
        avg_solve = np.mean(self.solve_times) if self.solve_times else 0.0
        avg_solve_time_norm = avg_solve / max(self.timeout_steps, 1)
        # 15. success_rate
        success_rate = self.levels_won / max(self.levels_attempted, 1)

        # Assemble and clip
        obs = np.array([
            time_elapsed_norm,
            time_since_success_norm,
            deaths_norm,
            retries_norm,
            hints_used_norm,
            steps_vs_optimal,
            keys_collected_frac,
            dist_to_exit_norm,
            grid_size_norm,
            wall_density_norm,
            num_keys_norm,
            num_enemies_norm,
            num_hints_norm,
            levels_completed_frac,
            avg_solve_time_norm,
            success_rate
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def render(self):
        """Renders the current game level as ASCII art."""
        if hasattr(self, 'game') and self.game is not None:
            return self.game.render_ascii()
        return ""

if __name__ == '__main__':
    env = DDAEnv()
    obs, info = env.reset(seed=42)
    print(f'Initial obs: {obs}')
    print(f'Initial info: {info}')
    
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f'Level {i+1}: action={action}, reward={reward:.2f}, solved={info.get("solved")}, '
              f'steps={info.get("solve_time")}, grid={info.get("grid_size")}')
        if terminated:
            break
    print(f'Total reward: {total_reward:.2f}')
