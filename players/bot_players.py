import sys
import os
import random
import itertools
from collections import deque
from typing import Optional, List, Tuple, Dict, Any
from abc import ABC, abstractmethod
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from envs.puzzle_game import Action, TileType


def _manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _find_tiles(grid: np.ndarray, tile_type: TileType) -> List[Tuple[int, int]]:
    """Find all positions of a given tile type in the grid."""
    indices = np.where(grid == tile_type.value)
    return list(zip(indices[0], indices[1]))


def _bfs_path(grid: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], inventory: set) -> List[Action]:
    """BFS from start to end on the grid.
    
    Walls are impassable. Doors are passable only if matching key is in inventory.
    Returns list of actions to follow, or empty list if unreachable.
    """
    if start == end:
        return []

    rows, cols = grid.shape
    queue = deque([(start, [])])
    visited = {start}
    
    # Mapping of doors to required keys
    door_to_key = {
        TileType.DOOR_R.value: 'R',
        TileType.DOOR_G.value: 'G',
        TileType.DOOR_B.value: 'B'
    }

    directions = [
        (Action.UP, (-1, 0)),
        (Action.DOWN, (1, 0)),
        (Action.LEFT, (0, -1)),
        (Action.RIGHT, (0, 1))
    ]

    while queue:
        (r, c), path = queue.popleft()

        for action, (dr, dc) in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor = (nr, nc)
                if neighbor not in visited:
                    tile_val = grid[nr, nc]
                    
                    # Check if passable
                    passable = True
                    if tile_val == TileType.WALL.value:
                        passable = False
                    elif tile_val in door_to_key:
                        req_key = door_to_key[tile_val]
                        if req_key not in inventory:
                            passable = False

                    if passable:
                        if neighbor == end:
                            return path + [action]
                        visited.add(neighbor)
                        queue.append((neighbor, path + [action]))
                        
    return []


class BaseBot(ABC):
    """Abstract base class for all bot players."""
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    @abstractmethod
    def choose_action(self, state: Dict[str, Any]) -> Action:
        """Choose an action given the current game state."""
        pass
    
    def play_level(self, game) -> Dict[str, Any]:
        """Play a complete level and return performance metrics."""
        state = game.get_state()
        while not state['done']:
            action = self.choose_action(state)
            state = game.step(action)
        return {
            'solved': state['won'],
            'steps': state['step_count'],
            'deaths': state['deaths'],
            'keys_collected': state['keys_collected'],
            'keys_total': state['keys_total'],
            'optimal_steps': state['optimal_steps'],
        }


class RandomBot(BaseBot):
    """Bot that chooses random actions. Very low skill, high variance."""
    def choose_action(self, state: Dict[str, Any]) -> Action:
        return self.rng.choice(list(Action))


class BFSBot(BaseBot):
    """Medium skill bot that uses BFS to find shortest path to objectives."""
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.current_path: List[Action] = []
        self.target: Optional[Tuple[int, int]] = None
        self.last_inventory_size = -1
        self.last_player_pos = None

    def choose_action(self, state: Dict[str, Any]) -> Action:
        grid = state['grid']
        player_pos = state['player_pos']
        inventory = state['inventory']
        
        # If inventory changed or position unexpectedly changed, re-evaluate path
        if len(inventory) != self.last_inventory_size or (self.current_path and self.last_player_pos and _manhattan_distance(player_pos, self.last_player_pos) > 1):
            self.current_path = []
            self.target = None
            self.last_inventory_size = len(inventory)

        self.last_player_pos = player_pos

        # If we have a path, check if it's still valid
        if self.current_path:
            return self.current_path.pop(0)
            
        # Need to find a new target
        keys = []
        keys.extend(_find_tiles(grid, TileType.KEY_R))
        keys.extend(_find_tiles(grid, TileType.KEY_G))
        keys.extend(_find_tiles(grid, TileType.KEY_B))
        
        # Find reachable keys
        reachable_keys = []
        for key_pos in keys:
            path = _bfs_path(grid, player_pos, key_pos, inventory)
            if path:
                reachable_keys.append((key_pos, path))
                
        if reachable_keys:
            # Sort by path length and pick the closest
            reachable_keys.sort(key=lambda x: len(x[1]))
            self.target = reachable_keys[0][0]
            self.current_path = reachable_keys[0][1]
        else:
            # Try to go to exit
            exit_pos = state['exit_pos']
            path = _bfs_path(grid, player_pos, exit_pos, inventory)
            if path:
                self.target = exit_pos
                self.current_path = path
            else:
                # Fallback to random
                return self.rng.choice(list(Action))
                
        if self.current_path:
            return self.current_path.pop(0)
        return self.rng.choice(list(Action))


class OptimalBot(BaseBot):
    """Near-optimal solver: evaluates all key collection orderings."""
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.computed_path: List[Action] = []
        self.path_computed = False
        self.last_player_pos = None

    def _compute_optimal_path(self, state: Dict[str, Any]):
        grid = state['grid']
        player_pos = state['player_pos']
        inventory = state['inventory']
        exit_pos = state['exit_pos']
        
        keys = []
        keys.extend(_find_tiles(grid, TileType.KEY_R))
        keys.extend(_find_tiles(grid, TileType.KEY_G))
        keys.extend(_find_tiles(grid, TileType.KEY_B))
        
        # If no keys, just go to exit
        if not keys:
            self.computed_path = _bfs_path(grid, player_pos, exit_pos, inventory)
            self.path_computed = True
            return

        best_path = None
        best_len = float('inf')
        
        # Try all permutations of keys
        for key_order in itertools.permutations(keys):
            curr_pos = player_pos
            curr_inv = set(inventory)
            curr_path = []
            possible = True
            
            for key_pos in key_order:
                path = _bfs_path(grid, curr_pos, key_pos, curr_inv)
                if not path and curr_pos != key_pos:
                    possible = False
                    break
                curr_path.extend(path)
                curr_pos = key_pos
                # We need to add the key to inventory to proceed
                if grid[key_pos] == TileType.KEY_R.value:
                    curr_inv.add('R')
                elif grid[key_pos] == TileType.KEY_G.value:
                    curr_inv.add('G')
                elif grid[key_pos] == TileType.KEY_B.value:
                    curr_inv.add('B')
                    
            if possible:
                # Finally, go to exit
                exit_path = _bfs_path(grid, curr_pos, exit_pos, curr_inv)
                if exit_path or curr_pos == exit_pos:
                    curr_path.extend(exit_path)
                    if len(curr_path) < best_len:
                        best_len = len(curr_path)
                        best_path = curr_path
                        
        if best_path is not None:
            self.computed_path = best_path
        else:
            # Fallback if no full path is found, just try to get to the exit directly if possible
            fallback = _bfs_path(grid, player_pos, exit_pos, inventory)
            if fallback:
                self.computed_path = fallback
            else:
                self.computed_path = []
                
        self.path_computed = True

    def choose_action(self, state: Dict[str, Any]) -> Action:
        player_pos = state['player_pos']
        if not self.path_computed or (self.last_player_pos and _manhattan_distance(player_pos, self.last_player_pos) > 1):
            self._compute_optimal_path(state)
            
        self.last_player_pos = player_pos

        if self.computed_path:
            return self.computed_path.pop(0)
        return self.rng.choice(list(Action))


class NoisyBFSBot(BaseBot):
    """
    Main bot for training.
    Takes a noise_prob parameter (0.0 to 1.0).
    At each step, with probability noise_prob, takes a random action instead of the BFS-optimal action.
    With probability (1 - noise_prob), follows BFS like BFSBot.
    """
    def __init__(self, noise_prob: float = 0.3, seed: Optional[int] = None):
        super().__init__(seed)
        self.noise_prob = max(0.0, min(1.0, noise_prob))
        self._bfs_bot = BFSBot(seed=seed)

    def choose_action(self, state: Dict[str, Any]) -> Action:
        if self.rng.random() < self.noise_prob:
            # Force BFS to recalculate path since we are taking a random step and diverting from its plan
            self._bfs_bot.current_path = []
            # Still update the BFS bot's state variables so it doesn't get confused
            self._bfs_bot.last_player_pos = state['player_pos']
            self._bfs_bot.last_inventory_size = len(state['inventory'])
            return self.rng.choice(list(Action))
        else:
            return self._bfs_bot.choose_action(state)


if __name__ == '__main__':
    from envs.puzzle_game import PuzzleGame, LevelConfig
    config = LevelConfig(grid_size=12, num_keys=2, num_enemies=0)
    
    for BotClass, name in [(RandomBot, 'Random'), (BFSBot, 'BFS'), (OptimalBot, 'Optimal'), (NoisyBFSBot, 'NoisyBFS(0.3)')]:
        game = PuzzleGame(config, seed=42)
        if name.startswith('Noisy'):
            bot = BotClass(noise_prob=0.3, seed=123)
        else:
            bot = BotClass(seed=123)
        result = bot.play_level(game)
        print(f'{name}: {result}')
