import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from collections import deque
import random
from typing import Optional, List, Tuple, Set, Dict
import itertools

class TileType(IntEnum):
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    EXIT = 3
    KEY_R = 4
    KEY_G = 5
    KEY_B = 6
    DOOR_R = 7
    DOOR_G = 8
    DOOR_B = 9
    ENEMY = 10
    HINT = 11

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

@dataclass
class LevelConfig:
    grid_size: int = 12
    wall_density: float = 0.3
    num_keys: int = 1
    num_enemies: int = 0
    num_hints: int = 0

    def __post_init__(self):
        self.grid_size = max(8, min(24, self.grid_size))
        self.wall_density = max(0.0, min(0.6, self.wall_density))
        self.num_keys = max(0, min(3, self.num_keys))
        self.num_enemies = max(0, min(5, self.num_enemies))
        self.num_hints = max(0, min(3, self.num_hints))

KEY_DOOR_PAIRS = {
    TileType.KEY_R: TileType.DOOR_R,
    TileType.KEY_G: TileType.DOOR_G,
    TileType.KEY_B: TileType.DOOR_B
}

KEY_COLORS = {
    TileType.KEY_R: 'R',
    TileType.KEY_G: 'G',
    TileType.KEY_B: 'B'
}

DOOR_TO_KEY = {v: k for k, v in KEY_DOOR_PAIRS.items()}

class PuzzleGame:
    """Main game engine class for the 2D grid-based key-door maze puzzle."""
    
    def __init__(self, config: LevelConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.grid = None
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.inventory: Set[str] = set()
        self.step_count = 0
        self.max_steps = self.config.grid_size * self.config.grid_size * 3
        self.done = False
        self.won = False
        self.deaths = 0
        self.hints_collected = 0
        self.keys_total = self.config.num_keys
        
        self.generate_level()

    def generate_level(self) -> None:
        """Generates a complete level with maze, player, exit, keys/doors, enemies, and hints."""
        for _ in range(10):
            # 1. Initialize grid filled with WALLs
            self.grid = np.full((self.config.grid_size, self.config.grid_size), TileType.WALL, dtype=np.int8)
            
            # 2. Generate maze
            self._generate_maze()
            
            # 3. Place player and exit
            empty_cells = self._find_empty_cells()
            if len(empty_cells) < 2:
                continue
                
            # Place player top-left-ish, exit bottom-right-ish
            empty_cells.sort(key=lambda p: p[0]**2 + p[1]**2)
            self.player_pos = empty_cells[0]
            self.exit_pos = empty_cells[-1]
            
            self.grid[self.player_pos] = TileType.PLAYER
            self.grid[self.exit_pos] = TileType.EXIT
            
            # 4. Place key-door pairs
            self._place_key_door_pairs()
            
            # 5. Place enemies
            self._place_enemies()
            
            # 6. Place hints
            self._place_hints()
            
            # 7. Validate solvability
            if self.validate_solvability():
                self.step_count = 0
                self.done = False
                self.won = False
                self.deaths = 0
                self.inventory = set()
                self.hints_collected = 0
                return
                
        # If failed 10 times, fallback to a clear path
        self.grid.fill(TileType.EMPTY)
        self.grid[0, 0] = TileType.PLAYER
        self.player_pos = (0, 0)
        self.grid[-1, -1] = TileType.EXIT
        self.exit_pos = (self.config.grid_size - 1, self.config.grid_size - 1)
        self.step_count = 0
        self.done = False
        self.won = False
        self.deaths = 0
        self.inventory = set()
        self.hints_collected = 0

    def _generate_maze(self) -> None:
        """DFS (recursive backtracker) maze generation."""
        if self.config.grid_size < 3:
            self.grid.fill(TileType.EMPTY)
            return

        start = (1, 1)
        self.grid[start] = TileType.EMPTY
        stack = [start]
        
        while stack:
            curr_r, curr_c = stack[-1]
            neighbors = []
            
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 1 <= nr < self.config.grid_size - 1 and 1 <= nc < self.config.grid_size - 1:
                    if self.grid[nr, nc] == TileType.WALL:
                        neighbors.append((nr, nc, curr_r + dr//2, curr_c + dc//2))
                        
            if neighbors:
                nr, nc, mr, mc = neighbors[self.rng.choice(len(neighbors))]
                self.grid[mr, mc] = TileType.EMPTY
                self.grid[nr, nc] = TileType.EMPTY
                stack.append((nr, nc))
            else:
                stack.pop()

        # Remove extra walls for open spaces based on density
        walls = [(r, c) for r in range(1, self.config.grid_size - 1) 
                 for c in range(1, self.config.grid_size - 1) 
                 if self.grid[r, c] == TileType.WALL]
        
        total_walls = len(walls)
        target_walls = int(total_walls * (self.config.wall_density / 0.6))
        walls_to_remove = max(0, total_walls - target_walls)
        
        if walls_to_remove > 0 and walls:
            indices = self.rng.choice(len(walls), walls_to_remove, replace=False)
            for idx in indices:
                r, c = walls[idx]
                self.grid[r, c] = TileType.EMPTY

    def _place_key_door_pairs(self) -> None:
        """Places keys and matching doors ensuring doors act as required bottlenecks."""
        keys_to_place = [TileType.KEY_R, TileType.KEY_G, TileType.KEY_B][:self.config.num_keys]
        doors_to_place = [TileType.DOOR_R, TileType.DOOR_G, TileType.DOOR_B][:self.config.num_keys]
        
        for k, d in zip(keys_to_place, doors_to_place):
            # Find a path to the exit ignoring doors
            path = self._get_path(self.player_pos, self.exit_pos, ignore_doors=True)
            if not path or len(path) < 4:
                continue

            # Pick door position along the middle of the path
            valid_door_spots = [p for p in path[1:-1] if self.grid[p] == TileType.EMPTY]
            if not valid_door_spots:
                continue

            door_pos = valid_door_spots[self.rng.choice(len(valid_door_spots))]
            self.grid[door_pos] = d

            # Now find a key position that is reachable from start WITHOUT passing through this door
            reachable_before_door = self._get_reachable(self.player_pos, ignore_doors=False)
            reachable_empty = [c for c in self._find_empty_cells() if c in reachable_before_door and c != door_pos]

            if reachable_empty:
                key_pos = reachable_empty[self.rng.choice(len(reachable_empty))]
                self.grid[key_pos] = k
            else:
                # Rollback door if no key spot
                self.grid[door_pos] = TileType.EMPTY

            # Check if exit is unreachable without key (confirming door is a true bottleneck)
            # If exit is still reachable without key, we can add wall framing or try another spot
            if self.exit_pos in self._get_reachable(self.player_pos, ignore_doors=False):
                # Lock alternative paths by placing extra wall segments around door if needed
                r, c = door_pos
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.config.grid_size and 0 <= nc < self.config.grid_size:
                        if self.grid[nr, nc] == TileType.EMPTY and (nr, nc) != self.player_pos and (nr, nc) != self.exit_pos:
                            self.grid[nr, nc] = TileType.WALL
                            # Re-verify if path still exists with key
                            if not self._get_path(self.player_pos, self.exit_pos, ignore_doors=True):
                                self.grid[nr, nc] = TileType.EMPTY  # revert if broke maze

    def _get_path(self, start: Tuple[int, int], end: Tuple[int, int], ignore_doors: bool) -> List[Tuple[int, int]]:
        """BFS to get a path from start to end."""
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            curr, path = queue.popleft()
            
            if curr == end:
                return path
                
            r, c = curr
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.config.grid_size and 0 <= nc < self.config.grid_size:
                    if (nr, nc) not in visited:
                        tile = self.grid[nr, nc]
                        is_passable = tile != TileType.WALL
                        if not ignore_doors and tile in [TileType.DOOR_R, TileType.DOOR_G, TileType.DOOR_B]:
                            is_passable = False
                            
                        if is_passable:
                            visited.add((nr, nc))
                            queue.append(((nr, nc), path + [(nr, nc)]))
        return []

    def _place_enemies(self) -> None:
        """Places enemies on the grid away from start."""
        empty_cells = self._find_empty_cells()
        valid_cells = [c for c in empty_cells if abs(c[0] - self.player_pos[0]) + abs(c[1] - self.player_pos[1]) > 3]
        
        for _ in range(min(self.config.num_enemies, len(valid_cells))):
            idx = self.rng.choice(len(valid_cells))
            pos = valid_cells.pop(idx)
            self.grid[pos] = TileType.ENEMY

    def _place_hints(self) -> None:
        """Places hint tiles on the grid."""
        empty_cells = self._find_empty_cells()
        for _ in range(min(self.config.num_hints, len(empty_cells))):
            idx = self.rng.choice(len(empty_cells))
            pos = empty_cells.pop(idx)
            self.grid[pos] = TileType.HINT

    def _find_empty_cells(self) -> List[Tuple[int, int]]:
        """Returns a list of coordinates for all EMPTY tiles."""
        return [(r, c) for r in range(self.config.grid_size) 
                for c in range(self.config.grid_size) 
                if self.grid[r, c] == TileType.EMPTY]

    def _get_reachable(self, start: Tuple[int, int], ignore_doors: bool = False) -> Set[Tuple[int, int]]:
        """BFS from start, return all reachable cells."""
        queue = deque([start])
        visited = {start}
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.config.grid_size and 0 <= nc < self.config.grid_size:
                    if (nr, nc) not in visited:
                        tile = self.grid[nr, nc]
                        if tile != TileType.WALL:
                            is_door = tile in [TileType.DOOR_R, TileType.DOOR_G, TileType.DOOR_B]
                            if not is_door or ignore_doors:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
        return visited

    def validate_solvability(self) -> bool:
        """
        Check that the exit is reachable from player position when ALL keys are collected.
        Also verify each key is reachable.
        """
        reachable_all_open = self._get_reachable(self.player_pos, ignore_doors=True)
        if self.exit_pos not in reachable_all_open:
            return False
            
        for r in range(self.config.grid_size):
            for c in range(self.config.grid_size):
                if self.grid[r, c] in KEY_COLORS:
                    if (r, c) not in reachable_all_open:
                        return False
        return True

    def step(self, action: Action) -> dict:
        """Executes one step in the game environment."""
        if self.done:
            return self.get_state()
            
        r, c = self.player_pos
        dr, dc = 0, 0
        if action == Action.UP: dr = -1
        elif action == Action.DOWN: dr = 1
        elif action == Action.LEFT: dc = -1
        elif action == Action.RIGHT: dc = 1
        
        nr, nc = r + dr, c + dc
        
        if 0 <= nr < self.config.grid_size and 0 <= nc < self.config.grid_size:
            target_tile = self.grid[nr, nc]
            can_move = False
            
            if target_tile == TileType.EMPTY:
                can_move = True
            elif target_tile in KEY_COLORS:
                self.inventory.add(KEY_COLORS[target_tile])
                can_move = True
            elif target_tile in DOOR_TO_KEY:
                req_key = KEY_COLORS[DOOR_TO_KEY[target_tile]]
                if req_key in self.inventory:
                    can_move = True
            elif target_tile == TileType.EXIT:
                can_move = True
                self.won = True
                self.done = True
            elif target_tile == TileType.ENEMY:
                can_move = True
                self.deaths += 1
                self.won = False
                self.done = True
            elif target_tile == TileType.HINT:
                self.hints_collected += 1
                can_move = True
                
            if can_move:
                self.grid[r, c] = TileType.EMPTY
                self.player_pos = (nr, nc)
                if not self.done:
                    self.grid[nr, nc] = TileType.PLAYER
                elif self.won:
                    self.grid[nr, nc] = TileType.PLAYER

        self.step_count += 1
        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            self.won = False
            
        return self.get_state()

    def get_state(self) -> dict:
        """Returns the current state of the game."""
        return {
            'grid': self.grid.copy(),
            'player_pos': self.player_pos,
            'exit_pos': self.exit_pos,
            'inventory': set(self.inventory),
            'step_count': self.step_count,
            'max_steps': self.max_steps,
            'done': self.done,
            'won': self.won,
            'deaths': self.deaths,
            'hints_collected': self.hints_collected,
            'keys_total': self.keys_total,
            'keys_collected': len(self.inventory),
            'optimal_steps': self.compute_optimal_path_length()
        }

    def compute_optimal_path_length(self) -> int:
        """BFS-based: compute shortest path from player to exit, collecting needed keys."""
        keys = []
        for r in range(self.config.grid_size):
            for c in range(self.config.grid_size):
                if self.grid[r, c] in KEY_COLORS:
                    keys.append((r, c))
                    
        if not keys:
            path_len = self._bfs_shortest_path(self.player_pos, self.exit_pos, passable_doors=set())
            return path_len if path_len != -1 else self.max_steps
            
        min_path_len = self.max_steps
        for perm in itertools.permutations(keys):
            curr_pos = self.player_pos
            total_len = 0
            curr_inventory = set(self.inventory)
            valid = True
            
            for key_pos in perm:
                passable = {KEY_DOOR_PAIRS[k] for k in KEY_DOOR_PAIRS if KEY_COLORS[k] in curr_inventory}
                dist = self._bfs_shortest_path(curr_pos, key_pos, passable_doors=passable)
                if dist == -1:
                    valid = False
                    break
                total_len += dist
                curr_pos = key_pos
                key_tile = self.grid[key_pos]
                curr_inventory.add(KEY_COLORS[key_tile])
                
            if valid:
                passable = {KEY_DOOR_PAIRS[k] for k in KEY_DOOR_PAIRS if KEY_COLORS[k] in curr_inventory}
                dist = self._bfs_shortest_path(curr_pos, self.exit_pos, passable_doors=passable)
                if dist != -1:
                    total_len += dist
                    min_path_len = min(min_path_len, total_len)
                    
        return min_path_len if min_path_len != self.max_steps else self.max_steps

    def _bfs_shortest_path(self, start: Tuple[int, int], end: Tuple[int, int], passable_doors: Set[TileType] = None) -> int:
        """BFS from start to end returning path length."""
        if passable_doors is None:
            passable_doors = set()
            
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            (r, c), dist = queue.popleft()
            
            if (r, c) == end:
                return dist
                
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.config.grid_size and 0 <= nc < self.config.grid_size:
                    if (nr, nc) not in visited:
                        tile = self.grid[nr, nc]
                        is_passable = tile != TileType.WALL
                        if tile in [TileType.DOOR_R, TileType.DOOR_G, TileType.DOOR_B]:
                            if tile not in passable_doors:
                                is_passable = False
                                
                        if is_passable:
                            visited.add((nr, nc))
                            queue.append(((nr, nc), dist + 1))
        return -1

    def render_ascii(self) -> str:
        """Returns a string representation of the grid."""
        char_map = {
            TileType.EMPTY: '.',
            TileType.WALL: '#',
            TileType.PLAYER: 'P',
            TileType.EXIT: 'E',
            TileType.KEY_R: 'r',
            TileType.KEY_G: 'g',
            TileType.KEY_B: 'b',
            TileType.DOOR_R: 'R',
            TileType.DOOR_G: 'G',
            TileType.DOOR_B: 'B',
            TileType.ENEMY: 'X',
            TileType.HINT: '?'
        }
        
        lines = []
        for r in range(self.config.grid_size):
            line = "".join(char_map.get(self.grid[r, c], '?') for c in range(self.config.grid_size))
            lines.append(line)
            
        status = f"Inv: {sorted(list(self.inventory))} | Step: {self.step_count}/{self.max_steps} | Done: {self.done} | Won: {self.won}"
        lines.append(status)
        return "\n".join(lines)

    def reset(self, config: Optional[LevelConfig] = None) -> dict:
        """Resets the environment."""
        if config is not None:
            self.config = config
        self.generate_level()
        return self.get_state()

if __name__ == '__main__':
    config = LevelConfig(grid_size=12, num_keys=2, num_enemies=1, num_hints=1)
    game = PuzzleGame(config, seed=42)
    print(game.render_ascii())
    print(f'Solvable: {game.validate_solvability()}')
    print(f'Optimal path: {game.compute_optimal_path_length()} steps')
