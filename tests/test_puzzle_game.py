"""Unit tests for the puzzle game engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from envs.puzzle_game import PuzzleGame, LevelConfig, TileType, Action, KEY_COLORS, KEY_DOOR_PAIRS


# ---- Fixtures ----

@pytest.fixture
def default_game():
    """A default 12x12 game with 1 key, no enemies."""
    config = LevelConfig(grid_size=12, num_keys=1, num_enemies=0, num_hints=0)
    return PuzzleGame(config, seed=42)


@pytest.fixture
def simple_game():
    """A minimal 8x8 game with no keys or enemies."""
    config = LevelConfig(grid_size=8, num_keys=0, num_enemies=0, num_hints=0)
    return PuzzleGame(config, seed=42)


# ---- Grid & generation tests ----

def test_grid_dimensions():
    """Grid shape should match config.grid_size."""
    for size in [8, 10, 12, 16]:
        config = LevelConfig(grid_size=size, num_keys=0)
        game = PuzzleGame(config, seed=42)
        assert game.grid.shape == (size, size), f"Expected ({size},{size}), got {game.grid.shape}"


def test_grid_has_player_and_exit(default_game):
    """Exactly one PLAYER and one EXIT tile should exist."""
    player_count = np.sum(default_game.grid == TileType.PLAYER)
    exit_count = np.sum(default_game.grid == TileType.EXIT)
    assert player_count == 1, f"Expected 1 PLAYER tile, found {player_count}"
    assert exit_count == 1, f"Expected 1 EXIT tile, found {exit_count}"


@pytest.mark.parametrize("seed", range(10))
def test_maze_solvable(seed):
    """Generated levels should always be solvable."""
    config = LevelConfig(grid_size=12, num_keys=2, num_enemies=0)
    game = PuzzleGame(config, seed=seed)
    assert game.validate_solvability(), f"Level with seed={seed} is not solvable"


# ---- Movement tests ----

def test_player_movement_basic(simple_game):
    """Player should move and step_count should increment."""
    game = simple_game
    initial_pos = game.player_pos
    initial_steps = game.step_count

    # Try all 4 directions, at least one should work
    moved = False
    for action in Action:
        r, c = game.player_pos
        dr = -1 if action == Action.UP else (1 if action == Action.DOWN else 0)
        dc = -1 if action == Action.LEFT else (1 if action == Action.RIGHT else 0)
        nr, nc = r + dr, c + dc
        if (0 <= nr < game.config.grid_size and 0 <= nc < game.config.grid_size
                and game.grid[nr, nc] == TileType.EMPTY):
            state = game.step(action)
            assert game.step_count == initial_steps + 1
            assert game.player_pos == (nr, nc)
            moved = True
            break

    if not moved:
        # If boxed in, at least step_count should increment
        game.step(Action.UP)
        assert game.step_count == initial_steps + 1


def test_wall_collision(simple_game):
    """Player should stay in place when moving into a wall."""
    game = simple_game
    r, c = game.player_pos

    # Find a direction with a wall
    for action in Action:
        dr = -1 if action == Action.UP else (1 if action == Action.DOWN else 0)
        dc = -1 if action == Action.LEFT else (1 if action == Action.RIGHT else 0)
        nr, nc = r + dr, c + dc
        if (0 <= nr < game.config.grid_size and 0 <= nc < game.config.grid_size
                and game.grid[nr, nc] == TileType.WALL):
            game.step(action)
            assert game.player_pos == (r, c), "Player moved through a wall!"
            return

    # If no adjacent wall found, place one
    for action in Action:
        dr = -1 if action == Action.UP else (1 if action == Action.DOWN else 0)
        dc = -1 if action == Action.LEFT else (1 if action == Action.RIGHT else 0)
        nr, nc = r + dr, c + dc
        if 0 <= nr < game.config.grid_size and 0 <= nc < game.config.grid_size:
            game.grid[nr, nc] = TileType.WALL
            game.step(action)
            assert game.player_pos == (r, c), "Player moved through a wall!"
            return


def test_boundary_collision():
    """Player should not move off the grid."""
    config = LevelConfig(grid_size=8, num_keys=0, num_enemies=0)
    game = PuzzleGame(config, seed=42)
    
    # Place player at corner
    old_r, old_c = game.player_pos
    game.grid[old_r, old_c] = TileType.EMPTY
    game.player_pos = (0, 0)
    game.grid[0, 0] = TileType.PLAYER

    # Try moving UP and LEFT — both should be out of bounds
    game.step(Action.UP)
    assert game.player_pos[0] >= 0
    game.step(Action.LEFT)
    assert game.player_pos[1] >= 0


# ---- Key/Door interaction tests ----

def test_key_pickup():
    """Walking onto a key should add it to inventory."""
    config = LevelConfig(grid_size=8, num_keys=0, num_enemies=0)
    game = PuzzleGame(config, seed=42)
    r, c = game.player_pos

    # Place a red key below the player
    for action, (dr, dc) in [(Action.DOWN, (1, 0)), (Action.RIGHT, (0, 1)),
                               (Action.UP, (-1, 0)), (Action.LEFT, (0, -1))]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < game.config.grid_size and 0 <= nc < game.config.grid_size:
            game.grid[nr, nc] = TileType.KEY_R
            state = game.step(action)
            assert 'R' in game.inventory, "Key R not picked up"
            return


def test_door_locked():
    """Player should not pass through a door without the matching key."""
    config = LevelConfig(grid_size=8, num_keys=0, num_enemies=0)
    game = PuzzleGame(config, seed=42)
    r, c = game.player_pos

    # Place a red door adjacent
    for action, (dr, dc) in [(Action.DOWN, (1, 0)), (Action.RIGHT, (0, 1))]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < game.config.grid_size and 0 <= nc < game.config.grid_size:
            game.grid[nr, nc] = TileType.DOOR_R
            game.step(action)
            assert game.player_pos == (r, c), "Player passed through locked door"
            return


def test_door_unlock():
    """Player should pass through a door when they have the matching key."""
    config = LevelConfig(grid_size=8, num_keys=0, num_enemies=0)
    game = PuzzleGame(config, seed=42)
    game.inventory.add('R')  # Give red key
    r, c = game.player_pos

    for action, (dr, dc) in [(Action.DOWN, (1, 0)), (Action.RIGHT, (0, 1))]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < game.config.grid_size and 0 <= nc < game.config.grid_size:
            game.grid[nr, nc] = TileType.DOOR_R
            game.step(action)
            assert game.player_pos == (nr, nc), "Player couldn't unlock door with key"
            return


# ---- Enemy & Exit tests ----

def test_enemy_kills():
    """Walking onto an enemy should kill the player."""
    config = LevelConfig(grid_size=8, num_keys=0, num_enemies=0)
    game = PuzzleGame(config, seed=42)
    r, c = game.player_pos

    for action, (dr, dc) in [(Action.DOWN, (1, 0)), (Action.RIGHT, (0, 1))]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < game.config.grid_size and 0 <= nc < game.config.grid_size:
            game.grid[nr, nc] = TileType.ENEMY
            game.step(action)
            assert game.done, "Game should be done after enemy contact"
            assert not game.won, "Player should not win after enemy contact"
            assert game.deaths >= 1, "Death count should be >= 1"
            return


def test_exit_wins():
    """Walking onto the exit should win the game."""
    config = LevelConfig(grid_size=8, num_keys=0, num_enemies=0)
    game = PuzzleGame(config, seed=42)
    r, c = game.player_pos

    for action, (dr, dc) in [(Action.DOWN, (1, 0)), (Action.RIGHT, (0, 1))]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < game.config.grid_size and 0 <= nc < game.config.grid_size:
            game.grid[nr, nc] = TileType.EXIT
            game.exit_pos = (nr, nc)
            game.step(action)
            assert game.done, "Game should be done after reaching exit"
            assert game.won, "Player should win after reaching exit"
            return


def test_timeout():
    """Game should timeout after max_steps."""
    config = LevelConfig(grid_size=8, num_keys=0, num_enemies=0)
    game = PuzzleGame(config, seed=42)
    game.max_steps = 5  # Override for test

    for _ in range(10):
        if game.done:
            break
        game.step(Action.UP)  # Will hit wall and stay in place

    assert game.done, "Game should be done after exceeding max_steps"


# ---- Utility tests ----

def test_optimal_path_positive():
    """Optimal path length should be > 0."""
    config = LevelConfig(grid_size=12, num_keys=1, num_enemies=0)
    game = PuzzleGame(config, seed=42)
    opt = game.compute_optimal_path_length()
    assert opt > 0, f"Optimal path length should be positive, got {opt}"


def test_level_config_clamping():
    """LevelConfig should clamp values to valid ranges."""
    config = LevelConfig(grid_size=3, num_keys=-1, num_enemies=100, wall_density=2.0)
    assert config.grid_size >= 8, f"grid_size should be >= 8, got {config.grid_size}"
    assert config.num_keys >= 0, f"num_keys should be >= 0, got {config.num_keys}"
    assert config.num_enemies <= 5, f"num_enemies should be <= 5, got {config.num_enemies}"
    assert config.wall_density <= 0.6, f"wall_density should be <= 0.6, got {config.wall_density}"


def test_render_ascii(default_game):
    """render_ascii should return a string with P and E."""
    ascii_str = default_game.render_ascii()
    assert isinstance(ascii_str, str)
    assert 'P' in ascii_str, "ASCII render missing player 'P'"
    assert 'E' in ascii_str, "ASCII render missing exit 'E'"


def test_reset():
    """reset() should generate a new valid level."""
    config = LevelConfig(grid_size=10, num_keys=1, num_enemies=0)
    game = PuzzleGame(config, seed=42)
    state1 = game.get_state()
    
    game.reset()
    state2 = game.get_state()
    
    assert state2['grid'].shape == (10, 10)
    assert not state2['done']
    assert state2['step_count'] == 0


def test_deterministic_seed():
    """Same seed should produce identical grids."""
    config = LevelConfig(grid_size=12, num_keys=2, num_enemies=1)
    game1 = PuzzleGame(config, seed=42)
    game2 = PuzzleGame(config, seed=42)
    assert np.array_equal(game1.grid, game2.grid), "Same seed produced different grids"


@pytest.mark.parametrize("seed", range(100, 120))
def test_multiple_seeds_solvable(seed):
    """All generated levels should be solvable regardless of seed."""
    config = LevelConfig(grid_size=12, num_keys=2, num_enemies=1, num_hints=1)
    game = PuzzleGame(config, seed=seed)
    assert game.validate_solvability(), f"Level with seed={seed} is not solvable"


# ---- Hint direction tests ----

def test_hint_direction_returns_tuple():
    """get_hint_direction should return (direction, target_name)."""
    config = LevelConfig(grid_size=10, num_keys=1, num_enemies=0, num_hints=1)
    game = PuzzleGame(config, seed=42)
    direction, target = game.get_hint_direction()
    assert isinstance(direction, str)
    assert isinstance(target, str)
    assert direction in ("↑", "↓", "←", "→")


def test_hint_direction_points_to_key():
    """When keys exist, hint should point toward a key."""
    config = LevelConfig(grid_size=10, num_keys=1, num_enemies=0, num_hints=1)
    game = PuzzleGame(config, seed=42)
    direction, target = game.get_hint_direction()
    # Should mention "Key" if uncollected keys exist
    assert "Key" in target or "Exit" in target


def test_hint_direction_points_to_exit_after_keys():
    """After all keys collected, hint should point to Exit."""
    config = LevelConfig(grid_size=10, num_keys=1, num_enemies=0, num_hints=1)
    game = PuzzleGame(config, seed=42)
    # Simulate collecting all keys
    game.inventory = {'R', 'G', 'B'}
    direction, target = game.get_hint_direction()
    assert target == "Exit"


def test_hint_activates_on_collection():
    """Stepping on a hint tile should activate hint_direction."""
    config = LevelConfig(grid_size=10, num_keys=1, num_enemies=0, num_hints=2)
    game = PuzzleGame(config, seed=42)
    # Find a hint tile and place player next to it
    hint_pos = None
    for r in range(config.grid_size):
        for c in range(config.grid_size):
            if game.grid[r, c] == TileType.HINT:
                hint_pos = (r, c)
                break
        if hint_pos:
            break
    
    if hint_pos:
        # Place player adjacent to hint
        r, c = hint_pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < config.grid_size and 0 <= nc < config.grid_size:
                if game.grid[nr, nc] == TileType.EMPTY:
                    game.grid[game.player_pos] = TileType.EMPTY
                    game.player_pos = (nr, nc)
                    game.grid[nr, nc] = TileType.PLAYER
                    # Player is at (nr, nc) = (r+dr, c+dc)
                    # To move toward hint at (r, c), step in direction (-dr, -dc)
                    if -dr == -1: action = Action.UP
                    elif -dr == 1: action = Action.DOWN
                    elif -dc == -1: action = Action.LEFT
                    else: action = Action.RIGHT
                    state = game.step(action)
                    assert state['hint_direction'] is not None
                    assert game.hints_remaining > 0
                    break


# ---- Danger zone tests ----

def test_danger_zones_empty_no_enemies():
    """No enemies should produce empty danger zones."""
    config = LevelConfig(grid_size=10, num_keys=0, num_enemies=0, num_hints=0)
    game = PuzzleGame(config, seed=42)
    assert game.get_danger_zones() == set()


def test_danger_zones_with_enemies():
    """Enemies should produce danger zones on adjacent cells."""
    config = LevelConfig(grid_size=12, num_keys=0, num_enemies=2, num_hints=0)
    game = PuzzleGame(config, seed=42)
    danger = game.get_danger_zones()
    # Danger zones should include enemy positions
    for r in range(config.grid_size):
        for c in range(config.grid_size):
            if game.grid[r, c] == TileType.ENEMY:
                assert (r, c) in danger, f"Enemy at ({r},{c}) not in danger zones"
    # Should also include adjacent non-wall cells
    assert len(danger) > 0


# ---- Feedback message tests ----

def test_feedback_on_key_pickup():
    """Picking up a key should produce feedback."""
    config = LevelConfig(grid_size=10, num_keys=1, num_enemies=0, num_hints=0)
    game = PuzzleGame(config, seed=42)
    # Find a key tile and place player adjacent
    key_pos = None
    for r in range(config.grid_size):
        for c in range(config.grid_size):
            if game.grid[r, c] in (TileType.KEY_R, TileType.KEY_G, TileType.KEY_B):
                key_pos = (r, c)
                break
        if key_pos:
            break
    
    if key_pos:
        r, c = key_pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < config.grid_size and 0 <= nc < config.grid_size:
                if game.grid[nr, nc] == TileType.EMPTY:
                    game.grid[game.player_pos] = TileType.EMPTY
                    game.player_pos = (nr, nc)
                    game.grid[nr, nc] = TileType.PLAYER
                    # Player is at (nr, nc) = (r+dr, c+dc)
                    # To move toward key at (r, c), step in direction (-dr, -dc)
                    if -dr == -1: action = Action.UP
                    elif -dr == 1: action = Action.DOWN
                    elif -dc == -1: action = Action.LEFT
                    else: action = Action.RIGHT
                    state = game.step(action)
                    assert state['feedback_type'] == 'key'
                    assert 'collected' in state['last_feedback']
                    break


def test_state_has_new_fields():
    """get_state should include hint_direction, last_feedback, and feedback_type."""
    config = LevelConfig(grid_size=10, num_keys=0, num_enemies=0, num_hints=0)
    game = PuzzleGame(config, seed=42)
    state = game.get_state()
    assert 'hint_direction' in state
    assert 'hint_target' in state
    assert 'last_feedback' in state
    assert 'feedback_type' in state

