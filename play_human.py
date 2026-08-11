"""
Interactive Pygame GUI for Human Players.

Play the 2D key-door puzzle game using keyboard arrow keys!
After each level, an AI Director (PPO or Heuristic) dynamically adjusts
the difficulty (grid size, keys, doors, enemies, hints) to match your skill level.
"""

import sys
import os
import argparse
import pygame
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.puzzle_game import PuzzleGame, LevelConfig, TileType, Action
from envs.dda_env import DDAAction, DDAEnv
from agents.eval_agent import HeuristicDDA, RLAgentDDA

# Colors (RGB)
COLOR_BACKGROUND = (30, 30, 40)
COLOR_GRID_LINE = (50, 50, 65)
COLOR_WALL = (60, 64, 80)
COLOR_EMPTY = (20, 22, 30)
COLOR_PLAYER = (50, 205, 50)     # Lime Green
COLOR_EXIT = (255, 215, 0)       # Gold
COLOR_KEY_R = (255, 75, 75)      # Red Key
COLOR_KEY_G = (75, 255, 75)      # Green Key
COLOR_KEY_B = (75, 150, 255)     # Blue Key
COLOR_DOOR_R = (180, 40, 40)     # Red Door
COLOR_DOOR_G = (40, 180, 40)     # Green Door
COLOR_DOOR_B = (40, 90, 180)     # Blue Door
COLOR_ENEMY = (235, 50, 90)      # Crimson Enemy
COLOR_HINT = (180, 100, 255)     # Purple Hint
COLOR_TEXT = (240, 240, 240)
COLOR_PANEL = (40, 44, 56)


class HumanGameWindow:
    def __init__(self, mode='heuristic', ppo_model_path=None, cell_size=40):
        pygame.init()
        pygame.display.set_caption("2D Key-Door Puzzle — DDA Interactive Mode")

        self.cell_size = cell_size
        self.mode = mode

        # Environment setup
        self.env = DDAEnv(config={
            'session_length': 20,
            't_min': 30,
            't_max': 120,
            'timeout_steps': 300
        })

        # Director setup
        if mode == 'ppo' and ppo_model_path:
            print(f"Loading PPO Director from: {ppo_model_path}")
            self.director = RLAgentDDA(ppo_model_path)
        else:
            print("Using Heuristic (Rule-Based) Director")
            self.director = HeuristicDDA(self.env.config)

        self.obs, self.info = self.env.reset(seed=42)
        self.game = self.env.game
        
        # Override bot step with human interaction
        self.level_step_count = 0
        self.level_start_ticks = pygame.time.get_ticks()
        self.current_action = None

    def render(self, screen, font):
        grid = self.game.grid
        grid_size = self.game.config.grid_size

        grid_px = grid_size * self.cell_size
        panel_width = 250
        width = grid_px + panel_width
        height = max(grid_px, 500)

        screen.fill(COLOR_BACKGROUND)

        # 1. Draw Grid Tiles
        for r in range(grid_size):
            for c in range(grid_size):
                tile = grid[r, c]
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)

                color = COLOR_EMPTY
                if tile == TileType.WALL:
                    color = COLOR_WALL
                elif tile == TileType.PLAYER:
                    color = COLOR_PLAYER
                elif tile == TileType.EXIT:
                    color = COLOR_EXIT
                elif tile == TileType.KEY_R:
                    color = COLOR_KEY_R
                elif tile == TileType.KEY_G:
                    color = COLOR_KEY_G
                elif tile == TileType.KEY_B:
                    color = COLOR_KEY_B
                elif tile == TileType.DOOR_R:
                    color = COLOR_DOOR_R
                elif tile == TileType.DOOR_G:
                    color = COLOR_DOOR_G
                elif tile == TileType.DOOR_B:
                    color = COLOR_DOOR_B
                elif tile == TileType.ENEMY:
                    color = COLOR_ENEMY
                elif tile == TileType.HINT:
                    color = COLOR_HINT

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, COLOR_GRID_LINE, rect, 1)

                # Draw Tile Text Label for Keys/Doors/Exit
                label = ""
                if tile == TileType.PLAYER: label = "P"
                elif tile == TileType.EXIT: label = "E"
                elif tile == TileType.KEY_R: label = "kR"
                elif tile == TileType.KEY_G: label = "kG"
                elif tile == TileType.KEY_B: label = "kB"
                elif tile == TileType.DOOR_R: label = "DR"
                elif tile == TileType.DOOR_G: label = "DG"
                elif tile == TileType.DOOR_B: label = "DB"
                elif tile == TileType.ENEMY: label = "X"
                elif tile == TileType.HINT: label = "?"

                if label:
                    txt_surface = font.render(label, True, (0, 0, 0) if tile in [TileType.PLAYER, TileType.EXIT, TileType.KEY_R, TileType.KEY_G, TileType.KEY_B] else COLOR_TEXT)
                    txt_rect = txt_surface.get_rect(center=rect.center)
                    screen.blit(txt_surface, txt_rect)

        # 2. Draw Side Panel
        panel_rect = pygame.Rect(grid_px, 0, panel_width, height)
        pygame.draw.rect(screen, COLOR_PANEL, panel_rect)

        # Panel Dashboard Info
        lines = [
            f"Mode: {self.mode.upper()}",
            f"Level: {self.env.levels_attempted + 1}/{self.env.session_length}",
            "-------------------",
            f"Grid Size: {self.game.config.grid_size}x{self.game.config.grid_size}",
            f"Keys: {self.game.config.num_keys}",
            f"Enemies: {self.game.config.num_enemies}",
            f"Hints: {self.game.config.num_hints}",
            "-------------------",
            f"Steps: {self.game.step_count}/{self.game.max_steps}",
            f"Inventory: {sorted(list(self.game.inventory))}",
            "-------------------",
            "Controls:",
            "ARROW KEYS: Move",
            "R: Reset Level",
            "Q: Quit"
        ]

        y_off = 20
        for line in lines:
            txt_surface = font.render(line, True, COLOR_TEXT)
            screen.blit(txt_surface, (grid_px + 15, y_off))
            y_off += 28

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()

        # Dynamic Window Sizing
        grid_px = self.game.config.grid_size * self.cell_size
        screen = pygame.display.set_mode((grid_px + 250, max(grid_px, 500)))
        font = pygame.font.SysFont("Consolas", 16, bold=True)

        running = True
        while running:
            clock.tick(30)
            action_taken = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action_taken = Action.UP
                    elif event.key == pygame.K_DOWN:
                        action_taken = Action.DOWN
                    elif event.key == pygame.K_LEFT:
                        action_taken = Action.LEFT
                    elif event.key == pygame.K_RIGHT:
                        action_taken = Action.RIGHT
                    elif event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        self.game.reset()

            # Handle Human Step
            if action_taken is not None and not self.game.done:
                state = self.game.step(action_taken)
                
                # Check level completion
                if self.game.done:
                    solved = self.game.won
                    solve_time = self.game.step_count
                    
                    # Update heuristic or observe
                    if isinstance(self.director, HeuristicDDA):
                        self.director.observe_result(solved, solve_time)

                    # Get DDA Director decision
                    director_action = self.director.choose_action(self.obs)

                    # Advance DDA Env
                    self.obs, reward, terminated, truncated, self.info = self.env.step(director_action)
                    self.game = self.env.game

                    if terminated:
                        print("Session Complete! Restarting new session...")
                        self.obs, self.info = self.env.reset()
                        self.game = self.env.game

                    # Adjust window size if grid size changed
                    grid_px = self.game.config.grid_size * self.cell_size
                    screen = pygame.display.set_mode((grid_px + 250, max(grid_px, 500)))

            self.render(screen, font)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Human Pygame Interactive Mode")
    parser.add_argument('--mode', choices=['heuristic', 'ppo'], default='heuristic',
                        help='Director mode: heuristic or ppo')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained PPO model zip')
    args = parser.parse_args()

    game_win = HumanGameWindow(mode=args.mode, ppo_model_path=args.model_path)
    game_win.run()


if __name__ == '__main__':
    main()
