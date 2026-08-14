"""
Interactive Pygame GUI for Human Players.

Play the 2D key-door puzzle game using keyboard arrow keys!
After each level, an AI Director (PPO or Heuristic) dynamically adjusts
the difficulty (grid size, keys, doors, enemies, hints) to match your skill level.

Features:
  - Start menu with difficulty presets
  - Level transition screens (win/lose/session complete)
  - Functional hints with direction arrows
  - Danger zone highlighting around enemies
  - Real-time feedback messages
  - Colored inventory display
  - DDA Director change log
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

# ─── Color Palette ───────────────────────────────────────────────────────────
BG = (18, 20, 28)
PANEL_BG = (28, 32, 44)
GRID_LINE = (40, 44, 58)
EMPTY = (12, 14, 20)
WALL = (48, 52, 68)
PLAYER = (46, 204, 113)      # Emerald
EXIT_GOLD = (241, 196, 15)   # Gold
KEY_R = (231, 76, 60)        # Red
KEY_G = (46, 204, 113)       # Green
KEY_B = (52, 152, 219)       # Blue
DOOR_R = (140, 35, 35)
DOOR_G = (35, 120, 35)
DOOR_B = (35, 75, 140)
ENEMY = (235, 45, 85)        # Coral
HINT_COLOR = (155, 89, 182)  # Amethyst
TEXT = (230, 234, 240)
TEXT_DIM = (140, 148, 165)
DANGER_TINT = (80, 20, 20)   # Subtle red tint for danger zones
OVERLAY_WIN = (46, 204, 113, 200)
OVERLAY_LOSE = (231, 76, 60, 200)
OVERLAY_SESSION = (52, 152, 219, 200)
ACCENT = (100, 220, 180)

# Feedback message colors
FB_COLORS = {
    'key': (46, 204, 113),
    'door_unlocked': (46, 204, 113),
    'door_locked': (231, 76, 60),
    'hint': (155, 89, 182),
    'enemy': (235, 45, 85),
    'exit': (241, 196, 15),
    'timeout': (231, 76, 60),
}

KEY_COLOR_MAP = {
    'R': KEY_R,
    'G': KEY_G,
    'B': KEY_B,
}

DDA_ACTION_NAMES = {
    DDAAction.KEEP_SAME: "No change",
    DDAAction.INCREASE_SIZE: "Grid size +2",
    DDAAction.DECREASE_SIZE: "Grid size -2",
    DDAAction.INCREASE_KEYS: "Keys +1",
    DDAAction.DECREASE_KEYS: "Keys -1",
    DDAAction.INCREASE_ENEMIES: "Enemies +1",
    DDAAction.DECREASE_ENEMIES: "Enemies -1",
    DDAAction.ADD_HINT: "Hints +1",
    DDAAction.REMOVE_HINT: "Hints -1",
}


class HumanGameWindow:
    """Full-featured interactive Pygame window with transitions, feedback, and DDA."""

    # ── Game states ──
    STATE_MENU = 'menu'
    STATE_PLAYING = 'playing'
    STATE_LEVEL_WIN = 'level_win'
    STATE_LEVEL_LOSE = 'level_lose'
    STATE_SESSION_DONE = 'session_done'

    def __init__(self, mode='heuristic', ppo_model_path=None, cell_size=42):
        pygame.init()
        pygame.display.set_caption("2D Key-Door Puzzle — DDA Interactive Mode")

        self.cell_size = cell_size
        self.mode = mode
        self.state = self.STATE_MENU

        # Fonts
        self.font = pygame.font.SysFont("Consolas", 15, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 28, bold=True)
        self.font_med = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 13)

        # Menu state
        self.menu_selection = 1  # 0=Easy, 1=Medium, 2=Hard
        self.ppo_model_path = ppo_model_path

        # Session tracking
        self.levels_played = 0
        self.levels_won_count = 0
        self.total_steps = 0
        self.last_dda_action = None
        self.feedback_text = None
        self.feedback_color = TEXT
        self.feedback_timer = 0

        # Environment and director (initialized on game start)
        self.env = None
        self.director = None
        self.obs = None
        self.info = None
        self.game = None

    def _init_game(self):
        """Initialize game environment based on menu selection."""
        presets = {
            0: {'session_length': 20, 't_min': 15, 't_max': 80, 'timeout_steps': 200,
                'initial_grid_size': 8, 'initial_num_keys': 0, 'initial_num_enemies': 0, 'initial_num_hints': 1},
            1: {'session_length': 20, 't_min': 30, 't_max': 120, 'timeout_steps': 300,
                'initial_grid_size': 10, 'initial_num_keys': 1, 'initial_num_enemies': 1, 'initial_num_hints': 1},
            2: {'session_length': 20, 't_min': 50, 't_max': 200, 'timeout_steps': 500,
                'initial_grid_size': 14, 'initial_num_keys': 2, 'initial_num_enemies': 2, 'initial_num_hints': 0},
        }
        config = presets[self.menu_selection]

        self.env = DDAEnv(config=config)
        if self.mode == 'ppo' and self.ppo_model_path:
            print(f"Loading PPO Director from: {self.ppo_model_path}")
            self.director = RLAgentDDA(self.ppo_model_path)
        else:
            print("Using Heuristic (Rule-Based) Director")
            self.director = HeuristicDDA(self.env.config)

        self.obs, self.info = self.env.reset(seed=42)
        self.game = self.env.game
        self.levels_played = 0
        self.levels_won_count = 0
        self.total_steps = 0
        self.last_dda_action = None
        self.state = self.STATE_PLAYING

    def _advance_level(self):
        """DDA Director chooses difficulty, advance to the next level."""
        solved = self.game.won
        solve_time = self.game.step_count
        deaths = self.game.deaths

        self.levels_played += 1
        self.total_steps += solve_time
        if solved:
            self.levels_won_count += 1

        if isinstance(self.director, HeuristicDDA):
            self.director.observe_result(solved, solve_time, deaths)

        director_action = self.director.choose_action(self.obs)
        self.last_dda_action = director_action

        self.obs, reward, terminated, truncated, self.info = self.env.step(director_action)
        self.game = self.env.game

        if terminated:
            self.state = self.STATE_SESSION_DONE
        else:
            self.state = self.STATE_PLAYING

    # ─── Rendering ──────────────────────────────────────────────────────

    def _draw_grid(self, screen):
        """Draws the game grid with danger zones and hint arrows."""
        grid = self.game.grid
        gs = self.game.config.grid_size
        danger_zones = self.game.get_danger_zones() if self.game.config.num_enemies > 0 else set()

        for r in range(gs):
            for c in range(gs):
                tile = grid[r, c]
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size,
                                   self.cell_size, self.cell_size)

                # Base color
                color = EMPTY
                if tile == TileType.WALL: color = WALL
                elif tile == TileType.PLAYER: color = PLAYER
                elif tile == TileType.EXIT: color = EXIT_GOLD
                elif tile == TileType.KEY_R: color = KEY_R
                elif tile == TileType.KEY_G: color = KEY_G
                elif tile == TileType.KEY_B: color = KEY_B
                elif tile == TileType.DOOR_R: color = DOOR_R
                elif tile == TileType.DOOR_G: color = DOOR_G
                elif tile == TileType.DOOR_B: color = DOOR_B
                elif tile == TileType.ENEMY: color = ENEMY
                elif tile == TileType.HINT: color = HINT_COLOR

                # Danger zone tint (red underlay for cells near enemies)
                if (r, c) in danger_zones and tile not in (TileType.WALL, TileType.ENEMY, TileType.PLAYER):
                    color = (
                        min(255, color[0] + 40),
                        max(0, color[1] - 10),
                        max(0, color[2] - 10),
                    )

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, GRID_LINE, rect, 1)

                # Tile labels
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
                    tc = (0, 0, 0) if tile in (TileType.PLAYER, TileType.EXIT,
                                                TileType.KEY_R, TileType.KEY_G, TileType.KEY_B) else TEXT
                    txt = self.font.render(label, True, tc)
                    screen.blit(txt, txt.get_rect(center=rect.center))

                # Danger zone marker
                if (r, c) in danger_zones and tile == TileType.EMPTY:
                    warn = self.font_small.render("!", True, (200, 60, 60))
                    screen.blit(warn, warn.get_rect(center=rect.center))

    def _draw_panel(self, screen, panel_x, height):
        """Draws the side info panel."""
        panel_rect = pygame.Rect(panel_x, 0, 260, height)
        pygame.draw.rect(screen, PANEL_BG, panel_rect)

        y = 15

        # Title
        title = self.font_med.render(f"Director: {self.mode.upper()}", True, ACCENT)
        screen.blit(title, (panel_x + 15, y)); y += 32

        # Divider
        pygame.draw.line(screen, GRID_LINE, (panel_x + 10, y), (panel_x + 250, y)); y += 10

        # Level info
        level_txt = f"Level {self.env.levels_attempted + 1} / {self.env.session_length}"
        screen.blit(self.font.render(level_txt, True, TEXT), (panel_x + 15, y)); y += 24

        # Current difficulty
        screen.blit(self.font.render("── Difficulty ──", True, TEXT_DIM), (panel_x + 15, y)); y += 22
        params = [
            f"Grid:    {self.game.config.grid_size}x{self.game.config.grid_size}",
            f"Keys:    {self.game.config.num_keys}",
            f"Enemies: {self.game.config.num_enemies}",
            f"Hints:   {self.game.config.num_hints}",
        ]
        for p in params:
            screen.blit(self.font.render(p, True, TEXT), (panel_x + 20, y)); y += 20

        y += 5
        pygame.draw.line(screen, GRID_LINE, (panel_x + 10, y), (panel_x + 250, y)); y += 10

        # Steps
        steps_txt = f"Steps: {self.game.step_count}/{self.game.max_steps}"
        screen.blit(self.font.render(steps_txt, True, TEXT), (panel_x + 15, y)); y += 24

        # Inventory with colored key circles
        screen.blit(self.font.render("Inventory:", True, TEXT_DIM), (panel_x + 15, y)); y += 22
        inv = sorted(list(self.game.inventory))
        if inv:
            kx = panel_x + 20
            for key_color in inv:
                c = KEY_COLOR_MAP.get(key_color, TEXT)
                pygame.draw.circle(screen, c, (kx + 8, y + 8), 8)
                kt = self.font_small.render(key_color, True, (0, 0, 0))
                screen.blit(kt, kt.get_rect(center=(kx + 8, y + 8)))
                kx += 28
            y += 24
        else:
            screen.blit(self.font_small.render("  (empty)", True, TEXT_DIM), (panel_x + 15, y)); y += 20

        # Hint direction indicator
        if self.game.hint_direction:
            y += 5
            pygame.draw.line(screen, GRID_LINE, (panel_x + 10, y), (panel_x + 250, y)); y += 10
            screen.blit(self.font.render("── Active Hint ──", True, HINT_COLOR), (panel_x + 15, y)); y += 22
            arrow_txt = f"  {self.game.hint_direction}  {self.game.hint_target}"
            screen.blit(self.font_med.render(arrow_txt, True, HINT_COLOR), (panel_x + 20, y)); y += 26
            remain_txt = f"  ({self.game.hints_remaining} steps left)"
            screen.blit(self.font_small.render(remain_txt, True, TEXT_DIM), (panel_x + 20, y)); y += 20

        # Last DDA action
        if self.last_dda_action is not None:
            y += 5
            pygame.draw.line(screen, GRID_LINE, (panel_x + 10, y), (panel_x + 250, y)); y += 10
            screen.blit(self.font.render("── Last DDA ──", True, TEXT_DIM), (panel_x + 15, y)); y += 22
            act_name = DDA_ACTION_NAMES.get(DDAAction(self.last_dda_action), "?")
            screen.blit(self.font.render(f"  {act_name}", True, ACCENT), (panel_x + 15, y)); y += 24

        # Feedback message
        if self.feedback_timer > 0:
            y += 5
            pygame.draw.line(screen, GRID_LINE, (panel_x + 10, y), (panel_x + 250, y)); y += 10
            fb = self.font.render(self.feedback_text or "", True, self.feedback_color)
            screen.blit(fb, (panel_x + 15, y)); y += 24

        # Controls at bottom
        ctrl_y = height - 80
        pygame.draw.line(screen, GRID_LINE, (panel_x + 10, ctrl_y), (panel_x + 250, ctrl_y)); ctrl_y += 8
        controls = [
            "ARROWS: Move",
            "R: Retry level",
            "Q: Quit",
        ]
        for ct in controls:
            screen.blit(self.font_small.render(ct, True, TEXT_DIM), (panel_x + 15, ctrl_y))
            ctrl_y += 16

    def _draw_overlay(self, screen, width, height, title, subtitle, color_rgb, details=None):
        """Draws a semi-transparent overlay screen (win/lose/session complete)."""
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((*color_rgb, 180))
        screen.blit(overlay, (0, 0))

        # Title
        title_surf = self.font_big.render(title, True, (255, 255, 255))
        screen.blit(title_surf, title_surf.get_rect(center=(width // 2, height // 2 - 60)))

        # Subtitle
        sub_surf = self.font_med.render(subtitle, True, (240, 240, 240))
        screen.blit(sub_surf, sub_surf.get_rect(center=(width // 2, height // 2 - 20)))

        # Detail lines
        if details:
            dy = height // 2 + 20
            for line in details:
                d_surf = self.font.render(line, True, (220, 225, 230))
                screen.blit(d_surf, d_surf.get_rect(center=(width // 2, dy)))
                dy += 24

        # Prompt
        prompt = "Press SPACE to continue"
        if self.state == self.STATE_SESSION_DONE:
            prompt = "Press SPACE for new session  |  Q to quit"
        p_surf = self.font.render(prompt, True, (200, 210, 220))
        screen.blit(p_surf, p_surf.get_rect(center=(width // 2, height - 40)))

    def _draw_menu(self, screen, width, height):
        """Draws the start menu."""
        screen.fill(BG)

        # Title
        title = self.font_big.render("2D Key-Door Puzzle", True, ACCENT)
        screen.blit(title, title.get_rect(center=(width // 2, 60)))

        sub = self.font_med.render("with Dynamic Difficulty Adjustment", True, TEXT_DIM)
        screen.blit(sub, sub.get_rect(center=(width // 2, 95)))

        # Difficulty presets
        presets = [
            ("Easy", "8x8 grid, 0 keys, 0 enemies", (46, 204, 113)),
            ("Medium", "10x10 grid, 1 key, 1 enemy", (241, 196, 15)),
            ("Hard", "14x14 grid, 2 keys, 2 enemies", (231, 76, 60)),
        ]

        box_w, box_h = 300, 60
        start_y = 150
        for i, (name, desc, color) in enumerate(presets):
            bx = (width - box_w) // 2
            by = start_y + i * (box_h + 15)
            rect = pygame.Rect(bx, by, box_w, box_h)

            if i == self.menu_selection:
                pygame.draw.rect(screen, color, rect, border_radius=8)
                pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=8)
                name_surf = self.font_med.render(f"> {name}", True, (0, 0, 0))
                desc_surf = self.font_small.render(desc, True, (30, 30, 30))
            else:
                pygame.draw.rect(screen, PANEL_BG, rect, border_radius=8)
                pygame.draw.rect(screen, GRID_LINE, rect, 1, border_radius=8)
                name_surf = self.font_med.render(f"  {name}", True, TEXT)
                desc_surf = self.font_small.render(desc, True, TEXT_DIM)

            screen.blit(name_surf, (bx + 15, by + 10))
            screen.blit(desc_surf, (bx + 15, by + 35))

        # Mode info
        mode_text = f"Director Mode: {self.mode.upper()}"
        if self.mode == 'ppo' and self.ppo_model_path:
            mode_text += f" ({os.path.basename(self.ppo_model_path)})"
        mode_surf = self.font.render(mode_text, True, TEXT_DIM)
        screen.blit(mode_surf, mode_surf.get_rect(center=(width // 2, start_y + 3 * (box_h + 15) + 20)))

        # Instructions
        instr = [
            "UP / DOWN: Select difficulty",
            "ENTER: Start game",
            "Q: Quit",
        ]
        iy = height - 70
        for ins in instr:
            s = self.font_small.render(ins, True, TEXT_DIM)
            screen.blit(s, s.get_rect(center=(width // 2, iy)))
            iy += 18

    # ─── Main Game Loop ─────────────────────────────────────────────────

    def run(self):
        clock = pygame.time.Clock()

        # Initial window at menu size
        menu_w, menu_h = 500, 450
        screen = pygame.display.set_mode((menu_w, menu_h))

        running = True
        while running:
            clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    # ── Menu state ──
                    if self.state == self.STATE_MENU:
                        if event.key == pygame.K_UP:
                            self.menu_selection = max(0, self.menu_selection - 1)
                        elif event.key == pygame.K_DOWN:
                            self.menu_selection = min(2, self.menu_selection + 1)
                        elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                            self._init_game()
                            # Resize window to fit grid
                            gpx = self.game.config.grid_size * self.cell_size
                            screen = pygame.display.set_mode((gpx + 260, max(gpx, 480)))
                        elif event.key == pygame.K_q:
                            running = False

                    # ── Playing state ──
                    elif self.state == self.STATE_PLAYING:
                        action_taken = None
                        if event.key == pygame.K_UP: action_taken = Action.UP
                        elif event.key == pygame.K_DOWN: action_taken = Action.DOWN
                        elif event.key == pygame.K_LEFT: action_taken = Action.LEFT
                        elif event.key == pygame.K_RIGHT: action_taken = Action.RIGHT
                        elif event.key == pygame.K_q: running = False
                        elif event.key == pygame.K_r:
                            # Retry current level (restart without advancing DDA)
                            self.game.reset()
                            self.feedback_text = "Level restarted"
                            self.feedback_color = TEXT_DIM
                            self.feedback_timer = 60

                        if action_taken is not None and not self.game.done:
                            self.game.step(action_taken)

                            # Show feedback from game engine
                            if self.game.last_feedback:
                                self.feedback_text = self.game.last_feedback
                                self.feedback_color = FB_COLORS.get(self.game.feedback_type, TEXT)
                                self.feedback_timer = 90

                            # Level ended?
                            if self.game.done:
                                if self.game.won:
                                    self.state = self.STATE_LEVEL_WIN
                                else:
                                    self.state = self.STATE_LEVEL_LOSE

                    # ── Win / Lose overlay ──
                    elif self.state in (self.STATE_LEVEL_WIN, self.STATE_LEVEL_LOSE):
                        if event.key == pygame.K_SPACE:
                            self._advance_level()
                            # Resize window for new grid size
                            gpx = self.game.config.grid_size * self.cell_size
                            screen = pygame.display.set_mode((gpx + 260, max(gpx, 480)))
                        elif event.key == pygame.K_q:
                            running = False

                    # ── Session complete ──
                    elif self.state == self.STATE_SESSION_DONE:
                        if event.key == pygame.K_SPACE:
                            self.state = self.STATE_MENU
                            screen = pygame.display.set_mode((menu_w, menu_h))
                        elif event.key == pygame.K_q:
                            running = False

            # ── Render ──
            if self.state == self.STATE_MENU:
                self._draw_menu(screen, menu_w, menu_h)

            elif self.state == self.STATE_PLAYING:
                gpx = self.game.config.grid_size * self.cell_size
                w = gpx + 260
                h = max(gpx, 480)
                screen.fill(BG)
                self._draw_grid(screen)
                self._draw_panel(screen, gpx, h)

                # Decrement feedback timer
                if self.feedback_timer > 0:
                    self.feedback_timer -= 1

            elif self.state == self.STATE_LEVEL_WIN:
                gpx = self.game.config.grid_size * self.cell_size
                w = gpx + 260
                h = max(gpx, 480)
                screen.fill(BG)
                self._draw_grid(screen)
                self._draw_panel(screen, gpx, h)

                details = [
                    f"Steps taken: {self.game.step_count}",
                    f"Keys collected: {len(self.game.inventory)}/{self.game.keys_total}",
                    f"Hints used: {self.game.hints_collected}",
                ]
                self._draw_overlay(screen, w, h, "Level Complete!", 
                                   f"Solved in {self.game.step_count} steps",
                                   (30, 140, 70), details)

            elif self.state == self.STATE_LEVEL_LOSE:
                gpx = self.game.config.grid_size * self.cell_size
                w = gpx + 260
                h = max(gpx, 480)
                screen.fill(BG)
                self._draw_grid(screen)
                self._draw_panel(screen, gpx, h)

                reason = "Killed by enemy!" if self.game.deaths > 0 else "Time's up!"
                details = [
                    f"Steps taken: {self.game.step_count}/{self.game.max_steps}",
                    f"Keys collected: {len(self.game.inventory)}/{self.game.keys_total}",
                ]
                self._draw_overlay(screen, w, h, "Level Failed",
                                   reason, (160, 30, 40), details)

            elif self.state == self.STATE_SESSION_DONE:
                gpx = self.game.config.grid_size * self.cell_size
                w = gpx + 260
                h = max(gpx, 480)
                screen.fill(BG)

                win_rate = (self.levels_won_count / max(self.levels_played, 1)) * 100
                avg_steps = self.total_steps / max(self.levels_played, 1)
                details = [
                    f"Levels Played: {self.levels_played}",
                    f"Levels Won: {self.levels_won_count}",
                    f"Win Rate: {win_rate:.0f}%",
                    f"Avg Steps: {avg_steps:.0f}",
                ]
                self._draw_overlay(screen, w, h, "Session Complete!",
                                   f"Director: {self.mode.upper()}", (30, 100, 160), details)

            pygame.display.flip()

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
