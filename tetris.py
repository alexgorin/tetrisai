import functools
import time
from typing import Tuple, Callable, Dict

import numpy as np
import pygame
import sys

from world import World, Config
from agent import IAgent, TetrisAction


class Color:
    BACKGROUND = (35,  35,  35)
    FIGURE = (100, 200, 115)
    BOARD = (50,  120, 52)


TIMER_EVENT = pygame.USEREVENT + 1


class Game:
    def __init__(self, config: Config):
        cell_size, rows, cols = config.CELL_SIZE, config.ROWS, config.COLS
        self.config = config
        self.width = cell_size * (cols + 6)
        self.height = cell_size * rows
        self.rlim = cell_size * cols
        self.bground_grid = np.asarray([
            [1 if x % 2 == y % 2 else 0 for x in range(cols)]
            for y in range(rows)
        ])
        self.is_game_over = False
        self.world = None
        self.key_actions = None
        self.game_over = False
        self.paused = False

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.default_font = pygame.font.Font(pygame.font.get_default_font(), 12)
        pygame.key.set_repeat(250, 25)
        pygame.event.set_blocked(pygame.MOUSEMOTION)

        self.init_game()

    def init_game(self):
        self.world = World.from_config(self.config)
        self.key_actions = self.create_key_actions()
        pygame.time.set_timer(TIMER_EVENT, 200)

    def draw_map_fragment(
            self, matrix: np.ndarray, offset_x: int, offset_y: int, cell_size: int, color: Tuple[int, int, int]
    ):
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        color,
                        pygame.Rect(
                            (offset_x + x) * cell_size,
                            (offset_y + y) * cell_size,
                            cell_size,
                            cell_size),
                        0)

    def toggle_pause(self):
        self.paused = not self.paused

    def start_game(self):
        if self.game_over:
            self.init_game()
            self.game_over = False

    def quit(self):
        pygame.display.update()
        sys.exit()

    def render_game_state(self):
        pygame.draw.line(self.screen,
                         (255, 255, 255),
                         (self.rlim + 1, 0),
                         (self.rlim + 1, self.height - 1))
        cell_size, cols = self.config.CELL_SIZE, self.config.COLS
        self.draw_map_fragment(self.bground_grid, 0, 0, cell_size, Color.BACKGROUND)
        self.draw_map_fragment(self.world.board.map_fragment, 0, 0, cell_size, Color.BOARD)
        self.draw_map_fragment(
            self.world.figure.map_fragment, self.world.figure_x, self.world.figure_y, cell_size, Color.FIGURE)
        self.draw_map_fragment(
            self.world.next_figure.map_fragment, cols + 1, 2, cell_size, Color.FIGURE)
        pygame.display.update()

    def if_not_paused(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.paused:
                return func(*args, **kwargs)

        return wrapper

    def create_key_actions(self) -> Dict[str, Callable]:
        general_actions = {
            'ESCAPE': self.quit,
            'p': self.toggle_pause,
            'SPACE': self.start_game,
        }

        game_actions = {
            event: self.if_not_paused(action)
            for event, action in {
                'LEFT': self.world.move_left,
                'RIGHT': self.world.move_right,
                'DOWN': self.world.move_down,
                'UP': self.world.rotate_figure,
                'RETURN': self.world.move_all_way_down
            }.items()
        }
        return {**general_actions, **game_actions}

    def run(self):
        self.game_over = False
        self.paused = False

        clock = pygame.time.Clock()
        while True:
            self.screen.fill((0, 0, 0))
            if self.game_over:
                break
            if self.paused:
                time.sleep(0.1)
                continue
            self.render_game_state()
            clock.tick(self.config.MAXFPS)

            for event in pygame.event.get():
                if event.type == TIMER_EVENT:
                    self.world.step()
                    if self.world.is_in_terminal_state():
                        self.game_over = True
                        break
                elif event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:
                    for key in self.key_actions:
                        if event.key == getattr(pygame, f"K_{key}"):
                            self.key_actions[key]()

    def run_agent(self, agent: IAgent):
        self.game_over = False
        self.paused = False

        clock = pygame.time.Clock()
        while True:
            if self.game_over:
                break

            self.screen.fill((0, 0, 0))
            self.render_game_state()

            clock.tick(self.config.MAXFPS)
            for event in pygame.event.get():
                if event.type == TIMER_EVENT:
                    self.world.step()
                    if self.world.is_in_terminal_state():
                        self.game_over = True
                        break
                    recent_action = None
                    while True:
                        action = agent.choose_action(self.world)
                        if recent_action is None or action == recent_action:
                            # agent.act(self.world, action)
                            action.apply(self.world)
                            recent_action = action
                        else:
                            self.screen.fill((0, 0, 0))
                            self.render_game_state()
                            time.sleep(0.2)
                            action.apply(self.world)
                            # agent.act(self.world, action)
                            break
                    pygame.event.clear()  # don't queue multiple events during debugging
                    time.sleep(0.2)
                elif event.type == pygame.QUIT:
                    self.quit()

#
#
# if __name__ == '__main__':
#     game = Game(Config())
#     # game.run()
#     game.run_agent(Agent())
