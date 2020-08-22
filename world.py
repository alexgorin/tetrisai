"""
Tetris world with all rules
"""
from dataclasses import dataclass
from typing import List

import numpy as np


class MapFragmentMixin:
    def __init__(self, map_fragment):
        self.map_fragment = map_fragment
        self._height = None
        self._width = None
        self.set_height_and_width()

    def set_height_and_width(self):
        self._height = self.map_fragment.shape[0]
        self._width = self.map_fragment.shape[1]

    def height(self):
        return self._height

    def width(self):
        return self._width


@dataclass
class Figure(MapFragmentMixin):
    def __init__(self, map_fragment: np.ndarray):
        super().__init__(map_fragment)

    def rotate_clockwise(self) -> None:
        self.map_fragment = np.rot90(self.map_fragment)
        self.set_height_and_width()

    def copy(self):
        return Figure(self.map_fragment.copy())

    def possible_orientations(self):
        unique_orientations = [self.copy()]
        figure_copy = self.copy()
        for _ in range(3):
            figure_copy.rotate_clockwise()
            if figure_copy not in unique_orientations:
                unique_orientations.append(figure_copy.copy())
        return unique_orientations

    def __eq__(self, other):
        return np.array_equal(self.map_fragment, other.map_fragment)

    def deepcopy(self):
        return Figure(self.map_fragment.copy())


tetris_figures = [Figure(np.asarray(e)) for e in (
    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 1, 1],
     [1, 1, 0]],

    [[1, 1, 0],
     [0, 1, 1]],

    [[1, 0, 0],
     [1, 1, 1]],

    [[0, 0, 1],
     [1, 1, 1]],

    [[1, 1, 1, 1]],

    [[1, 1],
     [1, 1]]
)]


class Random:
    def __init__(self, options_num: int):
        self.options_num = options_num

    def randint(self) -> int:
        return np.random.randint(0, self.options_num)


class FigureFactory:
    def __init__(self, figures: List[Figure], random: Random):
        self.figures = figures
        self.random = random

    def next(self) -> Figure:
        return self.figures[self.random.randint()].deepcopy()

    def deepcopy(self):
        return self


default_figure_factory = FigureFactory(tetris_figures, Random(len(tetris_figures)))


@dataclass
class Config:
    CELL_SIZE: int = 18
    COLS: int = 10
    ROWS: int = 22
    MAXFPS: int = 30
    FIGURE_FACTORY: FigureFactory = default_figure_factory


class Board(MapFragmentMixin):
    def __init__(self, map_fragment: np.ndarray):
        super().__init__(map_fragment)

    @classmethod
    def clean(cls, rows, columns):
        return cls(np.zeros((rows, columns)))

    def fix_figure(self, figure: Figure, x: int, y: int) -> None:
        self.map_fragment[y: y + figure.height(), x: x + figure.width()] += figure.map_fragment

    def intersects(self, figure: Figure, x: int, y: int) -> bool:
        return np.any(self.map_fragment[y: y + figure.height(), x: x + figure.width()] + figure.map_fragment > 1)

    def remove_full_lines(self) -> None:
        full_lines = self.map_fragment.all(axis=1)
        full_lines_count = np.count_nonzero(full_lines)
        if not full_lines_count:
            return

        self.map_fragment = np.concatenate([
            np.zeros((full_lines_count, self.width())),
            self.map_fragment[np.where(~full_lines)],
        ], axis=0)
        return self.remove_full_lines()

    def deepcopy(self):
        return Board(self.map_fragment.copy())


class World:
    def __init__(
            self, board: Board, figure: Figure, figure_x: int, figure_y: int, next_figure: Figure,
            figure_factory: FigureFactory
    ):
        self.board = board
        self.figure = figure
        self.figure_x = figure_x
        self.figure_y = figure_y
        self.next_figure = next_figure
        self.figure_factory = figure_factory

    @classmethod
    def from_config(cls, config: Config, figure_factory: FigureFactory = default_figure_factory):
        figure_x, figure_y = cls.new_figure_coordinates(config.COLS)
        return cls(
            Board.clean(config.ROWS, config.COLS),
            figure_factory.next(),
            figure_x,
            figure_y,
            figure_factory.next(),
            figure_factory,
        )

    @staticmethod
    def new_figure_coordinates(columns_num: int):
        return columns_num // 2 - 1, 0

    def _is_within_board(self, x, y):
        return 0 <= x <= self.board.width() and 0 <= y <= self.board.height()

    def _can_move_to(self, dx: int = 0, dy: int = 0) -> bool:
        if (
                not self._is_within_board(self.figure_x + dx, self.figure_y + dy)
                or not self._is_within_board(
                    self.figure_x + self.figure.width() + dx, self.figure_y + self.figure.height() + dy)
        ):
            return False

        for row_index, row in enumerate(self.figure.map_fragment):
            for col_index, value in enumerate(row):
                if value and self.board.map_fragment[self.figure_y + row_index + dy, self.figure_x + col_index + dx]:
                    return False
        return True

    def can_move_down(self) -> bool:
        return self._can_move_to(dy=1)

    def can_move_right(self) -> bool:
        return self._can_move_to(dx=1)

    def can_move_left(self) -> bool:
        return self._can_move_to(dx=-1)

    def move_down(self) -> bool:
        if self.can_move_down():
            self.figure_y += 1
            return True
        else:
            self.fix_figure()
            self.board.remove_full_lines()
            self.switch_figure()
            return False

    def move_all_way_down(self):
        while True:
            moved = self.move_down()
            if not moved:
                return

    def move_right(self) -> bool:
        if self.can_move_right():
            self.figure_x += 1
            return True
        else:
            return False

    def move_left(self) -> bool:
        if self.can_move_left():
            self.figure_x -= 1
            return True
        else:
            return False

    def fix_figure(self):
        self.board.fix_figure(self.figure, self.figure_x, self.figure_y)

    def switch_figure(self):
        self.figure_x, self.figure_y = self.new_figure_coordinates(self.board.width())
        self.figure = self.next_figure
        self.next_figure = self.figure_factory.next()

    def rotate_figure(self):
        self.figure.rotate_clockwise()

    def step(self):
        self.move_down()

    def is_in_terminal_state(self) -> bool:
        return self.board.intersects(self.figure, self.figure_x, self.figure_y)

    def deepcopy(self):
        return World(
            self.board.deepcopy(), self.figure.deepcopy(), self.figure_x, self.figure_y, self.next_figure.deepcopy(),
            self.figure_factory.deepcopy()
        )
