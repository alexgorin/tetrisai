import itertools
from dataclasses import dataclass
from enum import Enum
from typing import List, Iterator

from world import World, Figure


class IAction:
    def apply(self, world):
        raise NotImplementedError

    def apply_to_copy(self, world):
        world_copy = world.deepcopy()
        self.apply(world_copy)
        return world_copy


class HighLevelAction(IAction):
    def apply(self, world):
        for action in self.unroll(world):
            action.apply(world)

    def unroll(self, world) -> List[IAction]:
        raise NotImplementedError


class TetrisActionType(Enum):
    LEFT = 'left'
    RIGHT = 'right'
    ALL_WAY_DOWN = 'all_way_down'
    ROTATE = 'rotate'


class TetrisAction(IAction):
    def __init__(self, action_type: TetrisActionType):
        self._action_type = action_type

    def apply(self, world):
        {
            TetrisActionType.LEFT: world.move_left,
            TetrisActionType.RIGHT: world.move_right,
            TetrisActionType.ALL_WAY_DOWN: world.move_all_way_down,
            TetrisActionType.ROTATE: world.rotate_figure,
        }[self._action_type]()

    def __repr__(self):
        return str(self._action_type)

    def __eq__(self, other):
        return self._action_type == other._action_type

#
# class TransitionModel:
#     @staticmethod
#     def transition(world, action: IAction):
#         world_copy = world.deepcopy()
#         action.apply(world_copy)
#         return world_copy


@dataclass
class MoveToPosition(HighLevelAction):
    figure: Figure
    x: int

    def unroll(self, world) -> List[IAction]:
        return [TetrisAction(action_type) for action_type in self._plan(world, self.figure, self.x)]

    @classmethod
    def _plan(cls, world: World, figure: Figure, x: int) -> List[TetrisActionType]:
        return list(itertools.chain(
            cls._fit_figure_orientation(world, figure), cls._fit_figure_location(world, x)
        )) + [TetrisActionType.ALL_WAY_DOWN]

    @classmethod
    def _fit_figure_orientation(cls, world: World, figure: Figure) -> Iterator[TetrisActionType]:
        if world.figure != figure:
            figure_copy: Figure = world.figure.copy()
            while figure_copy != figure:
                yield TetrisActionType.ROTATE
                figure_copy.rotate_clockwise()

    @classmethod
    def _fit_figure_location(cls, world: World, x: int) -> Iterator[TetrisActionType]:
        if x < world.figure_x:
            yield from (TetrisActionType.LEFT for _ in range(world.figure_x - x))
        else:
            yield from (TetrisActionType.RIGHT for _ in range(x - world.figure_x))
