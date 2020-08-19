import itertools
from dataclasses import dataclass
from enum import Enum
from typing import List, Iterator, Callable, Optional

import numpy as np

from utility import Utility
from world import World, Figure
from action import IAction, TetrisAction, MoveToPosition
from state_tree import StateTree, Node
#
# class TetrisAction(Enum):
#     LEFT = 'left'
#     RIGHT = 'right'
#     ALL_WAY_DOWN = 'all_way_down'
#     ROTATE = 'rotate'
#
#
# @dataclass
# class HighLevelAction:
#     figure: Figure
#     figure_x: int
#
#     @staticmethod
#     def possible_actions(world: World) -> List:
#         actions = []
#         for figure in world.figure.possible_orientations():
#             for figure_x in range(world.board.width() - figure.width() + 1):
#                 actions.append(HighLevelAction(figure, figure_x))
#         return actions
#
#     def plan(self, world) -> List[TetrisAction]:
#         return list(itertools.chain(
#             self._fit_figure_orientation(world), self._fit_figure_location(world)
#         )) + [TetrisAction.ALL_WAY_DOWN]
#
#     def _fit_figure_orientation(self, world: World) -> Iterator[TetrisAction]:
#         if world.figure != self.figure:
#             figure_copy: Figure = world.figure.copy()
#             while figure_copy != self.figure:
#                 yield TetrisAction.ROTATE
#                 figure_copy.rotate_clockwise()
#
#     def _fit_figure_location(self, world: World) -> Iterator[TetrisAction]:
#         if self.figure_x < world.figure_x:
#             yield from (TetrisAction.LEFT for _ in range(world.figure_x - self.figure_x))
#         else:
#             yield from (TetrisAction.RIGHT for _ in range(self.figure_x - world.figure_x))
#
#
# def perform_action(world: World, action: TetrisAction):
#     {
#         TetrisAction.LEFT: world.move_left,
#         TetrisAction.RIGHT: world.move_right,
#         TetrisAction.ALL_WAY_DOWN: world.move_all_way_down,
#         TetrisAction.ROTATE: world.rotate_figure,
#     }[action]()
#
#
# class TransitionModel:
#     @staticmethod
#     def transition(world: World, action: TetrisAction) -> World:
#         world_copy = world.deepcopy()
#         perform_action(world_copy, action)
#         return world_copy
#
#     @staticmethod
#     def transitions(world: World, actions: List[TetrisAction]) -> World:
#         world_copy = world.deepcopy()
#         for action in actions:
#             perform_action(world_copy, action)
#         return world_copy


def random_utility(world: World):
    return np.random.randint(0, 10000)


def trivial_utility(world: World):
    empty_rows_count = 0
    first_non_empty_row_sum = 0
    for row in world.board.map_fragment:
        if row.any():
            first_non_empty_row_sum = row.sum()
            break
        else:
            empty_rows_count += 1

    return 100 * empty_rows_count + 1 / first_non_empty_row_sum


class IAgent:
    def choose_action(self, world) -> TetrisAction:
        raise NotImplementedError
    #
    # def act(self, world: World, action: TetrisAction):
    #     raise NotImplementedError


# class RandomAgent(IAgent):
#     def choose_action(self, world) -> TetrisAction:
#         return max([
#             (random_utility(TransitionModel.transition(world, possible_action)), possible_action)
#             for possible_action in TetrisAction
#         ], key=lambda e: e[0])[1]
#
#     def act(self, world: World, action: TetrisAction):
#         return perform_action(world, action)


def possible_actions(world: World) -> Iterator[MoveToPosition]:
    for figure in world.figure.possible_orientations():
        for figure_x in range(world.board.width() - figure.width() + 1):
            yield MoveToPosition(figure, figure_x)


class TetrisWorldNode(Node):
    def __init__(self, world: World, path: Optional[List[Node]] = None):
        super().__init__(path)
        self.world = world


class TetrisStateTree(StateTree):

    def expand_node(self, node: TetrisWorldNode) -> Iterator[Node]:
        for action in possible_actions(node.world):
            path = node.path + [action]
            yield TetrisWorldNode(action.apply_to_copy(node.world), path)


class ReflexiveHierarchicalAgent(IAgent):
    def __init__(self, utility: Callable):
        self._plan: List[TetrisAction] = []
        self.utility = utility

    def choose_action(self, world) -> TetrisAction:
        if not self._plan:
            self.extent_plan(world)
        return self._plan.pop(0)

    def extent_plan(self, world):
        self._plan.extend(self._new_plan(world))

    def _new_plan(self, world) -> List[TetrisAction]:
        state_tree = TetrisStateTree(TetrisWorldNode(world), lambda node: self.utility(node.world))
        return state_tree.max(depth_limit=1).path[0].unroll(world)

    # def _new_plan(self, world):
    #     plans = []
    #     for high_level_action in possible_actions(world):
    #         plan = high_level_action.unroll(world)
    #         utility, feature_values, weighted_feature_values = self.utility.value(
    #             transitions(world, plan))
    #         plans.append((utility, plan, feature_values, weighted_feature_values))
    #     best_plan = max(plans, key=lambda e: e[0])[1]
    #     return best_plan


class PlanningOneMoveHierarchicalAgent(ReflexiveHierarchicalAgent):
    def _new_plan(self, world) -> List[TetrisAction]:
        state_tree = TetrisStateTree(TetrisWorldNode(world), lambda node: self.utility(node.world))
        return state_tree.max(depth_limit=2).path[0].unroll(world)
