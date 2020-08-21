from multiprocessing import Pool
from typing import List, Iterator, Callable, Optional

import numpy as np

from action import TetrisAction, MoveToPosition
from state_tree import StateTree, Node, SimpleEvaluationStrategy, ParallelEvaluationStrategy
from world import World


def avg(elements: List):
    return sum(elements) / len(elements)


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
        self.evaluation_strategy = SimpleEvaluationStrategy(self.utility)

    def choose_action(self, world) -> TetrisAction:
        if not self._plan:
            self.extent_plan(world)
        return self._plan.pop(0)

    def extent_plan(self, world):
        self._plan.extend(self._new_plan(world))

    def _new_plan(self, world) -> List[TetrisAction]:
        state_tree = TetrisStateTree(
            TetrisWorldNode(world),
            self.evaluation_strategy
        )
        return state_tree.max(depth_limit=1).path[0].unroll(world)


class PlanningOneMoveHierarchicalAgent(ReflexiveHierarchicalAgent):
    def _new_plan(self, world) -> List[TetrisAction]:
        state_tree = TetrisStateTree(
            TetrisWorldNode(world),
            SimpleEvaluationStrategy(self.utility)
        )
        return state_tree.max(depth_limit=2).path[0].unroll(world)


class ProbabilisticPlanningHierarchicalAgent(ReflexiveHierarchicalAgent):
    def __init__(self, utility: Callable, processes: int = 10):
        super().__init__(utility)
        self.pool = Pool(processes=processes) if processes > 0 else None
        self.evaluation_strategy = ParallelEvaluationStrategy(self._probabilistic_utility)

    def __getstate__(self):
        return self._plan, self.utility

    def __setstate__(self, state):
        self._plan, self.utility = state

    def _probabilistic_utility(self, world: World):
        utilities_for_next_figure = []
        eval_strategy = SimpleEvaluationStrategy(self.utility)
        for figure in world.figure_factory.figures:
            world_copy = world.deepcopy()
            world_copy.figure = figure
            state_tree = TetrisStateTree(TetrisWorldNode(world_copy), eval_strategy)
            max_utility = max((value[0] for node, value in eval_strategy.node_values(state_tree.leaves(depth=1))))
            utilities_for_next_figure.append(max_utility)
        # TODO: weights if the distribution is not uniform
        return avg(utilities_for_next_figure)

    def _new_plan(self, world) -> List[TetrisAction]:
        state_tree = TetrisStateTree(
            TetrisWorldNode(world),
            self.evaluation_strategy,
        )
        return state_tree.max(depth_limit=2).path[0].unroll(world)


class LimitedProbabilisticPlanningHierarchicalAgent(ProbabilisticPlanningHierarchicalAgent):
    def __init__(self, utility: Callable, processes: int = 10):
        super().__init__(utility, processes)
        self.evaluation_strategy = ParallelEvaluationStrategy(utility)
        self.probabilistic_evaluation_strategy = ParallelEvaluationStrategy(self._probabilistic_utility, self.pool)

    def _new_plan(self, world) -> List[TetrisAction]:
        state_tree = TetrisStateTree(
            TetrisWorldNode(world),
            self.evaluation_strategy,
        )
        nodes_and_values = self.evaluation_strategy.node_values(state_tree.leaves(depth=2))
        top_rated_nodes_count = 10
        top_rated_nodes = [
            node for node, value in sorted(nodes_and_values, key=lambda e: e[1], reverse=True)[:top_rated_nodes_count]
        ]
        return max(
            self.probabilistic_evaluation_strategy.node_values(top_rated_nodes),
            key=lambda e: e[1],
        )[0].path[0].unroll(world)
