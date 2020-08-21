from abc import ABCMeta
from typing import List, Callable, Optional, Iterator, Tuple, Iterable
from queue import Queue
from multiprocessing import Pool


class Node:
    def __init__(self, path: Optional[List] = None):
        self.path = path or []

    def depth(self):
        return len(self.path)


class IFringe:
    def put(self, node: Node):
        raise NotImplementedError

    def get(self) -> Node:
        raise NotImplementedError

    def empty(self) -> bool:
        raise NotImplementedError


class FringeQueue(IFringe):
    def __init__(self):
        self.elements = Queue()

    def put(self, node: Node):
        self.elements.put(node)

    def get(self) -> Node:
        return self.elements.get()

    def empty(self) -> bool:
        return self.elements.empty()

    def clear(self):
        self.elements = Queue()


class FringeListQueue(IFringe):
    def __init__(self):
        self.elements = []

    def put(self, node: Node):
        self.elements.append(node)

    def get(self) -> Node:
        return self.elements.pop()

    def empty(self) -> bool:
        return not self.elements

    def clear(self):
        self.elements = []


class EvaluationStrategy:
    def __init__(self, utility: Callable[..., float]):
        self.utility = utility

    def node_values(self, nodes: Iterable[Node]) -> Iterable[Tuple[Node, float]]:
        raise NotImplementedError


class SimpleEvaluationStrategy(EvaluationStrategy):
    def node_values(self, nodes: Iterable[Node]) -> Iterable[Tuple[Node, float]]:
        return ((node, self.utility(node.world)) for node in nodes)


class ParallelEvaluationStrategy(EvaluationStrategy):
    def __init__(self, utility: Callable[..., float], pool: Pool = None):
        super().__init__(utility)
        self.pool = pool

    def __getstate__(self):
        return self.utility

    def __setstate__(self, state):
        self.utility = state
        self.pool = None

    def node_values(self, nodes: Iterable[Node]) -> Iterable[Tuple[Node, float]]:
        nodes_list = list(nodes)
        _map = self.pool.map if self.pool else map
        return zip(
            nodes_list,
            _map(self.utility, [node.world for node in nodes_list])
        )


class StateTree:
    def __init__(
            self, root_node: Node, evaluation_strategy: EvaluationStrategy,
            fringe_type: Callable[[], IFringe] = FringeQueue
    ):
        self.root_node = root_node
        self.evaluation_strategy = evaluation_strategy
        self.fringe = fringe_type()

    def leaves(self, depth: int) -> Iterator[Node]:
        self.fringe.put(self.root_node)
        while not self.fringe.empty():
            node = self.fringe.get()
            if node.depth() == depth:
                yield node
            else:
                children = self.expand_node(node)
                for child in children:
                    self.fringe.put(child)

    def max(self, depth_limit) -> Node:
        return max(
            self.evaluation_strategy.node_values(self.leaves(depth_limit)),
            key=lambda e: e[1]
        )[0]

    def expand_node(self, node) -> Iterator[Node]:
        raise NotImplementedError

#     def node_value(self, node: Node) -> Tuple[Node, float]:
#         return node, self.evaluate(node)
#
#
# class ParallelStateTree(StateTree, metaclass=ABCMeta):
#     def __init__(
#             self, root_node: Node, evaluate: Callable[[Node], float],
#             fringe_type: Callable[[], IFringe] = FringeListQueue,
#             pool: Pool = None,
#     ):
#         super().__init__(root_node, evaluate, fringe_type)
#         self.pool = pool
#
#     def __getstate__(self):
#         return self.root_node, self.evaluate, self.fringe
#
#     def __setstate__(self, state):
#         self.root_node, self.evaluate, self.fringe = state
#
#     def max(self, depth_limit) -> Node:
#         if self.pool:
#             nodes_and_values = self.pool.map(super().node_value, super().leaves(depth_limit))
#             return max(nodes_and_values, key=lambda e: e[1])[0]
#         else:
#             return super().max(depth_limit)
