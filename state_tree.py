from abc import ABCMeta
from typing import List, Callable, Optional, Iterator, Tuple
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


class StateTree:
    def __init__(
            self, root_node: Node, evaluate: Callable[[Node], float],
            fringe_type: Callable[[], IFringe] = FringeQueue
    ):
        self.root_node = root_node
        self.evaluate = evaluate
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
        return max(self.leaves(depth_limit), key=self.evaluate)

    def expand_node(self, node) -> Iterator[Node]:
        raise NotImplementedError

    def _evaluate(self, node: Node) -> Tuple[Node, float]:
        return node, self.evaluate(node)


class ParallelStateTree(StateTree, metaclass=ABCMeta):
    def __init__(
            self, root_node: Node, evaluate: Callable[[Node], float],
            fringe_type: Callable[[], IFringe] = FringeListQueue,
            pool: Pool = None,
    ):
        super().__init__(root_node, evaluate, fringe_type)
        self.pool = pool

    def __getstate__(self):
        return self.root_node, self.evaluate, self.fringe

    def __setstate__(self, state):
        self.root_node, self.evaluate, self.fringe = state

    def max(self, depth_limit) -> Node:
        if self.pool:
            nodes_and_values = self.pool.map(super()._evaluate, super().leaves(depth_limit))
            return max(nodes_and_values, key=lambda e: e[1])[0]
        else:
            return super().max(depth_limit)
