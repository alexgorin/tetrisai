from typing import List, Callable, Optional, Iterator
from queue import Queue


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


class StateTree:
    def __init__(
            self, root_node: Node, evaluate: Callable[[Node], float],
            fringe_type: Callable[[], IFringe] = FringeQueue
    ):
        self.root_node = root_node
        self.evaluate = evaluate
        self.fringe = fringe_type()
        self.fringe.put(self.root_node)

    def leaves(self, depth: int) -> Iterator[Node]:
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

