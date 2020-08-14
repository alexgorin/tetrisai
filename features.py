from typing import List

import numpy as np

from world import World


class Feature:
    def value(self, world: World) -> float:
        raise NotImplementedError


def get_empty_rows_count(world: World):
    count = 0
    for row in world.board.map_fragment:
        if row.any():
            break
        else:
            count += 1
    return count


class EmptyRowsCount(Feature):
    def value(self, world: World) -> float:
        empty_rows_count = get_empty_rows_count(world)

        threshold = world.board.height() / 3
        if empty_rows_count <= threshold:
            return empty_rows_count ** 2
        else:
            return threshold ** 2 + (empty_rows_count - threshold)


class HoleCount(Feature):
    def value(self, world: World) -> float:
        hole_count = 0
        filled_squares_above = np.zeros(world.board.width(), dtype=bool)
        for row in world.board.map_fragment:
            holes = np.logical_and(filled_squares_above, row == 0)
            hole_count += np.count_nonzero(holes)
            filled_squares_above = np.logical_or(filled_squares_above, row == 1)

        filled_rows_count = world.board.height() - get_empty_rows_count(world)
        return filled_rows_count / (hole_count + 1)


class FringeSmoothness(Feature):
    def value(self, world: World) -> float:
        fringe = self._fringe(world)
        discrepancies = 0
        for i in range(len(fringe) - 1):
            # discrepancies += abs(fringe[i] - fringe[i + 1])
            discrepancies += 1 if fringe[i + 1] != fringe[i] else 0
        return 1 / (discrepancies + 1)

    @staticmethod
    def _fringe(world: World) -> List[int]:
        # world_height = world.board.height()
        # fringe = np.asarray([world.board.height()] * world.board.width())
        #
        # # iterate from the bottom for optimization
        # for row_index, row in enumerate(world.board.map_fragment[::-1]):
        #     filled_cells = np.where(row == 1)
        #     if not filled_cells:
        #         break
        #     fringe[filled_cells] = world_height - row_index - 1

        fringe = []
        for col_index in range(world.board.width()):
            for row_index in range(world.board.height()):
                if world.board.map_fragment[row_index, col_index]:
                    fringe.append(row_index)
                    break
            else:
                fringe.append(world.board.height())
        return fringe


class AverageHeight(Feature):
    def value(self, world: World) -> float:
        heights_sum = 0
        filled_squares_count = 0
        for row_index, row in enumerate(world.board.map_fragment[::-1]):
            row_filled_squares_count = np.count_nonzero(row)
            if not row_filled_squares_count:
                break
            filled_squares_count += row_filled_squares_count
            heights_sum += row_index * row_filled_squares_count
        if not filled_squares_count:
            return 0
        avg_height = heights_sum / filled_squares_count
        return world.board.height() - avg_height
