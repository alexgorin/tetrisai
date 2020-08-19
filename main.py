from agent import ReflexiveHierarchicalAgent, PlanningOneMoveHierarchicalAgent
from features import FringeSmoothness, HoleCount, EmptyRowsCount, AverageHeight
from tetris import Game
from utility import Utility
from world import Config


def run_game():
    game = Game(Config())
    utility = Utility(
        [FringeSmoothness(), HoleCount(), EmptyRowsCount(), AverageHeight()],
        [3.2375932, 14.10950807, 22.32253916, 30.96122022]
    )
    game.run_agent(PlanningOneMoveHierarchicalAgent(utility))
    # game.run_agent(ReflexiveHierarchicalAgent(utility))


if __name__ == '__main__':
    run_game()