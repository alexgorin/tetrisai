"""
Simulating the games without actual rendering.
"""

from agent import ReflexiveHierarchicalAgent, IAgent, PlanningTwoMovesHierarchicalAgent, \
    ProbabilisticPlanningHierarchicalAgent, LimitedProbabilisticPlanningHierarchicalAgent
from features import FringeSmoothness, HoleCount, EmptyRowsCount, AverageHeight
from utility import Utility
from world import Config, World
import time
from multiprocessing import Pool
import cProfile as profile
import pstats
import io
import numpy as np


def simulate_game(world: World, agent: IAgent, max_iterations: int = 50000) -> int:
    for iterations_count in range(max_iterations):
        if world.is_in_terminal_state():
            break
        action = agent.choose_action(world)
        action.apply(world)

    return iterations_count


def run_simulation(config):
    world = World.from_config(config)
    utility = Utility(
        [FringeSmoothness(), HoleCount(), EmptyRowsCount(), AverageHeight()],
        [3.2375932, 14.10950807, 22.32253916, 30.96122022]
    )
    # agent = LimitedProbabilisticPlanningHierarchicalAgent(utility)
    # agent = ProbabilisticPlanningHierarchicalAgent(utility)
    agent = PlanningTwoMovesHierarchicalAgent(utility)
    # agent = ReflexiveHierarchicalAgent(utility)
    return simulate_game(world, agent)


def profile_simulation():
    np.random.seed(123)
    config = Config()
    start = time.time()
    pr = profile.Profile()
    with pr:
        iterations_count = run_simulation(config)

    print(f"{iterations_count} moves made")
    print(f"{time.time() - start}s passed.")

    s = io.StringIO()
    sortby = pstats.SortKey.TIME
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20)
    print(s.getvalue())


def run_simulation_batch(simulations_count=10, processes=10):
    config = Config()
    start = time.time()
    if processes > 1:
        with Pool(processes=min(simulations_count, processes)) as pool:
            results = pool.map(run_simulation, [config] * simulations_count)
    else:
        results = [run_simulation(config)]

    print(f"{time.time() - start}s passed.")
    print(results)
    print(f"Average score: {sum(results) / len(results)}")


if __name__ == '__main__':
    # run_simulation_batch()
    # profile_simulation()
    run_simulation_batch(simulations_count=1, processes=1)
