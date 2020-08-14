from agent import HighLevelAgent, IAgent
from features import FringeSmoothness, HoleCount, EmptyRowsCount, AverageHeight
from utility import Utility
from world import Config, World
import time
from multiprocessing import Pool
import cProfile as profile
import pstats
import io
import numpy as np


def simulate_game(world: World, agent: IAgent, max_iterations: int = 10000) -> int:
    for iterations_count in range(max_iterations):
        if world.is_in_terminal_state():
            break
        action = agent.choose_action(world)
        agent.act(world, action)

    return iterations_count


def run_simulation(config):
    # np.random.seed(123)
    world = World.from_config(config)
    # [16.10236407, 23.285073,  3.22214562])
    # [12.29972137, 24.01771438, 4.99028912]
    # [2.84854145, 10.82518706, 26.9453779, 27.70104868]
    # [4.37375974, 8.74747811, 21.85225262, 6.00195256]
    utility = Utility([FringeSmoothness(), HoleCount(), EmptyRowsCount(), AverageHeight()], [ 3.2375932 , 14.10950807, 22.32253916, 30.96122022])
    agent = HighLevelAgent(utility)
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


def run_simulation_batch():
    config = Config()
    simulations_count = 50
    start = time.time()

    with Pool(processes=10) as pool:
        results = pool.map(run_simulation, [config] * simulations_count)

    print(f"{time.time() - start}s passed.")
    print(results)
    print(f"Average score: {sum(results) / len(results)}")


if __name__ == '__main__':
    run_simulation_batch()
    # profile_simulation()