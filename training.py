import itertools
import time
from multiprocessing import Pool

import numpy as np
import pygad

from agent import HighLevelAgent, IAgent
from features import FringeSmoothness, HoleCount, EmptyRowsCount, AverageHeight
from utility import Utility
from world import Config, World
import functools


def simulate_game(world: World, agent: IAgent, max_iterations: int = 10000) -> int:
    for iterations_count in range(max_iterations):
        if world.is_in_terminal_state():
            break
        action = agent.choose_action(world)
        agent.act(world, action)

    return iterations_count


pool: Pool = None

seed = 1234


def run_simulation(index, weights):
    config = Config()
    np.random.seed(seed + index)
    world = World.from_config(config)
    utility = Utility([FringeSmoothness(), HoleCount(), EmptyRowsCount(), AverageHeight()], weights)
    agent = HighLevelAgent(utility)
    return simulate_game(world, agent)


def fitness_func(weights, _):
    simulations_count = 10
    results = pool.starmap(run_simulation, [(i, weights) for i in range(simulations_count)])
    return sum(results) / len(results)


def train():
    num_generations = 10
    sol_per_pop = 100
    num_parents_mating = sol_per_pop * 2 // 3
    num_genes = 4

    init_range_low = 1
    init_range_high = 30
    parent_selection_type = "sss"
    keep_parents = sol_per_pop // 4
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_num_genes = 1

    start = time.time()

    def on_generation(ga_instance):
        global seed
        seed += 100
        np.random.seed(seed)
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print(f"{time.time() - start}s passed")
        # print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

    global pool
    pool = Pool(10)
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_num_genes=mutation_num_genes,
                           mutation_probability=0.3,
                           on_generation=on_generation,
                           random_mutation_min_val=-5,
                           random_mutation_max_val=5,)

    ga_instance.run()
    print(f"Best solution: {ga_instance.best_solution()}")
    print(f"{time.time() - start}s passed.")
    pool.close()


if __name__ == "__main__":
    train()