"""Uniform random binary-mask search for comparison with GWO."""

import math
import random
from time import perf_counter

from gwo import GWOResult


def run_random_search(objective, dimensions, evaluations, seed=None):
    """Maximize an objective with independent Bernoulli(0.5) feature masks.

    Every sampled mask counts against the budget, including duplicates and empty
    masks. History contains best-so-far fitness after each objective call.
    """
    for name, value in (("dimensions", dimensions), ("evaluations", evaluations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1.")
    start = perf_counter()
    rng = random.Random(seed)
    best_position, best_accuracy = [], None
    best_fitness = -math.inf
    history = []
    for _ in range(evaluations):
        position = [rng.randrange(2) for _ in range(dimensions)]
        fitness = float(objective(position.copy()))
        if not math.isfinite(fitness):
            raise ValueError("The objective function must return a finite fitness.")
        if fitness > best_fitness:
            best_position = position.copy()
            best_fitness = fitness
            best_accuracy = getattr(objective, "last_accuracy", None)
        history.append(best_fitness)
    return GWOResult(best_position, best_fitness, history,
                     (perf_counter() - start) * 1000, best_accuracy)
