"""Binary Grey Wolf Optimizer for selecting on/off parameters."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Callable, List, Optional, Sequence


@dataclass
class GreyWolf:
    position: List[int]
    fitness: float


@dataclass
class GWOResult:
    best_position: List[int]
    best_fitness: float
    history: List[float]
    execution_time_ms: float


def count_enabled(position: Sequence[int]) -> float:
    """Example only: minimize the number of enabled parameters."""
    return float(sum(position))


def _evaluate(objective: Callable[[Sequence[int]], float],
              position: List[int]) -> GreyWolf:
    fitness = float(objective(position.copy()))
    if not math.isfinite(fitness):
        raise ValueError("The objective function must return a finite fitness.")
    return GreyWolf(position, fitness)


def _update_leaders(leaders: List[GreyWolf], wolf: GreyWolf) -> None:
    # Store independent snapshots, and shift older leaders down when needed.
    for index, leader in enumerate(leaders):
        if wolf.fitness < leader.fitness:
            leaders.insert(index, GreyWolf(wolf.position.copy(), wolf.fitness))
            del leaders[3:]
            return


def run_gwo(
    objective: Callable[[Sequence[int]], float] = count_enabled,
    dimensions: int = 2,
    num_wolves: int = 30,
    max_iterations: int = 100,
    seed: Optional[int] = None,
    output_path: Optional[str] = "results.txt",
) -> GWOResult:
    """Minimize objective and optionally write iteration fitness to a file.

    Defaults are example settings because ObjectiveFunction.hpp was not supplied.
    dimensions is the number of on/off parameters. Positions contain integer
    bits, initialized independently with equal probability of 0 or 1. Continuous
    GWO proposals are clipped to [0, 1] and sampled as probabilities of enabling
    each parameter. Leaders update after every evaluation and store the three
    best evaluations seen so far (ties are allowed).
    """
    for name, value, minimum in (
        ("dimensions", dimensions, 1),
        ("num_wolves", num_wolves, 3),
        ("max_iterations", max_iterations, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")

    start = perf_counter()
    rng = random.Random(seed)
    leaders = [GreyWolf([], math.inf) for _ in range(3)]
    wolves = []
    for _ in range(num_wolves):
        wolf = _evaluate(objective, [rng.randrange(2)
                                     for _ in range(dimensions)])
        wolves.append(wolf)
        _update_leaders(leaders, wolf)

    history = []
    for iteration in range(max_iterations):
        a = 2.0 - iteration / max_iterations * 2.0
        for index, wolf in enumerate(wolves):
            position = []
            for dimension in range(dimensions):
                candidates = []
                for leader in leaders:
                    r1, r2 = rng.random(), rng.random()
                    coefficient_a = 2.0 * a * r1 - a
                    coefficient_c = 2.0 * r2
                    distance = abs(coefficient_c * leader.position[dimension]
                                   - wolf.position[dimension])
                    candidates.append(leader.position[dimension]
                                      - coefficient_a * distance)
                probability_on = max(0.0, min(1.0, sum(candidates) / 3.0))
                position.append(int(rng.random() < probability_on))
            wolves[index] = _evaluate(objective, position)
            _update_leaders(leaders, wolves[index])
        history.append(leaders[0].fitness)

    if output_path is not None:
        with Path(output_path).open("w", encoding="utf-8") as output:
            for iteration, fitness in enumerate(history, start=1):
                output.write(f"{iteration}: {fitness:.6g}\n")

    return GWOResult(
        leaders[0].position.copy(), leaders[0].fitness, history,
        (perf_counter() - start) * 1000.0,
    )


def print_results(result: GWOResult) -> None:
    print(f"Best Parameters: {result.best_position}")
    print(f"Best Fitness: {result.best_fitness:.10f}")
    print(f"Execution Time: {result.execution_time_ms:.2f} milliseconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", "--parameters", type=int, default=2,
                        help="number of binary on/off parameters")
    parser.add_argument("--wolves", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", default="results.txt")
    args = parser.parse_args()
    print_results(run_gwo(dimensions=args.dimensions, num_wolves=args.wolves,
                          max_iterations=args.iterations, seed=args.seed,
                          output_path=args.output))


if __name__ == "__main__":
    main()
