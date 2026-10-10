"""Binary Grey Wolf Optimizer for selecting on/off parameters."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Callable, List, Optional, Sequence

import numpy as np
from name import wolf_name
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionObjective:
    """Score a feature mask on a fixed validation split (lower is better)."""

    def __init__(self, X, y, feature_weight: float = 0.05,
                 validation_size: float = 0.2, seed: int = 42):
        if not 0 < feature_weight < 1:
            raise ValueError("feature_weight must be between 0 and 1.")
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a two-dimensional feature matrix.")
        self.dimensions = X.shape[1]
        self.feature_weight = feature_weight
        self.seed = seed
        self.X_train, self.X_validation, self.y_train, self.y_validation = (
            train_test_split(X, y, test_size=validation_size,
                             random_state=seed, stratify=y)
        )
        self.last_accuracy = None

    def __call__(self, position: Sequence[int]) -> float:
        mask = np.asarray(position)
        if mask.shape != (self.dimensions,) or not np.isin(mask, [0, 1]).all():
            raise ValueError("Each feature must have one binary selection bit.")
        selected = mask.astype(bool)
        self.last_accuracy = None
        if not selected.any():
            return 2.0  # Worse than every valid subset; no model can be fitted.
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=self.seed),
        )
        model.fit(self.X_train[:, selected], self.y_train)
        self.last_accuracy = float(model.score(
            self.X_validation[:, selected], self.y_validation))
        return ((1 - self.feature_weight) * (1 - self.last_accuracy)
                + self.feature_weight * selected.sum() / self.dimensions)


@dataclass
class GreyWolf:
    position: List[int]
    fitness: float
    accuracy: Optional[float] = None
    name: str = ""


@dataclass
class GWOResult:
    best_position: List[int]
    best_fitness: float
    history: List[float]
    execution_time_ms: float
    best_accuracy: Optional[float] = None
    best_name: str = ""


def count_enabled(position: Sequence[int]) -> float:
    """Example only: minimize the number of enabled parameters."""
    return float(sum(position))


def _evaluate(objective: Callable[[Sequence[int]], float],
              position: List[int], name: str = "") -> GreyWolf:
    fitness = float(objective(position.copy()))
    if not math.isfinite(fitness):
        raise ValueError("The objective function must return a finite fitness.")
    return GreyWolf(position, fitness, getattr(objective, "last_accuracy", None), name)


def _update_leaders(leaders: List[GreyWolf], wolf: GreyWolf) -> None:
    # Store independent snapshots, and shift older leaders down when needed.
    for index, leader in enumerate(leaders):
        if wolf.fitness < leader.fitness:
            leaders.insert(index, GreyWolf(wolf.position.copy(), wolf.fitness,
                                           wolf.accuracy, wolf.name))
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
    for index in range(num_wolves):
        wolf = _evaluate(objective, [rng.randrange(2)
                                     for _ in range(dimensions)], wolf_name(index))
        wolves.append(wolf)
        _update_leaders(leaders, wolf)

    history = []
    reports = []
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
            wolves[index] = _evaluate(objective, position, wolf.name)
            _update_leaders(leaders, wolves[index])
            if output_path is not None:
                updated = wolves[index]
                report = (f"Iteration {iteration + 1} | {updated.name} | "
                          f"Fitness: {updated.fitness:.6g} | "
                          f"Selected Features: {sum(position)}/{dimensions}")
                if updated.accuracy is not None:
                    report += f" | Validation Accuracy: {updated.accuracy:.2%}"
                reports.append(report)
        history.append(leaders[0].fitness)

    if output_path is not None:
        with Path(output_path).open("w", encoding="utf-8") as output:
            for iteration, fitness in enumerate(history, start=1):
                output.write(f"{iteration}: {fitness:.6g}\n")
            output.write("\nWolf updates:\n")
            for report in reports:
                output.write(report + "\n")

    return GWOResult(
        leaders[0].position.copy(), leaders[0].fitness, history,
        (perf_counter() - start) * 1000.0,
        leaders[0].accuracy,
        leaders[0].name,
    )


def select_features(X, y, feature_weight: float = 0.05,
                    validation_size: float = 0.2, seed: int = 42,
                    num_wolves: int = 30, max_iterations: int = 100,
                    output_path: Optional[str] = "results.txt") -> GWOResult:
    """Select features using logistic regression validation accuracy.

    Supply development data only. Keep final test data outside the GWO search.
    Increasing feature_weight favors smaller subsets over accuracy.
    """
    objective = LogisticRegressionObjective(
        X, y, feature_weight, validation_size, seed)
    return run_gwo(objective, objective.dimensions, num_wolves,
                   max_iterations, seed, output_path)


def print_results(result: GWOResult) -> None:
    if result.best_name:
        print(f"Best Wolf: {result.best_name}")
    print(f"Best Parameters: {result.best_position}")
    print(f"Selected Features: {sum(result.best_position)}/{len(result.best_position)}")
    if result.best_accuracy is not None:
        print(f"Validation Accuracy: {result.best_accuracy:.2%}")
    print(f"Best Fitness: {result.best_fitness:.10f}")
    print(f"Execution Time: {result.execution_time_ms:.2f} milliseconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-weight", type=float, default=0.05,
                        help="fitness weight for feature count (default: 0.05)")
    parser.add_argument("--wolves", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results.txt")
    args = parser.parse_args()
    data = load_breast_cancer()
    print("Example dataset: breast cancer")
    print_results(select_features(data.data, data.target,
                                  feature_weight=args.feature_weight,
                                  num_wolves=args.wolves,
                                  max_iterations=args.iterations, seed=args.seed,
                                  output_path=args.output))


if __name__ == "__main__":
    main()
