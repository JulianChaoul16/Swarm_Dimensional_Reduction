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


def _wolf_details(wolf: GreyWolf) -> str:
    accuracy = "N/A" if wolf.accuracy is None else f"{wolf.accuracy:.2%}"
    return (f"Wolf: {wolf.name}\n"
            f"Parameters: {wolf.position}\n"
            f"Selected Features: {sum(wolf.position)}/{len(wolf.position)}\n"
            f"Validation Accuracy: {accuracy}\n"
            f"Fitness: {wolf.fitness:.10f}\n")


def _result_summary(result: GWOResult) -> str:
    lines = []
    if result.best_name:
        lines.append(f"Best Wolf: {result.best_name}")
    lines.append(f"Best Parameters: {result.best_position}")
    lines.append(f"Selected Features: {sum(result.best_position)}/{len(result.best_position)}")
    if result.best_accuracy is not None:
        lines.append(f"Validation Accuracy: {result.best_accuracy:.2%}")
    lines.append(f"Best Fitness: {result.best_fitness:.10f}")
    lines.append(f"Execution Time: {result.execution_time_ms:.2f} milliseconds")
    return "\n".join(lines)


def run_gwo(
    objective: Callable[[Sequence[int]], float] = count_enabled,
    dimensions: int = 2,
    num_wolves: int = 30,
    max_iterations: int = 100,
    seed: Optional[int] = None,
    output_path: Optional[str] = "results.txt",
    detailed_output_path: Optional[str] = None,
) -> GWOResult:
    """Minimize objective and write round summaries and detailed wolf changes.

    By default, detailed_result.txt is saved beside output_path. Set output_path
    to None to disable file output, or supply detailed_output_path for a custom
    detailed log (including when output_path is None).

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

    if detailed_output_path is None and output_path is not None:
        detailed_output_path = str(Path(output_path).with_name("detailed_result.txt"))
    if (output_path is not None and detailed_output_path is not None
            and Path(output_path).resolve() == Path(detailed_output_path).resolve()):
        raise ValueError("Regular and detailed results must use different files.")

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
    round_reports = []
    reports = []
    if detailed_output_path is not None:
        reports.append("Initial pack:\n")
        reports.extend(_wolf_details(wolf) for wolf in wolves)
    for iteration in range(max_iterations):
        if detailed_output_path is not None:
            reports.append(f"Round {iteration + 1}:\n")
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
            if detailed_output_path is not None:
                updated = wolves[index]
                enabled = [i for i, (old, new) in enumerate(zip(wolf.position, position))
                           if old == 0 and new == 1]
                disabled = [i for i, (old, new) in enumerate(zip(wolf.position, position))
                            if old == 1 and new == 0]
                report = (_wolf_details(updated)
                          + f"Previous Parameters: {wolf.position}\n"
                          + f"Features Enabled (zero-based indices): {enabled}\n"
                          + f"Features Disabled (zero-based indices): {disabled}\n"
                          + f"Previous Fitness: {wolf.fitness:.10f}\n"
                          + f"Fitness Change (negative is better): {updated.fitness - wolf.fitness:+.10f}\n"
                          + f"Selected Feature Count Change: {sum(position) - sum(wolf.position):+d}\n")
                if wolf.accuracy is not None and updated.accuracy is not None:
                    report += (f"Previous Validation Accuracy: {wolf.accuracy:.2%}\n"
                               f"Accuracy Change (percentage points): {(updated.accuracy - wolf.accuracy) * 100:+.4f}\n")
                else:
                    previous_accuracy = "N/A" if wolf.accuracy is None else f"{wolf.accuracy:.2%}"
                    report += f"Previous Validation Accuracy: {previous_accuracy}\nAccuracy Change: N/A\n"
                reports.append(report)
        history.append(leaders[0].fitness)
        if output_path is not None:
            current_best = min(wolves, key=lambda wolf: wolf.fitness)
            round_reports.append(
                f"Round {iteration + 1}\n"
                + "Best wolf this round:\n" + _wolf_details(current_best)
                + "Best found so far:\n" + _wolf_details(leaders[0]))

    result = GWOResult(
        leaders[0].position.copy(), leaders[0].fitness, history,
        (perf_counter() - start) * 1000.0,
        leaders[0].accuracy,
        leaders[0].name,
    )
    if output_path is not None:
        with Path(output_path).open("w", encoding="utf-8") as output:
            for report in round_reports:
                output.write(report + "\n")
            output.write("Final Result:\n" + _result_summary(result) + "\n")
    if detailed_output_path is not None:
        with Path(detailed_output_path).open("w", encoding="utf-8") as output:
            for report in reports:
                output.write(report + "\n")
    return result


def select_features(X, y, feature_weight: float = 0.05,
                    validation_size: float = 0.2, seed: int = 42,
                    num_wolves: int = 30, max_iterations: int = 100,
                    output_path: Optional[str] = "results.txt",
                    detailed_output_path: Optional[str] = None) -> GWOResult:
    """Select features using logistic regression validation accuracy.

    Supply development data only. Keep final test data outside the GWO search.
    Increasing feature_weight favors smaller subsets over accuracy.
    """
    objective = LogisticRegressionObjective(
        X, y, feature_weight, validation_size, seed)
    return run_gwo(objective, objective.dimensions, num_wolves,
                   max_iterations, seed, output_path, detailed_output_path)


def print_results(result: GWOResult) -> None:
    print(_result_summary(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-weight", type=float, default=0.05,
                        help="fitness weight for feature count (default: 0.05)")
    parser.add_argument("--wolves", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results.txt")
    parser.add_argument("--detailed-output", default=None,
                        help="detailed log path (default: detailed_result.txt beside --output)")
    args = parser.parse_args()
    data = load_breast_cancer()
    print("Example dataset: breast cancer")
    print_results(select_features(data.data, data.target,
                                  feature_weight=args.feature_weight,
                                  num_wolves=args.wolves,
                                  max_iterations=args.iterations, seed=args.seed,
                                  output_path=args.output,
                                  detailed_output_path=args.detailed_output))


if __name__ == "__main__":
    main()
