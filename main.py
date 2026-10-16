"""Load a dataset, run a feature-selection experiment, and report results."""

import argparse
import csv
from pathlib import Path
from typing import Optional

from dataset import load_csv
from fitness import LogisticRegressionObjective
from gwo import GWOResult, print_results, run_gwo


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="path to a CSV with column headers")
    parser.add_argument("--target", required=True, help="name of the class label column")
    parser.add_argument("--delimiter", default=",", help="CSV separator (default: comma)")
    parser.add_argument("--validation-size", type=float, default=0.2,
                        help="fraction of rows used for validation (default: 0.2)")
    parser.add_argument("--feature-weight", type=float, default=0.05,
                        help="nonnegative feature penalty coefficient (default: 0.05)")
    parser.add_argument("--wolves", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results.txt")
    parser.add_argument("--detailed-output", default=None,
                        help="detailed log path (default: detailed_result.txt beside --output)")
    args = parser.parse_args()
    try:
        X, y, feature_names = load_csv(args.dataset, args.target, args.delimiter)
        baseline = LogisticRegressionObjective(X, y,
                                               feature_weight=args.feature_weight,
                                               validation_size=args.validation_size,
                                               seed=args.seed)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    print(f"Dataset: {args.dataset} ({X.shape[0]} rows, {X.shape[1]} features)")
    print(f"Target: {args.target}")
    print(f"Feature order: {feature_names}")
    baseline_fitness = baseline([1] * baseline.dimensions)
    print(f"All-Features Baseline Validation Accuracy: {baseline.last_accuracy:.2%}")
    print(f"All-Features Baseline Fitness: {baseline_fitness:.10f}")
    result = select_features(X, y,
                             feature_weight=args.feature_weight,
                             validation_size=args.validation_size,
                             num_wolves=args.wolves,
                             max_iterations=args.iterations, seed=args.seed,
                             output_path=args.output,
                             detailed_output_path=args.detailed_output)
    print_results(result)
    selected_names = [name for name, bit in zip(feature_names, result.best_position) if bit]
    print(f"Selected columns: {selected_names}")
    with Path(args.output).open("a", encoding="utf-8") as output:
        output.write(f"Dataset: {args.dataset}\nTarget: {args.target}\n")
        output.write(f"Feature order: {feature_names}\nSelected columns: {selected_names}\n")


if __name__ == "__main__":
    main()
