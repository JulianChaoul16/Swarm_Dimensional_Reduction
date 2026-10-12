"""Load a dataset, run a feature-selection experiment, and report results."""

import argparse
from typing import Optional

from sklearn.datasets import load_breast_cancer

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
    parser.add_argument("--feature-weight", type=float, default=0.05,
                        help="nonnegative feature penalty coefficient (default: 0.05)")
    parser.add_argument("--wolves", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results.txt")
    parser.add_argument("--detailed-output", default=None,
                        help="detailed log path (default: detailed_result.txt beside --output)")
    args = parser.parse_args()
    data = load_breast_cancer()
    print("Example dataset: breast cancer")
    baseline = LogisticRegressionObjective(data.data, data.target,
                                           feature_weight=args.feature_weight,
                                           seed=args.seed)
    baseline_fitness = baseline([1] * baseline.dimensions)
    print(f"All-Features Baseline Validation Accuracy: {baseline.last_accuracy:.2%}")
    print(f"All-Features Baseline Fitness: {baseline_fitness:.10f}")
    print_results(select_features(data.data, data.target,
                                  feature_weight=args.feature_weight,
                                  num_wolves=args.wolves,
                                  max_iterations=args.iterations, seed=args.seed,
                                  output_path=args.output,
                                  detailed_output_path=args.detailed_output))


if __name__ == "__main__":
    main()
