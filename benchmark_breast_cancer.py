"""Reproduce the README benchmark with an untouched test split."""

import csv
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import load_csv
from fitness import LogisticRegressionObjective
from gwo import run_gwo
from random_search import run_random_search


class TrackedObjective:
    """Count every objective call and retain best-so-far fitness."""

    def __init__(self, objective):
        self.objective = objective
        self.history = []

    @property
    def last_accuracy(self):
        return self.objective.last_accuracy

    def __call__(self, position):
        fitness = self.objective(position)
        self.history.append(max(self.history[-1], fitness) if self.history else fitness)
        return fitness


def main():
    data = load_breast_cancer()
    # Exercise the same CSV loader used for user-provided datasets.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "breast_cancer.csv"
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([*data.feature_names, "target"])
            writer.writerows([*row, int(label)] for row, label in zip(data.data, data.target))
        X, y, names = load_csv(path, "target")

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    baseline = LogisticRegressionObjective(X_dev, y_dev, seed=42)
    baseline_fitness = baseline([1] * X.shape[1])
    wolves, iterations, seed = 30, 100, 42
    budget = wolves * (iterations + 1)  # Initial pack plus every update.
    gwo_objective = TrackedObjective(LogisticRegressionObjective(X_dev, y_dev, seed=seed))
    random_objective = TrackedObjective(LogisticRegressionObjective(X_dev, y_dev, seed=seed))
    print(f"Running GWO: {budget} objective evaluations", flush=True)
    result = run_gwo(gwo_objective, X.shape[1], num_wolves=wolves,
                     max_iterations=iterations, seed=seed, output_path=None)
    print(f"Running random search: {budget} objective evaluations", flush=True)
    random_result = run_random_search(random_objective, X.shape[1], budget, seed=seed)
    assert len(gwo_objective.history) == len(random_objective.history) == budget

    def test_accuracy(mask):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        model.fit(X_dev[:, mask], y_dev)
        return float(model.score(X_test[:, mask], y_test))

    def summarize(search_result, objective):
        mask = np.asarray(search_result.best_position, dtype=bool)
        return {
            "features": int(mask.sum()), "validation_accuracy": search_result.best_accuracy,
            "fitness": search_result.best_fitness, "test_accuracy": test_accuracy(mask),
            "feature_names": [name for name, bit in zip(names, mask) if bit],
            "mask": search_result.best_position, "evaluations": len(objective.history),
            "convergence": [
                {"evaluations": count, "best_fitness": objective.history[count - 1]}
                for count in range(wolves, budget + 1, wolves)
            ],
        }

    report = {
        "dataset": "sklearn.datasets.load_breast_cancer",
        "rows": len(y), "total_features": len(names),
        "development_rows": len(y_dev), "search_training_rows": len(baseline.y_train),
        "validation_rows": len(baseline.y_validation), "test_rows": len(y_test),
        "seed": 42, "wolves": 30, "iterations": 100, "feature_weight": 0.05,
        "evaluation_budget_per_search": budget,
        "random_search_distribution": "independent Bernoulli(0.5) per feature",
        "budget_policy": "Every objective call counts, including duplicate and empty masks; no caching. Baseline and final test fits are outside the search budget.",
        "baseline": {"features": len(names), "validation_accuracy": baseline.last_accuracy,
                     "fitness": baseline_fitness,
                     "test_accuracy": test_accuracy(np.ones(len(names), dtype=bool))},
        "selected": summarize(result, gwo_objective),
        "random_search": summarize(random_result, random_objective),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__},
    }
    destination = Path(__file__).resolve().parent / "benchmarks" / "breast_cancer.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination}")
    for name in ("baseline", "selected", "random_search"):
        row = report[name]
        print(f"{name}: {row['features']} features, validation={row['validation_accuracy']:.2%}, "
              f"test={row['test_accuracy']:.2%}, fitness={row['fitness']:.10f}")


if __name__ == "__main__":
    main()
