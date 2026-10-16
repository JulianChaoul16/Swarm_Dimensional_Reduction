"""Render the committed benchmark's convergence curves without rerunning ML."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    directory = Path(__file__).resolve().parent / "benchmarks"
    report = json.loads((directory / "breast_cancer.json").read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    for key, label, color in (("selected", "GWO", "#1769aa"),
                              ("random_search", "Random search", "#b95000")):
        points = report[key]["convergence"]
        axis.step([point["evaluations"] for point in points],
                  [point["best_fitness"] for point in points],
                  where="post", label=label, color=color, linewidth=2)
    axis.axhline(report["baseline"]["fitness"], color="#555555", linestyle="--",
                 label="All-features baseline")
    axis.set(xlabel="Objective evaluations (including the initial pack)",
             ylabel="Best-so-far validation fitness (higher is better)",
             title=f"Breast cancer feature search: equal {report['evaluation_budget_per_search']:,}-evaluation budgets",
             xlim=(0, report["evaluation_budget_per_search"]))
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    fig.text(0.5, -0.035,
             f"Seed {report['seed']} · accuracy − {report['feature_weight']} × selected / total features · recorded every {report['wolves']} calls",
             ha="center", fontsize=9)
    destination = directory / "breast_cancer_convergence.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
