# Breast cancer benchmark artifacts

`breast_cancer.json` is measured output from `python benchmark_breast_cancer.py`.
It replaces the old `results.txt` and `test_results.txt` snapshots, which contained
unlabeled numbers from an earlier experiment and were not accuracy evidence.
Normal CLI runs still write their own `results.txt` and detailed logs.

- `baseline`: logistic regression with all 30 features.
- `selected`: the subset selected by GWO.
- `random_search`: the best subset from independent Bernoulli(0.5) random masks.
- `validation_accuracy` and `test_accuracy`: fractions, from 0 to 1.
- `fitness`: validation accuracy minus `0.05 * selected_features / 30`; higher is better.
- `mask`: binary values in scikit-learn's original feature order; `feature_names`
  lists the enabled columns.
- `evaluations`: objective calls, including initialization, duplicate masks, and
  empty masks. Each search receives exactly 3,030 calls; neither caches scores.
- `convergence`: best-so-far validation fitness at calls 30, 60, ..., 3,030.
  The first point includes GWO initialization. Points use equal call counts,
  rather than equating an optimizer iteration with one random draw.
- `environment`: measured runtime library versions. Wall-clock duration is not
  the comparison budget.

Both searches use the same seed, data splits, classifier, scaling, penalty,
and initial mask distribution. Since both initialize masks using Python's seeded
random generator, their first 30 candidates are identical. Subsequent GWO
candidates use leader-guided updates; random search keeps sampling independently.
The baseline fit and final test fits are outside the search budget. Test results
never guide the search or selection of which method to report.

`breast_cancer_convergence.png` is generated from the JSON by
`python plot_benchmark.py` after installing `requirements-benchmark.txt`. Its
curves show validation fitness, not test accuracy or evidence across seeds.
This single-seed comparison does not establish statistical superiority. In this
run GWO has higher penalized validation fitness and fewer features, while random
search has higher test accuracy.
