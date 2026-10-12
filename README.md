# Swarm_Dimensional_Reduction
Using a GWO to reduce the number of features in an ML model

### Big Question
What is the minimum number of features an ML model can have while still having a relatively high accuracy score?

### Algorithm

Every "Wolf" has n number of parameters dependent on the dimensions of model
5 wolfs 

### Logistic regression feature selection

Install dependencies with `python -m pip install -r requirements.txt`.
Run the built-in breast cancer example with:

```sh
python main.py --wolves 5 --iterations 20 --seed 42
```

For your own numeric classification data:

```python
from main import select_features

result = select_features(X, y, num_wolves=5, max_iterations=20)
selected_columns = [i for i, bit in enumerate(result.best_position) if bit]
X_selected = X[:, selected_columns]
```

Each wolf is scored initially and after every update using logistic regression
on its selected columns. Scaling is fitted only on training data. All wolves
use the same stratified training/validation split for comparable scores.
Fitness is maximized:
`validation_accuracy - 0.05 * (selected_features / total_features)`.
The breast cancer dataset has 30 features, so the denominator is 30 for this
experiment; custom datasets use their own total feature count. Accuracy is a
fraction between 0 and 1. Use `--feature-weight` to vary the nonnegative penalty
coefficient (zero disables the penalty). Empty masks receive fitness
`-1 - feature_weight` and are not fitted. The returned `best_accuracy` is validation
accuracy, not an independent test score. Keep final test data separate from
the data passed to `select_features`; repeated search can overfit validation data.
This is a heuristic tradeoff, not a guarantee of the smallest possible subset
or a minimum accuracy threshold. `results.txt` records the best wolf in each round
and the best found so far, with parameters, feature count, fitness, and validation
accuracy. It ends with the full final summary, including execution time.
`detailed_result.txt` records the initial pack and every wolf's details each round:
current and previous parameters, enabled/disabled feature indices (zero-based),
fitness changes, feature-count changes, and accuracy changes in percentage points.
A positive fitness change means improvement; unavailable accuracy is marked N/A.
Use `--output` and `--detailed-output` to customize the paths. By default the
detailed file is saved beside the regular file. For Python calls, `output_path=None`
disables both files unless `detailed_output_path` is explicitly supplied.
Harry Potter names come from `name.py` and stay with
each wolf throughout the run. Larger packs reuse names with numbered suffixes.
The final summary identifies the best wolf, also available as `result.best_name`.

### Module responsibilities

- `main.py` loads the dataset, evaluates an all-features baseline, runs the
  experiment, and reports results. The baseline uses the same validation split.
- `fitness.py` evaluates feature masks with logistic regression and applies the
  feature-count penalty.
- `gwo.py` initializes the pack, updates wolf positions and leaders, and records
  per-round results and detailed wolf changes.

Run experiments with `main.py`. For custom datasets, import `select_features`
from `main`. The optimizer accepts any finite objective where higher is better.
