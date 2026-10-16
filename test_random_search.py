"""Checks for the matched-budget random-search baseline."""

import unittest

from random_search import run_random_search


class RandomSearchTests(unittest.TestCase):
    def test_budget_includes_every_sample_and_tracks_best(self):
        observations = []

        def objective(position):
            observations.append(position.copy())
            return -sum(position)

        result = run_random_search(objective, dimensions=4, evaluations=25, seed=42)
        self.assertEqual(len(observations), 25)
        self.assertEqual(len(result.history), 25)
        self.assertTrue(all(len(mask) == 4 and set(mask) <= {0, 1} for mask in observations))
        scores = [-sum(mask) for mask in observations]
        self.assertEqual(result.best_fitness, max(scores))
        self.assertEqual(result.best_fitness, -sum(result.best_position))
        self.assertEqual(result.history, [max(scores[:i]) for i in range(1, 26)])

    def test_same_seed_repeats_search(self):
        first = run_random_search(sum, 8, 50, seed=7)
        second = run_random_search(sum, 8, 50, seed=7)
        self.assertEqual(first.best_position, second.best_position)
        self.assertEqual(first.best_fitness, second.best_fitness)
        self.assertEqual(first.history, second.history)


if __name__ == "__main__":
    unittest.main()
