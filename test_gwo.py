"""Optimizer behavior checks using inexpensive, observable objectives."""

import random
import unittest

from gwo import run_gwo


class RecordingObjective:
    def __init__(self):
        self.positions = []
        self.scores = []

    def __call__(self, position):
        self.positions.append(position.copy())
        # Distinguish masks and avoid a test tied to a particular optimal mask.
        score = sum((index + 1) * bit for index, bit in enumerate(position))
        self.scores.append(score)
        return score


class GWOTests(unittest.TestCase):
    def test_seed_reproduces_entire_search_independent_of_global_rng(self):
        first_objective = RecordingObjective()
        first = run_gwo(first_objective, dimensions=9, num_wolves=5,
                        max_iterations=12, seed=42, output_path=None)
        state = random.getstate()
        try:
            random.seed(999)
            for _ in range(100):
                random.random()
            second_objective = RecordingObjective()
            second = run_gwo(second_objective, dimensions=9, num_wolves=5,
                             max_iterations=12, seed=42, output_path=None)
        finally:
            random.setstate(state)
        self.assertEqual(first_objective.positions, second_objective.positions)
        self.assertEqual(first.best_position, second.best_position)
        self.assertEqual(first.best_fitness, second.best_fitness)
        self.assertEqual(first.best_accuracy, second.best_accuracy)
        self.assertEqual(first.best_name, second.best_name)
        self.assertEqual(first.history, second.history)
        # Elapsed time is deliberately excluded from reproducibility assertions.

    def test_every_evaluated_mask_is_binary_and_has_requested_dimensions(self):
        for dimensions in (1, 7, 30):
            for seed in (0, 42, 123):
                with self.subTest(dimensions=dimensions, seed=seed):
                    objective = RecordingObjective()
                    result = run_gwo(objective, dimensions=dimensions, num_wolves=4,
                                     max_iterations=10, seed=seed, output_path=None)
                    self.assertEqual(len(objective.positions), 4 * 11)
                    for mask in objective.positions + [result.best_position]:
                        self.assertEqual(len(mask), dimensions)
                        self.assertTrue(all(type(bit) is int and bit in (0, 1) for bit in mask))

    def test_best_result_and_history_match_all_observed_evaluations(self):
        objective = RecordingObjective()
        wolves, iterations = 6, 15
        result = run_gwo(objective, dimensions=11, num_wolves=wolves,
                         max_iterations=iterations, seed=7, output_path=None)
        self.assertEqual(result.best_fitness, max(objective.scores))
        self.assertIn(result.best_position,
                      [mask for mask, score in zip(objective.positions, objective.scores)
                       if score == result.best_fitness])
        expected_history = [max(objective.scores[:wolves * (iteration + 2)])
                            for iteration in range(iterations)]
        self.assertEqual(result.history, expected_history)
        self.assertEqual(result.history[-1], result.best_fitness)
        self.assertTrue(all(a <= b for a, b in zip(result.history, result.history[1:])))

    def test_initial_best_and_its_metadata_survive_worse_later_evaluations(self):
        class DecliningObjective:
            def __init__(self):
                self.positions = []
                self.last_accuracy = None

            def __call__(self, position):
                self.positions.append(position.copy())
                self.last_accuracy = 1 / len(self.positions)
                return 100 - len(self.positions)

        objective = DecliningObjective()
        result = run_gwo(objective, dimensions=8, num_wolves=3,
                         max_iterations=10, seed=42, output_path=None)
        self.assertEqual(result.best_fitness, 99)
        self.assertEqual(result.best_position, objective.positions[0])
        self.assertEqual(result.best_accuracy, 1.0)
        self.assertEqual(result.best_name, "Harry Potter")
        self.assertEqual(result.history, [99] * 10)
        self.assertEqual(len(objective.positions), 33)

    def test_zero_iterations_returns_best_initial_candidate(self):
        objective = RecordingObjective()
        result = run_gwo(objective, dimensions=8, num_wolves=5,
                         max_iterations=0, seed=4, output_path=None)
        self.assertEqual(len(objective.positions), 5)
        self.assertEqual(result.history, [])
        self.assertEqual(result.best_fitness, max(objective.scores))
        first_best = objective.scores.index(max(objective.scores))
        self.assertEqual(result.best_position, objective.positions[first_best])

    def test_objective_cannot_mutate_stored_positions(self):
        class MutatingObjective(RecordingObjective):
            def __call__(self, position):
                score = super().__call__(position)
                position[:] = [99] * len(position)
                return score

        objective = MutatingObjective()
        result = run_gwo(objective, dimensions=8, num_wolves=5,
                         max_iterations=10, seed=42, output_path=None)
        self.assertTrue(set(result.best_position) <= {0, 1})
        self.assertEqual(result.best_fitness, max(objective.scores))
        self.assertIn(result.best_position, objective.positions)


if __name__ == "__main__":
    unittest.main()
